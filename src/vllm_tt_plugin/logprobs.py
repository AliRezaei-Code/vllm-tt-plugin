# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import torch
from vllm.v1.outputs import LogprobsTensors


def build_logprobs_from_topk(
    top_k_logprobs: torch.Tensor,
    top_k_indices: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    max_num_logprobs: int,
) -> LogprobsTensors:
    """Build LogprobsTensors from device top-K logprobs.

    Device always computes top-32 logprobs sorted descending.
    This function trims to max_num_logprobs which is in range (0-20) to
    match the OpenAI API limit, then packs into LogprobsTensors format
    expected by the downstream vLLM pipeline.
    """
    sz = top_k_logprobs.shape[0]
    n = max_num_logprobs

    # Cast both tensors to int64 to avoid uint32/int32 promotion errors.
    if sampled_token_ids.dim() == 1:
        sampled_token_ids = sampled_token_ids.unsqueeze(-1)
    sampled_expanded = sampled_token_ids.to(torch.int64)
    # ttnn.sampling selects from the same fixed top-32 candidate set returned
    # here. If that device invariant changes, the sampled rank can be wrong.
    match_mask = top_k_indices.to(torch.int64) == sampled_expanded
    ranks = match_mask.int().argmax(dim=-1)

    if ranks.dim() < top_k_logprobs.dim():
        ranks = ranks.unsqueeze(-1)
    sampled_logprob = top_k_logprobs.gather(1, ranks.long())

    logprob_token_ids = torch.zeros(sz, n + 1, dtype=torch.int32)
    logprobs_values = torch.zeros(sz, n + 1, dtype=torch.float32)
    logprob_token_ids[:, 0] = sampled_token_ids.squeeze(-1)
    logprobs_values[:, 0] = sampled_logprob.squeeze(-1)
    logprob_token_ids[:, 1 : n + 1] = top_k_indices[:, :n].to(torch.int32)
    logprobs_values[:, 1 : n + 1] = top_k_logprobs[:, :n].to(torch.float32)

    selected_token_ranks = ranks.squeeze(-1).to(torch.int32)
    return LogprobsTensors(
        logprob_token_ids,
        logprobs_values,
        selected_token_ranks,
    )


def build_device_logprobs(
    tt_log_probs: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    sampled_token_ids: torch.Tensor,
    rows: torch.Tensor,
    max_num_logprobs: int,
) -> LogprobsTensors:
    """Pack logprobs for device-sampled tokens at ``rows``.

    ``sampled_token_ids`` is the already-selected ``[n]`` sampled tokens (one
    per row); ``rows`` indexes the device output's batch dimension. Covers the
    top-K device path (gpt-oss returns a sorted top-32 set) and the
    single-sampled-logprob path (one logprob per row).
    """
    n = sampled_token_ids.shape[0]
    if isinstance(tt_log_probs, tuple):
        top_k_logprobs, top_k_indices = tt_log_probs
        return build_logprobs_from_topk(
            top_k_logprobs=top_k_logprobs[rows],
            top_k_indices=top_k_indices[rows],
            sampled_token_ids=sampled_token_ids,
            max_num_logprobs=max_num_logprobs,
        )
    sampled_log_probs = tt_log_probs[rows].reshape(n)
    return LogprobsTensors(
        logprob_token_ids=sampled_token_ids.reshape(n, 1).to(torch.int32),
        logprobs=sampled_log_probs.reshape(n, 1).to(torch.float32),
        selected_token_ranks=torch.full((n,), -1, dtype=torch.int32),
    )


def build_prompt_logprobs(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    num_prompt_logprobs: int,
) -> LogprobsTensors:
    """Pack per-prompt-position logprobs into upstream's ``LogprobsTensors``.

    ``logits`` is ``[num_positions, vocab_size]``: the logits at each prompt
    position, already conditioned on that position's prefix.
    ``target_token_ids`` is ``[num_positions]`` and is *already shifted* -- the
    token whose probability is being reported at position ``i`` is
    ``prompt_token_ids[i + 1]``, because the logprob at position ``i`` is the
    probability of the token that *follows* it. The shift is the caller's
    job because it needs the full token table and the request's starting
    offset; see :func:`shift_prompt_target_token_ids`.

    Column 0 of the result is always the target token and its logprob, so the
    caller can report the real token even when the model ranked it outside the
    top-K. Columns 1.. are the top-K alternatives by logit.

    ``num_prompt_logprobs`` follows upstream: ``-1`` requests the whole
    vocabulary, any other non-negative value requests that many alternatives.

    This mirrors upstream ``compute_topk_scores`` (vllm/v1/worker/gpu/sample/
    logprob.py at v0.26.0), which is triton-backed and therefore device-only.
    The rank column is upstream's ``_ranks_kernel`` definition: the count of
    vocab entries whose logit is greater than or equal to the target's, so
    ties share a rank.
    """
    if logits.dim() != 2:
        raise ValueError(
            f"build_prompt_logprobs expects [num_positions, vocab_size] "
            f"logits, got shape {tuple(logits.shape)}"
        )
    if target_token_ids.dim() != 1 or target_token_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            f"target_token_ids must be 1-D with one entry per position: got "
            f"shape {tuple(target_token_ids.shape)} for "
            f"{logits.shape[0]} position(s)"
        )
    if num_prompt_logprobs < -1:
        raise ValueError(
            f"num_prompt_logprobs must be -1 (all vocabulary) or a "
            f"non-negative count, got {num_prompt_logprobs}"
        )

    num_positions, vocab_size = logits.shape
    target = target_token_ids.to(torch.int64)
    # A target outside the vocabulary would surface as an opaque torch
    # RuntimeError from the gather below, naming an index and nothing else.
    if target.numel() and (int(target.min()) < 0 or int(target.max()) >= vocab_size):
        raise ValueError(
            f"target_token_ids must lie in [0, {vocab_size}) to index a "
            f"vocabulary of size {vocab_size}, but the range is "
            f"[{int(target.min())}, {int(target.max())}]"
        )
    log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)

    requested = vocab_size if num_prompt_logprobs == -1 else num_prompt_logprobs
    if requested > 0:
        top_indices = torch.topk(logits, requested, dim=-1).indices
        token_ids = torch.cat((target.unsqueeze(-1), top_indices), dim=1)
    else:
        token_ids = target.unsqueeze(-1)

    values = log_probs.gather(-1, token_ids)

    # Upstream's _ranks_kernel: 1-based rank of the target under a descending
    # sort, with ties counted as greater-or-equal so tied tokens agree.
    target_logits = logits.gather(-1, target.unsqueeze(-1))
    ranks = (logits >= target_logits).sum(dim=-1).to(torch.int32)

    return LogprobsTensors(
        logprob_token_ids=token_ids.to(torch.int32),
        logprobs=values.to(torch.float32),
        selected_token_ranks=ranks,
    )


def shift_prompt_target_token_ids(
    token_ids: torch.Tensor,
    num_computed_tokens: int,
    prompt_len: int,
    num_scheduled_tokens: int,
    prompt_chunked: bool,
) -> torch.Tensor:
    """The target token id for each prompt position in this chunk.

    ``token_ids`` is the request's token table up to
    ``num_computed_tokens + num_scheduled_tokens``. Upstream's
    ``get_prompt_logprobs_token_ids`` reads position
    ``num_computed_tokens + 1 + i`` for the ``i``-th scheduled token, so the
    report is shifted one position into the future.

    When the prompt is not chunked the final scheduled position predicts the
    first generated token rather than a prompt token, so upstream drops it
    (``end_idx -= 1``). This helper applies the same rule and returns only the
    positions that have a real target.
    """
    length = num_scheduled_tokens - (0 if prompt_chunked else 1)
    if length <= 0:
        return token_ids.new_empty((0,), dtype=torch.int64)
    start = num_computed_tokens + 1
    end = min(start + length, prompt_len + 1, token_ids.shape[0])
    if end <= start:
        return token_ids.new_empty((0,), dtype=torch.int64)
    return token_ids[start:end].to(torch.int64)
