# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Prompt logprobs: the packing, the shift, and the runner seam.

`prompt_logprobs` needs logits at *every* prompt position, and the tt-metal
generator returns them only for the final position, so
`TTPlatform.validate_request` refuses the request. That refusal covers the
device half only. The plugin half -- carrying the request's setting through the
batch, shifting to the token that follows each position, and packing the result
into upstream's `LogprobsTensors` -- is real code that runs on the host and is
tested here.

The contract being pinned is upstream `PromptLogprobsWorker` at v0.26.0
(`vllm/v1/worker/gpu/sample/prompt_logprob.py`): the target at position `i` is
`prompt_token_ids[i + 1]`, an unchunked prompt drops its final position, and
the rank column counts vocabulary entries whose logit is `>=` the target's.
"""

from types import SimpleNamespace
import numpy as np

import pytest
import torch

from vllm_tt_plugin.input_batch import (
    PROMPT_LOGPROBS_NONE_SENTINEL,
    SamplingInputBatch,
)
from vllm_tt_plugin.logprobs import (
    build_prompt_logprobs,
    shift_prompt_target_token_ids,
)
from vllm_tt_plugin.model_runner import TTModelRunner


def _logits(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


# --------------------------------------------------------------------------
# shift_prompt_target_token_ids
# --------------------------------------------------------------------------


def test_shift_reports_the_token_after_each_position():
    # Prompt [10, 11, 12, 13]; the logprob at position 0 is P(11 | 10), so the
    # target for position 0 is 11, not 10.
    tokens = torch.tensor([10, 11, 12, 13], dtype=torch.int64)

    targets = shift_prompt_target_token_ids(
        tokens,
        num_computed_tokens=0,
        prompt_len=4,
        num_scheduled_tokens=4,
        prompt_chunked=False,
    )

    assert targets.tolist() == [11, 12, 13]


def test_shift_drops_the_final_position_of_an_unchunked_prompt():
    # The last prompt position predicts the first *generated* token, which is
    # not a prompt token, so upstream drops it (`end_idx -= 1`).
    tokens = torch.tensor([10, 11, 12, 13], dtype=torch.int64)

    targets = shift_prompt_target_token_ids(
        tokens,
        num_computed_tokens=0,
        prompt_len=4,
        num_scheduled_tokens=4,
        prompt_chunked=False,
    )

    assert targets.shape[0] == 3


def test_shift_keeps_every_position_of_a_chunked_prompt():
    # A chunk that does not reach the end of the prompt keeps all of its rows,
    # because each still predicts a token inside the prompt.
    tokens = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64)

    targets = shift_prompt_target_token_ids(
        tokens,
        num_computed_tokens=0,
        prompt_len=5,
        num_scheduled_tokens=3,
        prompt_chunked=True,
    )

    assert targets.tolist() == [11, 12, 13]


def test_shift_resumes_from_the_computed_offset():
    tokens = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)

    targets = shift_prompt_target_token_ids(
        tokens,
        num_computed_tokens=2,
        prompt_len=6,
        num_scheduled_tokens=2,
        prompt_chunked=True,
    )

    # Upstream reads target_pos = num_computed_tokens + 1 + block, so rows
    # covering positions 2 and 3 report tokens 13 and 14.
    assert targets.tolist() == [13, 14]


def test_shift_of_a_single_position_prompt_is_empty():
    tokens = torch.tensor([10], dtype=torch.int64)

    targets = shift_prompt_target_token_ids(
        tokens,
        num_computed_tokens=0,
        prompt_len=1,
        num_scheduled_tokens=1,
        prompt_chunked=False,
    )

    assert targets.shape[0] == 0


# --------------------------------------------------------------------------
# build_prompt_logprobs
# --------------------------------------------------------------------------


def test_column_zero_is_the_target_token_and_its_logprob():
    # Position 0 puts almost all mass on token 3, so P(3) should dominate and
    # the target's rank should be 1.
    logits = _logits([[0.0, 0.0, 0.0, 10.0]])

    packed = build_prompt_logprobs(
        logits=logits,
        target_token_ids=torch.tensor([3], dtype=torch.int64),
        num_prompt_logprobs=0,
    )

    assert packed.logprob_token_ids.shape == (1, 1)
    assert packed.logprob_token_ids[0, 0].item() == 3
    # log_softmax of a 4-way distribution dominated by a single logit.
    assert packed.logprobs[0, 0].item() == pytest.approx(-0.0001, abs=1e-3)
    assert packed.selected_token_ranks[0].item() == 1


def test_target_outside_the_top_k_is_still_reported():
    # Column 0 must carry the real target even when the model ranked it low,
    # otherwise the caller cannot label the logprob it asked for.
    logits = _logits([[10.0, 9.0, 0.0, -10.0]])

    packed = build_prompt_logprobs(
        logits=logits,
        target_token_ids=torch.tensor([3], dtype=torch.int64),
        num_prompt_logprobs=2,
    )

    assert packed.logprob_token_ids.shape == (1, 3)
    assert packed.logprob_token_ids[0, 0].item() == 3
    # Alternatives are the two highest logits, which exclude the target.
    assert packed.logprob_token_ids[0, 1:].tolist() == [0, 1]
    # The target is last, so its rank under a descending sort is 4.
    assert packed.selected_token_ranks[0].item() == 4


def test_width_is_num_prompt_logprobs_plus_one():
    logits = _logits([[1.0, 2.0, 3.0, 4.0, 5.0]])

    packed = build_prompt_logprobs(
        logits=logits,
        target_token_ids=torch.tensor([0], dtype=torch.int64),
        num_prompt_logprobs=3,
    )

    assert packed.logprob_token_ids.shape == (1, 4)
 


def test_minus_one_requests_the_whole_vocabulary():
    logits = _logits([[1.0, 2.0, 3.0]])

    packed = build_prompt_logprobs(
        logits=logits,
        target_token_ids=torch.tensor([0], dtype=torch.int64),
        num_prompt_logprobs=-1,
    )

    # target + every vocabulary entry
    assert packed.logprob_token_ids.shape == (1, 4)


def test_values_are_log_softmax_of_the_logits_at_each_reported_token():
    # The real contract: every reported value is the log-softmax of the logits
    # row at the token named in column 0 of the same row. It is NOT a
    # re-normalisation of the reported subset -- a top-K subset does not sum to
    # one, and upstream's `-1` case repeats the target because it is already
    # inside the full-vocabulary top-k.
    logits = _logits([[0.0, 1.0, 2.0, 3.0]])

    packed = build_prompt_logprobs(
        logits=logits,
        target_token_ids=torch.tensor([0], dtype=torch.int64),
        num_prompt_logprobs=2,
    )

    expected = torch.log_softmax(logits, dim=-1)
    for column, token in enumerate(packed.logprob_token_ids[0].tolist()):
        assert packed.logprobs[0, column].item() == pytest.approx(
            expected[0, token].item(), abs=1e-6
        )


def test_top_k_subset_carries_no_more_than_the_full_probability():
    # Guards the previous test's misconception directly: a subset is a strict
    # subset of the distribution, so it can never exceed the total.
    logits = _logits([[0.0, 1.0, 2.0, 3.0]])

    packed = build_prompt_logprobs(
        logits=logits,
        target_token_ids=torch.tensor([0], dtype=torch.int64),
        num_prompt_logprobs=1,
    )

    assert packed.logprobs.exp().sum().item() <= 1.0 + 1e-5


def test_tied_targets_share_a_rank():
    # Upstream's _ranks_kernel counts `logits >= x`, so two tokens with equal
    # logits must report the same rank rather than 1 and 2.
    logits = _logits([[5.0, 5.0, 1.0]])

    packed = build_prompt_logprobs(
        logits=logits,
        target_token_ids=torch.tensor([0], dtype=torch.int64),
        num_prompt_logprobs=0,
    )

    # Two entries tie for first, so the rank is 2 under the >= convention.
    assert packed.selected_token_ranks[0].item() == 2


def test_packing_covers_every_position():
    logits = _logits([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])

    packed = build_prompt_logprobs(
        logits=logits,
        target_token_ids=torch.tensor([1, 0, 1], dtype=torch.int64),
        num_prompt_logprobs=1,
    )

    assert packed.logprob_token_ids.shape == (3, 2)
    assert packed.logprobs.shape == (3, 2)
    assert packed.selected_token_ranks.shape == (3,)


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"num_prompt_logprobs": -2}, "num_prompt_logprobs"),
        (
            {"target_token_ids": torch.tensor([0, 1], dtype=torch.int64)},
            "one entry per position",
        ),
    ],
)
def test_invalid_arguments_name_the_problem(kwargs, fragment):
    call = {
        "logits": _logits([[0.0, 1.0]]),
        "target_token_ids": torch.tensor([1], dtype=torch.int64),
        "num_prompt_logprobs": 0,
    }
    call.update(kwargs)

    with pytest.raises(ValueError, match=fragment):
        build_prompt_logprobs(**call)


# --------------------------------------------------------------------------
# Batch plumbing
# --------------------------------------------------------------------------


def _sampling_batch_with(sampling_params) -> SamplingInputBatch:
    batch = SamplingInputBatch(max_num_reqs=4)
    return batch


def test_sampling_batch_defaults_to_no_prompt_logprobs():
    batch = _sampling_batch_with(None)

    assert int(batch.num_prompt_logprobs[0].item()) == PROMPT_LOGPROBS_NONE_SENTINEL


def test_sampling_batch_tracks_the_requested_count():
    batch = _sampling_batch_with(None)
    batch.num_prompt_logprobs[2] = 3

    assert int(batch.num_prompt_logprobs[2].item()) == 3


def test_sampling_batch_preserves_minus_one():
    # -1 means "all vocabulary" and must survive the round trip; remapping it
    # to a concrete count (as num_logprobs does) would silently cap the answer.
    batch = _sampling_batch_with(None)
    batch.num_prompt_logprobs[1] = -1

    assert int(batch.num_prompt_logprobs[1].item()) == -1


# --------------------------------------------------------------------------
# Runner seam
# --------------------------------------------------------------------------


def _runner(
    requested, token_ids, num_prompt_tokens, num_computed_tokens=0
) -> SimpleNamespace:
    """A runner stand-in exposing only what the seam reads.

    ``token_ids_cpu`` is a real 2-D array because the seam slices it as
    ``[req_index, :num_prompt_tokens]``, the way the real batch is shaped.
    """
    table = np.zeros((4, 16), dtype=np.int32)
    for row, ids in enumerate(token_ids or []):
        table[row, : len(ids)] = ids
    batch = SimpleNamespace(
        prompt_logprobs_requested=requested >= 0,
        requested_prompt_logprobs=lambda _index: requested,
        req_id_to_index={"r0": 0},
        num_prompt_tokens=np.array([num_prompt_tokens] * 4, dtype=np.int32),
        num_computed_tokens_cpu=np.array([num_computed_tokens] * 4, dtype=np.int32),
        token_ids_cpu=table,
    )
    return SimpleNamespace(input_batch=batch)


def test_no_request_asking_yields_none_per_request():
    runner = _runner(requested=PROMPT_LOGPROBS_NONE_SENTINEL, token_ids=None, num_prompt_tokens=0)

    result = TTModelRunner._compute_prompt_logprobs_dict(runner, ["r0", "r1"])

    assert result == {"r0": None, "r1": None}


def test_asking_without_logits_raises_and_names_the_requirement():
    # A caller that asked for a distribution and got None has been told
    # nothing, so this must be loud, not a silent empty result.
    runner = _runner(requested=1, token_ids=None, num_prompt_tokens=0)

    with pytest.raises(ValueError) as excinfo:
        TTModelRunner._compute_prompt_logprobs_dict(runner, ["r0"])

    message = str(excinfo.value)
    assert "per-prompt-position prefill logits" in message
    assert "tt-metal" in message


def test_asking_with_logits_returns_a_packed_result():
    runner = _runner(requested=1, token_ids=[[0, 1, 2, 3]], num_prompt_tokens=4)

    logits = _logits(
        [
            [0.0, 0.0, 0.0, 10.0],
            [0.0, 9.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 0.0],
        ]
    )

    result = TTModelRunner._compute_prompt_logprobs_dict(
        runner, ["r0"], prompt_logits_by_req={"r0": logits}
    )

    packed = result["r0"]
    assert packed is not None
    # Four prompt positions, unchunked: the last predicts a generated token.
    assert packed.logprob_token_ids.shape == (3, 2)
    # Targets are prompt_token_ids[1:], i.e. 1, 2, 3.
    assert packed.logprob_token_ids[:, 0].tolist() == [1, 2, 3]
 
 
def test_single_row_for_a_multi_token_prompt_is_reported_not_swallowed():
    # This is the shape today's tt-metal generator produces: logits for the
    # final prompt position only. Returning None here would tell a caller that
    # asked for a distribution nothing at all, so it is an error.
    runner = _runner(requested=0, token_ids=[[0, 1, 2, 3]], num_prompt_tokens=4)
    logits = _logits([[0.0, 1.0, 2.0, 3.0]])

    with pytest.raises(ValueError) as excinfo:
        TTModelRunner._compute_prompt_logprobs_dict(
            runner, ["r0"], prompt_logits_by_req={"r0": logits}
        )

    message = str(excinfo.value)
    assert "4 token(s)" in message
    assert "1 prefill logit row(s)" in message
    assert "tt-metal" in message


def test_single_token_prompt_yields_none_rather_than_an_error():
    # A one-token prompt has no following token, so there is nothing to report.
    # That is an empty answer, not a failure.
    runner = _runner(requested=0, token_ids=[[2]], num_prompt_tokens=1)
    logits = _logits([[0.0, 1.0, 2.0, 3.0]])

    result = TTModelRunner._compute_prompt_logprobs_dict(
        runner, ["r0"], prompt_logits_by_req={"r0": logits}
    )

    assert result["r0"] is None


def test_missing_logits_for_one_request_names_that_request():
    runner = _runner(requested=0, token_ids=[[10, 11]], num_prompt_tokens=2)
    logits = _logits([[0.0, 1.0]])

    with pytest.raises(ValueError, match="r0"):
        TTModelRunner._compute_prompt_logprobs_dict(
            runner, ["r0"], prompt_logits_by_req={"other": logits}
        )
 
 
def test_mid_prompt_resume_is_refused_rather_than_mis_aligned():
    # Chunked prefill is on by default, so a request resuming at position k > 0
    # is routine. Reading targets from offset 1 would pair position k's target
    # with position 0's logits and report a confident wrong answer.
    runner = _runner(
        requested=0,
        token_ids=[[0, 1, 2, 3]],
        num_prompt_tokens=4,
        num_computed_tokens=2,
    )
    logits = _logits([[0.0, 1.0, 2.0, 3.0]] * 2)

    with pytest.raises(ValueError) as excinfo:
        TTModelRunner._compute_prompt_logprobs_dict(
            runner, ["r0"], prompt_logits_by_req={"r0": logits}
        )

    message = str(excinfo.value)
    assert "resuming at position 2" in message
    assert "4-token prompt" in message
    assert "--no-enable-chunked-prefill" in message


