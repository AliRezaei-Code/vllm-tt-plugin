# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""SamplingParams the TT backend honours, and the ones it must not drop.

The distinction that matters: controls routed to *host sampling* are honoured
(`compat_sampling_required` sends logprobs, bad_words, structured_outputs,
logit_bias, allowed_token_ids and min_tokens to the CPU), so they are not
refused. Controls with no path at any output width were previously accepted and
then ignored, which is worse than a refusal: a client gets output that does not
match what it asked for.
"""

import contextlib
from types import SimpleNamespace

import pytest
from vllm.sampling_params import RepetitionDetectionParams, SamplingParams

from vllm_tt_plugin.feature_support import (
    unsupported_block_output_params,
    unsupported_request_params,
)

# --- honoured: routed to host sampling, so never refused --------------------


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(max_tokens=8, logprobs=1),
        SamplingParams(max_tokens=8, min_p=0.05),
        SamplingParams(max_tokens=8, min_tokens=3),
        SamplingParams(max_tokens=8, bad_words=["forbidden"]),
        SamplingParams(max_tokens=8, logit_bias={100: -5.0}),
        SamplingParams(max_tokens=8, allowed_token_ids=[1, 2, 3]),
        SamplingParams(max_tokens=8, stop=["END"]),
        SamplingParams(max_tokens=8, seed=1234),
    ],
    ids=[
        "logprobs",
        "min_p",
        "min_tokens",
        "bad_words",
        "logit_bias",
        "allowed_token_ids",
        "stop",
        "seed",
    ],
)
def test_host_sampling_control_is_not_refused(params):
    """These reach the CPU sampler, so accepting them is correct."""
    assert unsupported_request_params(params) == []


# --- refused: no path at any output width ----------------------------------

SILENTLY_DROPPED = [
    ("thinking_token_budget", SamplingParams(max_tokens=8, thinking_token_budget=256)),
    (
        "repetition_detection",
        SamplingParams(
            max_tokens=8,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=4, min_count=3
            ),
        ),
    ),
    ("extra_args", SamplingParams(max_tokens=8, extra_args={"custom": True})),
]


@pytest.mark.parametrize(
    ("field", "params"), SILENTLY_DROPPED, ids=[n for n, _ in SILENTLY_DROPPED]
)
def test_dropped_control_is_refused_with_its_own_name(field, params):
    unsupported = unsupported_request_params(params)
    assert len(unsupported) == 1, unsupported
    assert field in unsupported[0]


def test_compat_sampling_covers_exactly_the_host_sampled_controls():
    """Keep the refusal list and the host-sampling list from drifting apart.

    Anything compat_sampling_required sends to the CPU is honoured, so it must
    not appear in the refusal list. This is the pairing that makes both true at
    once rather than by inspection.
    """
    from vllm_tt_plugin.platform import TTPlatform

    honoured = [
        SamplingParams(max_tokens=8, logprobs=1),
        SamplingParams(max_tokens=8, min_p=0.05),
        SamplingParams(max_tokens=8, min_tokens=3),
        SamplingParams(max_tokens=8, bad_words=["x"]),
        SamplingParams(max_tokens=8, logit_bias={1: -1.0}),
        SamplingParams(max_tokens=8, allowed_token_ids=[1]),
    ]
    for params in honoured:
        # logprobs=1 on 1 device still needs host sampling; multi-device does not
        # return top-k at all, so both shapes must stay unrefused.
        assert unsupported_request_params(params) == [], params
        assert isinstance(
            TTPlatform.compat_sampling_required(params, num_devices=1), bool
        )


# --- n > 1: upstream machinery, not a gap -----------------------------------


def test_n_greater_than_one_passes_token_at_a_time_validation():
    """The parent request is validated, then fanned out upstream.

    `AsyncLLM` calls `process_inputs` (which calls `validate_request`) with the
    parent's params, and only afterwards expands children each carrying `n = 1`
    (async_llm.py:388-397, parallel_sampling.py:73-74). The runner therefore
    never sees n != 1.
    """
    assert unsupported_request_params(SamplingParams(max_tokens=8, n=3)) == []


def test_n_greater_than_one_is_refused_for_block_output_models():
    """A block-output model owns its sampling and never sees the fan-out."""
    unsupported = unsupported_block_output_params(SamplingParams(max_tokens=8, n=3))
    assert any(line.startswith("n=") for line in unsupported), unsupported


# --- prompt_logprobs: still refused, and the reason is recorded -------------


def test_prompt_logprobs_is_still_refused_by_the_platform():
    """Not yet implemented; the refusal names the device.

    The runner hard-codes prompt_logprobs_dict to None
    (model_runner.py:1950, 2477-2489) because no tt-metal generator exposes
    prompt-position logits yet. A paired tt-metal change is required before the
    refusal can go.
    """
    from vllm_tt_plugin.platform import TTPlatform

    with pytest.raises(ValueError, match="prompt_logprobs"):
        TTPlatform.validate_request(
            {"prompt_token_ids": [1, 2, 3], "prompt_embeds": None},
            SamplingParams(max_tokens=8, prompt_logprobs=1),
        )


def test_max_logprobs_clamp_applies_and_warns_with_both_values(caplog):
    """The 32 -> 20 downgrade must be visible, not silent.

    The clamp lives near the top of `_apply_check_and_update_config`, so the
    config is mutated before anything later in that method can fail. Assert the
    side effect rather than the method's completion, which depends on model
    registration this host cannot perform.
    """
    import logging

    from vllm_tt_plugin.platform import TTPlatform

    config = SimpleNamespace(
        additional_config={},
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        model_config=SimpleNamespace(
            max_logprobs=32,
            runner="auto",
            enable_return_routed_experts=False,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        structured_outputs_config=SimpleNamespace(disable_any_whitespace=False),
    )
    # vLLM configures its logger tree with propagate=False, so records never
    # reach the root logger where caplog listens; attach its handler directly.
    # The plugin logger is vllm.tt.platform, not the module path.
    logger = logging.getLogger("vllm.tt.platform")
    logger.addHandler(caplog.handler)
    try:
        with (
            caplog.at_level(logging.WARNING, logger="vllm.tt.platform"),
            # The clamp already ran; later steps need a real registered model.
            contextlib.suppress(Exception),
        ):
            TTPlatform._apply_check_and_update_config(config)
    finally:
        logger.removeHandler(caplog.handler)

    assert config.model_config.max_logprobs == 20
    messages = {r.getMessage() for r in caplog.records}
    assert any("32" in m and "20" in m for m in messages), messages


def test_sampling_params_field_the_plugin_never_reads_is_documented_not_hidden():
    """Guard the boundary: these are the fields the refusal list does NOT cover.

    Listed so that adding one of them to the plugin is a deliberate act. They
    are response-shaping rather than sampling controls, and detokenization runs
    host-side, so they are recorded as findings rather than refused.
    """
    params = SamplingParams(max_tokens=8)
    for attr in ("output_kind", "skip_reading_prefix_cache", "detokenize"):
        assert hasattr(params, attr), attr
    assert unsupported_request_params(params) == []
