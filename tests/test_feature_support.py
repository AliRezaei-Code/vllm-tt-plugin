# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Config-time refusals for upstream features the TT backend does not serve.

Each test asserts the *mechanism*: that the offending value and the CLI flag
the operator typed both appear in the refusal. Prose is deliberately not
asserted, so rewording a message does not break the suite while losing the
value or the flag does.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from vllm_tt_plugin.feature_support import (
    TT_SUPPORTED_CACHE_DTYPES,
    TT_SUPPORTED_RUNNERS,
    unsupported_block_output_params,
    unsupported_request_params,
    validate_all,
    verify_quantization,
)

# Sections the override helper knows how to patch.
_SECTIONS = {
    "speculative_config",
    "lora_config",
    "kv_transfer_config",
    "ec_transfer_config",
    "model_config",
    "cache_config",
    "parallel_config",
}


def _config(**overrides):
    """A config double with every field the validators read, all supported."""
    config = SimpleNamespace(
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        model_config=SimpleNamespace(
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
    )
    for key, value in overrides.items():
        section, _, attr = key.partition(".")
        if not attr:
            assert section in _SECTIONS, f"unknown config section {section!r}"
            setattr(config, section, value)
        else:
            setattr(getattr(config, section), attr, value)
    return config


# (id, overrides, value the message must quote, flag the message must name)
REFUSALS = [
    (
        "speculative_config",
        {"speculative_config": SimpleNamespace(num_speculative_tokens=4)},
        "num_speculative_tokens=4",
        "--speculative-config",
    ),
    (
        "tensor_parallel_size",
        {"parallel_config.tensor_parallel_size": 2},
        "tensor_parallel_size=2",
        "--tensor-parallel-size",
    ),
    (
        "pipeline_parallel_size",
        {"parallel_config.pipeline_parallel_size": 4},
        "pipeline_parallel_size=4",
        "--pipeline-parallel-size",
    ),
    (
        "decode_context_parallel_size",
        {"parallel_config.decode_context_parallel_size": 2},
        "decode_context_parallel_size=2",
        "--decode-context-parallel-size",
    ),
    (
        "prefill_context_parallel_size",
        {"parallel_config.prefill_context_parallel_size": 2},
        "prefill_context_parallel_size=2",
        "--prefill-context-parallel-size",
    ),
    (
        "enable_expert_parallel",
        {"parallel_config.enable_expert_parallel": True},
        "enable_expert_parallel=True",
        "--enable-expert-parallel",
    ),
    (
        "enable_eplb",
        {"parallel_config.enable_eplb": True},
        "enable_eplb=True",
        "--enable-eplb",
    ),
    (
        "lora_config",
        {"lora_config": SimpleNamespace(max_lora_rank=16)},
        "max_lora_rank=16",
        "--enable-lora",
    ),
    (
        "cache_dtype",
        {"cache_config.cache_dtype": "fp8"},
        "fp8",
        "--kv-cache-dtype",
    ),
    (
        "kv_transfer_config",
        {"kv_transfer_config": SimpleNamespace(kv_connector="NixlConnector")},
        "NixlConnector",
        "--kv-transfer-config",
    ),
    (
        "ec_transfer_config",
        {"ec_transfer_config": SimpleNamespace(connector="LMCacheConnector")},
        "LMCacheConnector",
        "--kv-transfer-config",
    ),
    (
        "runner",
        {"model_config.runner": "pooling"},
        "pooling",
        "--runner",
    ),
    (
        "enable_return_routed_experts",
        {"model_config.enable_return_routed_experts": True},
        "enable_return_routed_experts=True",
        "--enable-return-routed-experts",
    ),
]


@pytest.mark.parametrize(
    ("overrides", "value", "flag"),
    [(o, v, f) for _, o, v, f in REFUSALS],
    ids=[name for name, _, _, _ in REFUSALS],
)
def test_unserved_feature_is_refused_with_value_and_flag(overrides, value, flag):
    """Every refusal quotes the offending value and names the CLI flag."""
    with pytest.raises(ValueError) as excinfo:
        validate_all(object, _config(**overrides))
    message = str(excinfo.value)
    assert value in message, message
    assert flag in message, message


@pytest.mark.parametrize("cache_dtype", sorted(TT_SUPPORTED_CACHE_DTYPES), ids=str)
def test_supported_cache_dtype_is_accepted(cache_dtype):
    """The refusal is about the value, not about cache_config being present."""
    validate_all(object, _config(**{"cache_config.cache_dtype": cache_dtype}))


@pytest.mark.parametrize("runner", sorted(TT_SUPPORTED_RUNNERS), ids=str)
def test_supported_runner_is_accepted(runner):
    validate_all(object, _config(**{"model_config.runner": runner}))


def test_a_fully_supported_config_is_not_refused():
    """The gate must not fire on a config that asks for nothing unserved."""
    validate_all(object, _config())


def test_refusals_run_in_a_deterministic_order():
    """Two unserved features produce the same first refusal every time.

    Order matters because an operator fixes one problem per launch: reporting
    the same first refusal every run is what makes the sequence debuggable.
    """
    config = _config(
        **{
            "parallel_config.tensor_parallel_size": 2,
            "cache_config.cache_dtype": "fp8",
        }
    )
    messages = set()
    for _ in range(3):
        with pytest.raises(ValueError) as excinfo:
            validate_all(object, config)
        messages.add(str(excinfo.value))
    assert len(messages) == 1
    assert "--tensor-parallel-size" in messages.pop()


# Every CacheDType literal the pinned vLLM declares other than the three TT
# supports, so a future CacheDType addition shows up as an uncovered value
# rather than a silent acceptance.
UNSUPPORTED_CACHE_DTYPES = [
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
]


def test_unsupported_cache_dtype_list_matches_the_pinned_alias():
    """The hand-listed literals must still be the ones the pin declares."""
    import typing

    from vllm.config.cache import CacheDType

    declared = set(typing.get_args(CacheDType))
    assert declared == TT_SUPPORTED_CACHE_DTYPES | set(UNSUPPORTED_CACHE_DTYPES)


@pytest.mark.parametrize("cache_dtype", UNSUPPORTED_CACHE_DTYPES, ids=str)
def test_every_other_cache_dtype_literal_is_refused(cache_dtype):
    """Not just fp8: every device encoding TT cannot allocate is refused."""
    with pytest.raises(ValueError, match="kv-cache-dtype"):
        validate_all(object, _config(**{"cache_config.cache_dtype": cache_dtype}))


def test_verify_quantization_names_the_method_and_the_flag():
    """Quantization is refused at ModelConfig construction, before the gate."""
    with pytest.raises(ValueError) as excinfo:
        verify_quantization(object, "fp8")
    message = str(excinfo.value)
    assert "fp8" in message
    assert "--quantization" in message


def test_refusal_survives_python_dash_O():
    """A refusal that `python -O` removes is not a refusal.

    The three original refusals were `assert` statements, which -O strips. Run
    the real validator under -O in a subprocess and require the same ValueError.
    """
    script = (
        "import sys\n"
        "from types import SimpleNamespace\n"
        "from vllm_tt_plugin.feature_support import validate_all\n"
        "cfg = SimpleNamespace(\n"
        "    speculative_config=None,\n"
        "    lora_config=SimpleNamespace(max_lora_rank=8),\n"
        "    kv_transfer_config=None,\n"
        "    ec_transfer_config=None,\n"
        "    model_config=SimpleNamespace(\n"
        "        runner='auto', enable_return_routed_experts=False),\n"
        "    cache_config=SimpleNamespace(cache_dtype='auto'),\n"
        "    parallel_config=SimpleNamespace(\n"
        "        tensor_parallel_size=1, pipeline_parallel_size=1,\n"
        "        enable_expert_parallel=False, enable_eplb=False,\n"
        "        decode_context_parallel_size=1,\n"
        "        prefill_context_parallel_size=1),\n"
        ")\n"
        "try:\n"
        "    validate_all(object, cfg)\n"
        "except ValueError:\n"
        "    print('REFUSED')\n"
        "    sys.exit(0)\n"
        "print('NOT REFUSED')\n"
        "sys.exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "REFUSED" in result.stdout, (
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )


def test_block_output_refusal_is_a_superset_of_the_general_one():
    """A block-output model owns its sampling, so it refuses strictly more."""
    params = SamplingParams(
        max_tokens=8,
        n=2,
        logprobs=3,
        min_tokens=2,
        bad_words=["nope"],
    )
    general = unsupported_request_params(params)
    block = unsupported_block_output_params(params)
    assert set(general).issubset(set(block))
    # n is the block-only control: upstream never fans a block request out.
    assert not any(line.startswith("n=") for line in general)
    assert any(line.startswith("n=") for line in block)


def test_n_greater_than_one_is_not_refused_for_token_at_a_time_models():
    """n > 1 is upstream machinery, not a gap.

    Upstream validates the parent request and only then fans it out into
    independent n == 1 children (async_llm.py:388-397), so the runner never
    sees n != 1 and refusing it would break a request that already works.
    """
    assert unsupported_request_params(SamplingParams(max_tokens=8, n=4)) == []


@pytest.mark.parametrize(
    ("factory", "expected_field"),
    [
        (
            lambda: SamplingParams(max_tokens=4, thinking_token_budget=128),
            "thinking_token_budget",
        ),
        (
            lambda: SamplingParams(
                max_tokens=4,
                repetition_detection=SimpleNamespace(max_pattern_size=4, min_count=3),
            ),
            "repetition_detection",
        ),
        (lambda: SamplingParams(max_tokens=4, extra_args={"k": 1}), "extra_args"),
    ],
    ids=["thinking_token_budget", "repetition_detection", "extra_args"],
)
def test_silently_dropped_request_control_is_refused(factory, expected_field):
    """A control the backend ignores must not reach the sampler unnoticed."""
    unsupported = unsupported_request_params(factory())
    assert any(expected_field in line for line in unsupported), unsupported
    assert unsupported_request_params(SamplingParams(max_tokens=4)) == []
