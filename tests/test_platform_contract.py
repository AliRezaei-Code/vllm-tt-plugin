# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Platform members TTPlatform adds, and the ones it deliberately leaves alone.

Two groups. The first is what the plugin now overrides: quantization
verification and the punica wrapper. The second is the set the gap matrix marks
`no override` because the base class is already correct or the member is never
reached — those are asserted as *absent* so a future well-meaning override is a
deliberate act rather than an accident.
"""

import pytest

# NOTE: `vllm_tt_plugin.platform` is imported inside each test, not here. A
# module-scope import of it lands while `vllm.platforms` is still
# half-initialized by the plugin entry point, which raises "cannot import name
# 'current_platform'" when this file is collected first. AGENTS.md section 8
# calls this out; deferring the import is the documented fix.


def _TTPlatform():
    """Import late: see the NOTE above."""
    from vllm_tt_plugin.platform import TTPlatform

    return TTPlatform


def test_verify_quantization_refuses_every_method():
    """TT serves no weight-quantization method, so all of them are refused.

    `ModelConfig._verify_quantization` only calls this when a quantization is
    actually set, so an unquantized launch is unaffected.
    """
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

    assert QUANTIZATION_METHODS, "the pinned vLLM declares no quant methods"
    for method in sorted(QUANTIZATION_METHODS):
        with pytest.raises(ValueError) as excinfo:
            _TTPlatform().verify_quantization(method)
        message = str(excinfo.value)
        assert method in message
        assert "--quantization" in message


def test_supported_quantization_list_stays_empty():
    """The base list holds method names, so 'empty' is the honest TT answer.

    Setting it to dtype strings would be a category error, and setting it to
    any real method would be a lie.
    """
    assert _TTPlatform().supported_quantization == []


def test_get_punica_wrapper_names_the_reason():
    """A bare NotImplementedError from the base is not diagnosable."""
    with pytest.raises(NotImplementedError) as excinfo:
        _TTPlatform().get_punica_wrapper()
    message = str(excinfo.value)
    assert "LoRA" in message
    assert "TT backend" in message


@pytest.mark.parametrize(
    "name",
    [
        # Unreachable, so left alone on purpose. See the gap matrix.
        "get_attn_backend_cls",  # only attention/selector.py:185; no vLLM Attention
        "get_device_communicator_cls",  # only parallel_state.py:485; never at TP=1
        "get_cache_block_size_bytes",  # no caller anywhere in the 0.26.0 tree
        "sleep",
        "wake_up",
        "execute_dummy_batch",
    ],
)
def test_unreachable_platform_member_is_not_overridden(name):
    """Base stubs here are correct; an override would be unreachable code."""
    assert name not in _TTPlatform().__dict__, name


def test_nixl_members_keep_their_empty_base_defaults():
    """P/D is refused at config time, so the nixl defaults are the right answer."""
    assert _TTPlatform().get_nixl_supported_devices() == {}
    assert _TTPlatform().get_nixl_memory_type() is None


def test_sleep_mode_is_refused_by_upstream_not_by_the_plugin():
    """No override, because upstream already refuses --enable-sleep-mode.

    `Platform.is_sleep_mode_available` is True only for CUDA/ROCm/XPU, and
    `ModelConfig.__post_init__` raises for anything else. Assert the predicate
    the upstream guard reads, so a change to _TTPlatform()._enum cannot silently
    turn sleep mode on.
    """
    instance = _TTPlatform()()
    assert instance.is_sleep_mode_available() is False


def test_dtype_capability_predicates_stay_false():
    """TT advertises no fp8/mx support; the base defaults are correct."""
    assert _TTPlatform().supports_fp8() is False
    assert _TTPlatform().supports_mx() is False
    assert _TTPlatform().support_static_graph_mode() is False
    assert _TTPlatform().support_deep_gemm() is False
