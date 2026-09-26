# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Prefix caching: the capability gate, and what a host test cannot prove.

`supports_prefix_caching` is a model capability the plugin consumes. The gate is
the capability, not the flag, and that part is provable here — nothing tested it
before, because every existing scheduler test runs with caching off.

## What is deliberately NOT asserted here

The obvious proof — schedule an identical prompt twice, assert the second is
scheduled fewer tokens — is **not asserted here**, and the reason is recorded so
it is not re-derived. Measured on this host with a `supports_prefix_caching`
model, a `TTScheduler` built the way `tests/test_block_scheduler.py` builds one:

* the manager reports `enable_caching is True`;
* `Request.block_hashes` computes correctly (4 hashes for a 64-token prompt at
  block_size 16), so the block-hash contract is not the problem;
* yet `get_computed_blocks()` reports 0 cached tokens for a second identical
  prompt, and the second request is always scheduled for its full prompt length.

**The cause is not identified.** A first hypothesis — that `KVCacheConfig` with
`kv_cache_tensors=[]` yields an empty block pool — is wrong: the pool does have
`num_blocks` (18 for a 256-token model at block_size 16), and an earlier
`get_block_ids()` returning `([],)` was taken *after* the first request had been
freed, so its blocks had already returned to the pool. That observation was not
evidence of an empty pool.

So a host test currently cannot distinguish "prefix caching works" from "prefix
caching is broken" on this path, and asserting either would be a guess. The
honest options are a device test in `tests/tt`, or a host harness that supplies
real KV tensors; both are recorded as open work rather than faked here.
"""

import pathlib
from contextlib import contextmanager
from unittest.mock import patch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)

MAX_MODEL_LEN = 256
LOCAL_MODEL_CONFIG = pathlib.Path(__file__).parent / "model_configs" / "qwen2"


class _ModelWithPrefixCaching:
    """A tt-metal-style model class that declares the capability."""

    model_capabilities = {"supports_prefix_caching": True}


class _ModelWithoutPrefixCaching:
    """A tt-metal-style model class that does not."""

    model_capabilities: dict = {}


@contextmanager
def _stub_model_resolution(model_cls):
    """Resolve the stub instead of a real architecture, and keep platform state
    fresh. Mirrors tests/test_block_scheduler.py.
    """
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(model_cls, None),
        ),
    ):
        yield


def _resolve_with(model_cls, *, enable_prefix_caching: bool = True) -> VllmConfig:
    """Build a config through the real platform hook and hand it back."""
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=MAX_MODEL_LEN,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    with _stub_model_resolution(model_cls):
        return VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )


def test_declaring_the_capability_keeps_prefix_caching_enabled():
    """A model that declares `supports_prefix_caching` keeps the flag set."""
    config = _resolve_with(_ModelWithPrefixCaching)
    assert config.cache_config.enable_prefix_caching is True


def test_omitting_the_capability_disables_prefix_caching():
    """The gate: an undeclared capability turns the feature off.

    This is the AGENTS.md section 6 rule — absent keys default to False — and it
    is why a plugin can honour `enable_prefix_caching` without serving it. The
    platform logs "Prefix caching is not supported in TT backend ... disabling
    it" when it takes this branch.
    """
    config = _resolve_with(_ModelWithoutPrefixCaching)
    assert config.cache_config.enable_prefix_caching is False
