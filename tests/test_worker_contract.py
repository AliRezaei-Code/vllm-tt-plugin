# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The WorkerBase members TTWorker was missing, and what they now do.

`get_model` is the one that mattered: `WorkerBase.get_model` raised
`NotImplementedError`, which also broke `get_model_inspection` — called
unconditionally by `entrypoints/llm.py` through
`collective_rpc("get_model_inspection")` — and `apply_model`, called at
`llm.py:603`. The LoRA members are a different case: reachable only if a LoRA
config bypasses the config-time refusal, so they must still fail loudly rather
than anonymously.
"""

from types import SimpleNamespace

import pytest
from torch import nn

from vllm_tt_plugin.worker import TTWorker


def _worker(model=None):
    worker = TTWorker.__new__(TTWorker)
    worker.model_runner = SimpleNamespace(model=model)
    return worker


class _Tiny(nn.Module):
    """A stand-in for a tt-metal generator: any nn.Module will do."""


def test_get_model_returns_the_loaded_generator():
    model = _Tiny()
    assert _worker(model).get_model() is model


def test_get_model_before_load_raises_rather_than_attribute_error():
    """Upstream reaches get_model after load_model, but say so if it doesn't."""
    with pytest.raises(RuntimeError, match="load_model"):
        _worker(None).get_model()


def test_apply_model_works_through_get_model():
    """apply_model is WorkerBase's default fn(self.get_model()); both must work."""
    model = _Tiny()
    assert TTWorker.apply_model(_worker(model), type) is _Tiny


def test_get_model_inspection_works_through_get_model():
    """The unconditional collective_rpc path: llm.py:896 -> get_model_inspection.

    Call the inherited WorkerBase method, not format_model_inspection directly:
    the point is that WorkerBase.get_model_inspection reaches get_model at all.
    """
    rendered = TTWorker.get_model_inspection(_worker(_Tiny()))
    assert "_Tiny" in rendered


LORA_REFUSALS = [
    ("add_lora", lambda w: w.add_lora(SimpleNamespace(lora_int_id=7, name="a")), "7"),
    ("remove_lora", lambda w: w.remove_lora(3), "3"),
    ("pin_lora", lambda w: w.pin_lora(5), "5"),
    ("list_loras", lambda w: w.list_loras(), "LoRA"),
]


@pytest.mark.parametrize(
    ("name", "call", "expected"),
    [(n, c, e) for n, c, e in LORA_REFUSALS],
    ids=[n for n, _, _ in LORA_REFUSALS],
)
def test_lora_members_refuse_loudly_naming_the_id(name, call, expected):
    """A LoRA RPC that slips past the config gate must name the adapter."""
    with pytest.raises(NotImplementedError) as excinfo:
        call(_worker(_Tiny()))
    message = str(excinfo.value)
    assert "LoRA is not supported by the TT backend" in message
    assert expected in message


def test_no_lora_member_is_inherited_from_the_base():
    """Guard against a silent regression back to the base stubs.

    WorkerBase raises a bare `NotImplementedError` with no message, so a test
    asserting only "it raises" would pass either way. Assert the message.
    """
    worker = _worker(_Tiny())
    for call, expected in (
        (lambda: worker.add_lora(SimpleNamespace(lora_int_id=1)), "1"),
        (lambda: worker.remove_lora(1), "1"),
        (lambda: worker.pin_lora(1), "1"),
        (lambda: worker.list_loras(), "LoRA"),
    ):
        with pytest.raises(NotImplementedError) as excinfo:
            call()
        assert "TT backend" in str(excinfo.value), expected


def test_get_cache_block_size_bytes_is_not_implemented():
    """Deliberate: nothing in vLLM 0.26.0 calls it.

    `WorkerBase.get_cache_block_size_bytes` docstring says "Used in speculative
    decoding", but the only occurrence of the name in the whole 0.26.0 tree is
    that definition. Implementing it would add unreachable code.
    """
    assert "get_cache_block_size_bytes" not in TTWorker.__dict__


@pytest.mark.parametrize("name", ["sleep", "wake_up"])
def test_sleep_members_are_unreachable_on_tt(name):
    """Deliberate: upstream refuses --enable-sleep-mode before the worker exists.

    `ModelConfig.__post_init__` raises "Sleep mode is not supported on current
    platform." for any platform that is not CUDA/ROCm/XPU, and TT is OOT, so
    `sleep` and `wake_up` can never be reached. Unlike the members below, these
    have a config-time gate, so the question is closed rather than open.
    """
    assert name not in TTWorker.__dict__, name


def test_execute_dummy_batch_is_a_known_unimplemented_dp_member():
    """NOT unreachable, unlike the sleep members. A real, recorded gap.

    `execute_dummy_batch` is the data-parallelism surface upstream's plugin
    design doc names, and it is reached on DP deployments:
    `LLMEngine.has_unfinished_requests_dp` sets `should_execute_dummy_batch`
    when a rank has nothing unfinished while peers do (llm_engine.py:196-202),
    `step()` fires it (:297-299), and `core.py:2049-2055` fires it again from
    inside `run_busy_loop` -- the DP busy loop -- whenever a step executed
    nothing. Both land on `collective_rpc("execute_dummy_batch")` via
    `executor/abstract.py:249-250`, which reaches `WorkerWrapperBase.__getattr__`.

    So a TT standard-DP rank that idles while a peer runs gets an
    `AttributeError` from inside a worker RPC loop today. It is deliberately
    unimplemented: the GPU version is a real dummy forward sized by
    `model_runner.uniform_decode_query_len`, holding DP ranks in collective
    lockstep, and a wrong one produces a DP hang rather than a clean error.
    There is no Tenstorrent device here to validate a mesh forward against, so
    this test records the gap rather than pretending it is closed. Tracked as
    Finding F3 in .deep-research/notes/gap-matrix.md; it needs a paired
    tt-metal change and a device test.
    """
    assert "execute_dummy_batch" not in TTWorker.__dict__
