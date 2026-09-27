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


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("sleep", (1,), "level=1"),
        ("wake_up", (None,), "tags=None"),
    ],
)
def test_sleep_members_refuse_by_name_not_by_attribute_error(name, args, expected):
    """Reachable by string dispatch, so the failure must name itself.

    Neither member is declared on `WorkerBase`. Upstream reaches them by
    `Executor.collective_rpc("sleep", ...)` (`executor/abstract.py:323,343`),
    whose `method` parameter is typed `str | Callable[[WorkerBase], _R]`
    (`:155`), so the string resolves on the worker and a missing override
    surfaces as a bare `AttributeError` from inside a worker RPC loop.

    The serving path is closed — `ModelConfig.__post_init__` refuses
    `enable_sleep_mode` because it gates on
    `current_platform.is_sleep_mode_available()` (`model.py:546`), which is
    False for TT. That closes the *serving* question, not the dispatch one: the
    offline `LLM` API calls `llm.sleep()` / `llm.wake_up()` from
    `benchmarks/throughput.py`, so the members are still reachable and a
    refusal by name is what a user there should get.
    """
    assert name in TTWorker.__dict__, (
        f"TTWorker must answer {name} by name; upstream dispatches it by "
        f"string, so a missing method is an AttributeError in a worker loop"
    )
    worker = _worker(_Tiny())
    with pytest.raises(NotImplementedError) as excinfo:
        getattr(worker, name)(*args)
    message = str(excinfo.value)
    assert "TT backend" in message, message
    assert expected in message, message
    assert "--enable-sleep-mode" in message, message


def test_execute_dummy_batch_refuses_by_name_not_by_attribute_error():
    """Reachable on the DP path, so the failure must name itself.

    `execute_dummy_batch` is not on `WorkerBase`; upstream reaches it by
    `Executor.collective_rpc("execute_dummy_batch")` string dispatch, which
    lands on `WorkerWrapperBase.__getattr__`. Left unimplemented, a TT
    standard-DP rank that idles while a peer runs gets a bare `AttributeError`
    from inside a worker RPC loop. It raises `NotImplementedError` instead, and
    says which DP condition triggered it and what to do about it.

    The functional implementation is deliberately absent: the reference version
    is a real dummy forward sized by `model_runner.uniform_decode_query_len`
    whose job is holding DP ranks in collective lockstep, and a wrong one causes
    a DP hang rather than a clean error. Finding F3 in the gap matrix.
    """
    assert "execute_dummy_batch" in TTWorker.__dict__, (
        "TTWorker must answer execute_dummy_batch by name; upstream dispatches "
        "it by string, so a missing method is an AttributeError in a worker loop"
    )
    worker = _worker(_Tiny())
    with pytest.raises(NotImplementedError) as excinfo:
        worker.execute_dummy_batch()
    message = str(excinfo.value)
    assert "TT backend" in message, message
    assert "data-parallel" in message, message
    assert "--data-parallel-size 1" in message, message
