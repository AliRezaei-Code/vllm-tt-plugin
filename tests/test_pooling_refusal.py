# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Pooling requests must be refused by a check that survives `python -O`.

`validate_runner` gates on the DECLARED `ModelConfig.runner` field, whose default
is `"auto"`, so it passes. Upstream then resolves the concrete runner type from
the architecture and can land on `"pooling"` (`ModelConfig._get_runner_type`).
That leaves two guards in `input_batch.py` as the last places a pooling request
can be refused on the TT backend -- and both were `assert`, which `python -O`
strips. Under `-O` a pooling request would proceed with `sampling_params is
None` and fail somewhere deeper, with a message that says nothing about pooling.

These tests assert the refusal, and assert it under `-O` in a subprocess, so the
hole cannot reopen.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_tt_plugin.input_batch import InputBatch, build_cached_request_state


def _new_req_data(**overrides):
    fields = dict(
        req_id="req-pool",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        pooling_params=SimpleNamespace(task="embed"),
        sampling_params=None,
        block_ids=(0,),
        num_computed_tokens=3,
        lora_request=None,
        prompt_embeds=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_new_pooling_request_is_refused_with_a_reason():
    with pytest.raises(ValueError) as excinfo:
        build_cached_request_state(_new_req_data())
    message = str(excinfo.value)
    assert "Pooling requests are not supported" in message, message
    assert "sampling_params" in message, message
    assert "runner" in message, message


def test_a_generation_request_is_still_accepted():
    """The refusal must not fire on a normal generation request."""
    state = build_cached_request_state(
        _new_req_data(
            pooling_params=None,
            sampling_params=SamplingParams(max_tokens=4),
        )
    )
    assert state.req_id == "req-pool"
    assert state.sampling_params is not None


def test_pooling_refusal_survives_python_dash_O():
    """The regression test: an `assert` here would vanish under -O."""
    script = (
        "import sys\n"
        "from types import SimpleNamespace\n"
        "from vllm_tt_plugin.input_batch import build_cached_request_state\n"
        "d = SimpleNamespace(\n"
        "    req_id='req-pool', prompt_token_ids=[1, 2, 3], mm_features=None,\n"
        "    pooling_params=SimpleNamespace(task='embed'), sampling_params=None,\n"
        "    block_ids=(0,), num_computed_tokens=3, lora_request=None,\n"
        "    prompt_embeds=None)\n"
        "try:\n"
        "    build_cached_request_state(d)\n"
        "except ValueError as exc:\n"
        "    print('REFUSED' if 'ooling' in str(exc) else 'WRONG ERROR')\n"
        "    sys.exit(0)\n"
        "print('NOT REFUSED')\n"
        "sys.exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        env={**__import__("os").environ, "PYTHONPATH": "ci/host-stubs"},
    )
    assert "REFUSED" in result.stdout, (
        f"stdout={result.stdout[-800:]!r} stderr={result.stderr[-800:]!r}"
    )


def test_no_assert_guards_the_pooling_paths():
    """Narrow to the two functions that guard pooling, not the whole module.

    `input_batch.py` still has pre-existing asserts elsewhere. AGENTS.md 2.4
    says those are debt and section 7 says not to convert them in an unrelated
    change, so this test pins only the paths this change owns.
    """
    import ast
    import pathlib

    import vllm_tt_plugin.input_batch as mod

    source = pathlib.Path(mod.__file__)
    tree = ast.parse(source.read_text())
    guarded = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(isinstance(sub, ast.Assert) for sub in ast.walk(node))
    }
    assert "build_cached_request_state" not in guarded, (
        "build_cached_request_state regressed to an assert; python -O strips it, "
        "so the pooling refusal would vanish"
    )
    assert "add_request" not in guarded, (
        "add_request regressed to an assert; python -O strips it, so an "
        "over-subscribed batch would write past every per-request buffer"
    )


def _batch():
    return InputBatch(
        max_num_reqs=2,
        max_model_len=32,
        max_num_batched_tokens=32,
        vocab_size=64,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )


def _state(req_id="r0"):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=(0,),
        num_computed_tokens=3,
        output_token_ids=[],
    )


def test_add_request_refuses_an_oversubscribed_row():
    """`req_index` past `max_num_reqs` must raise, not scribble past the buffers.

    `model_runner.py:813` supplies `req_index` from the slots freed by removals
    and falls back to `None` (meaning "append at num_reqs"), so adding more
    requests than the batch has room for lands here. The bound is on a
    caller-supplied integer that nothing upstream bounds, which is why it is a
    real invariant rather than a structurally impossible state.
    """
    batch = _batch()
    with pytest.raises(IndexError) as excinfo:
        batch.add_request(_state(), req_index=batch.max_num_reqs)
    message = str(excinfo.value)
    assert "max_num_reqs" in message, message
    assert "r0" in message, message


def test_add_request_row_guard_survives_python_dash_O():
    """The regression test: the `req_index` bound as an `assert` vanished here.

    Built with `object.__new__` because `InputBatch.__init__` needs a device
    mesh, which a bare host process cannot make. The guard reads one attribute,
    so this exercises the real method rather than a stand-in for it.
    """
    script = (
        "import sys\n"
        "from vllm.sampling_params import SamplingParams\n"
        "from vllm.v1.worker.gpu_input_batch import CachedRequestState\n"
        "from vllm_tt_plugin.input_batch import InputBatch\n"
        "request = CachedRequestState(\n"
        "    req_id='r0', prompt_token_ids=[1, 2, 3], mm_features=None,\n"
        "    sampling_params=SamplingParams(temperature=0.0), generator=None,\n"
        "    block_ids=(0,), num_computed_tokens=3, output_token_ids=[])\n"
        "batch = object.__new__(InputBatch)\n"
        "batch.max_num_reqs = 2\n"
        "try:\n"
        "    InputBatch.add_request(batch, request, req_index=2)\n"
        "except IndexError as exc:\n"
        "    print('REFUSED' if 'max_num_reqs' in str(exc) else 'WRONG ERROR')\n"
        "    sys.exit(0)\n"
        "print('NOT REFUSED')\n"
        "sys.exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        env={**__import__("os").environ, "PYTHONPATH": "ci/host-stubs"},
    )
    assert "REFUSED" in result.stdout, (
        f"stdout={result.stdout[-800:]!r} stderr={result.stderr[-800:]!r}"
    )
