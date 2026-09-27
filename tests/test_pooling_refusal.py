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

from vllm_tt_plugin.input_batch import build_cached_request_state


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
