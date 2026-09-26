# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Speculative decoding: the contract, and the refusal that guards it.

`spec_decode.py` and `spec_accept.py` are merged and tested, but nothing in the
runtime imports them except each other and `tests/spec/`, and
`check_and_update_config` refuses `speculative_config`. So the contract exists
and the runtime cannot use it. The drafter half is in flight as open PR #131
(model drafter), #127 and #130, with #142 on top of #131; the int64-seed fix
#is #141. Those branches are large (the drafter branch is ~14.6k lines across 51
files, mostly device tests under tests/tt/spec) and are under review by their
authors, so this branch does not merge them — see
.deep-research/notes/gap-matrix.md.

The invariant worth a permanent test is not "the drafter is wired", which is
not true yet and is not this branch's job. It is: **the plugin must never
accept a speculative config it cannot serve.** An accepted
`speculative_config` with no reachable drafter would return unverified tokens,
which is the one outcome worse than a refusal.
"""

import ast
import pathlib

import pytest
from vllm.sampling_params import SamplingParams

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "vllm_tt_plugin"
SPEC_MODULES = ("spec_decode", "spec_accept")
RUNTIME_ENTRY = PLUGIN_ROOT / "model_runner.py"


def _imported_names(path: pathlib.Path) -> set[str]:
    """Every module name imported by `path`, at any nesting depth."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(alias.name for alias in node.names)
    return names


def _spec_modules_in_runtime_graph() -> set[str]:
    """Spec modules reachable from the runner, following plugin imports.

    Walks the import graph rather than grepping source text, so moving an import
    does not break the check and a runtime that stops importing the contract
    does not slip past it.
    """
    seen: set[str] = set()
    frontier = [RUNTIME_ENTRY]
    visited: set[pathlib.Path] = set()
    while frontier:
        path = frontier.pop()
        if path in visited or not path.exists():
            continue
        visited.add(path)
        for name in _imported_names(path):
            for spec in SPEC_MODULES:
                # Matches `from .spec_decode import X` and
                # `from vllm_tt_plugin.spec_decode import X` alike.
                if name.endswith(spec):
                    seen.add(spec)
            if name.startswith("vllm_tt_plugin."):
                candidate = PLUGIN_ROOT / (name.split(".", 1)[1] + ".py")
                if candidate.exists():
                    frontier.append(candidate)
    return seen


def test_drafter_is_not_yet_wired_so_the_config_must_stay_refused():
    """The invariant, stated as the pair that must never disagree.

    Either the runtime reaches the speculative contract and may accept a spec
    config, or it must refuse one. Accepting a config while the drafter is
    unreachable would return unverified tokens.
    """
    from types import SimpleNamespace

    from vllm_tt_plugin.feature_support import validate_speculative_decoding

    reachable = _spec_modules_in_runtime_graph()
    spec_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=4)
    )

    if reachable:
        # Drafter wired: refusing would be the bug this test exists to prevent.
        validate_speculative_decoding(object, spec_config)
    else:
        with pytest.raises(ValueError, match="speculative"):
            validate_speculative_decoding(object, spec_config)


def test_the_speculative_config_refusal_names_the_flags():
    """The refusal must tell the operator what to unset."""
    from types import SimpleNamespace

    from vllm_tt_plugin.feature_support import validate_speculative_decoding

    with pytest.raises(ValueError) as excinfo:
        validate_speculative_decoding(
            object,
            SimpleNamespace(
                speculative_config=SimpleNamespace(num_speculative_tokens=4)
            ),
        )
    message = str(excinfo.value)
    assert "num_speculative_tokens=4" in message
    assert "--speculative-config" in message
    assert "--speculative-model" in message


def test_absent_speculative_config_passes_the_gate():
    from types import SimpleNamespace

    from vllm_tt_plugin.feature_support import validate_speculative_decoding

    validate_speculative_decoding(object, SimpleNamespace(speculative_config=None))


@pytest.mark.parametrize("spec", SPEC_MODULES)
def test_spec_module_imports_without_a_tt_device(spec):
    """The contract must stay importable on a host with no Tenstorrent device."""
    module = __import__(f"vllm_tt_plugin.{spec}", fromlist=["__name__"])
    assert module.__name__ == f"vllm_tt_plugin.{spec}"


def test_spec_accept_uses_the_pinned_upstream_sampler():
    """The accept walk leans on upstream's top-k/top-p, so the coupling is real.

    `spec_accept.py` imports `apply_top_k_top_p_pytorch` from
    `vllm.v1.sample.ops.topk_topp_sampler`; if that symbol moves, the module
    breaks at import and the contract is no longer testable.
    """
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    assert callable(apply_top_k_top_p_pytorch)
    source = (PLUGIN_ROOT / "spec_accept.py").read_text()
    assert "apply_top_k_top_p_pytorch" in source


def test_prompt_logprobs_remains_refused_until_ttmetal_exposes_prefill_logits():
    """The other unimplemented feature, pinned so it cannot be forgotten.

    `model_runner.py` hard-codes `prompt_logprobs_dict` to None
    (lines 1950, 2477-2489) because no tt-metal generator exposes
    prompt-position logits. The paired tt-metal change is required first
    (AGENTS.md section 6).
    """
    from vllm_tt_plugin.platform import TTPlatform

    with pytest.raises(ValueError, match="prompt_logprobs"):
        TTPlatform.validate_request(
            {"prompt_token_ids": [1, 2, 3], "prompt_embeds": None},
            SamplingParams(max_tokens=8, prompt_logprobs=1),
        )
