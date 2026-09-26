# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Proof for `_gather_multi_modal_inputs`, which had no host coverage.

The function is the plugin's whole multimodal contract and it is load-bearing:
it builds the per-request image tensors the tt-metal generator receives. With no
host test, nothing proves the alignment rule — that the returned lists follow
*persistent batch* order rather than new-request order — which is the one thing
that is easy to get subtly wrong and impossible to notice until a real image
request returns the wrong image.

Constructed with `TTModelRunner.__new__`, the pattern the rest of the suite uses
for this class, because a real runner needs a mesh device.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.model_runner import TTModelRunner


def _feature(modality="image", pixel_values=None, grid=None, data=...):
    """A MultiModalFeatureSpec stand-in."""
    if data is not ...:
        return SimpleNamespace(modality=modality, data=data)
    payload = {"pixel_values": SimpleNamespace(data=pixel_values)}
    if grid is not None:
        payload["image_grid_thw"] = SimpleNamespace(data=grid)
    return SimpleNamespace(modality=modality, data=payload)


def _runner(req_ids, request_features):
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids))
    runner.requests = {
        req_id: SimpleNamespace(mm_features=features)
        for req_id, features in zip(req_ids, request_features)
    }
    return runner


def test_gather_returns_persistent_batch_order_not_new_request_order():
    """The whole point: order follows input_batch.req_ids.

    Selecting a subset out of order must still return entries in batch order for
    the requested indices, because the model input tensors are built that way.
    """
    pv0, pv1, pv2 = torch.zeros(3), torch.ones(3), torch.full((3,), 2.0)
    req_ids = ["a", "b", "c"]
    runner = _runner(
        req_ids,
        [
            [_feature(pixel_values=pv0)],
            [_feature(pixel_values=pv1)],
            [_feature(pixel_values=pv2)],
        ],
    )

    gathered = runner._gather_multi_modal_inputs()

    assert [entry[0] for entry in gathered["pixel_values"]] == [pv0, pv1, pv2]
    # Same selection, same order: this is the regression the test exists for.
    subset = runner._gather_multi_modal_inputs([2, 0])
    assert [entry[0] for entry in subset["pixel_values"]] == [pv2, pv0]


def test_text_only_request_gives_none_in_both_keys():
    """A request with no features is None, not an empty list.

    An empty list would be read downstream as "a request with zero images",
    which is a different thing from "a request with no image inputs".
    """
    pv = torch.zeros(3)
    runner = _runner(["text", "image"], [None, [_feature(pixel_values=pv)]])

    gathered = runner._gather_multi_modal_inputs()

    assert gathered["pixel_values"][0] is None
    assert gathered["image_grid_thw"][0] is None
    assert gathered["pixel_values"][1] == [pv]
    assert set(gathered) == {"pixel_values", "image_grid_thw"}


def test_multiple_images_per_request_stay_aligned_with_each_other():
    """pixel_values[i] and image_grid_thw[i] must describe the same images."""
    a, b = torch.zeros(2), torch.ones(2)
    ga, gb = torch.tensor([1, 1, 3]), torch.tensor([1, 1, 5])
    runner = _runner(
        ["two"],
        [[_feature(pixel_values=a, grid=ga), _feature(pixel_values=b, grid=gb)]],
    )

    gathered = runner._gather_multi_modal_inputs()

    assert gathered["pixel_values"][0] == [a, b]
    assert [t.tolist() for t in gathered["image_grid_thw"][0]] == [
        [1, 1, 3],
        [1, 1, 5],
    ]


def test_feature_without_data_yields_none_not_a_crash():
    """A feature carrying no data must not raise mid-gather.

    The whole batch is dropped if one slot raises, so a None placeholder is
    what keeps the rest of the batch usable.
    """
    pv = torch.zeros(3)
    runner = _runner(
        ["empty", "real"], [[_feature(data=None)], [_feature(pixel_values=pv)]]
    )

    gathered = runner._gather_multi_modal_inputs()

    assert gathered["pixel_values"][0] == [None]
    assert gathered["image_grid_thw"][0] == [None]
    assert gathered["pixel_values"][1] == [pv]


def test_missing_grid_yields_none_for_that_feature_only():
    """image_grid_thw is optional per feature; pixel_values is not."""
    pv = torch.zeros(3)
    runner = _runner(["req"], [[_feature(pixel_values=pv)]])

    gathered = runner._gather_multi_modal_inputs()

    assert gathered["pixel_values"][0] == [pv]
    assert gathered["image_grid_thw"][0] == [None]


def test_non_image_modality_is_refused():
    """The image-only contract, which the docstring states and nothing enforced."""
    runner = _runner(
        ["video"], [[_feature(modality="video", pixel_values=torch.zeros(3))]]
    )

    with pytest.raises(NotImplementedError, match="images"):
        runner._gather_multi_modal_inputs()


def test_empty_batch_returns_empty_lists():
    runner = _runner([], [])

    gathered = runner._gather_multi_modal_inputs()

    assert gathered == {"pixel_values": [], "image_grid_thw": []}
