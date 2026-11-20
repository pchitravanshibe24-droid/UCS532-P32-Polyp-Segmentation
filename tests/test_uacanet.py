"""UACA equations, spatial alignment, supervision and end-to-end inference."""

import numpy as np
import pytest
import torch

from src.inference import run_inference_on_image
from src.losses import BCEIoULoss
from src.models.uacanet import UACANet, aggregate_context, region_maps


def test_region_definitions_and_pixel_feature_alignment():
    probability = torch.tensor([[[[0, 0.25, 0.5, 0.75, 1]]]])
    regions = region_maps(probability)
    expected = torch.tensor([[[[0, 0, 0, 0.25, 0.5]],
                              [[0.5, 0.25, 0, 0, 0]],
                              [[0, 0.25, 0.5, 0.25, 0]]]])
    torch.testing.assert_close(regions, expected)
    torch.testing.assert_close(regions.sum(dim=1), torch.full((1, 1, 5), 0.5))
    features = torch.tensor([[[[1, 2, 3, 4, 5]], [[10, 20, 30, 40, 50]]]], dtype=torch.float)
    torch.testing.assert_close(aggregate_context(features, regions),
                               torch.tensor([[[3.5, 35], [1, 10], [3, 30]]]))


@pytest.mark.parametrize("uncertain_context", [True, False])
def test_uacanet_odd_shapes_and_gradients(uncertain_context):
    torch.set_num_threads(2)
    model = UACANet(8, 8, uncertain_context)
    result = model(torch.rand(2, 3, 35, 49))
    assert len(result.logits) == 4
    assert all(logits.shape == (2, 1, 35, 49) for logits in result.logits)
    assert result.uncertainty.shape == (2, 1, 35, 49)
    assert 0 <= result.uncertainty.min() <= result.uncertainty.max() <= 0.5
    for logits in result.logits:
        logits.retain_grad()
    loss, components = BCEIoULoss()(result, torch.zeros_like(result.logits[-1]))
    assert components["loss_total"] == pytest.approx(components["loss_bce"] + components["loss_iou"])
    loss.backward()
    assert all(logits.grad.abs().sum() > 0 for logits in result.logits)
    assert model.stem[0].weight.grad.abs().sum() > 0


def test_single_head_bce_plus_soft_iou():
    logits = torch.zeros((1, 1, 2, 2))
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    loss, _ = BCEIoULoss()(logits, target)
    assert loss.item() == pytest.approx(np.log(2) + 1 - 1.5 / 3.5)


def test_inference_uses_frozen_threshold_and_preserves_resolution():
    model = UACANet(8, 8)
    rgb = np.zeros((37, 57, 3), dtype=np.uint8)
    result = run_inference_on_image(model, rgb, (32, 48), 0.7)
    assert result["uncertainty_map"].shape == (37, 57)
    assert result["uncertainty_overlay"].shape == rgb.shape
    np.testing.assert_array_equal(result["binary_mask"], result["mean_probability"] >= 0.7)
    with pytest.raises(ValueError, match="MC dropout"):
        run_inference_on_image(model, rgb, (32, 48), mc_passes=2)
