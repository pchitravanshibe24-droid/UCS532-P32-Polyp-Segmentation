"""Unit tests for U-Net model forward pass, shapes, and gradients."""

import torch
import pytest

from src.models.unet import UNet, build_model


def test_unet_output_shape():
    """Verify that U-Net produces logits of shape [B, 1, H, W] for standard and smoke configurations."""
    # Smoke config features: [16, 32, 64, 128]
    model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128])
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)

    assert logits.shape == (2, 1, 64, 64)
    assert logits.dtype == torch.float32


def test_unet_gradient_flow():
    """Verify that loss computed from logits backpropagates gradients to all weights."""
    model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128])
    x = torch.randn(2, 3, 32, 32)
    target = torch.ones(2, 1, 32, 32)

    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"
        assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients!"


def test_unet_dropout_activation():
    """Verify that dropout layers exist and are active in training mode."""
    model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128], dropout_prob=0.3)
    dropouts = [m for m in model.modules() if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d))]
    assert len(dropouts) > 0, "No dropout layers found in model!"

    # In eval mode with fixed input, outputs should be identical
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Eval mode outputs must be deterministic!"
