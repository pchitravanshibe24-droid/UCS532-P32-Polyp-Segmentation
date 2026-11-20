"""Unit tests for segmentation losses (BCE and Soft Dice)."""

import torch
import pytest

from src.losses import CombinedLoss, SoftDiceLoss


def test_soft_dice_loss_bounds_and_empty_masks():
    """Verify Dice loss behavior on perfect match, opposite match, and empty masks."""
    dice_fn = SoftDiceLoss(smooth=1.0e-6)

    # 1. Perfect match (all ones)
    targets = torch.ones(1, 1, 32, 32)
    probs_perfect = torch.ones(1, 1, 32, 32)
    loss_perfect = dice_fn(probs_perfect, targets)
    assert abs(loss_perfect.item()) < 1.0e-4, f"Perfect match loss should be ~0, got {loss_perfect.item()}"

    # 2. Total mismatch
    probs_worst = torch.zeros(1, 1, 32, 32)
    loss_worst = dice_fn(probs_worst, targets)
    assert abs(loss_worst.item() - 1.0) < 1.0e-4, f"Complete mismatch loss should be ~1, got {loss_worst.item()}"

    # 3. Both empty (healthy colon: no polyp present, predicted no polyp)
    empty_targets = torch.zeros(1, 1, 32, 32)
    empty_probs = torch.zeros(1, 1, 32, 32)
    loss_empty_match = dice_fn(empty_probs, empty_targets)
    assert abs(loss_empty_match.item()) < 1.0e-4, f"Empty target and prediction match should be ~0, got {loss_empty_match.item()}"


def test_combined_loss_backward():
    """Verify CombinedLoss computes finite loss and gradients backpropagate smoothly."""
    loss_fn = CombinedLoss(lambda_bce=0.5, lambda_dice=0.5)

    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 16, 16)).float()

    total_loss, components = loss_fn(logits, targets)

    assert torch.isfinite(total_loss)
    assert "loss_total" in components
    assert "loss_bce" in components
    assert "loss_dice" in components

    total_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
