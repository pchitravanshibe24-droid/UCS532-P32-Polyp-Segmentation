"""Unit tests for Monte Carlo Dropout uncertainty calculations and boundary analysis."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.unet import UNet
from src.uncertainty import (
    analyze_boundary_uncertainty,
    compute_predictive_entropy,
    compute_uncertainty_error_bins,
    enable_mc_dropout,
    restore_model_eval,
    run_mc_dropout_inference,
)


def test_mc_dropout_batchnorm_state():
    """Verify that enable_mc_dropout keeps BatchNorm in eval mode while activating Dropout."""
    model = UNet(in_channels=3, out_channels=1, features=[16, 32], dropout_prob=0.3)
    enable_mc_dropout(model)

    # All BatchNorm layers must be in eval mode
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert not m.training, "BatchNorm2d must remain in eval mode during MC Dropout!"
        elif isinstance(m, (nn.Dropout, nn.Dropout2d)):
            assert m.training, "Dropout layers must be active in training mode during MC passes!"

    # Restore eval mode
    restore_model_eval(model)
    for m in model.modules():
        assert not m.training, "All modules must return to eval mode after restore_model_eval!"


def test_predictive_entropy_bounds():
    """Verify entropy values at extremes (p=0, p=1) and maximum (p=0.5)."""
    # Natural log base
    p_extreme = np.array([0.0, 1.0, 0.5])
    ent_nat = compute_predictive_entropy(p_extreme, log_base="natural")

    assert ent_nat[0] < 1.0e-3, f"Entropy at p=0 should be ~0, got {ent_nat[0]}"
    assert ent_nat[1] < 1.0e-3, f"Entropy at p=1 should be ~0, got {ent_nat[1]}"
    assert abs(ent_nat[2] - np.log(2)) < 1.0e-3, f"Max entropy at p=0.5 should be ln(2)~0.693, got {ent_nat[2]}"

    # Binary log base
    ent_bin = compute_predictive_entropy(p_extreme, log_base="binary")
    assert abs(ent_bin[2] - 1.0) < 1.0e-3, f"Max binary entropy at p=0.5 should be 1.0 bit, got {ent_bin[2]}"


def test_stochastic_forward_passes_and_variance():
    """Verify that MC passes produce stochastic variation and finite non-negative variance."""
    model = UNet(in_channels=3, out_channels=1, features=[16, 32], dropout_prob=0.5)
    x = torch.randn(1, 3, 32, 32)

    res = run_mc_dropout_inference(model, x, mc_passes=10, log_base="natural")

    mean_p = res["mean_probability"]
    var_p = res["mc_variance"]
    entropy = res["predictive_entropy"]
    passes = res["stochastic_passes"]

    assert passes.shape == (10, 32, 32)
    assert mean_p.shape == (32, 32)
    assert var_p.shape == (32, 32)
    assert entropy.shape == (32, 32)

    # Variance must be non-negative
    assert np.all(var_p >= 0.0)
    assert np.all(mean_p >= 0.0) and np.all(mean_p <= 1.0)

    # Passes must not all be identical due to active dropout
    diff = np.max(np.abs(passes[0] - passes[1]))
    assert diff > 0.0, "Stochastic passes with active dropout should not be identical!"


def test_boundary_analysis_disjoint_regions():
    """Verify boundary band, interior, and background masks are disjoint."""
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[15:35, 15:35] = 1  # 20x20 square polyp

    uncertainty = np.random.uniform(0, 1, (50, 50)).astype(np.float32)
    res = analyze_boundary_uncertainty(uncertainty, gt, band_radius=3)

    b_mask = res["boundary_mask"]
    i_mask = res["interior_mask"]
    bg_mask = res["background_mask"]

    # All 3 regions must be mutually disjoint
    assert not np.any(b_mask & i_mask)
    assert not np.any(b_mask & bg_mask)
    assert not np.any(i_mask & bg_mask)

    # Union covers entire image
    assert np.all(b_mask | i_mask | bg_mask)
    assert res["boundary_pixels"] > 0
    assert res["interior_pixels"] > 0
    assert res["background_pixels"] > 0


def test_uncertainty_error_binning():
    """Verify quantile binning produces requested bin count with valid error rates."""
    u = np.linspace(0.0, 1.0, 100)
    # Higher uncertainty has higher probability of error
    e = (u > 0.6).astype(int)

    bin_res = compute_uncertainty_error_bins(u, e, num_bins=5)
    bins = bin_res["bins"]

    assert len(bins) == 5
    for b in bins:
        assert 0.0 <= b["error_rate"] <= 1.0
        assert b["pixel_count"] > 0

    # Upper bins should have higher error rate than lowest bin
    assert bins[-1]["error_rate"] >= bins[0]["error_rate"]
    assert not np.isnan(bin_res["error_detection_auroc"])
