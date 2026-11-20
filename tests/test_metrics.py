"""Unit tests for segmentation metrics on hand-crafted toy masks and edge cases."""

import numpy as np
import pytest

from src.metrics import compute_hausdorff_95, compute_single_sample_metrics


def test_hd95_rejects_single_distant_outlier_as_maximum():
    target = np.zeros((40, 40), dtype=np.uint8)
    target[2:12, 2:12] = 1
    prediction = target.copy()
    prediction[35, 35] = 1
    assert compute_hausdorff_95(prediction, target) == 0.0


def test_perfect_overlap():
    """Verify metrics on a perfect segmentation match."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 1  # 4 pixels

    m = compute_single_sample_metrics(mask, mask)
    assert m["dice"] == 1.0
    assert m["iou"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["pixel_error"] == 0.0


def test_hand_calculated_partial_overlap():
    """Verify metrics against exact hand-computable formula.
    
    Target: 4 pixels at [0:2, 0:2]
    Pred:   2 pixels at [0:1, 0:2]
    TP = 2, FP = 0, FN = 2
    Precision = 2 / 2 = 1.0
    Recall = 2 / 4 = 0.5
    Dice = 2*2 / (4 + 2) = 4/6 = 0.666667
    IoU = 2 / (2 + 0 + 2) = 0.5
    Pixel error = (0 + 2) / 16 = 0.125
    """
    target = np.zeros((4, 4), dtype=np.uint8)
    target[0:2, 0:2] = 1

    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[0:1, 0:2] = 1

    m = compute_single_sample_metrics(pred, target)
    assert abs(m["dice"] - 4.0 / 6.0) < 1.0e-5
    assert abs(m["iou"] - 0.5) < 1.0e-5
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert abs(m["pixel_error"] - 2.0 / 16.0) < 1.0e-5


def test_zero_overlap():
    """Verify metrics when prediction and ground truth are completely disjoint."""
    target = np.zeros((4, 4), dtype=np.uint8)
    target[0:2, 0:2] = 1

    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[2:4, 2:4] = 1

    m = compute_single_sample_metrics(pred, target)
    assert m["dice"] == 0.0
    assert m["iou"] == 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["pixel_error"] == 8.0 / 16.0


def test_empty_mask_edge_cases():
    """Verify edge case handling when target, prediction, or both are empty."""
    empty = np.zeros((4, 4), dtype=np.uint8)
    non_empty = np.zeros((4, 4), dtype=np.uint8)
    non_empty[0:2, 0:2] = 1

    # Case 1: Both empty
    m_both = compute_single_sample_metrics(empty, empty)
    assert m_both["dice"] == 1.0
    assert m_both["iou"] == 1.0
    assert m_both["pixel_error"] == 0.0

    # Case 2: Target empty, pred non-empty (False Positives only)
    m_fp = compute_single_sample_metrics(non_empty, empty)
    assert m_fp["dice"] == 0.0
    assert m_fp["iou"] == 0.0
    assert m_fp["precision"] == 0.0
    assert m_fp["pixel_error"] == 4.0 / 16.0

    # Case 3: Target non-empty, pred empty (False Negatives only)
    m_fn = compute_single_sample_metrics(empty, non_empty)
    assert m_fn["dice"] == 0.0
    assert m_fn["iou"] == 0.0
    assert m_fn["recall"] == 0.0
    assert m_fn["pixel_error"] == 4.0 / 16.0
