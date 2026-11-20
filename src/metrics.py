"""Segmentation metrics: Dice, IoU, Precision, Recall, F1, and Hausdorff Distance (HD95).

Implements strict handling of edge cases (empty masks, zero denominators, and macro-aggregation).
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def compute_confusion_matrix_elements(
    pred_binary: np.ndarray,
    target_binary: np.ndarray
) -> Tuple[int, int, int, int]:
    """Compute True Positives, False Positives, False Negatives, and True Negatives.
    
    Args:
        pred_binary: Binary predicted mask (values in {0, 1}).
        target_binary: Binary ground truth mask (values in {0, 1}).
        
    Returns:
        Tuple of (tp, fp, fn, tn).
    """
    pred_b = (pred_binary > 0).astype(bool)
    targ_b = (target_binary > 0).astype(bool)

    tp = int(np.sum(pred_b & targ_b))
    fp = int(np.sum(pred_b & (~targ_b)))
    fn = int(np.sum((~pred_b) & targ_b))
    tn = int(np.sum((~pred_b) & (~targ_b)))

    return tp, fp, fn, tn


def compute_hausdorff_95(
    pred_binary: np.ndarray,
    target_binary: np.ndarray
) -> float:
    """Compute symmetric 95th percentile Hausdorff Distance in pixel units.
    
    Args:
        pred_binary: Binary predicted mask.
        target_binary: Binary ground truth mask.
        
    Returns:
        HD95 distance in pixels. Returns 0.0 if both masks are empty, or np.nan if one is empty.
    """
    pred = np.asarray(pred_binary, dtype=bool)
    target = np.asarray(target_binary, dtype=bool)

    # Edge cases: empty masks
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")

    pred_edge = pred & ~binary_erosion(pred)
    target_edge = target & ~binary_erosion(target)
    forward = distance_transform_edt(~target_edge)[pred_edge]
    backward = distance_transform_edt(~pred_edge)[target_edge]
    return float(max(np.percentile(forward, 95), np.percentile(backward, 95)))


def compute_single_sample_metrics(
    pred_binary: Union[np.ndarray, torch.Tensor],
    target_binary: Union[np.ndarray, torch.Tensor],
    compute_hd95: bool = False
) -> Dict[str, float]:
    """Compute Dice, IoU, Precision, Recall, F1, and Pixel Error for a single binary image pair.
    
    Args:
        pred_binary: Array or tensor of shape [H, W] or [1, H, W] with values in {0, 1}.
        target_binary: Array or tensor of shape [H, W] or [1, H, W] with values in {0, 1}.
        compute_hd95: Whether to compute HD95 distance.
        
    Returns:
        Dictionary of computed metrics.
    """
    if isinstance(pred_binary, torch.Tensor):
        pred_arr = pred_binary.detach().cpu().squeeze().numpy()
    else:
        pred_arr = np.squeeze(pred_binary)

    if isinstance(target_binary, torch.Tensor):
        targ_arr = target_binary.detach().cpu().squeeze().numpy()
    else:
        targ_arr = np.squeeze(target_binary)

    pred_b = (pred_arr > 0).astype(np.uint8)
    targ_b = (targ_arr > 0).astype(np.uint8)
    if pred_b.shape != targ_b.shape or pred_b.size == 0:
        raise ValueError("Prediction and target must have the same nonempty shape.")

    tp, fp, fn, tn = compute_confusion_matrix_elements(pred_b, targ_b)
    total_pixels = tp + fp + fn + tn

    # Edge case 1: Both ground truth and prediction are empty (correctly predicted no polyp)
    if tp == 0 and fp == 0 and fn == 0:
        metrics = {
            "dice": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "pixel_error": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": tn,
        }
    # Edge case 2: Target is empty, but prediction has false positives
    elif (tp + fn) == 0 and fp > 0:
        metrics = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pixel_error": float(fp / total_pixels) if total_pixels > 0 else 0.0,
            "tp": 0, "fp": fp, "fn": 0, "tn": tn,
        }
    # Edge case 3: Target has polyp, but prediction is empty (false negatives only)
    elif (tp + fp) == 0 and fn > 0:
        metrics = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pixel_error": float(fn / total_pixels) if total_pixels > 0 else 0.0,
            "tp": 0, "fp": 0, "fn": fn, "tn": tn,
        }
    else:
        dice = float(2.0 * tp / (2.0 * tp + fp + fn))
        iou = float(tp / (tp + fp + fn))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = dice  # Mathematically equivalent to Dice for binary classification
        pixel_error = float((fp + fn) / total_pixels) if total_pixels > 0 else 0.0

        metrics = {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pixel_error": pixel_error,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    if compute_hd95:
        metrics["hd95"] = compute_hausdorff_95(pred_b, targ_b)

    return metrics


def compute_batch_dice(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Fast vectorized batch Dice calculation for validation monitoring.
    
    Args:
        probs: Sigmoid probability tensor [B, 1, H, W].
        targets: Binary ground truth tensor [B, 1, H, W].
        threshold: Binarization threshold.
        
    Returns:
        Mean Dice score across the batch.
    """
    preds = (probs >= threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = torch.sum(preds_flat * targets_flat, dim=1)
    cardinality = torch.sum(preds_flat, dim=1) + torch.sum(targets_flat, dim=1)

    # Empty mask policy: if cardinality is 0 (both empty), Dice is 1.0
    both_empty = (cardinality == 0)
    dice_per_sample = torch.where(
        both_empty,
        torch.ones_like(intersection),
        (2.0 * intersection) / (cardinality + 1e-7)
    )

    return float(torch.mean(dice_per_sample).item())
