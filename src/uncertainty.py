"""Monte Carlo Dropout uncertainty estimation, boundary analysis, and error calibration."""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def enable_mc_dropout(model: nn.Module) -> None:
    """Activate Dropout modules in training mode while strictly keeping BatchNorm in eval mode.
    
    This prevents running mean/variance statistics leakage during Monte Carlo stochastic inference.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def restore_model_eval(model: nn.Module) -> None:
    """Restore all modules to deterministic evaluation mode."""
    model.eval()


def compute_predictive_entropy(
    mean_prob: np.ndarray,
    log_base: str = "natural",
    eps: float = 1.0e-7
) -> np.ndarray:
    """Compute binary predictive entropy from mean probability map.
    
    H(p) = -p*log(p) - (1-p)*log(1-p)
    
    Args:
        mean_prob: Numpy array with probabilities in [0, 1].
        log_base: 'natural' (max value ln(2) ~ 0.6931) or 'binary' (max value 1.0).
        eps: Small epsilon for numerical clipping.
        
    Returns:
        Entropy map matching input shape.
    """
    p = np.clip(mean_prob, eps, 1.0 - eps)
    if log_base == "binary":
        entropy = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    else:
        entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    return entropy.astype(np.float32)


def run_mc_dropout_inference(
    model: nn.Module,
    image_tensor: torch.Tensor,
    mc_passes: int = 20,
    device: Optional[torch.device] = None,
    log_base: str = "natural",
    eps: float = 1.0e-7
) -> Dict[str, np.ndarray]:
    """Execute T stochastic forward passes with active dropout and compute uncertainty maps.
    
    Args:
        model: Trained PyTorch U-Net.
        image_tensor: Input tensor of shape [1, 3, H, W] or [3, H, W].
        mc_passes: Number of stochastic passes T (default: 20).
        device: Torch device (CPU or CUDA).
        log_base: 'natural' or 'binary'.
        eps: Epsilon for numerical stability.
        
    Returns:
        Dictionary containing raw numpy arrays:
            - 'mean_probability': [H, W] in [0, 1]
            - 'mc_variance': [H, W] in [0, 0.25]
            - 'predictive_entropy': [H, W]
            - 'stochastic_passes': [T, H, W]
    """
    if device is None:
        device = next(model.parameters()).device

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]

    image_tensor = image_tensor.to(device)

    # Enable MC Dropout while keeping BatchNorm in eval mode
    enable_mc_dropout(model)

    passes = []
    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)
            passes.append(probs.squeeze().cpu().numpy())

    # Restore deterministic evaluation mode
    restore_model_eval(model)

    stochastic_stack = np.stack(passes, axis=0)  # [T, H, W]
    mean_prob = np.mean(stochastic_stack, axis=0)  # [H, W]
    variance = np.var(stochastic_stack, axis=0)    # [H, W]
    entropy = compute_predictive_entropy(mean_prob, log_base=log_base, eps=eps)

    return {
        "mean_probability": mean_prob,
        "mc_variance": variance,
        "predictive_entropy": entropy,
        "stochastic_passes": stochastic_stack,
    }


def analyze_boundary_uncertainty(
    uncertainty_map: np.ndarray,
    gt_binary_mask: np.ndarray,
    band_radius: int = 5
) -> Dict[str, Any]:
    """Analyze uncertainty distribution across boundary band, interior, and background regions.
    
    Args:
        uncertainty_map: 2D numpy array [H, W] (entropy or variance).
        gt_binary_mask: 2D binary numpy array [H, W] with {0, 1}.
        band_radius: Dilation/erosion kernel size in pixels (default: 5).
        
    Returns:
        Dictionary with mean uncertainty per region, pixel counts, and region masks.
    """
    gt_u8 = (gt_binary_mask > 0).astype(np.uint8) * 255
    ksize = band_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    dilation = cv2.dilate(gt_u8, kernel)
    erosion = cv2.erode(gt_u8, kernel)

    # Region segmentations
    interior_mask = (erosion > 0)
    boundary_mask = (dilation > 0) & (~interior_mask)
    background_mask = (dilation == 0)

    mean_boundary = float(np.mean(uncertainty_map[boundary_mask])) if np.any(boundary_mask) else 0.0
    mean_interior = float(np.mean(uncertainty_map[interior_mask])) if np.any(interior_mask) else 0.0
    mean_background = float(np.mean(uncertainty_map[background_mask])) if np.any(background_mask) else 0.0

    return {
        "mean_boundary_uncertainty": mean_boundary,
        "mean_interior_uncertainty": mean_interior,
        "mean_background_uncertainty": mean_background,
        "boundary_pixels": int(np.sum(boundary_mask)),
        "interior_pixels": int(np.sum(interior_mask)),
        "background_pixels": int(np.sum(background_mask)),
        "boundary_mask": boundary_mask,
        "interior_mask": interior_mask,
        "background_mask": background_mask,
    }


def compute_uncertainty_error_bins(
    uncertainty_scores: np.ndarray,
    pixel_errors: np.ndarray,
    num_bins: int = 10
) -> Dict[str, Any]:
    """Compute error rates across equal-count deciles of uncertainty.
    
    Tests the error-uncertainty association hypothesis: do higher uncertainty deciles exhibit higher error rates?
    
    Args:
        uncertainty_scores: 1D array of flattened pixel uncertainty values.
        pixel_errors: 1D binary array of flattened pixel error values (0 or 1).
        num_bins: Number of bins (default: 10 deciles).
        
    Returns:
        Dictionary with bin edges, mean uncertainties, error rates, and descriptive AUROC.
    """
    u_flat = uncertainty_scores.ravel()
    e_flat = pixel_errors.ravel().astype(int)

    # Compute AUROC of error detection if both 0 and 1 exist in errors
    if len(np.unique(e_flat)) > 1:
        try:
            error_auroc = float(roc_auc_score(e_flat, u_flat))
        except Exception:
            error_auroc = float("nan")
    else:
        error_auroc = float("nan")

    # Quantile binning
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(u_flat, quantiles)

    # Ensure strictly increasing edges to avoid empty bins on flat distributions
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.linspace(u_flat.min(), u_flat.max() + 1e-6, num_bins + 1)

    bin_records = []
    actual_num_bins = len(bin_edges) - 1

    for i in range(actual_num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == actual_num_bins - 1:
            in_bin = (u_flat >= low) & (u_flat <= high)
        else:
            in_bin = (u_flat >= low) & (u_flat < high)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            error_count = int(np.sum(e_flat[in_bin]))
            error_rate = float(error_count / bin_count)
            mean_u = float(np.mean(u_flat[in_bin]))
        else:
            error_count = 0
            error_rate = 0.0
            mean_u = float((low + high) / 2.0)

        bin_records.append({
            "bin_index": i + 1,
            "uncertainty_min": float(low),
            "uncertainty_max": float(high),
            "mean_uncertainty": mean_u,
            "pixel_count": bin_count,
            "error_count": error_count,
            "error_rate": error_rate,
        })

    return {
        "bins": bin_records,
        "error_detection_auroc": error_auroc,
    }
