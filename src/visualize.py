"""Visualization engine for multi-panel diagnostic figures, uncertainty maps, and error calibration."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.utils import ensure_dir


def plot_multipanel_diagnostic(
    image_rgb: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    overlay_img: np.ndarray,
    mean_prob: np.ndarray,
    variance_map: np.ndarray,
    entropy_map: np.ndarray,
    output_path: Path,
    title: str = "Diagnostic Evaluation Panel"
) -> None:
    """Generate an 8-panel comprehensive diagnostic figure for a single test case.
    
    Panels:
        1. Input Image
        2. Ground Truth Mask
        3. Predicted Binary Mask
        4. Overlay
        5. Mean Probability Map
        6. Predictive Variance Map
        7. Predictive Entropy Map
        8. Pixel Error Map (Red=False Positive, Blue=False Negative)
    """
    has_gt = gt_mask is not None
    num_cols = 4
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))

    # 1. Input Image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("1. Input Endoscopic Image")
    axes[0, 0].axis("off")

    # 2. Ground Truth
    if has_gt:
        axes[0, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title("2. Ground Truth Mask")
    else:
        axes[0, 1].text(0.5, 0.5, "Ground Truth N/A", ha="center", va="center")
        axes[0, 1].set_title("2. Ground Truth Mask")
    axes[0, 1].axis("off")

    # 3. Predicted Mask
    axes[0, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("3. Predicted Mask")
    axes[0, 2].axis("off")

    # 4. Overlay
    axes[0, 3].imshow(overlay_img)
    axes[0, 3].set_title("4. Prediction Overlay")
    axes[0, 3].axis("off")

    # 5. Probability Map
    im5 = axes[1, 0].imshow(mean_prob, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("5. Mean Probability Map")
    axes[1, 0].axis("off")
    fig.colorbar(im5, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 6. Variance Map
    im6 = axes[1, 1].imshow(variance_map, cmap="plasma")
    axes[1, 1].set_title("6. Predictive Variance Map")
    axes[1, 1].axis("off")
    fig.colorbar(im6, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 7. Entropy Map
    im7 = axes[1, 2].imshow(entropy_map, cmap="inferno")
    axes[1, 2].set_title("7. Predictive Entropy Map")
    axes[1, 2].axis("off")
    fig.colorbar(im7, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # 8. Pixel Error Map
    if has_gt:
        # Error map: FP = red, FN = blue, Correct = gray
        error_vis = np.zeros((*pred_mask.shape[:2], 3), dtype=np.uint8)
        error_vis[:] = [230, 230, 230]  # Light gray background for correct pixels

        fp = (pred_mask == 1) & (gt_mask == 0)
        fn = (pred_mask == 0) & (gt_mask == 1)

        error_vis[fp] = [230, 30, 30]   # Red for False Positive
        error_vis[fn] = [30, 80, 230]   # Blue for False Negative

        axes[1, 3].imshow(error_vis)
        axes[1, 3].set_title("8. Error Map (Red=FP, Blue=FN)")
    else:
        axes[1, 3].text(0.5, 0.5, "Error Map N/A (No GT)", ha="center", va="center")
        axes[1, 3].set_title("8. Error Map")
    axes[1, 3].axis("off")

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_vs_uncertainty_bins(
    bin_records: List[Dict[str, Any]],
    output_path: Path,
    auroc: Optional[float] = None
) -> None:
    """Plot Error Rate versus Uncertainty Deciles."""
    indices = [b["bin_index"] for b in bin_records]
    rates = [b["error_rate"] * 100.0 for b in bin_records]
    means = [b["mean_uncertainty"] for b in bin_records]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "crimson"
    ax1.set_xlabel("Uncertainty Decile Bin (Low to High)")
    ax1.set_ylabel("Empirical Pixel Error Rate (%)", color=color)
    bars = ax1.bar(indices, rates, color=color, alpha=0.7, edgecolor="black", width=0.6)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, max(rates + [10.0]) * 1.2)

    # Add error rate text above bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

    # Line overlay for mean uncertainty
    ax2 = ax1.twinx()
    color2 = "navy"
    ax2.set_ylabel("Mean Predictive Entropy / Uncertainty", color=color2)
    ax2.plot(indices, means, color=color2, marker="o", lw=2, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color2)

    title = "Error Rate vs. Uncertainty Decile Bins"
    if auroc is not None and not np.isnan(auroc):
        title += f" (AUROC = {auroc:.3f})"
    plt.title(title, fontsize=12)

    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Error-uncertainty calibration plot saved to: {output_path}")


def plot_boundary_uncertainty_breakdown(
    boundary_results: Dict[str, Any],
    output_path: Path
) -> None:
    """Plot bar chart comparing uncertainty in boundary, interior, and background regions."""
    regions = ["Boundary Band", "Polyp Interior", "Background"]
    values = [
        boundary_results.get("mean_boundary_uncertainty", 0.0),
        boundary_results.get("mean_interior_uncertainty", 0.0),
        boundary_results.get("mean_background_uncertainty", 0.0),
    ]
    colors = ["darkorange", "forestgreen", "slategray"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(regions, values, color=colors, edgecolor="black", width=0.55)
    ax.set_ylabel("Mean Predictive Uncertainty")
    ax.set_title("Uncertainty Across Morphological Anatomical Regions")
    ax.set_ylim(0, max(values + [0.1]) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Boundary uncertainty plot saved to: {output_path}")
