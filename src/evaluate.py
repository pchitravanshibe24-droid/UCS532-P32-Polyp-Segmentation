"""Evaluate a frozen checkpoint on held-out images and save traceable diagnostics."""

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch

from src.data.dataset import PolypDataset
from src.data.transforms import binarize_mask
from src.inference import color_heatmap, load_model_from_checkpoint, run_inference_on_image
from src.metrics import compute_single_sample_metrics
from src.uncertainty import analyze_boundary_uncertainty, compute_uncertainty_error_bins
from src.utils import ensure_dir, get_device, load_config, set_seed

METRICS = ("dice", "iou", "precision", "recall", "f1", "mae")
REGIONS = ("boundary", "interior", "background")


def save_panels(path, panels):
    """Save labeled RGB panels with an identical scale and preserved aspect ratio."""
    labeled = []
    for label, image in panels:
        height, width = image.shape[:2]
        scale = 200 / max(height, width)
        thumbnail = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        if thumbnail.ndim == 2:
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2RGB)
        tile = np.zeros((230, 208, 3), dtype=np.uint8)
        y, x = 24 + (200 - thumbnail.shape[0]) // 2, (208 - thumbnail.shape[1]) // 2
        tile[y:y + thumbnail.shape[0], x:x + thumbnail.shape[1]] = thumbnail
        cv2.putText(tile, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(tile)
    image = np.concatenate(labeled, axis=1)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write {path}")


def evaluate_test_set(config_path, checkpoint_override=None, mc_passes_override=None):
    config = load_config(config_path)
    torch.set_num_threads(config["project"].get("threads", 4))
    seed = config["project"]["seed"]
    set_seed(seed)
    device = get_device(config["project"].get("device", "cpu"))
    paths = config["paths"]
    model, checkpoint = load_model_from_checkpoint(checkpoint_override or paths["best_checkpoint"], device, config)
    inference = config["inference"]
    threshold = checkpoint.get("threshold", inference.get("evaluation_threshold", 0.5))
    passes = mc_passes_override if mc_passes_override is not None else inference.get("mc_passes", 1)
    size = tuple(config["data"]["image_size"])
    mask_threshold = config["data"].get("mask_threshold", 127)
    dataset = PolypDataset(paths["test_split"])
    metrics_dir = ensure_dir(paths["metrics_dir"])
    figures_dir = ensure_dir(paths["figures_dir"])
    native_panels = ensure_dir(figures_dir / "test_panels")
    derived_panels = ensure_dir(figures_dir / "mask_panels")
    print(f"Test pairs: {len(dataset)} | Seed {seed} | threshold {threshold} | checkpoint epoch {checkpoint['epoch']}")

    rows, uncertainties, entropies, errors = [], [], [], []
    for idx in range(len(dataset)):
        rgb, mask, _, _, _ = dataset.load_raw(idx)
        gt = binarize_mask(mask, mask_threshold)
        result = run_inference_on_image(model, rgb, size, threshold, passes, device, inference.get("log_base", "natural"))
        pred, probability = result["binary_mask"], result["mean_probability"]
        uaca = result["uncertainty_map"]
        entropy = result["entropy_map"]
        ambiguity = uaca if uaca is not None else entropy
        case = f"case_{idx + 1:03d}"
        metrics = compute_single_sample_metrics(pred, gt)
        row = {
            "case": case, **metrics, "mae": float(np.abs(probability - gt).mean()),
            "gt_area": float(gt.mean()), "pred_area": float(pred.mean()),
            "mean_uncertainty": float(ambiguity.mean()), "mean_entropy": float(entropy.mean()),
        }
        grid = size[::-1]
        small_map = cv2.resize(ambiguity, grid)
        small_gt = cv2.resize(gt, grid, interpolation=cv2.INTER_NEAREST)
        small_pred = cv2.resize(pred, grid, interpolation=cv2.INTER_NEAREST)
        stats = analyze_boundary_uncertainty(small_map, small_gt, band_radius=2)
        for region in REGIONS:
            row[f"{region}_uncertainty"] = (
                stats[f"mean_{region}_uncertainty"] if stats[f"{region}_pixels"] else None
            )
        uncertainties.append(small_map.ravel())
        entropies.append(cv2.resize(entropy, grid).ravel())
        errors.append((small_pred != small_gt).ravel())
        rows.append(row)

        name, intensity = ("UACA uncertainty", 2 * uaca) if uaca is not None else ("Predictive entropy", entropy)
        panels = [
            ("Ground truth", gt * 255), ("Prediction", pred * 255),
            ("Probability", color_heatmap(probability)), (name, color_heatmap(intensity)),
        ]
        overlay = result.get("uncertainty_overlay", result["overlay"])
        save_panels(native_panels / f"{case}.png", [
            ("Original", rgb), *panels, ("Uncertainty overlay" if uaca is not None else "Mask overlay", overlay),
        ])
        save_panels(derived_panels / f"{case}.png", panels)

    frame = pd.DataFrame(rows)
    frame.to_csv(metrics_dir / "test_metrics.csv", index=False)
    pooled_errors = np.concatenate(errors)
    scores = [("uncertainty", np.concatenate(uncertainties)), ("entropy", np.concatenate(entropies))]
    summary = {
        "dataset_name": config["project"]["name"], "seed": seed, "device": str(device),
        "test_sample_count": len(frame), "threshold": threshold,
        "epoch": checkpoint["epoch"], "validation_dice": checkpoint["best_val_dice"],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "mc_passes": passes, "uncertainty_kind": "UACA" if uaca is not None else "predictive_entropy",
        "metrics_macro_mean": {metric: float(frame[metric].mean()) for metric in METRICS},
        "metrics_macro_std": {metric: float(frame[metric].std(ddof=0)) for metric in METRICS},
    }
    for name, values in scores:
        binned = compute_uncertainty_error_bins(values, pooled_errors)
        pd.DataFrame(binned["bins"]).to_csv(metrics_dir / f"{name}_bins.csv", index=False)
        auroc = binned["error_detection_auroc"]
        summary[f"{name}_error_auroc"] = auroc if math.isfinite(auroc) else None
    summary["region_uncertainty"] = {
        region: float(frame[f"{region}_uncertainty"].dropna().mean())
        if frame[f"{region}_uncertainty"].notna().any() else None
        for region in REGIONS
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(" | ".join(f"{metric} {summary['metrics_macro_mean'][metric]:.6f}" for metric in METRICS))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--mc-passes", type=int)
    args = parser.parse_args()
    evaluate_test_set(args.config, args.checkpoint, args.mc_passes)
