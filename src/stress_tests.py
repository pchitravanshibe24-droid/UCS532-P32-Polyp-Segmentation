"""Apply configured image-only corruptions to a fixed held-out checkpoint."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import PolypDataset
from src.data.transforms import binarize_mask
from src.inference import load_model_from_checkpoint, run_inference_on_image
from src.metrics import compute_single_sample_metrics
from src.utils import ensure_dir, get_device, load_config, set_seed


def apply_gaussian_blur(img_rgb, kernel_size=5, sigma=1.5):
    return cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), sigma)


def apply_brightness_shift(img_rgb, factor=1.0):
    return np.clip(img_rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img_rgb, std=15.0, seed=42):
    rng = np.random.default_rng(seed)
    return np.clip(img_rgb + rng.normal(0, std, img_rgb.shape), 0, 255).astype(np.uint8)


def apply_contrast_shift(img_rgb, factor=1.0):
    image = img_rgb.astype(np.float32)
    mean = image.mean(axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def apply_scale_resizing(img_rgb, scale=0.5):
    height, width = img_rgb.shape[:2]
    shrunk = cv2.resize(img_rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA)
    return cv2.resize(shrunk, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_jpeg_compression(img_rgb, quality=40):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    okay, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not okay:
        raise OSError("JPEG encoding failed.")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


CORRUPTIONS = {
    "gaussian_blur": apply_gaussian_blur, "brightness": apply_brightness_shift,
    "gaussian_noise": apply_gaussian_noise, "contrast": apply_contrast_shift,
    "scale": apply_scale_resizing, "jpeg_compression": apply_jpeg_compression,
}


def run_stress_testing(config_path, checkpoint_override=None, selected=None):
    config = load_config(config_path)
    torch.set_num_threads(config["project"].get("threads", 4))
    device = get_device(config["project"].get("device", "cpu"))
    paths = config["paths"]
    model, checkpoint = load_model_from_checkpoint(checkpoint_override or paths["best_checkpoint"], device, config)
    threshold = checkpoint.get("threshold", config["inference"].get("evaluation_threshold", 0.5))
    size = tuple(config["data"]["image_size"])
    mask_threshold = config["data"].get("mask_threshold", 127)
    dataset = PolypDataset(paths["test_split"])
    metrics = ensure_dir(paths["metrics_dir"])
    figures = ensure_dir(paths["figures_dir"])
    seed = config.get("stress_tests", {}).get("seed", config["project"]["seed"])
    set_seed(seed)
    factors = config["stress_tests"]["factors"]
    options = selected or [(factor, severity) for factor, severities in factors.items() for severity in severities]
    conditions = [("clean", None, {})] + [
        (f"{factor}/{severity}", CORRUPTIONS[factor], factors[factor][severity]) for factor, severity in options
    ]
    print(f"Stress test | seed {seed} | threshold {threshold} | test pairs {len(dataset)}")
    records = []
    for idx in range(len(dataset)):
        image, mask, _, _, _ = dataset.load_raw(idx)
        gt = binarize_mask(mask, mask_threshold)
        for condition, transform, parameters in conditions:
            kwargs = dict(parameters)
            if transform is apply_gaussian_noise:
                kwargs.setdefault("seed", seed + idx)
            changed = transform(image, **kwargs) if transform else image
            out = run_inference_on_image(model, changed, size, threshold, device=device)
            scores = compute_single_sample_metrics(out["binary_mask"], gt)
            uncertainty = out["uncertainty_map"]
            if uncertainty is None:
                uncertainty = out["entropy_map"]
            records.append({"case": f"case_{idx + 1:03d}", "condition": condition,
                            "dice": scores["dice"], "iou": scores["iou"],
                            "mean_uncertainty": float(uncertainty.mean())})

    frame = pd.DataFrame(records)
    clean = frame[frame["condition"] == "clean"].set_index("case")
    for column in ("dice", "iou", "mean_uncertainty"):
        frame[f"delta_{column}"] = frame[column] - frame["case"].map(clean[column])
    frame.to_csv(metrics / "stress_per_image.csv", index=False)
    grid = frame.groupby("condition", sort=False).agg(
        count=("case", "count"), mean_dice=("dice", "mean"), mean_iou=("iou", "mean"),
        mean_uncertainty=("mean_uncertainty", "mean"), delta_dice=("delta_dice", "mean"),
        delta_uncertainty=("delta_mean_uncertainty", "mean"),
    ).reset_index()
    grid.to_csv(metrics / "stress_grid.csv", index=False)
    print(grid.to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for axis, column, title in (
        (axes[0], "delta_dice", "Dice"), (axes[1], "delta_uncertainty", "Mean ambiguity"),
    ):
        subset = grid[grid["condition"] != "clean"]
        axis.barh(subset["condition"], subset[column], color="#34798a")
        axis.axvline(0, color="black", lw=0.7)
        axis.set_xlabel(f"Mean paired change in {title}")
    fig.savefig(figures / "stress_test_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    run_stress_testing(args.config, args.checkpoint)
