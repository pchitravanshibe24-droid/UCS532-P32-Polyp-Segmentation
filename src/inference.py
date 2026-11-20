"""Shared inference for evaluation and the local demo."""

from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.transforms import image_to_tensor, resize_rgb
from src.models import SegmentationOutput, UACANet, build_model, final_logits
from src.uncertainty import compute_predictive_entropy, run_mc_dropout_inference


def load_model_from_checkpoint(checkpoint_path, device=None, config=None):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}. Run bash run_all.sh first.")
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    effective_config = checkpoint.get("config") or config
    if not effective_config:
        raise ValueError("Checkpoint has no model configuration.")
    if config and any(
        effective_config.get(section, {}).get(key) != config.get(section, {}).get(key)
        for section, key in (("model", "architecture"), ("data", "image_size"), ("data", "mask_threshold"))
    ):
        raise ValueError("Checkpoint architecture or preprocessing does not match the evaluation configuration.")
    model = build_model(effective_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def create_polyp_overlay(image_rgb, binary_mask, color=(255, 30, 30), alpha=0.45):
    opacity = (alpha * (binary_mask > 0))[..., None]
    return np.rint(image_rgb * (1 - opacity) + np.asarray(color) * opacity).astype(np.uint8)


def color_heatmap(values):
    intensity = np.asarray(np.clip(values, 0, 1) * 255, dtype=np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(intensity, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def uncertainty_overlay(image_rgb, uncertainty):
    opacity = np.clip(uncertainty, 0, 0.5)[..., None]
    heatmap = color_heatmap(2 * uncertainty)
    return np.rint(image_rgb * (1 - opacity) + heatmap * opacity).astype(np.uint8)


def run_inference_on_image(model, image_rgb, target_size=(256, 256), threshold=0.5,
                           mc_passes=1, device=None, log_base="natural"):
    """Return original-resolution probabilities, segmentation and uncertainty maps."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image.")
    if mc_passes < 1 or not 0 <= threshold <= 1:
        raise ValueError("Invalid inference threshold or pass count.")
    device = device or next(model.parameters()).device
    size = image_rgb.shape[1::-1]
    tensor = image_to_tensor(resize_rgb(image_rgb, target_size)).unsqueeze(0).to(device)
    model.eval()
    uncertainty = None
    if mc_passes > 1:
        if isinstance(model, UACANet) or not any(
            isinstance(layer, torch.nn.Dropout2d) and layer.p > 0 for layer in model.modules()
        ):
            raise ValueError("MC dropout requires a trained U-Net with active dropout layers.")
        result = run_mc_dropout_inference(model, tensor, mc_passes, device, log_base)
        probability = result["mean_probability"]
        variance = result["mc_variance"]
    else:
        with torch.inference_mode():
            output = model(tensor)
            probability = final_logits(output).sigmoid()[0, 0].cpu().numpy()
            if isinstance(output, SegmentationOutput):
                uncertainty = cv2.resize(output.uncertainty[0, 0].cpu().numpy(), size)
        variance = np.zeros_like(probability)
    probability = cv2.resize(probability, size)
    variance = cv2.resize(variance, size)
    entropy = compute_predictive_entropy(probability, log_base=log_base)
    binary = (probability >= threshold).astype(np.uint8)
    result = {
        "binary_mask": binary, "mean_probability": probability,
        "entropy_map": entropy, "variance_map": variance, "uncertainty_map": uncertainty,
        "overlay": create_polyp_overlay(image_rgb, binary),
        "metrics_summary": {
            "polyp_area_percentage": float(100 * binary.mean()),
            "mean_entropy": float(entropy.mean()), "max_entropy": float(entropy.max()),
            "mean_variance": float(variance.mean()), "max_variance": float(variance.max()),
            "mc_passes": mc_passes, "threshold": threshold,
        },
    }
    if uncertainty is not None:
        result["uncertainty_overlay"] = uncertainty_overlay(image_rgb, uncertainty)
    return result
