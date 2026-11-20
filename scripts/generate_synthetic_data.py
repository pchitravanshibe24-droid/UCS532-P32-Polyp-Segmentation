"""Synthetic endoscopic data generator for fast CPU testing and smoke verification.

Generates simulated colonoscopy frames with synthetic mucosal textures, specular highlights,
and realistic deformed polyp masks, including negative controls (clean mucosa without polyps).
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.utils import ensure_dir, load_config, set_seed


def generate_synthetic_colon_frame(
    height: int = 256,
    width: int = 256,
    has_polyp: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize a single endoscopic image and binary ground truth mask.
    
    Args:
        height: Image height.
        width: Image width.
        has_polyp: Whether to include a polyp in this frame.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (image_rgb_uint8, mask_uint8_binary_0_255).
    """
    rng = np.random.RandomState(seed)
    
    # 1. Base mucosal background: reddish-pink gradient with radial falloff
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2.0, width / 2.0
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    vignette = 1.0 - 0.45 * (dist_from_center / max_dist)
    
    # Base pink/red mucosa tones
    base_r = rng.uniform(160, 210)
    base_g = rng.uniform(55, 95)
    base_b = rng.uniform(65, 110)
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:, :, 0] = base_r
    img[:, :, 1] = base_g
    img[:, :, 2] = base_b
    
    # Add low-frequency tissue texture folds
    num_folds = rng.randint(3, 7)
    for _ in range(num_folds):
        fx = rng.uniform(0.01, 0.05)
        fy = rng.uniform(0.01, 0.05)
        phase = rng.uniform(0, 2 * np.pi)
        wave = np.sin(x * fx + y * fy + phase) * rng.uniform(10, 25)
        img[:, :, 0] += wave
        img[:, :, 1] += wave * 0.4
        img[:, :, 2] += wave * 0.5
        
    # High-frequency mucosal texture
    noise = rng.normal(0, 7, (height, width, 3))
    img += noise
    
    # Apply vignette / lumen shading
    img *= vignette[:, :, np.newaxis]
    
    # 2. Specular reflections (moist mucosal surface)
    num_highlights = rng.randint(4, 10)
    for _ in range(num_highlights):
        hx = rng.randint(int(width * 0.15), int(width * 0.85))
        hy = rng.randint(int(height * 0.15), int(height * 0.85))
        hradius = rng.randint(2, 6)
        cv2.circle(img, (hx, hy), hradius, (255, 250, 245), -1)
        
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 3. Polyp simulation
    if has_polyp:
        # Random center in mucosal region
        px = rng.randint(int(width * 0.25), int(width * 0.75))
        py = rng.randint(int(height * 0.25), int(height * 0.75))
        axes = (rng.randint(int(width * 0.10), int(width * 0.22)),
                rng.randint(int(height * 0.08), int(height * 0.18)))
        angle = rng.uniform(0, 180)
        
        # Create polyp mask using an irregular ellipse
        poly_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(poly_mask, (px, py), axes, angle, 0, 360, 255, -1)
        
        # Deform the contour slightly with morphological jitter for organic shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        poly_mask = cv2.morphologyEx(poly_mask, cv2.MORPH_OPEN, kernel)
        
        mask = poly_mask
        
        # Modify the image inside the polyp region: slightly more vascular/red/raised
        polyp_indices = mask > 0
        img[polyp_indices, 0] = np.clip(img[polyp_indices, 0] * 1.15 + rng.uniform(10, 25), 0, 255)
        img[polyp_indices, 1] = np.clip(img[polyp_indices, 1] * 0.85 - rng.uniform(5, 15), 0, 255)
        img[polyp_indices, 2] = np.clip(img[polyp_indices, 2] * 0.90, 0, 255)
        
        # Polyp boundary shading (creates 3D bulbous effect)
        boundary_band = cv2.Canny(mask, 100, 200)
        boundary_band = cv2.dilate(boundary_band, kernel, iterations=1)
        img[boundary_band > 0] *= 0.75
        
    # Clip and convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask


def generate_synthetic_dataset(
    output_dir: Path,
    num_samples: int = 20,
    height: int = 64,
    width: int = 64,
    negative_sample_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Path, Path]:
    """Generate a batch of synthetic colonoscopy images and binary masks.
    
    Args:
        output_dir: Base directory to save synthetic dataset.
        num_samples: Total number of frames to generate.
        height: Image height.
        width: Image width.
        negative_sample_ratio: Fraction of frames without any polyp (empty mask).
        seed: Random seed.
        
    Returns:
        Tuple of (images_dir_path, masks_dir_path).
    """
    images_dir = ensure_dir(output_dir / "images")
    masks_dir = ensure_dir(output_dir / "masks")
    
    set_seed(seed)
    rng = np.random.RandomState(seed)
    
    print(f"Generating {num_samples} synthetic polyp samples in {output_dir}...")
    num_negatives = int(num_samples * negative_sample_ratio)
    
    for i in range(num_samples):
        # Determine whether this sample has a polyp
        has_polyp = (i >= num_negatives)
        sample_seed = seed + i * 13
        
        img, mask = generate_synthetic_colon_frame(
            height=height,
            width=width,
            has_polyp=has_polyp,
            seed=sample_seed
        )
        
        sample_name = f"synthetic_polyp_{i+1:04d}"
        img_path = images_dir / f"{sample_name}.jpg"
        mask_path = masks_dir / f"{sample_name}.png"
        
        # Save image as RGB JPG and mask as single-channel PNG
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), mask)
        
    print(f"Successfully generated {num_samples} synthetic pairs ({num_samples - num_negatives} positive, {num_negatives} negative).")
    return images_dir, masks_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic colonoscopy data for testing.")
    parser.add_argument("--config", type=str, default="configs/smoke_test.yaml", help="Path to config YAML.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of synthetic pairs to generate.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    data_root = Path(config["paths"]["data_root"])
    img_h, img_w = config["data"]["image_size"]
    seed = config["project"]["seed"]
    
    generate_synthetic_dataset(
        output_dir=data_root,
        num_samples=args.num_samples,
        height=img_h,
        width=img_w,
        seed=seed
    )
