"""Unit tests for paired data transformations."""

import numpy as np
import pytest
import torch

from src.data.transforms import PairedTransforms


def test_deterministic_transforms():
    """Verify that deterministic inference transforms output correct shapes and value ranges."""
    transforms = PairedTransforms(image_size=(64, 64), is_train=False)

    # Create dummy 120x80 image and mask
    raw_img = np.random.randint(0, 256, (80, 120, 3), dtype=np.uint8)
    raw_mask = np.zeros((80, 120), dtype=np.uint8)
    raw_mask[20:50, 30:70] = 255

    img_tensor, mask_tensor = transforms(raw_img, raw_mask)

    # Shape tests
    assert img_tensor.shape == (3, 64, 64)
    assert mask_tensor.shape == (1, 64, 64)

    # Type tests
    assert img_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.float32

    # Value range tests
    assert img_tensor.min() >= 0.0
    assert img_tensor.max() <= 1.0

    # Strictly binary mask values
    unique_vals = torch.unique(mask_tensor).tolist()
    assert all(val in [0.0, 1.0] for val in unique_vals)


def test_augmentation_mask_binarity():
    """Verify that geometric augmentations preserve strict binary mask values (nearest-neighbor)."""
    transforms = PairedTransforms(
        image_size=(64, 64),
        is_train=True,
        horizontal_flip_prob=0.8,
        vertical_flip_prob=0.8,
        rotation_limit=25.0,
        scale_limit=0.2,
        brightness_contrast_prob=0.8,
        color_jitter_prob=0.8,
    )

    raw_img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[30:70, 30:70] = 255

    for _ in range(10):
        img_tensor, mask_tensor = transforms(raw_img, raw_mask)
        unique_vals = torch.unique(mask_tensor).tolist()
        for v in unique_vals:
            assert v in (0.0, 1.0), f"Found non-binary value {v} in augmented mask!"


def test_empty_mask_preservation():
    """Verify that an empty mask (no polyp) remains all zeros after transform."""
    transforms = PairedTransforms(image_size=(64, 64), is_train=True)
    raw_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    empty_mask = np.zeros((50, 50), dtype=np.uint8)

    img_tensor, mask_tensor = transforms(raw_img, empty_mask)
    assert torch.sum(mask_tensor).item() == 0.0
