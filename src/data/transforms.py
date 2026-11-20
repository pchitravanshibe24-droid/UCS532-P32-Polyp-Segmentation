"""Data transformation and augmentation pipelines.

Implements paired geometric transforms and image-only photometric perturbations.
Guarantees nearest-neighbor interpolation and strict binarization for masks.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def image_to_tensor(rgb):
    return torch.from_numpy((rgb.astype(np.float32) / 255).transpose(2, 0, 1))


def resize_rgb(rgb, image_size):
    height, width = image_size
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)


def binarize_mask(mask, threshold=127):
    # Also accept lossless masks explicitly encoded as zero and one.
    return (mask > (0 if mask.max() <= 1 else threshold)).astype(np.uint8)


class PairedTransforms:
    """Paired image and mask transform pipeline with guaranteed nearest-neighbor mask interpolation."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        is_train: bool = False,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        rotation_limit: float = 15.0,
        scale_limit: float = 0.1,
        brightness_contrast_prob: float = 0.3,
        color_jitter_prob: float = 0.2,
        mask_threshold: int = 127
    ):
        self.image_size = tuple(image_size)  # (height, width)
        self.is_train = is_train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_limit = rotation_limit
        self.scale_limit = scale_limit
        self.brightness_contrast_prob = brightness_contrast_prob
        self.color_jitter_prob = color_jitter_prob
        self.mask_threshold = mask_threshold

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to image and mask.
        
        Args:
            image: uint8 RGB numpy array [H, W, 3].
            mask: uint8 grayscale numpy array [H, W].
            
        Returns:
            Tuple of:
                - image_tensor: torch.float32 tensor [3, H_target, W_target] in range [0.0, 1.0]
                - mask_tensor: torch.float32 tensor [1, H_target, W_target] with binary values {0.0, 1.0}
        """
        target_h, target_w = self.image_size
        
        # 1. Resize: bilinear for image, nearest-neighbor for mask
        image = resize_rgb(image, self.image_size)
        mask = binarize_mask(mask, self.mask_threshold)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # 2. Training Augmentations
        if self.is_train:
            # Random Horizontal Flip
            if np.random.rand() < self.horizontal_flip_prob:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
                
            # Random Vertical Flip
            if np.random.rand() < self.vertical_flip_prob:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
                
            # Random Affine (Rotation and Scale)
            if self.rotation_limit > 0 or self.scale_limit > 0:
                angle = np.random.uniform(-self.rotation_limit, self.rotation_limit)
                scale = np.random.uniform(1.0 - self.scale_limit, 1.0 + self.scale_limit)
                center = (target_w / 2.0, target_h / 2.0)
                matrix = cv2.getRotationMatrix2D(center, angle, scale)
                
                image = cv2.warpAffine(
                    image, matrix, (target_w, target_h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
                mask = cv2.warpAffine(
                    mask, matrix, (target_w, target_h),
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                
            # Photometric perturbations (IMAGE ONLY!)
            if np.random.rand() < self.brightness_contrast_prob:
                alpha = np.random.uniform(0.85, 1.15)  # Contrast
                beta = np.random.uniform(-20, 20)      # Brightness
                image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                
            if np.random.rand() < self.color_jitter_prob:
                # Mild channel shift
                shifts = np.random.uniform(-10, 10, size=3)
                image = np.clip(image.astype(np.float32) + shifts, 0, 255).astype(np.uint8)
                
        # 3. Ensure mask is strictly binary {0.0, 1.0}
        binary_mask = mask.astype(np.float32)
        
        # 4. Normalize image to [0.0, 1.0] and convert to Torch tensors
        mask_float = binary_mask[np.newaxis, :, :]                           # [1, H, W]
        
        image_tensor = image_to_tensor(image)
        mask_tensor = torch.from_numpy(mask_float).float()
        
        return image_tensor, mask_tensor


def get_train_transforms(config: Dict[str, Any]) -> PairedTransforms:
    """Build paired training transform pipeline from configuration."""
    aug_cfg = config.get("augmentation", {})
    return PairedTransforms(
        image_size=config["data"]["image_size"],
        is_train=aug_cfg.get("enabled", True),
        horizontal_flip_prob=aug_cfg.get("horizontal_flip_prob", 0.5),
        vertical_flip_prob=aug_cfg.get("vertical_flip_prob", 0.5),
        rotation_limit=aug_cfg.get("rotation_limit", 15.0),
        scale_limit=aug_cfg.get("scale_limit", 0.1),
        brightness_contrast_prob=aug_cfg.get("brightness_contrast_prob", 0.3),
        color_jitter_prob=aug_cfg.get("color_jitter_prob", 0.2),
        mask_threshold=config["data"].get("mask_threshold", 127)
    )


def get_inference_transforms(config: Dict[str, Any]) -> PairedTransforms:
    """Build deterministic validation and testing transform pipeline."""
    return PairedTransforms(
        image_size=config["data"]["image_size"],
        is_train=False,
        mask_threshold=config["data"].get("mask_threshold", 127)
    )
