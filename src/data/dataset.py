"""PyTorch Dataset implementation for Polyp Segmentation."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data.transforms import binarize_mask, image_to_tensor


class PolypDataset(Dataset):
    """Dataset for loading paired colonoscopy images and binary polyp masks."""
    
    def __init__(
        self,
        split_source: Union[str, Path, List[Dict[str, Any]]],
        transforms: Optional[Callable[[np.ndarray, np.ndarray], Any]] = None,
        base_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize PolypDataset.
        
        Args:
            split_source: Path to split CSV file or list of pair dictionaries.
            transforms: Callable transform taking (image, mask) and returning (image_tensor, mask_tensor).
            base_dir: Optional base directory to resolve relative paths in CSV.
        """
        self.transforms = transforms
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        
        if isinstance(split_source, (str, Path)):
            csv_path = Path(split_source)
            if not csv_path.is_file():
                raise FileNotFoundError(f"Split CSV file not found: {csv_path}")
            df = pd.read_csv(csv_path)
            self.samples = df.to_dict("records")
        elif isinstance(split_source, list):
            self.samples = split_source
        else:
            raise TypeError(f"Unsupported split_source type: {type(split_source)}")
            
        if len(self.samples) == 0:
            raise ValueError("PolypDataset initialized with 0 samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute() and p.exists():
            return p
        # Check relative to base_dir
        candidate = self.base_dir / p
        if candidate.exists():
            return candidate
        # Check as given
        if p.exists():
            return p
        raise FileNotFoundError(f"Dataset file does not exist: {path_str} (checked {candidate} and {p})")

    def load_raw(self, idx):
        sample = self.samples[idx]
        pair_id = sample.get("pair_id", str(idx))
        
        image_path = self._resolve_path(sample["image_path"])
        mask_path = self._resolve_path(sample["mask_path"])
        
        # Load image (BGR to RGB)
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Failed to read image at {image_path}")
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Load mask (Grayscale)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise IOError(f"Failed to read mask at {mask_path}")
        if image_rgb.shape[:2] != mask_gray.shape:
            raise ValueError(f"Image and mask dimensions differ for {pair_id}.")
        return image_rgb, mask_gray, pair_id, image_path, mask_path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_rgb, mask_gray, pair_id, image_path, mask_path = self.load_raw(idx)
            
        # Apply transforms if provided
        if self.transforms:
            image_tensor, mask_tensor = self.transforms(image_rgb, mask_gray)
        else:
            # Fallback simple tensor conversion
            image_tensor = image_to_tensor(image_rgb)
            mask_tensor = torch.from_numpy(binarize_mask(mask_gray).astype(np.float32)).unsqueeze(0)
            
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "pair_id": pair_id,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
