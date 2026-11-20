"""Strict image-mask pairing, integrity verification, and dataset auditing."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from src.data.transforms import binarize_mask


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_files_by_stem(directory: Path, allowed_exts: Set[str]) -> Dict[str, Path]:
    """Index files in a directory by their lowercased stem.
    
    Args:
        directory: Directory to search.
        allowed_exts: Set of allowed extensions (e.g. {'.jpg', '.png'}).
        
    Returns:
        Mapping from filename stem to absolute Path.
    """
    stem_map: Dict[str, Path] = {}
    if not directory.is_dir():
        return stem_map
        
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in allowed_exts:
            stem = p.stem.lower()
            if stem in stem_map:
                raise ValueError(f"Duplicate filename stem detected in {directory}: {p.name} and {stem_map[stem].name}")
            stem_map[stem] = p
            
    return stem_map


def audit_single_pair(
    image_path: Path,
    mask_path: Path,
    mask_threshold: int = 127
) -> Dict[str, Any]:
    """Inspect and audit a single image-mask pair.
    
    Verifies readability, spatial dimension compatibility, unique mask values,
    and foreground polyp area fraction.
    
    Args:
        image_path: Path to image file.
        mask_path: Path to mask file.
        mask_threshold: Threshold to convert mask to binary (default: 127).
        
    Returns:
        Audit dictionary with dimensions, unique values, and validity status.
    """
    pair_id = image_path.stem
    result: Dict[str, Any] = {
        "pair_id": pair_id,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "is_valid": False,
        "error": None,
    }
    
    # 1. Read Image
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            img_mode = img.mode
            img.load()
    except Exception as e:
        result["error"] = f"Unreadable image file {image_path}: {e}"
        return result
        
    # 2. Read Mask
    try:
        mask_cv = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_cv is None:
            raise ValueError("cv2.imread returned None")
    except Exception as e:
        result["error"] = f"Unreadable mask file {mask_path}: {e}"
        return result
        
    mask_height, mask_width = mask_cv.shape[:2]
    
    # 3. Spatial Alignment Check
    if (img_width, img_height) != (mask_width, mask_height):
        result["error"] = (
            f"Dimension mismatch for pair '{pair_id}': "
            f"Image is ({img_width}x{img_height}) while Mask is ({mask_width}x{mask_height})."
        )
        return result
        
    # 4. Mask Statistics
    unique_vals = [int(v) for v in np.unique(mask_cv)]
    binary_mask = binarize_mask(mask_cv, mask_threshold)
    fg_pixels = int(np.sum(binary_mask))
    total_pixels = mask_width * mask_height
    fg_ratio = float(fg_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    result.update({
        "is_valid": True,
        "width": img_width,
        "height": img_height,
        "image_mode": img_mode,
        "unique_mask_values": unique_vals,
        "foreground_pixels": fg_pixels,
        "foreground_ratio": fg_ratio,
        "is_empty_mask": (fg_pixels == 0),
    })
    return result


def find_image_mask_pairs(
    images_dir: Path,
    masks_dir: Path,
    mask_threshold: int = 127
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scan directories, pair images with masks by filename stem, and audit all pairs.
    
    Args:
        images_dir: Directory containing images.
        masks_dir: Directory containing masks.
        mask_threshold: Binarization threshold for masks.
        
    Returns:
        Tuple of (list_of_valid_pairs, audit_summary_dict).
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
        
    image_files = find_files_by_stem(images_dir, SUPPORTED_IMAGE_EXTS)
    mask_files = find_files_by_stem(masks_dir, SUPPORTED_IMAGE_EXTS)
    
    all_stems = sorted(set(image_files.keys()).union(set(mask_files.keys())))
    
    valid_pairs: List[Dict[str, Any]] = []
    unmatched_images: List[str] = []
    unmatched_masks: List[str] = []
    corrupt_or_mismatched: List[Dict[str, Any]] = []
    
    for stem in all_stems:
        has_img = stem in image_files
        has_mask = stem in mask_files
        
        if has_img and not has_mask:
            unmatched_images.append(str(image_files[stem]))
            continue
        if has_mask and not has_img:
            unmatched_masks.append(str(mask_files[stem]))
            continue
            
        # Both present: audit the pair
        audit_res = audit_single_pair(image_files[stem], mask_files[stem], mask_threshold)
        if audit_res["is_valid"]:
            valid_pairs.append(audit_res)
        else:
            corrupt_or_mismatched.append(audit_res)
            
    summary: Dict[str, Any] = {
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "total_images_discovered": len(image_files),
        "total_masks_discovered": len(mask_files),
        "valid_pairs_count": len(valid_pairs),
        "unmatched_images_count": len(unmatched_images),
        "unmatched_masks_count": len(unmatched_masks),
        "corrupt_or_mismatched_count": len(corrupt_or_mismatched),
        "unmatched_images": unmatched_images,
        "unmatched_masks": unmatched_masks,
        "errors": corrupt_or_mismatched,
    }
    
    return valid_pairs, summary


def audit_and_pair_dataset(
    images_dir: Path,
    masks_dir: Path,
    output_report_path: Optional[Path] = None,
    mask_threshold: int = 127
) -> List[Dict[str, Any]]:
    """High-level function to pair and audit dataset, saving report if requested.
    
    Raises:
        ValueError if no valid pairs exist or if unmatched pairs are detected.
    """
    valid_pairs, summary = find_image_mask_pairs(images_dir, masks_dir, mask_threshold)
    
    if output_report_path:
        output_path = Path(output_report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
    if summary["unmatched_images_count"] > 0 or summary["unmatched_masks_count"] > 0:
        print(f"Warning: Discovered {summary['unmatched_images_count']} unmatched images and {summary['unmatched_masks_count']} unmatched masks.")
        
    if len(valid_pairs) == 0:
        raise ValueError(
            f"No valid image-mask pairs found between {images_dir} and {masks_dir}. "
            "Please check file extensions and paths."
        )
        
    return valid_pairs
