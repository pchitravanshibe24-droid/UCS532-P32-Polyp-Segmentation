import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.pairing import find_image_mask_pairs
from src.utils import ensure_dir, load_config, set_seed


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for exact duplicate and integrity checks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_splits(
    images_dir: Path,
    masks_dir: Path,
    splits_dir: Path,
    ratios: List[float] = [0.70, 0.15, 0.15],
    seed: int = 42,
    mask_threshold: int = 127
) -> Dict[str, pd.DataFrame]:
    """Create reproducible, disjoint train/val/test splits with SHA-256 leakage verification.
    
    Args:
        images_dir: Images folder path.
        masks_dir: Masks folder path.
        splits_dir: Target folder to store CSV files.
        ratios: [train_ratio, val_ratio, test_ratio] summing to 1.0.
        seed: Random seed.
        mask_threshold: Binarization threshold.
        
    Returns:
        Dict of DataFrames: {"train": df, "val": df, "test": df}.
    """
    if len(ratios) != 3 or min(ratios) <= 0 or abs(sum(ratios) - 1) >= 1e-5:
        raise ValueError("Expected three positive split ratios summing to one.")
    
    set_seed(seed)
    valid_pairs, summary = find_image_mask_pairs(images_dir, masks_dir, mask_threshold)
    
    if not valid_pairs:
        raise ValueError("Cannot create splits: 0 valid image-mask pairs discovered.")
    if any(summary[key] for key in ("unmatched_images_count", "unmatched_masks_count", "corrupt_or_mismatched_count")):
        raise ValueError(f"Cannot create splits due to invalid or unmatched pairs: {summary}")
        
    print(f"Discovered {len(valid_pairs)} valid pairs. Computing file hashes for leakage checks...")
    
    # Enrich pairs with file hashes
    unique_pairs = {}
    for p in valid_pairs:
        p["image_hash"] = compute_file_sha256(Path(p["image_path"]))
        p["mask_hash"] = compute_file_sha256(Path(p["mask_path"]))
        existing = unique_pairs.setdefault(p["image_hash"], p)
        if existing["mask_hash"] != p["mask_hash"]:
            raise ValueError(f"Identical images have different masks: {existing['pair_id']} and {p['pair_id']}")
    duplicate_count = len(valid_pairs) - len(unique_pairs)
    valid_pairs = list(unique_pairs.values())
        
    # Shuffle deterministically
    indices = np.arange(len(valid_pairs))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    
    n_total = len(valid_pairs)
    n_train = int(round(ratios[0] * n_total))
    n_val = int(round(ratios[1] * n_total))
    # Test gets the remainder to guarantee all samples are allocated
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("The requested ratios produce an empty split.")
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_pairs = [valid_pairs[i] for i in train_indices]
    val_pairs = [valid_pairs[i] for i in val_indices]
    test_pairs = [valid_pairs[i] for i in test_indices]
    
    # 1. Check ID disjointness
    train_ids = {p["pair_id"] for p in train_pairs}
    val_ids = {p["pair_id"] for p in val_pairs}
    test_ids = {p["pair_id"] for p in test_pairs}
    
    assert train_ids.isdisjoint(val_ids), "CRITICAL: Train and Val splits overlap in IDs!"
    assert train_ids.isdisjoint(test_ids), "CRITICAL: Train and Test splits overlap in IDs!"
    assert val_ids.isdisjoint(test_ids), "CRITICAL: Val and Test splits overlap in IDs!"
    
    # Check exact file duplication across splits.
    train_hashes = {p["image_hash"] for p in train_pairs}
    val_hashes = {p["image_hash"] for p in val_pairs}
    test_hashes = {p["image_hash"] for p in test_pairs}
    
    hash_leakage = (train_hashes & val_hashes) | (train_hashes & test_hashes) | (val_hashes & test_hashes)
    if hash_leakage:
        raise RuntimeError("Identical image hashes across splits.")
    print(f"Split seed {seed}; removed {duplicate_count} exact duplicate pairs; zero image hashes overlap.")
        
    # Convert to DataFrames
    cols = ["pair_id", "image_path", "mask_path", "image_hash", "mask_hash", "width", "height", "foreground_ratio", "is_empty_mask"]
    dfs = {
        "train": pd.DataFrame(train_pairs)[cols],
        "val": pd.DataFrame(val_pairs)[cols],
        "test": pd.DataFrame(test_pairs)[cols],
    }
    
    ensure_dir(splits_dir)
    for name, df in dfs.items():
        csv_path = splits_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {name} split ({len(df)} samples) to: {csv_path}")
        
    # Save metadata
    meta = {
        "seed": seed,
        "ratios": ratios,
        "total_samples": n_total,
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "test_count": len(test_pairs),
        "disjoint_verified": True,
        "duplicate_hashes_detected": len(hash_leakage),
        "duplicate_pairs_removed": duplicate_count,
        "archive_pairs_audited": summary["valid_pairs_count"],
        "split_fingerprints": {
            name: hashlib.sha256(df[["pair_id", "image_hash", "mask_hash"]].to_csv(index=False, lineterminator="\n").encode()).hexdigest()
            for name, df in dfs.items()
        },
    }
    with open(splits_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    return dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic train/val/test splits.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--data-root", type=str, default=None, help="Optional data root override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    ratios = cfg["data"]["split_ratios"]
    splits_dir = Path(cfg["paths"]["splits_dir"])
    
    if args.data_root:
        root = Path(args.data_root)
        images_dir = root / "images"
        masks_dir = root / "masks"
    else:
        images_dir = Path(cfg["paths"]["images_dir"])
        masks_dir = Path(cfg["paths"]["masks_dir"])
        
    generate_splits(
        images_dir=images_dir,
        masks_dir=masks_dir,
        splits_dir=splits_dir,
        ratios=ratios,
        seed=seed,
        mask_threshold=cfg["data"].get("mask_threshold", 127)
    )
