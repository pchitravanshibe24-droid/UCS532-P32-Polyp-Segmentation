"""Unit tests for split generation determinism and data leakage prevention."""

import tempfile
from shutil import copyfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.create_splits import generate_splits


def test_exact_duplicate_removal_and_conflicting_masks(mock_dataset_for_splits):
    images, masks, splits = mock_dataset_for_splits
    copyfile(images / "case_000.jpg", images / "duplicate.jpg")
    copyfile(masks / "case_000.png", masks / "duplicate.png")
    result = generate_splits(images, masks, splits)
    assert sum(map(len, result.values())) == 10
    assert all("duplicate" not in split["pair_id"].tolist() for split in result.values())
    cv2.imwrite(str(masks / "duplicate.png"), np.zeros((50, 50), dtype=np.uint8))
    with pytest.raises(ValueError, match="Identical images have different masks"):
        generate_splits(images, masks, splits)


@pytest.fixture
def mock_dataset_for_splits():
    """Create a temporary dataset with 10 pairs for split testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        splits_dir = tmp_path / "splits"
        images_dir.mkdir()
        masks_dir.mkdir()

        for i in range(10):
            stem = f"case_{i:03d}"
            # Unique pixels to guarantee distinct hashes
            img = np.full((50, 50, 3), i * 20, dtype=np.uint8)
            mask = np.zeros((50, 50), dtype=np.uint8)
            mask[10:30, 10:30] = 255
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), img)
            cv2.imwrite(str(masks_dir / f"{stem}.png"), mask)

        yield images_dir, masks_dir, splits_dir


def test_splits_disjointness_and_ratios(mock_dataset_for_splits):
    """Verify that generated splits are strictly disjoint and allocate correct sample counts."""
    images_dir, masks_dir, splits_dir = mock_dataset_for_splits

    dfs = generate_splits(
        images_dir=images_dir,
        masks_dir=masks_dir,
        splits_dir=splits_dir,
        ratios=[0.60, 0.20, 0.20],
        seed=42,
    )

    train_df = dfs["train"]
    val_df = dfs["val"]
    test_df = dfs["test"]

    assert len(train_df) == 6
    assert len(val_df) == 2
    assert len(test_df) == 2

    train_ids = set(train_df["pair_id"])
    val_ids = set(val_df["pair_id"])
    test_ids = set(test_df["pair_id"])

    # Disjointness checks
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Total union should be all 10
    assert len(train_ids | val_ids | test_ids) == 10


def test_splits_determinism(mock_dataset_for_splits):
    """Verify that identical seed reproduces the exact same split assignments."""
    images_dir, masks_dir, splits_dir = mock_dataset_for_splits

    dfs1 = generate_splits(
        images_dir=images_dir,
        masks_dir=masks_dir,
        splits_dir=splits_dir / "run1",
        ratios=[0.70, 0.15, 0.15],
        seed=123,
    )

    dfs2 = generate_splits(
        images_dir=images_dir,
        masks_dir=masks_dir,
        splits_dir=splits_dir / "run2",
        ratios=[0.70, 0.15, 0.15],
        seed=123,
    )

    assert dfs1["train"]["pair_id"].tolist() == dfs2["train"]["pair_id"].tolist()
    assert dfs1["val"]["pair_id"].tolist() == dfs2["val"]["pair_id"].tolist()
    assert dfs1["test"]["pair_id"].tolist() == dfs2["test"]["pair_id"].tolist()
