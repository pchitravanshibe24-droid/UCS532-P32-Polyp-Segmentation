"""Unit tests for dataset pairing and integrity auditing."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.pairing import audit_single_pair, find_image_mask_pairs


@pytest.fixture
def temp_pairing_env():
    """Create a temporary directory structure with mock images and masks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        # 1. Valid matching pair (100x100)
        img1 = np.full((100, 100, 3), 180, dtype=np.uint8)
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:60, 20:60] = 255
        cv2.imwrite(str(images_dir / "case_001.jpg"), img1)
        cv2.imwrite(str(masks_dir / "case_001.png"), mask1)

        # 2. Unmatched image (no mask)
        img2 = np.full((100, 100, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(images_dir / "case_orphan_img.jpg"), img2)

        # 3. Unmatched mask (no image)
        mask3 = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(masks_dir / "case_orphan_mask.png"), mask3)

        # 4. Dimension mismatch pair (image 100x100, mask 120x80)
        img4 = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask4 = np.zeros((80, 120), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "case_mismatch.jpg"), img4)
        cv2.imwrite(str(masks_dir / "case_mismatch.png"), mask4)

        yield images_dir, masks_dir


def test_pairing_discovery_and_classification(temp_pairing_env):
    """Test that pairing identifies matched, unmatched, and mismatched pairs."""
    images_dir, masks_dir = temp_pairing_env
    valid_pairs, summary = find_image_mask_pairs(images_dir, masks_dir)

    assert summary["total_images_discovered"] == 3
    assert summary["total_masks_discovered"] == 3
    assert summary["valid_pairs_count"] == 1
    assert valid_pairs[0]["pair_id"] == "case_001"
    assert valid_pairs[0]["width"] == 100
    assert valid_pairs[0]["height"] == 100
    assert valid_pairs[0]["foreground_pixels"] == 1600

    # Check unmatched
    assert summary["unmatched_images_count"] == 1
    assert "case_orphan_img" in summary["unmatched_images"][0]
    assert summary["unmatched_masks_count"] == 1
    assert "case_orphan_mask" in summary["unmatched_masks"][0]

    # Check corrupt/mismatched
    assert summary["corrupt_or_mismatched_count"] == 1
    assert "Dimension mismatch" in summary["errors"][0]["error"]


def test_empty_mask_flagging():
    """Verify that empty masks are correctly flagged without failing validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_p = tmp_path / "img.jpg"
        mask_p = tmp_path / "mask.png"

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.imwrite(str(img_p), img)
        cv2.imwrite(str(mask_p), mask)

        res = audit_single_pair(img_p, mask_p)
        assert res["is_valid"] is True
        assert res["is_empty_mask"] is True
        assert res["foreground_pixels"] == 0
        assert res["foreground_ratio"] == 0.0
