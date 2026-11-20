"""Dataset integration with portable, generated fixtures."""

import cv2
import numpy as np
import pandas as pd
import torch

from src.data.dataset import PolypDataset
from src.data.transforms import PairedTransforms


def test_polyp_dataset_loads_relative_csv_and_binary_mask(tmp_path):
    image = np.full((40, 52, 3), 100, dtype=np.uint8)
    mask = np.zeros((40, 52), dtype=np.uint8)
    mask[10:20, 15:30] = 1
    assert cv2.imwrite(str(tmp_path / "image.png"), image)
    assert cv2.imwrite(str(tmp_path / "mask.png"), mask)
    split = tmp_path / "split.csv"
    pd.DataFrame([{"pair_id": "case", "image_path": "image.png", "mask_path": "mask.png"}]).to_csv(split, index=False)
    dataset = PolypDataset(split, base_dir=tmp_path, transforms=PairedTransforms((32, 48)))
    sample = dataset[0]
    assert sample["pair_id"] == "case"
    assert sample["image"].shape == (3, 32, 48)
    assert sample["mask"].shape == (1, 32, 48)
    assert sample["image"].dtype == sample["mask"].dtype == torch.float32
    assert sample["mask"].unique().tolist() == [0, 1]
