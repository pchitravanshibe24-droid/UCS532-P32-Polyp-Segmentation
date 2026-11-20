"""Exercise the real demo callback and downloadable results without local artifacts."""

from zipfile import ZipFile

import numpy as np
import torch

from app import create_demo, predict_gradio
from src.models import build_model


def test_gradio_prediction(tmp_path):
    torch.set_num_threads(2)
    config = {"model": {"architecture": "uacanet", "channels": 8, "backbone_width": 8},
              "project": {"threads": 2}, "data": {"image_size": [32, 48]}, "inference": {}}
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": build_model(config).state_dict(), "config": config,
                "threshold": 0.4}, checkpoint)
    image = np.full((45, 63, 3), 130, dtype=np.uint8)
    mask, probability, heatmap, overlay, stats, download = predict_gradio(image, checkpoint)
    assert mask.shape == probability.shape == image.shape[:2]
    assert heatmap.shape == overlay.shape == image.shape
    assert "0.40" in stats
    with ZipFile(download) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            "segmentation.png", "probability_16bit.png", "uaca_uncertainty_16bit.png",
            "uncertainty_heatmap.png", "uncertainty_overlay.png", "README.txt",
        }
    assert create_demo(checkpoint).get_config_file()["components"]
