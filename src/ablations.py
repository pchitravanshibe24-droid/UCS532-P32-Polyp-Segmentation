"""Configurations for the shared-split U-Net comparison and UACA ablation."""

from copy import deepcopy
VARIANTS = ("uacanet", "unet", "no_uncertain")


def experiment_config(base, variant, run_dir, checkpoint_dir):
    config = deepcopy(base)
    if variant == "no_uncertain":
        config["model"]["use_uncertain_context"] = False
    if variant == "unet":
        config["model"] = {"architecture": "unet", "features": [8, 16, 32], "dropout_prob": 0.0}
    paths = config["paths"]
    for split in ("train", "val", "test"):
        paths[f"{split}_split"] = (run_dir / "splits" / f"{split}.csv").as_posix()
    paths.update(
        splits_dir=(run_dir / "splits").as_posix(),
        checkpoints_dir=checkpoint_dir.as_posix(),
        best_checkpoint=(checkpoint_dir / "best_model.pt").as_posix(),
        metrics_dir=(run_dir / variant / "metrics").as_posix(),
        figures_dir=(run_dir / variant / "figures").as_posix(),
    )
    return config
