"""Train segmentation models and select checkpoints and thresholds on validation."""

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PolypDataset
from src.data.transforms import get_inference_transforms, get_train_transforms
from src.losses import build_loss
from src.metrics import compute_batch_dice
from src.models import build_model, final_logits
from src.utils import ensure_dir, get_device, load_config, print_environment_info, set_seed


def run_epoch(model, loader, criterion, device, *, optimizer=None, thresholds=()):
    training = optimizer is not None
    model.train(training)
    totals = {}
    dice = {threshold: 0.0 for threshold in thresholds}
    count = len(loader.dataset)
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if training:
                optimizer.zero_grad()
            output = model(images)
            loss, components = criterion(output, masks)
            if training:
                loss.backward()
                optimizer.step()
            for name, value in components.items():
                key = name.removesuffix("_total")
                totals[key] = totals.get(key, 0.0) + value * len(images)
            if thresholds:
                probabilities = final_logits(output).sigmoid()
                for threshold in thresholds:
                    dice[threshold] += compute_batch_dice(probabilities, masks, threshold) * len(images)

    prefix = "train" if training else "val"
    metrics = {f"{prefix}_{name}": value / count for name, value in totals.items()}
    if thresholds:
        winner = max(thresholds, key=lambda t: (dice[t], -abs(t - 0.5), -t))
        metrics.update(val_dice=dice[winner] / count, threshold=winner)
        metrics.update({f"val_dice_at_{t:.2f}": dice[t] / count for t in thresholds})
    return metrics


def plot_and_save_curves(history, output_path):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for field, label in (("train_loss", "Training"), ("val_loss", "Validation")):
        axes[0].plot(df.epoch, df[field], label=label)
    axes[0].legend()
    axes[0].set_ylabel("Supervised loss")
    axes[1].plot(df.epoch, df.val_dice, color="seagreen")
    axes[1].set_ylabel("Best validation Dice on threshold grid")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_training(config_path):
    config = load_config(config_path)
    seed = config["project"]["seed"]
    torch.set_num_threads(config["project"].get("threads", 4))
    device = get_device(config["project"].get("device", "cpu"))
    # CUDA bilinear backward may require a nondeterministic kernel on some builds.
    set_seed(seed, strict=device.type == "cpu")
    print_environment_info(config)
    paths = config["paths"]
    checkpoint = Path(paths["best_checkpoint"])
    ensure_dir(checkpoint.parent)
    metrics_dir = ensure_dir(paths["metrics_dir"])
    figures_dir = ensure_dir(paths["figures_dir"])
    training = config["training"]
    loader_kwargs = {
        "batch_size": training["batch_size"],
        "num_workers": config["data"].get("num_workers", 0),
    }
    train_data = PolypDataset(paths["train_split"], transforms=get_train_transforms(config))
    val_data = PolypDataset(paths["val_split"], transforms=get_inference_transforms(config))
    train_loader = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    val_loader = DataLoader(val_data, **loader_kwargs)
    model = build_model(config).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Train pairs: {len(train_data)} | Validation pairs: {len(val_data)} | Trainable parameters: {params}")
    criterion = build_loss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training.get("weight_decay", 1e-4))
    schedule = training.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=schedule.get("factor", 0.5), patience=schedule.get("patience", 5),
        min_lr=schedule.get("min_lr", 1e-6),
    )
    inference = config["inference"]
    thresholds = tuple(inference.get("threshold_grid", [inference.get("evaluation_threshold", 0.5)]))
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError("Validation thresholds must lie in (0, 1).")
    early = training.get("early_stopping", {})
    best_dice, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, training["epochs"] + 1):
        train = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val = run_epoch(model, val_loader, criterion, device, thresholds=thresholds)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val["val_loss"])
        history.append({"epoch": epoch, "lr": lr, **train, **val})
        pd.DataFrame(history).to_csv(metrics_dir / "train_history.csv", index=False)
        print(f"Epoch {epoch:02d} | train loss {train['train_loss']:.4f} | "
              f"validation loss {val['val_loss']:.4f} | validation Dice {val['val_dice']:.4f} | "
              f"threshold {val['threshold']:.2f}", flush=True)
        if val["val_dice"] > best_dice:
            best_dice, best_epoch = val["val_dice"], epoch
            temporary = checkpoint.with_suffix(f".{uuid4().hex}.tmp")
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(), "config": config,
                "seed": seed, "threshold": val["threshold"], "best_val_dice": best_dice,
                "trainable_parameters": params,
            }, temporary)
            os.replace(temporary, checkpoint)
        if early.get("enabled") and epoch - best_epoch >= early["patience"]:
            break
    plot_and_save_curves(history, figures_dir / "training_curves.png")
    print(f"Selected epoch {best_epoch} | validation Dice {best_dice:.6f} | checkpoint {checkpoint}")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    run_training(parser.parse_args().config)
