"""Reproduce the shared-split benchmark from the official archive."""

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import yaml

from scripts.create_splits import generate_splits
from scripts.generate_synthetic_data import generate_synthetic_dataset
from src.ablations import VARIANTS, experiment_config
from src.train import run_training
from src.utils import load_config

DATA_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ARCHIVE_HASH = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"


def prepare_kvasir(root):
    """Verify the official archive and install only the paired image and mask files."""
    root.parent.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "kvasir-seg.zip"
    if not archive.exists():
        download = archive.with_suffix(".part")
        print(f"Downloading the official Kvasir-SEG archive from {DATA_URL}", flush=True)
        with urllib.request.urlopen(DATA_URL, timeout=90) as response, download.open("wb") as output:
            shutil.copyfileobj(response, output)
        download.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ARCHIVE_HASH:
        raise ValueError(f"Unexpected dataset archive SHA-256: {digest}")
    with ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            parts = entry.filename.split("/")
            if len(parts) != 3 or parts[0] != "Kvasir-SEG" or parts[1] not in ("images", "masks"):
                continue
            if not parts[2] or parts[2] in (".", "..") or any(char in parts[2] for char in ("\\", ":")):
                continue
            target = root / parts[1] / parts[2]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(entry) as source, target.open("wb") as stream:
                shutil.copyfileobj(source, stream)
    print(f"Kvasir-SEG archive SHA-256 {digest}", flush=True)


def run(args):
    started = time.perf_counter()
    base = load_config("configs/smoke_test.yaml" if args.smoke else "configs/default.yaml")
    if args.smoke:
        base["model"] = {"architecture": "uacanet", "channels": 8, "backbone_width": 8}
        base["loss"] = {"type": "bce_iou"}
        base["inference"]["mc_passes"] = 1
        base["project"]["threads"] = 2
    if args.device:
        base["project"]["device"] = args.device
    label = "smoke" if args.smoke else "kvasir"
    run_dir = Path("outputs") / label
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(base["paths"]["data_root"])
    if args.stage in ("all", "train"):
        if args.smoke:
            generate_synthetic_dataset(root, num_samples=20, height=64, width=64, seed=base["project"]["seed"])
        else:
            prepare_kvasir(root)
        generate_splits(root / "images", root / "masks", run_dir / "splits",
                        base["data"]["split_ratios"], base["project"]["seed"], base["data"]["mask_threshold"])
        if not args.smoke:
            metadata = json.loads((run_dir / "splits" / "split_metadata.json").read_text())
            if metadata["archive_pairs_audited"] != 1000:
                raise ValueError("Expected 1,000 paired images and masks in Kvasir-SEG.")
    variants = ("uacanet",) if args.smoke else VARIANTS
    for variant in variants:
        config_path = run_dir / f"{variant}.yaml"
        if args.stage in ("all", "train"):
            config = experiment_config(base, variant, run_dir, Path("checkpoints") / label / variant)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"Training {variant} | Seed {base['project']['seed']}", flush=True)
            variant_start = time.perf_counter()
            run_training(config_path)
            print(f"Training runtime {variant}: {time.perf_counter() - variant_start:.2f} s", flush=True)
        if args.stage in ("all", "evaluate"):
            from src.evaluate import evaluate_test_set

            print(f"Evaluating {variant}", flush=True)
            evaluate_test_set(config_path)
    if not args.smoke and args.stage in ("all", "evaluate"):
        from scripts.summarize_results import summarize
        from src.stress_tests import run_stress_testing

        summarize(run_dir)
        run_stress_testing(run_dir / "uacanet.yaml", selected=[
            ("gaussian_blur", "moderate"), ("brightness", "darker"),
        ])
    print(f"{label} {args.stage} runtime: {time.perf_counter() - started:.2f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    run(parser.parse_args())
