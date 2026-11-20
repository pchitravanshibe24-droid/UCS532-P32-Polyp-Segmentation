"""Build the report tables and derived diagnostic figures from recorded outputs."""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ablations import VARIANTS
from src.evaluate import METRICS
from src.utils import ensure_dir

NAMES = {"uacanet": "Compact UACANet", "unet": "U-Net", "no_uncertain": "UACANet without uncertain context"}


def summarize(run_dir):
    run_dir = Path(run_dir)
    tables = ensure_dir(run_dir / "tables")
    figures = ensure_dir(run_dir / "figures")
    rows, models = [], {}
    for variant in VARIANTS:
        metrics = run_dir / variant / "metrics"
        record = json.loads((metrics / "summary.json").read_text())
        models[variant] = pd.read_csv(metrics / "test_metrics.csv")
        rows.append({
            "model": variant, "name": NAMES[variant], "n": record["test_sample_count"],
            "epoch": record["epoch"], "threshold": record["threshold"],
            "parameters": record["trainable_parameters"],
            **{f"{metric}_mean": record["metrics_macro_mean"][metric] for metric in METRICS},
            **{f"{metric}_std": record["metrics_macro_std"][metric] for metric in METRICS},
        })
    table = pd.DataFrame(rows)
    table.to_csv(tables / "main_table.csv", index=False)
    print(table[["name", "epoch", "threshold", *(f"{m}_mean" for m in METRICS)]].to_string(index=False))

    main = models["uacanet"].set_index("case")
    comparisons = []
    for variant in ("unet", "no_uncertain"):
        compared = models[variant].set_index("case")
        if not main.index.equals(compared.index):
            raise ValueError("Model rows must have identical case ordering.")
        paired = main["dice"] - compared["dice"]
        comparisons.append({"reference": variant, "count": len(paired), "mean_delta_dice": paired.mean(),
                            "median_delta_dice": paired.median(), "wins": int((paired > 0).sum()),
                            "ties": int((paired == 0).sum()), "losses": int((paired < 0).sum())})
    pd.DataFrame(comparisons).to_csv(tables / "paired_dice.csv", index=False)

    area_cutoff = float(pd.read_csv(run_dir / "splits" / "train.csv")["foreground_ratio"].median())
    slices = []
    for variant, frame in models.items():
        for label, subset in frame.groupby(np.where(frame["gt_area"] < area_cutoff, "smaller", "larger")):
            slices.append({"model": variant, "slice": label, "cutoff": area_cutoff, "count": len(subset),
                           "mean_dice": subset.dice.mean(), "mean_iou": subset.iou.mean(),
                           "mean_recall": subset.recall.mean()})
    pd.DataFrame(slices).to_csv(tables / "size_slice.csv", index=False)

    worst = main.sort_values(["dice", "case"]).head(3)
    best = main.sort_values(["dice", "case"], ascending=[False, True]).head(1)
    cases = pd.concat([worst.assign(selection="lowest Dice"), best.assign(selection="highest Dice")])
    cases = cases.assign(
        dominant_error=np.select([cases.fn > cases.fp, cases.fp > cases.fn],
                                 ["false negatives", "false positives"], default="tie"),
    )
    cases.to_csv(tables / "failure_cases.csv")
    for case in cases.index.unique():
        shutil.copyfile(run_dir / "uacanet" / "figures" / "mask_panels" / f"{case}.png", figures / f"{case}.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    for metric, axis in zip(("uncertainty", "entropy"), axes):
        bins = pd.read_csv(run_dir / "uacanet" / "metrics" / f"{metric}_bins.csv")
        axis.plot(bins.bin_index, bins.error_rate, "o-", color="#34798a")
        axis.set(xlabel=f"{'UACA' if metric == 'uncertainty' else 'Entropy'} quantile bin",
                 ylabel="Pixel error rate", ylim=(0, max(0.05, 1.1 * bins.error_rate.max())))
        axis.grid(alpha=0.25)
    fig.savefig(figures / "uncertainty_error.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
