"""Compare an independently computed main table to the recorded reference."""

import argparse

import numpy as np
import pandas as pd

COLUMNS = ("dice", "iou", "precision", "recall", "f1", "mae")
IDENTIFIERS = ["n", "epoch", "threshold", "parameters"]


def compare(reference_path, actual_path):
    reference = pd.read_csv(reference_path).set_index("model")
    actual = pd.read_csv(actual_path).set_index("model")
    if not reference.index.equals(actual.index):
        raise ValueError("Reference and replay must contain the same model rows in the same order.")
    if not actual[["n", "parameters"]].equals(reference[["n", "parameters"]]):
        raise ValueError("Replay test counts or model sizes differ from the reference.")
    print("Frozen epoch and threshold")
    print(reference[IDENTIFIERS].join(actual[IDENTIFIERS], lsuffix="_reference", rsuffix="_replay").to_string())
    kinds = [f"{column}_{stat}" for stat in ("mean", "std") for column in COLUMNS]
    delta = actual[kinds] - reference[kinds]
    if not np.isfinite(delta.to_numpy()).all():
        raise ValueError("Replay or reference metrics are nonfinite.")
    print("Metric delta (replay minus recorded)")
    print(delta.to_string(float_format=lambda x: f"{x:+.8f}"))
    print(f"Largest absolute delta: {delta.abs().to_numpy().max():.8f}")
    print(f"All means and standard deviations match at four decimals: "
          f"{reference[kinds].round(4).equals(actual[kinds].round(4))}")
    return delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("actual")
    args = parser.parse_args()
    compare(args.reference, args.actual)
