"""Summarize macro-F1 over the five seeds for the source-IP experiment."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = pd.read_csv(args.metrics)
    summary = (
        rows.groupby(["split", "model", "set"])["macro_f1"]
        .agg(macro_f1_mean="mean", macro_f1_sd="std", n_seeds="count")
        .reset_index()
    )
    summary.to_csv(args.output, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
