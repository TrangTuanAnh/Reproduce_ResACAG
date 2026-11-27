"""Summarize macro-F1 over the five seeds for the source-IP experiment."""
import argparse
from pathlib import Path

import pandas as pd


def paired_contrasts(rows):
    flow_rand, flow_grp = ("RAND", "flow"), ("GRP", "flow")
    id_rand, id_grp = ("RAND", "flow+id"), ("GRP", "flow+id")
    values = rows.pivot(index=["model", "seed"], columns=["split", "set"], values="macro_f1")
    required = [flow_rand, flow_grp, id_rand, id_grp]
    if any(column not in values for column in required) or values[required].isna().any().any():
        raise ValueError("E9 contrasts require flow and flow+id in RAND and GRP for each model and seed.")
    return (
        pd.DataFrame({
            "grp_minus_rand_flow_pp": 100 * (values[flow_grp] - values[flow_rand]),
            "add_identity_rand_pp": 100 * (values[id_rand] - values[flow_rand]),
            "add_identity_grp_pp": 100 * (values[id_grp] - values[flow_grp]),
            "seed_declines_grp": values[id_grp] < values[flow_grp],
        })
        .groupby(level="model")
        .agg(grp_minus_rand_flow_pp=("grp_minus_rand_flow_pp", "mean"),
             add_identity_rand_pp=("add_identity_rand_pp", "mean"),
             add_identity_grp_pp=("add_identity_grp_pp", "mean"),
             seed_declines_grp=("seed_declines_grp", "sum"),
             n_seeds=("seed_declines_grp", "size"))
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--contrasts", type=Path, help="Optional CSV of per-model E9 contrasts.")
    args = parser.parse_args()
    rows = pd.read_csv(args.metrics)
    summary = (
        rows.groupby(["split", "model", "set"])["macro_f1"]
        .agg(macro_f1_mean="mean", macro_f1_sd="std", n_seeds="count")
        .reset_index()
    )
    summary.to_csv(args.output, index=False, encoding="utf-8")
    if args.contrasts:
        paired_contrasts(rows).to_csv(args.contrasts, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
