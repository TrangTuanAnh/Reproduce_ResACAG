"""Reproduce the paired contrasts from the four fixed source-IP configurations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


CONTRASTS = {
    "past_minus_flow": ("flow+topo_c", "flow"),
    "past_minus_shuffled": ("flow+topo_c", "flow+topo_r"),
    "all_minus_past": ("flow+topo_t", "flow+topo_c"),
    "identity_minus_flow": ("flow+id", "flow"),
}


def summarize(rows):
    results = []
    for name, model_rows in rows.groupby("model", sort=True):
        scores = model_rows.pivot(index=["cfg", "seed"], columns="set", values="macro_f1")
        for contrast, (left, right) in CONTRASTS.items():
            delta = 100 * (scores[left] - scores[right])
            per_config = delta.groupby(level="cfg").mean()
            k, n = len(per_config), len(delta)
            if k != 4 or n != 12 or delta.isna().any() or not (delta.groupby(level="cfg").size() == 3).all():
                raise ValueError(f"Expected three matched seeds per configuration for {name}, {contrast}.")
            center = per_config.mean()
            cfg_margin = t.ppf(0.975, k - 1) * per_config.std(ddof=1) / np.sqrt(k)
            left_rows = model_rows[model_rows["set"] == left]
            ratio = (left_rows["n_test"] / left_rows["n_train"]).mean()
            split_margin = t.ppf(0.975, n - 1) * delta.std(ddof=1) * np.sqrt(1 / n + ratio)
            cfg_lo, cfg_hi = center - cfg_margin, center + cfg_margin
            split_lo, split_hi = delta.mean() - split_margin, delta.mean() + split_margin
            results.append({
                "model": name, "contrast": contrast, "mean_pp": center,
                "configs_improved": int((per_config > 0).sum()),
                "splits_improved": int((delta > 0).sum()),
                "cfg_lo": cfg_lo, "cfg_hi": cfg_hi, "split_lo": split_lo, "split_hi": split_hi,
                "wide_lo": min(cfg_lo, split_lo), "wide_hi": max(cfg_hi, split_hi),
                **{cfg: score for cfg, score in per_config.items()},
            })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summarize(pd.read_csv(args.metrics)).to_csv(args.summary, index=False)


if __name__ == "__main__":
    main()
