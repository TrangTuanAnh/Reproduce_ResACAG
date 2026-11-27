"""Repeat the source-IP check over all four Brute-force/DoS IP configurations."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from flow_splits import grp_split
from run_source_ip_experiments import C, build_sets, load, metrics, model


SEEDS = (301, 302, 303)
MODELS = ("dt", "hgb", "rf")
SETS = ("flow", "flow+topo_c", "flow+topo_r", "flow+topo_t", "flow+id")


def plans(data):
    y, src, addresses = data["y"], data["src"], data["srcip"]
    address_to_id = dict(zip(addresses, src))
    two_ip_classes = [sorted(set(addresses[y == c])) for c in (1, 2)]
    if any(len(ips) != 2 for ips in two_ip_classes):
        raise ValueError("Brute-force and DoS must each have two source IPs.")
    for a, b in product(range(2), repeat=2):
        fixed = {address_to_id[ip]: int(position == test_position)
                 for ips, test_position in zip(two_ip_classes, (a, b))
                 for position, ip in enumerate(ips)}
        for seed in SEEDS:
            train, test, audit = grp_split(y, src, seed, initial_sides=fixed)
            if len(audit) != C or not all(row["supported"] for row in audit.values()):
                raise ValueError(f"At least one class is absent in C{a}{b} seed {seed}.")
            yield f"C{a}{b}", seed, train, test, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Save class counts without fitting models.")
    args = parser.parse_args()
    data, _ = load(args.data_root / "cse_cic" / "cse_cic.parquet")
    splits = list(plans(data))
    args.out.mkdir(parents=True, exist_ok=False)
    support = [{"cfg": config, "seed": seed, "class_id": cls,
                "train_flows": row["train_flows"], "test_flows": row["test_flows"], "source_ips": row["ips"]}
               for config, seed, _, _, audit in splits for cls, row in audit.items()]
    pd.DataFrame(support).to_csv(args.out / "class_support.csv", index=False)
    if args.check_only:
        return

    y = data["y"]
    cache = {}
    results = []
    for config, seed, train, test, _ in splits:
        inputs = build_sets(data, seed, cache)
        for name, feature_set in product(MODELS, SETS):
            x = inputs[feature_set]
            tree = model(name, seed).fit(x[train], y[train])
            proba = np.zeros((len(test), C))
            proba[:, np.asarray(tree.classes_, int)] = tree.predict_proba(x[test])
            results.append(dict(split="GRPF", cfg=config, seed=seed, model=name, set=feature_set,
                                n_train=len(train), n_test=len(test),
                                **metrics(y[test], proba.argmax(axis=1), proba)))
        print(f"Completed {config}, seed {seed}", flush=True)
    pd.DataFrame(results).to_csv(args.out / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
