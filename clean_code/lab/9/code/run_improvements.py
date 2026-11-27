"""Exploratory offline checks for two proposed IDS changes on the report dataset."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/reproduce"))
sys.path.insert(0, str(ROOT / "lab/7/code"))

from resacag.data import (CSE_CLASSES, CSE_NON_FEATURES, ENDPOINTS, TrainMinMax,
                          endpoint_ids, sha256, split_indices, write_json)
from resacag.experiment import evaluate, metric_dict, seed_all
from resacag.graph import FlowGraph, node2vec
from resacag.models import EdgeClassifier, ResACAGLayer
from flow_splits import grp_split


REPORT_HASH = "ea1736a132a5ffb5d1b93eae48b5ba4c17433655dbc621236dbf9b3a9fcee8c7"
HISTORY_COLS = ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts"]
WINDOW_US = 300_000_000
BATCH_SIZE = 500


class SimilarityGatedLayer(ResACAGLayer):
    """Replace the mean with a learned cosine-weighted mean of sampled edges."""

    def __init__(self, feature_dim, layer):
        super().__init__(feature_dim, 32, layer, 0.5)
        self.similarity_logit_scale = nn.Parameter(torch.zeros(()))
        self.similarity_logit_bias = nn.Parameter(torch.zeros(()))

    def forward(self, own, neighbor, edge, mask, topology):
        masked = edge * mask.unsqueeze(-1)
        reference = masked.sum(dim=1, keepdim=True) - masked
        cosine = F.cosine_similarity(edge, reference, dim=-1)
        weights = torch.sigmoid(self.similarity_logit_scale * cosine + self.similarity_logit_bias)
        weights = (weights * mask).unsqueeze(-1)
        messages = torch.cat([edge, neighbor], dim=-1) if self.layer > 0 else edge
        aggregated = (self.message(messages) * weights).sum(1) / weights.sum(1).clamp_min(1e-8)
        attended = self.attention(aggregated)
        gate = torch.sigmoid(self.gate(attended))
        fused = gate * attended + (1 - gate) * topology
        broadcast = 0.5 * (fused + fused.mean(dim=0, keepdim=True))
        return F.relu(self.update(torch.cat([own, broadcast, attended], dim=-1)))


def read_flows(path):
    actual_hash = sha256(path)
    if actual_hash != REPORT_HASH:
        raise ValueError(f"Dataset differs from report: {actual_hash}")
    names = json.loads(path.with_name("feature_names.json").read_text(encoding="utf-8"))["feature_names"]
    if len(set(names)) != 77 or set(names) & CSE_NON_FEATURES:
        raise ValueError("Expected 77 flow-only features")
    frame = pd.read_parquet(path, columns=[*names, *ENDPOINTS, "start_us", "label"], use_threads=False)
    values = frame[names].to_numpy(dtype=np.float32)
    labels = frame["label"].to_numpy(dtype=np.int64)
    return frame, names, values, labels


def write_csv(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def score_row(split, seed, model, variant, scores, seconds, epoch=""):
    infil = scores["per_class"]["Infiltration"]
    return {"split": split, "seed": seed, "model": model, "variant": variant,
            "test_flows": scores["n"], "macro_f1": scores["f1_macro"], "weighted_f1": scores["f1_weighted"],
            "infiltration_recall": infil["recall"], "infiltration_precision": infil["precision"],
            "infiltration_support": int(infil["support"]), "best_epoch": epoch, "seconds": round(seconds, 2)}


def fit_gnn(seed, variant, graphs, features, labels, topology, run, max_epochs):
    seed_all(seed)
    device = topology.device
    feature_dim = features["train"].shape[1]
    model = EdgeClassifier(feature_dim, 32, len(CSE_CLASSES))
    if variant == "cosine_gate":
        with torch.random.fork_rng(devices=[]):
            for i, original in enumerate(model.layers):
                gated = SimilarityGatedLayer(feature_dim, i)
                report = gated.load_state_dict(original.state_dict(), strict=False)
                assert set(report.missing_keys) == {"similarity_logit_scale", "similarity_logit_bias"}
                assert not report.unexpected_keys
                model.layers[i] = gated
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    fanouts = graphs["train"].fanouts()
    targets = torch.as_tensor(labels["train"], device=device)
    best, stale, best_epoch, best_weights = (-1.0, float("-inf")), 0, 0, None
    history = []
    start = time.perf_counter()
    print(f"Starting {variant}, seed {seed}, fanouts {fanouts}", flush=True)
    for epoch in range(max_epochs):
        model.train()
        epoch_start = time.perf_counter()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(targets))
        for i in range(0, len(order), BATCH_SIZE):
            batch = order[i:i + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch, graphs["train"], features["train"], topology, fanouts, rng)
            loss = F.cross_entropy(logits, targets[torch.as_tensor(batch, device=device)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
        pred, val_loss, _, _ = evaluate(model, graphs["validation"], features["validation"], topology,
                                        labels["validation"], fanouts, BATCH_SIZE, seed + 10000)
        val_score = metric_dict(labels["validation"], pred, CSE_CLASSES)["f1_macro"]
        entry = {"epoch": epoch + 1, "macro_f1": val_score, "loss": val_loss,
                 "seconds": round(time.perf_counter() - epoch_start, 2)}
        history.append(entry)
        write_json(run / "validation.json", history)
        print(f"{variant} {seed} {entry}", flush=True)
        candidate = (val_score, -val_loss)
        if candidate > best:
            best, stale, best_epoch = candidate, 0, epoch + 1
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 3:
            break
    model.load_state_dict(best_weights)
    pred, _, _, _ = evaluate(model, graphs["test"], features["test"], topology, labels["test"], fanouts,
                             BATCH_SIZE, seed + 10000)
    scores = metric_dict(labels["test"], pred, CSE_CLASSES)
    write_json(run / "metrics.json", {"test": scores, "best_epoch": best_epoch,
                                     "gate_parameters": {n: float(v) for n, v in best_weights.items()
                                                         if "similarity_logit" in n}})
    np.savez_compressed(run / "predictions.npz", y=labels["test"], prediction=pred)
    return score_row("stratified", seed, "ResACAG", variant, scores, time.perf_counter() - start, best_epoch)


def run_gate(path, cache, out, seeds, device, max_epochs):
    frame, _, x, y = read_flows(path)
    endpoints = endpoint_ids(frame)
    del frame
    settings = {"graph": "all_splits_structure", "walks_per_node": 2, "length": 10,
                "window": 2, "negative_samples": 3, "p": 1.0, "q": 1.0}
    cache_origin = "regenerated" if cache is None else "provided"
    if cache is None:
        embedding, _ = node2vec(FlowGraph(endpoints), 32, 42, device, epochs=2)
        cache = out / "node2vec.npy"
        np.save(cache, embedding)
        write_json(out / "node2vec_key.json", {"config": {
            "name": "cse_cic", "sha256": REPORT_HASH, "seed": 42, "hidden": 32,
            "node2vec_epochs": 2, "node2vec": settings,
        }})
    else:
        key = json.loads(cache.with_name("node2vec_key.json").read_text(encoding="utf-8"))["config"]
        if (key["sha256"] not in (REPORT_HASH, f"{REPORT_HASH}|filter=None")
                or key["seed"] != 42 or key["node2vec_epochs"] != 2 or key["hidden"] != 32
                or any(key["node2vec"].get(k) != v for k, v in settings.items())):
            raise ValueError("Expected the seed-42 full-structure node2vec cache for the report dataset.")
        embedding = np.load(cache, allow_pickle=False)
    if embedding.shape != (int(endpoints.max()) + 1, 32) or not np.isfinite(embedding).all():
        raise ValueError("Unexpected node2vec shape or non-finite values.")
    topology = torch.as_tensor(embedding, device=device)
    write_json(out / "protocol.json", {
        "dataset_sha256": REPORT_HASH, "seeds": seeds, "device": str(device), "max_epochs": max_epochs,
        "code_sha256": sha256(Path(__file__)), "torch": str(torch.__version__),
        "topology": "seed-42 node2vec on unlabeled full structure, shared by all matched pairs",
        "node2vec_sha256": sha256(cache), "node2vec_origin": cache_origin,
        "checkpoint": "validation macro-F1 then cross-entropy, patience 3",
        "gate": "logistic weights of cosine to other sampled peers, initialized to uniform; both layers",
    })
    rows = []
    for seed in seeds:
        splits = split_indices(y, "stratified", seed)
        scaler = TrainMinMax().fit(x[splits["train"]])
        graphs, features, labels = {}, {}, {}
        for part, idx in splits.items():
            features[part] = torch.as_tensor(scaler.transform(x[idx]), device=device)
            graphs[part] = FlowGraph(endpoints[idx], len(embedding))
            labels[part] = y[idx]
        for variant in ("mean", "cosine_gate"):
            row = fit_gnn(seed, variant, graphs, features, labels, topology, out / f"{seed}_{variant}", max_epochs)
            rows.append(row)
            write_csv(out / "scores.csv", rows)
            print(row, flush=True)


def history_vectors(groups, start, end, values):
    """Count and average completed flows of the same endpoint role in a trailing window."""
    vectors = np.zeros((len(groups), values.shape[1] + 1), dtype=np.float32)
    order = np.lexsort((end, groups))
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    for idx in np.split(order, boundaries):
        times = end[idx]
        low = np.searchsorted(times, start[idx] - WINDOW_US, side="left")
        high = np.searchsorted(times, start[idx], side="left")
        count = high - low
        prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values[idx], axis=0, dtype=np.float64)])
        vectors[idx, 0] = np.log1p(count)
        vectors[idx, 1:] = (prefix[high] - prefix[low]) / np.maximum(count, 1)[:, None]
    return vectors


def run_history(path, out, seeds):
    frame, names, values, labels = read_flows(path)
    timed = frame["start_us"].notna().to_numpy()
    start = frame.loc[timed, "start_us"].to_numpy(dtype=np.float64)
    duration = frame.loc[timed, "Flow Duration"].to_numpy(dtype=np.float64)
    assert np.isfinite(duration).all() and (duration >= 0).all()
    end = start + duration
    source_ip = frame.loc[timed, "Src IP"].to_numpy()
    dest_ip = frame.loc[timed, "Dst IP"].to_numpy()
    del frame
    x, y = values[timed], labels[timed]
    del values
    x[~np.isfinite(x)] = np.nan
    history_values = x[:, [names.index(name) for name in HISTORY_COLS]]
    assert np.isfinite(history_values).all()
    history_values = np.log1p(np.maximum(history_values, 0))
    write_json(out / "protocol.json", {
        "dataset_sha256": REPORT_HASH, "seeds": seeds, "timed_rows": len(y),
        "undated_rows": int((~timed).sum()), "code_sha256": sha256(Path(__file__)),
        "models": ["Decision Tree", "HistGradientBoosting"], "split": "Lab 7 GRP (original IP ordering)",
        "history_columns": HISTORY_COLS, "window_us": WINDOW_US,
        "history_rule": "within each split independently; previous completions in [start-300s, start), no labels",
        "vectors": "separate source/destination IP-role log count and mean log flow features; empty -> zero",
    })
    rows, coverage = [], []
    source_groups = pd.factorize(source_ip)[0]
    for seed in seeds:
        train, test, audit = grp_split(y, source_groups, seed)
        assert len(np.intersect1d(source_ip[train], source_ip[test])) == 0
        assert all(row["supported"] for row in audit.values())
        write_json(out / f"{seed}_split.json", {str(k): v for k, v in audit.items()})
        train_hist, test_hist = (np.column_stack([history_vectors(ips[idx], start[idx], end[idx], history_values[idx])
                                                for ips in (source_ip, dest_ip)]) for idx in (train, test))
        src_history = test_hist[:, 0] > 0
        dst_history = test_hist[:, len(HISTORY_COLS) + 1] > 0
        for subset, mask in (("all", np.ones(len(test), dtype=bool)),
                             ("Infiltration", y[test] == CSE_CLASSES.index("Infiltration"))):
            coverage.append({"seed": seed, "subset": subset, "n": int(mask.sum()),
                             "src_has_history": float(src_history[mask].mean()),
                             "dst_has_history": float(dst_history[mask].mean()),
                             "either_has_history": float((src_history | dst_history)[mask].mean())})
        write_csv(out / "coverage.csv", coverage)
        train_x, test_x = x[train], x[test]
        for variant, x_train, x_test in (
            ("flow", train_x, test_x),
            ("flow_history", np.column_stack([train_x, train_hist]), np.column_stack([test_x, test_hist])),
        ):
            models = (DecisionTreeClassifier(class_weight="balanced", random_state=seed),
                      HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1,
                          l2_regularization=1, class_weight="balanced", early_stopping=False, random_state=seed))
            for model in models:
                tick = time.perf_counter()
                model.fit(x_train, y[train])
                pred = model.predict(x_test)
                scores = metric_dict(y[test], pred, CSE_CLASSES)
                name = type(model).__name__
                write_json(out / f"{seed}_{variant}_{name}.json", scores)
                row = score_row("GRP", seed, name, variant, scores, time.perf_counter() - tick)
                rows.append(row)
                write_csv(out / "scores.csv", rows)
                print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["gate", "history"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", required=True, type=int, nargs="+")
    parser.add_argument("--node2vec", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=12)
    args = parser.parse_args()
    if args.node2vec and args.experiment != "gate":
        parser.error("--node2vec is only used by the gate experiment")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.experiment == "gate":
        run_gate(args.data, args.node2vec, args.out, args.seeds, torch.device(device), args.max_epochs)
    else:
        run_history(args.data, args.out, args.seeds)


if __name__ == "__main__":
    main()
