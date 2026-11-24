"""Reproducible training, validation-only selection, offline robustness and profiling."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from .data import TrainMinMax, class_counts, load_dataset, sha256, split_indices, write_json
from .graph import FlowGraph, node2vec
from .models import EdgeClassifier, VARIANTS


def code_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for file in sorted(root.glob("*.py")):
        h.update(file.name.encode())
        h.update(file.read_bytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def metric_dict(y, pred, classes):
    labels = list(range(len(classes)))
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_weighted": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "per_class": classification_report(y, pred, labels=labels, target_names=classes,
                                               output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(), "n": len(y)}


@torch.no_grad()
def evaluate(model, graph, features, topology, labels, fanouts, batch_size, seed, embeddings=False):
    model.eval()
    rng = np.random.default_rng(seed)
    predictions, encoded, total_loss = [], [], 0.0
    start = time.perf_counter()
    for start_index in range(0, len(labels), batch_size):
        idx = np.arange(start_index, min(start_index + batch_size, len(labels)))
        logits, emb = model(idx, graph, features, topology, fanouts, rng)
        loss = F.cross_entropy(logits, torch.as_tensor(labels[idx], device=features.device))
        total_loss += float(loss) * len(idx)
        predictions.append(logits.argmax(1).cpu().numpy())
        if embeddings:
            encoded.append(emb.cpu().numpy())
    return np.concatenate(predictions), total_loss / len(labels), time.perf_counter() - start, (
        np.concatenate(encoded) if embeddings else None)


def prepare(args, name, seed, output, device):
    ds = load_dataset(name, Path(args.data_root))
    original_rows = len(ds.y)
    selection = np.arange(original_rows)
    if args.max_rows and args.max_rows < len(selection):
        selection, _ = train_test_split(selection, train_size=args.max_rows, random_state=seed, stratify=ds.y)
        selection.sort()
        ds.x, ds.y, ds.endpoints = ds.x[selection], ds.y[selection], ds.endpoints[selection]
    splits = split_indices(ds.y, args.split, seed)
    sequential = split_indices(ds.y, "sequential", seed)
    output.mkdir(parents=True, exist_ok=True)
    # node2vec is unsupervised; "all_splits_structure" walks the unlabelled flows of every split
    # (transductive), "train" only the train graph (held-out unseen nodes get zero embedding).
    walk_scope = "all_splits_structure" if name in args.transductive_node2vec else "train"
    dataset_key = {"name": name, "sha256": ds.metadata["sha256"], "seed": seed,
                   "split": args.split, "max_rows": args.max_rows, "hidden": args.hidden,
                   "node2vec_epochs": args.node2vec_epochs,
                   "node2vec": {"walks_per_node": 2, "length": 10, "window": 2, "negative_samples": 3,
                                "p": 1.0, "q": 1.0, "negative_distribution": "uniform_active_train_nodes"}}
    if walk_scope != "train":
        # Train-only keys stay byte-identical to the historical runs, so saved embeddings remain reusable.
        dataset_key["node2vec"].update(graph=walk_scope, negative_distribution="uniform_active_nodes_all_splits")
    key_json = json.dumps(dataset_key, sort_keys=True)
    fingerprint = hashlib.sha256(key_json.encode()).hexdigest()
    scaler = TrainMinMax().fit(ds.x[splits["train"]])
    scaler.save(output / "scaler.npz")
    np.savez(output / "splits.npz", **{k: selection[v] for k, v in splits.items()})
    graphs, features, labels = {}, {}, {}
    num_nodes = int(ds.endpoints.max()) + 1
    for part, indices in splits.items():
        graphs[part] = FlowGraph(ds.endpoints[indices], num_nodes)
        features[part] = torch.as_tensor(scaler.transform(ds.x[indices]), device=device)
        labels[part] = ds.y[indices]
    metadata = {**ds.metadata, "original_rows": original_rows, "used_rows": len(ds.y),
                "feature_names": ds.feature_names, "classes": ds.classes, "dataset_config": dataset_key,
                "counts": class_counts(ds.y, ds.classes),
                "splits": {k: class_counts(v, ds.classes) for k, v in labels.items()},
                "sequential_diagnostic": {k: class_counts(ds.y[v], ds.classes) for k, v in sequential.items()},
                "nonfinite_feature_values": int((~np.isfinite(ds.x)).sum()),
                "graphs": {k: {"edges": len(g.endpoints), "active_nodes": len(g.active_nodes),
                                 "mean_node_degree": g.node_degree_mean,
                                 "adaptive_S": g.edge_neighbor_mean} for k, g in graphs.items()},
                "split_policy": "independent graphs; train-only node2vec; held-out unseen nodes get zero embedding"
                if walk_scope == "train" else
                "independent graphs; node2vec walks all splits' unlabelled structure (transductive); no labels",
                "node2vec_fingerprint": fingerprint}
    seen = set(np.unique(ds.y[splits["train"]]))
    unseen = sorted(set(np.unique(ds.y)) - seen)
    metadata["missing_train_classes"] = [ds.classes[i] for i in unseen]
    write_json(output / "data_audit.json", metadata)
    if unseen and not args.allow_missing_classes:
        raise ValueError(f"Train has no examples for {metadata['missing_train_classes']}. See data_audit.json; use stratified split or explicitly --allow-missing-classes.")
    emb_path, key_path = output / "node2vec.npy", output / "node2vec_key.json"
    if emb_path.exists() and key_path.exists() and json.loads(key_path.read_text())["fingerprint"] == fingerprint:
        topology = np.load(emb_path, allow_pickle=False)
        print(f"{name}: reusing saved node2vec ({walk_scope})", flush=True)
    else:
        walk_graph = graphs["train"] if walk_scope == "train" else FlowGraph(ds.endpoints, num_nodes)
        topology, loss = node2vec(walk_graph, args.hidden, seed, device, epochs=args.node2vec_epochs)
        np.save(emb_path, topology)
        write_json(key_path, {"fingerprint": fingerprint, "loss": loss, "config": dataset_key})
    metadata["unseen_node_fraction"] = {k: float(np.mean(graphs["train"].degree[g.active_nodes] == 0))
                                         for k, g in graphs.items()}
    write_json(output / "data_audit.json", metadata)
    return ds, graphs, features, labels, torch.as_tensor(topology, device=device), splits, selection


def profile_cpu(model, graph, feature, topology, fanouts, batch_size, seed):
    """One training forward/backward batch, no optimizer step, on CPU."""
    replica = copy.deepcopy(model).cpu().train()
    data = feature.cpu()
    topo = topology.cpu()
    idx = np.arange(min(batch_size, len(graph.endpoints)))
    rng = np.random.default_rng(seed)
    # Profiler overhead differs from wall time; report separately.
    start = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        logits, _ = replica(idx, graph, data, topo, fanouts, rng)
        # Deterministic arbitrary labels suffice for profiling backward, not metric evaluation.
        F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long)).backward()
    return {"self_cpu_operator_seconds": sum(e.self_cpu_time_total for e in prof.key_averages()) / 1e6,
            "profiled_wall_seconds": time.perf_counter() - start, "batch_size": len(idx),
            "system_memory_percent": psutil.virtual_memory().percent,
            "process_rss_mib": psutil.Process().memory_info().rss / 2**20,
            "note": "CPU forward/backward only; excludes optimizer; current hardware, not original paper timing"}


def train_variant(args, name, variant, seed, output, ds, graphs, features, labels, topology, splits, selection):
    directory = output / variant
    directory.mkdir(exist_ok=True)
    strategy = "fixed" if variant == "sampling_s1" else "degree" if variant == "sampling_s2" else "adaptive"
    if variant in ["e_graphsage", "e_graphsage_m", "gat", "e_resgat", "graph_transformer", "mlp"]:
        strategy = "fixed"
    fanouts = graphs["train"].fanouts(strategy, tuple(args.caps), tuple(args.fixed))
    config = {"dataset": name, "variant": variant, "seed": seed, "hidden": args.hidden,
              "learning_rate": 0.01 if name == "darknet" else 0.007, "dropout": 0.5,
              "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
              "split": args.split, "max_rows": args.max_rows, "strategy": strategy,
              "fanouts": fanouts, "code_sha256": code_hash(), "edge_update": args.edge_update,
              "node2vec_epochs": args.node2vec_epochs, "data_sha256": ds.metadata["sha256"],
              "node2vec_graph": "all_splits_structure" if name in args.transductive_node2vec else "train",
              "weight_decay": 0.0, "loss": "unweighted_cross_entropy", "clip_gradient_norm": 5.0,
              "checkpoint_selection": "validation macro-F1 then loss", "profile_cpu": args.profile_cpu}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError(f"Completed run {directory} has different code/config; choose a new output directory.")
        print(f"SKIP completed {name}/{variant}/seed={seed}", flush=True)
        return
    seed_all(seed)
    model = EdgeClassifier(features["train"].shape[1], args.hidden, len(ds.classes),
                           dropout=0.5, edge_update=args.edge_update, **VARIANTS[variant]).to(features["train"].device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    write_json(directory / "config.json", config)
    checkpoint = directory / "best.pt"
    best, stale, history, times = (-1.0, float("-inf")), 0, [], []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    if features["train"].is_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(labels["train"]))
        loss_sum = 0.0
        epoch_start = time.perf_counter()
        for offset in range(0, len(order), args.batch_size):
            batch = order[offset:offset + args.batch_size]
            batch_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch, graphs["train"], features["train"], topology, fanouts, rng)
            loss = F.cross_entropy(logits, torch.as_tensor(labels["train"][batch], device=logits.device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {name}/{variant}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            times.append(time.perf_counter() - batch_start)
            peak_rss = max(peak_rss, process.memory_info().rss)
        pred, val_loss, _, _ = evaluate(model, graphs["validation"], features["validation"], topology,
                                        labels["validation"], fanouts, args.batch_size, seed + 10000)
        score = metric_dict(labels["validation"], pred, ds.classes)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / len(order), "validation_loss": val_loss,
               "validation_f1_macro": score["f1_macro"], "validation_f1_weighted": score["f1_weighted"],
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        write_json(directory / "history.json", history)
        print(f"{name}/{variant} seed={seed} epoch={epoch+1} loss={row['train_loss']:.4f} "
              f"val_macro={score['f1_macro']:.4f} seconds={row['seconds']:.1f}", flush=True)
        candidate = (score["f1_macro"], -val_loss)
        if candidate > best:
            best, stale = candidate, 0
            # Tensor-only state allows weights_only=True when loading.
            torch.save(model.state_dict(), checkpoint)
            write_json(directory / "best_epoch.json", {"epoch": epoch + 1, "validation_macro": best[0]})
        else:
            stale += 1
        if stale >= args.patience:
            break
    training_seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(checkpoint, map_location=features["train"].device, weights_only=True))
    pred, loss, eval_seconds, embeddings = evaluate(model, graphs["test"], features["test"], topology,
                                                    labels["test"], fanouts, args.batch_size, seed + 10000,
                                                    embeddings=variant == "resacag")
    results = {"signature": signature, "config": config, "test": metric_dict(labels["test"], pred, ds.classes),
               "test_loss": loss, "test_seconds": eval_seconds, "train_validation_seconds": training_seconds,
               "epochs_completed": len(history), "train_batch_wall_seconds_mean": float(np.mean(times)),
               "process_peak_rss_mib_sampled": peak_rss / 2**20, "parameters": sum(p.numel() for p in model.parameters()),
               "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if features["train"].is_cuda else 0,
               "topology": ds.metadata["topology"], "scope": "full_available_dataset" if not args.max_rows else "subset"}
    np.savez_compressed(directory / "predictions.npz", y=labels["test"], prediction=pred,
                        original_row_ids=selection[splits["test"]])
    if embeddings is not None:
        indices = np.random.default_rng(seed).choice(len(pred), min(2000, len(pred)), replace=False)
        np.savez_compressed(directory / "visualization.npz", raw=features["test"][indices].cpu().numpy(),
                            embedding=embeddings[indices], y=labels["test"][indices],
                            original_row_ids=selection[splits["test"]][indices])
    results["noise"] = {"0.0": results["test"]}
    # Nested, shared corruption sets, evaluated only after validation checkpoint selection.
    noise_rng = np.random.default_rng(seed + 20000)
    noise_order = noise_rng.permutation(len(labels["test"]))
    replacement = noise_rng.random(features["test"].shape, dtype=np.float32)
    for fraction in [0.1, 0.2, 0.3]:
        noisy = features["test"].clone()
        corrupted = noise_order[:int(fraction * len(noise_order))]
        noisy[corrupted] = torch.as_tensor(replacement[corrupted], device=noisy.device)
        noise_pred, _, _, _ = evaluate(model, graphs["test"], noisy, topology, labels["test"],
                                       fanouts, args.batch_size, seed + 10000)
        results["noise"][str(fraction)] = metric_dict(labels["test"], noise_pred, ds.classes)
        np.savez_compressed(directory / f"noise_{int(fraction*100)}.npz", prediction=noise_pred,
                            corrupted_local_rows=corrupted)
        del noisy
    if args.profile_cpu:
        results["cpu_profile"] = profile_cpu(model, graphs["train"], features["train"], topology,
                                               fanouts, args.batch_size, seed)
    # metrics.json is the completion marker, written after all requested evaluations.
    write_json(metrics_path, results)
    print(f"DONE {name}/{variant} F1w={results['test']['f1_weighted']:.4f} F1m={results['test']['f1_macro']:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--data-root", default=str(root / "dataset"))
    p.add_argument("--output", default=str(root.parents[0] / "lab" / "1" / "runs"))
    p.add_argument("--datasets", nargs="+", default=["darknet", "cse_reference"], choices=["darknet", "cse_reference", "cse_cic"])
    p.add_argument("--transductive-node2vec", nargs="*", default=["cse_cic"], choices=["darknet", "cse_reference", "cse_cic"],
                   help="Datasets whose node2vec walks all splits' unlabelled topology (lab/5 uses cse_cic); pass no value for train-only")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--caps", nargs=2, type=int, default=[8, 4])
    p.add_argument("--fixed", nargs=2, type=int, default=[4, 2])
    p.add_argument("--node2vec-epochs", type=int, default=2)
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows; positive marks a subset, never full reproduction")
    p.add_argument("--split", choices=["sequential", "stratified"], default="stratified")
    p.add_argument("--edge-update", choices=["static", "neighbor"], default="neighbor")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--profile-cpu", action="store_true")
    p.add_argument("--allow-missing-classes", action="store_true")
    args = p.parse_args()
    if args.hidden % 4 or min(args.epochs, args.patience, args.batch_size, args.node2vec_epochs, *args.caps, *args.fixed) < 1:
        p.error("hidden must be divisible by 4; epochs, patience, batch, walks and fanouts must be positive")
    torch.set_num_threads(2)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "platform": platform.platform(), "device": device,
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                   "dependencies": {x: importlib.metadata.version(x) for x in
                                    ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "psutil"]},
                   "arguments": vars(args), "code_sha256": code_hash(), "torch_threads": 2,
                   "determinism": "seeded; deterministic algorithms warn_only; CUDA numerical variation remains possible"}
    write_json(out / "environment.json", environment)
    for name in args.datasets:
        for seed in args.seeds:
            seed_all(seed)
            output = out / name / f"seed_{seed}"
            prepared = prepare(args, name, seed, output, device)
            for variant in args.variants:
                train_variant(args, name, variant, seed, output, *prepared)
            del prepared
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
