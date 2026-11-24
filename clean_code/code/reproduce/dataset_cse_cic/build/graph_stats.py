"""Topology diagnostics: does the graph carry label signal?

Edge homophily of flow e=(u,v) with label l is the share of the other flows at u or v
that also have label l. A random graph gives roughly the class prior; an informative
graph gives much more. Reported overall, per class, and against the label prior.

Usage: python -I graph_stats.py DATASET_PARQUET REFERENCE_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Benign", "Brute-force", "DoS", "DDoS", "Web Attack", "Botnet", "Infiltration"]


def stats(src, dst, y):
    k = len(CLASSES)
    n_src = src.max() + 1
    nodes = np.concatenate([src, dst + n_src])
    labels = np.concatenate([y, y])
    n = nodes.max() + 1
    deg = np.bincount(nodes, minlength=n)
    cnt = np.zeros((n, k), dtype=np.int64)
    np.add.at(cnt, (nodes, labels), 1)
    u, v = src, dst + n_src
    same = cnt[u, y] - 1 + cnt[v, y] - 1
    total = deg[u] - 1 + deg[v] - 1
    has = total > 0
    h = np.where(has, same / np.maximum(total, 1), np.nan)
    prior = np.bincount(y, minlength=k) / len(y)
    per_class = {CLASSES[c]: {"edges": int((y == c).sum()),
                              "with_neighbors": float(has[y == c].mean()) if (y == c).any() else None,
                              "homophily": float(np.nanmean(h[y == c])) if has[y == c].any() else None,
                              "prior": float(prior[c])} for c in range(k)}
    active = deg > 0
    return {"edges": int(len(y)), "nodes": int(active.sum()),
            "src_nodes": int(len(np.unique(src))), "dst_nodes": int(len(np.unique(dst))),
            "node_degree_mean": float(deg[active].mean()), "node_degree_max": int(deg.max()),
            "edges_with_neighbor_edges": float(has.mean()),
            "edge_homophily_mean": float(np.nanmean(h)),
            "expected_if_random": float((prior ** 2).sum() / 1.0),
            "per_class": per_class}


def main(dataset, ref_dir, out_json):
    df = pd.read_parquet(dataset)
    y = df["label"].to_numpy()
    src = pd.factorize(df["Src IP"].astype(str) + ":" + df["Src Port"].astype(str))[0]
    dst = pd.factorize(df["Dst IP"].astype(str) + ":" + df["Dst Port"].astype(str))[0]
    srcip = pd.factorize(df["Src IP"].astype(str))[0]
    dstip = pd.factorize(df["Dst IP"].astype(str))[0]
    adj = np.load(Path(ref_dir) / "adj_random.npy", allow_pickle=False)[df["eres_row"].to_numpy()]
    rsrc = pd.factorize(adj[:, 0])[0]
    rdst = pd.factorize(adj[:, 1])[0]
    result = {"observed_ip_port": stats(src, dst, y), "observed_ip_only": stats(srcip, dstip, y),
              "eresgat_random": stats(rsrc, rdst, y)}
    Path(out_json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, r in result.items():
        print(f"{name:18s} nodes={r['nodes']:7d} deg_mean={r['node_degree_mean']:.2f} deg_max={r['node_degree_max']:6d} "
              f"with_nbrs={r['edges_with_neighbor_edges']:.3f} homophily={r['edge_homophily_mean']:.3f} "
              f"(random~{r['expected_if_random']:.3f})")
        print("    " + "  ".join(f"{c}:{v['homophily']:.2f}" if v['homophily'] is not None else f"{c}:-"
                                  for c, v in r["per_class"].items()))


if __name__ == "__main__":
    main(*sys.argv[1:4])
