"""Baseline tra nhãn theo đầu kết nối (không dùng đặc trưng luồng) trên cả hai bộ, split seed 42 như báo cáo."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

root = Path(sys.argv[1])
ENDPOINTS = ["Src IP", "Src Port", "Dst IP", "Dst Port"]


def darknet_labels(frame):
    subtype = frame["Label.1"].astype(str).str.lower().str.replace("-", "", regex=False).str.replace(" ", "", regex=False)
    lookup = {"audiostreaming": 1, "browsing": 2, "chat": 3, "email": 4, "p2p": 5, "filetransfer": 6, "videostreaming": 7, "voip": 8}
    labels = subtype.map(lookup)
    labels[frame["Label"].isin(["Non-Tor", "NonVPN"])] = 0
    return labels.to_numpy(dtype=np.int64)


def split(y, seed=42):
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    test, val = train_test_split(rest, test_size=0.4, random_state=seed, stratify=y[rest])
    return np.sort(train), np.sort(test)


def lookup(keys, y, train, test):
    table = pd.DataFrame({"k": keys[train], "y": y[train]}).groupby("k")["y"].agg(lambda s: s.value_counts().idxmax())
    pred = pd.Series(keys[test]).map(table)
    seen = pred.notna().to_numpy()
    return pred.fillna(0).astype(int).to_numpy(), seen


def run(name, frame, y, k):
    train, test = split(y)
    keys = {
        "Dst IP:cổng": (frame["Dst IP"].astype(str) + ":" + frame["Dst Port"].astype(str)).to_numpy(),
        "Dst IP": frame["Dst IP"].astype(str).to_numpy(),
        "Src IP": frame["Src IP"].astype(str).to_numpy(),
    }
    labels = list(range(k))
    out = {}
    for kname, kv in keys.items():
        p, seen = lookup(kv, y, train, test)
        out[kname] = {"f1_weighted": 100 * f1_score(y[test], p, labels=labels, average="weighted", zero_division=0),
                      "f1_macro": 100 * f1_score(y[test], p, labels=labels, average="macro", zero_division=0),
                      "seen_rate": 100 * seen.mean()}
    print(name, json.dumps({a: {b: round(c, 2) for b, c in v.items()} for a, v in out.items()}, ensure_ascii=False))
    return out


df = pd.read_csv(root / "darknet_original" / "Darknet.csv", low_memory=False)
run("darknet", df, darknet_labels(df), 9)
fc = pd.read_parquet(root / "cse_cic" / "cse_cic.parquet")
run("cse_cic", fc, fc["label"].to_numpy(dtype=np.int64), 7)
