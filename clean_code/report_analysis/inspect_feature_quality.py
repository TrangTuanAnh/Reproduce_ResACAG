"""Số liệu bổ sung cho EDA của cả hai bộ: cột hằng, cột không hợp lệ, cặp tương quan, IP chưa gặp, trùng với Benign."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path(sys.argv[1])
runs = {"darknet": sys.argv[2], "cse_cic": sys.argv[3]}
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
    return np.sort(train), np.sort(val), np.sort(test)


def group_ids(x):
    x = np.ascontiguousarray(x.astype(np.float32))
    x = np.where(np.isnan(x), np.float32(-9.87654e30), x)
    void = x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()
    return np.unique(void, return_inverse=True)[1].astype(np.int64)


out = {}
df = pd.read_csv(root / "darknet_original" / "Darknet.csv", low_memory=False)
excluded = set(ENDPOINTS + ["Flow ID", "Timestamp", "Label", "Label.1"])
feats = [c for c in df.columns if c not in excluded]
frames = {"darknet": (df, feats, darknet_labels(df))}
fc = pd.read_parquet(root / "cse_cic" / "cse_cic.parquet")
fe = json.loads((root / "cse_cic" / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]
frames["cse_cic"] = (fc, fe, fc["label"].to_numpy(dtype=np.int64))

for name, (frame, features, y) in frames.items():
    x = frame[features].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    r = {}
    arr = x.to_numpy()
    fin = np.isfinite(arr)
    r["invalid_column_names"] = {c: int((~fin[:, j]).sum()) for j, c in enumerate(features) if (~fin[:, j]).any()}
    const = []
    for j, c in enumerate(features):
        v = arr[fin[:, j], j]
        if len(v) and v.min() == v.max() and fin[:, j].all():
            const.append(c)
    r["constant_columns"] = const
    train, val, test = split(y)
    # tương quan tuyến tính tuyệt đối > 0,99 trên train (bỏ cột hằng, giá trị không hữu hạn -> NaN)
    xt = x.iloc[train].replace([np.inf, -np.inf], np.nan)
    xt = xt[[c for c in features if c not in const]]
    corr = xt.corr().abs()
    iu = np.triu_indices(len(corr), 1)
    vals = corr.to_numpy()[iu]
    r["pairs_over_0_99"] = int((vals > 0.99).sum())
    r["pairs_equal_1"] = int((vals > 0.999999).sum())
    # trùng với train: nhãn Benign cụ thể
    gid = group_ids(arr)
    benign_train = set(gid[train][y[train] == 0])
    yt, gt = y[test], gid[test]
    in_benign = np.array([g in benign_train for g in gt])
    r["test_overlap_with_benign_by_class"] = [int((in_benign & (yt == c)).sum()) for c in range(y.max() + 1)]
    r["test_by_class"] = np.bincount(yt, minlength=y.max() + 1).tolist()
    # IP chưa gặp
    sip = frame["Src IP"].astype(str).to_numpy()
    dip = frame["Dst IP"].astype(str).to_numpy()
    spk = (frame["Src IP"].astype(str) + ":" + frame["Src Port"].astype(str)).to_numpy()
    tr_sip = set(sip[train]); tr_dip = set(dip[train]); tr_spk = set(spk[train])
    s_ip_new = np.array([a not in tr_sip for a in sip[test]])
    s_pk_new = np.array([a not in tr_spk for a in spk[test]])
    r["test_src_ipport_new"] = int(s_pk_new.sum())
    r["test_src_ipport_new_but_ip_seen"] = int((s_pk_new & ~s_ip_new).sum())
    r["test_src_ip_new"] = int(s_ip_new.sum())
    r["test_src_ip_new_rate"] = float(s_ip_new.mean())
    r["test_n"] = int(len(test))
    out[name] = r

# kết quả từng lớp từ metrics
for name, p in runs.items():
    m = json.loads((Path(p) / "metrics.json").read_text(encoding="utf-8"))
    out[name]["per_class_result"] = m["test"]["per_class"]
    out[name]["accuracy"] = m["test"]["accuracy"]
    out[name]["f1_weighted"] = m["test"]["f1_weighted"]
    out[name]["f1_macro"] = m["test"]["f1_macro"]
    out[name]["confusion_matrix"] = m["test"]["confusion_matrix"]

Path(sys.argv[4]).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k, r in out.items():
    print(k, {a: r[a] for a in ("invalid_column_names", "constant_columns", "pairs_over_0_99", "pairs_equal_1",
                                "test_overlap_with_benign_by_class", "test_src_ipport_new", "test_src_ipport_new_but_ip_seen",
                                "test_src_ip_new", "test_src_ip_new_rate", "test_n")})
