"""Tính lại các thống kê EDA cho CẢ HAI bộ dữ liệu (darknet, cse_cic) với cùng một định nghĩa.

Cách chạy:  python -I summarize_datasets.py DATA_ROOT OUT_JSON
DATA_ROOT chứa darknet_original/Darknet.csv và cse_cic/{cse_cic.parquet,feature_names.json}.
Chia tập: phân tầng theo nhãn, 50% train, 20% validation, 30% test, seed 42
(giống resacag/data.py::split_indices).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

root, out_json = Path(sys.argv[1]), Path(sys.argv[2])

DARK_CLASSES = ["Benign", "Audio Stream", "Browsing", "Chat", "Email", "P2P", "Transfer", "Video Stream", "VOIP"]
CSE_CLASSES = ["Benign", "Brute-force", "DoS", "DDoS", "Web Attack", "Botnet", "Infiltration"]
ENDPOINTS = ["Src IP", "Src Port", "Dst IP", "Dst Port"]


def darknet_labels(frame):
    subtype = frame["Label.1"].astype(str).str.lower().str.replace("-", "", regex=False).str.replace(" ", "", regex=False)
    lookup = {"audiostreaming": 1, "browsing": 2, "chat": 3, "email": 4, "p2p": 5,
              "filetransfer": 6, "videostreaming": 7, "voip": 8}
    benign = frame["Label"].isin(["Non-Tor", "NonVPN"])
    labels = subtype.map(lookup)
    labels[benign] = 0
    assert not labels.isna().any()
    return labels.to_numpy(dtype=np.int64)


def endpoints(frame):
    s = pd.factorize(frame["Src IP"].astype(str) + ":" + frame["Src Port"].astype(str))[0]
    d = pd.factorize(frame["Dst IP"].astype(str) + ":" + frame["Dst Port"].astype(str))[0]
    return s.astype(np.int64), d.astype(np.int64)


def split(y, seed=42):
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    test, val = train_test_split(rest, test_size=0.4, random_state=seed, stratify=y[rest])
    return np.sort(train), np.sort(val), np.sort(test)


def group_ids(x):
    """Mỗi dòng đủ 77 giá trị (float32) -> một mã nhóm; NaN/inf được ghi nhận riêng."""
    x = np.ascontiguousarray(x.astype(np.float32))
    x = np.where(np.isnan(x), np.float32(-9.87654e30), x)
    void = x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()
    return np.unique(void, return_inverse=True)[1].astype(np.int64)


def analyse(name, classes, x, y, src, dst, extra=None):
    k = len(classes)
    n = len(y)
    res = {"name": name, "classes": classes, "rows": int(n), "features": int(x.shape[1])}
    res["class_counts"] = np.bincount(y, minlength=k).tolist()
    train, val, test = split(y)
    res["split"] = {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))}

    # --- chất lượng đặc trưng
    finite = np.isfinite(x)
    res["invalid_values"] = int((~finite).sum())
    res["invalid_rows"] = int((~finite).any(axis=1).sum())
    res["invalid_columns"] = int((~finite).any(axis=0).sum())
    const = 0
    for j in range(x.shape[1]):
        col = x[:, j]
        v = col[np.isfinite(col)]
        if (len(v) == 0) or (v.min() == v.max() and finite[:, j].all()):
            const += 1
    res["constant_columns"] = const

    # --- nhóm trùng đặc trưng
    gid = group_ids(x)
    g = gid.max() + 1
    cnt_all = np.zeros((g, k), dtype=np.int32)
    np.add.at(cnt_all, (gid, y), 1)
    size = cnt_all.sum(1)
    res["groups"] = int(g)
    res["duplicate_rows_beyond_first"] = int((size - 1).sum())
    multi = (cnt_all > 0).sum(1) > 1
    res["multilabel_groups"] = int(multi.sum())
    res["multilabel_rows"] = int(size[multi].sum())
    in_multi = multi[gid]
    res["multilabel_rows_by_class"] = [int(((y == c) & in_multi).sum()) for c in range(k)]

    cnt_tr = np.zeros((g, k), dtype=np.int32)
    np.add.at(cnt_tr, (gid[train], y[train]), 1)
    gt = gid[test]
    yt = y[test]
    seen = cnt_tr[gt].sum(1) > 0
    other = (cnt_tr[gt].sum(1) - cnt_tr[gt, yt]) > 0
    res["overlap_test_total"] = int(seen.sum())
    res["overlap_test_rate"] = float(seen.mean())
    res["test_by_class"] = np.bincount(yt, minlength=k).tolist()
    res["overlap_by_class"] = [float(seen[yt == c].mean()) if (yt == c).any() else None for c in range(k)]
    res["overlap_other_label_by_class"] = [float(other[yt == c].mean()) if (yt == c).any() else None for c in range(k)]
    res["overlap_other_label_counts"] = [int(other[yt == c].sum()) for c in range(k)]
    res["overlap_counts"] = [int(seen[yt == c].sum()) for c in range(k)]
    res["overlap_other_label_total"] = int(other.sum())

    # --- cấu trúc đồ thị
    n_src = src.max() + 1
    nodes = np.concatenate([src, dst + n_src])
    labels2 = np.concatenate([y, y])
    nn = nodes.max() + 1
    deg = np.bincount(nodes, minlength=nn)
    active = deg > 0
    res["graph"] = {"nodes": int(active.sum()), "deg1_nodes": int((deg == 1).sum()),
                    "deg_mean": float(deg[active].mean()), "deg_max": int(deg.max()),
                    "src_nodes": int(len(np.unique(src))), "dst_nodes": int(len(np.unique(dst)))}
    cnt = np.zeros((nn, k), dtype=np.int64)
    np.add.at(cnt, (nodes, labels2), 1)
    u, v = src, dst + n_src
    same = cnt[u, y] - 1 + cnt[v, y] - 1
    total = deg[u] - 1 + deg[v] - 1
    has = total > 0
    h = np.where(has, same / np.maximum(total, 1), np.nan)
    res["homophily_overall"] = float(np.nanmean(h))
    res["homophily_by_class"] = [float(np.nanmean(h[y == c])) if has[y == c].any() else None for c in range(k)]
    res["with_neighbors_by_class"] = [float(has[y == c].mean()) for c in range(k)]

    # --- nút chưa thấy trong train (test, seed 42)
    tr_src = np.zeros(n_src, dtype=bool)
    tr_src[src[train]] = True
    n_dst = dst.max() + 1
    tr_dst = np.zeros(n_dst, dtype=bool)
    tr_dst[dst[train]] = True
    s_seen = tr_src[src[test]]
    d_seen = tr_dst[dst[test]]
    both = s_seen & d_seen
    one = s_seen ^ d_seen
    none = ~s_seen & ~d_seen
    res["unseen_by_class"] = {
        "both_seen": [float(both[yt == c].mean()) for c in range(k)],
        "one_unseen": [float(one[yt == c].mean()) for c in range(k)],
        "both_unseen": [float(none[yt == c].mean()) for c in range(k)]}
    res["unseen_total"] = {"both_seen": float(both.mean()), "one_unseen": float(one.mean()), "both_unseen": float(none.mean()),
                           "src_unseen": float((~s_seen).mean())}
    test_nodes = np.unique(np.concatenate([src[test], dst[test] + n_src]))
    tr_nodes = np.unique(np.concatenate([src[train], dst[train] + n_src]))
    new_nodes = np.setdiff1d(test_nodes, tr_nodes)
    res["test_nodes"] = int(len(test_nodes))
    res["test_new_nodes"] = int(len(new_nodes))
    res["test_new_node_rate"] = float(len(new_nodes) / len(test_nodes))

    if extra:
        res.update(extra(train, val, test))
    return res


def time_info(order_values, valid, y, k):
    """Chia theo thứ tự thời gian 50/20/30 trên các dòng có thời điểm hợp lệ."""
    idx = np.flatnonzero(valid)
    idx = idx[np.lexsort((idx, order_values[idx]))]
    a, b = int(0.5 * len(idx)), int(0.7 * len(idx))  # 50% huấn luyện, 20% chọn mô hình, 30% kiểm tra
    parts = {"train": idx[:a], "val": idx[a:b], "test": idx[b:]}
    return {"used_rows": int(len(idx)), "dropped_rows": int((~valid).sum()),
            "dropped_by_class": [int(((y == c) & ~valid).sum()) for c in range(k)],
            "table": {p: np.bincount(y[i], minlength=k).tolist() for p, i in parts.items()}}


out = {}

# ----------------------------------------------------------------- darknet
df = pd.read_csv(root / "darknet_original" / "Darknet.csv", low_memory=False)
excluded = set(ENDPOINTS + ["Flow ID", "Timestamp", "Label", "Label.1"])
features = [c for c in df.columns if c not in excluded]
xd = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
yd = darknet_labels(df)
sd, dd = endpoints(df)

ts = df["Timestamp"].astype(str)
t_clear = pd.to_datetime(ts, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
valid_d = t_clear.notna().to_numpy()
tsec = (t_clear.astype("int64") // 10**9).to_numpy()
tsec = np.where(valid_d, tsec, 0)

month = t_clear.dt.strftime("%Y-%m")
month_counts = {}
for m in sorted(month.dropna().unique()):
    month_counts[m] = np.bincount(yd[(month == m).to_numpy()], minlength=len(DARK_CLASSES)).tolist()
day = t_clear.dt.strftime("%Y-%m-%d")

out["darknet"] = analyse("darknet", DARK_CLASSES, xd, yd, sd, dd,
                         extra=lambda tr, va, te: {"time_split": time_info(tsec, valid_d, yd, len(DARK_CLASSES))})
out["darknet"]["by_month"] = month_counts
out["darknet"]["timestamp_unreadable_rows"] = int((~valid_d).sum())
out["darknet"]["n_days"] = int(day.nunique())

# ----------------------------------------------------------------- cse_cic
frame = pd.read_parquet(root / "cse_cic" / "cse_cic.parquet")
fe = json.loads((root / "cse_cic" / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]
xc = frame[fe].to_numpy(dtype=np.float32)
yc = frame["label"].to_numpy(dtype=np.int64)
sc, dc = endpoints(frame)
start = frame["start_us"].to_numpy(dtype=np.float64)
valid_c = np.isfinite(start)
out["cse_cic"] = analyse("cse_cic", CSE_CLASSES, xc, yc, sc, dc,
                         extra=lambda tr, va, te: {"time_split": time_info(np.where(valid_c, start, 0), valid_c, yc, len(CSE_CLASSES))})
days = sorted(frame["day"].astype(str).unique())
out["cse_cic"]["by_day"] = {d: np.bincount(yc[(frame["day"].astype(str) == d).to_numpy()], minlength=len(CSE_CLASSES)).tolist() for d in days}

out_json.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k_, r in out.items():
    print(k_, {a: r[a] for a in ("rows", "split", "invalid_rows", "invalid_values", "constant_columns", "groups",
                                 "duplicate_rows_beyond_first", "multilabel_groups", "multilabel_rows",
                                 "overlap_test_total", "overlap_test_rate", "homophily_overall",
                                 "test_nodes", "test_new_nodes", "test_new_node_rate")})
    print("  graph", r["graph"])
    print("  unseen_total", r["unseen_total"])
