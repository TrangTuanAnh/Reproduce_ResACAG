"""Map every E-ResGAT CSE-CIC row (edge_feat_scaled.npy) back to its row in the
original AWS "Processed Traffic Data for ML Algorithms" CSVs.

E-ResGAT (prepare_CIC.ipynb) concatenated, in this order:
  50000 random rows 02-14, 50000 random rows 02-15, 50000 random rows 02-21,
  all 362 non-Benign rows 02-22 (file order), all 566 non-Benign rows 02-23 (file order),
  50000 random rows 02-28 (after drop_duplicates/dropna), 50000 random rows 03-02,
then dropped Dst Port and Timestamp and z-scored each of the 77 columns.
The 928 web rows are known exactly, so they pin the z-score (mean, std) of each
column; inverting the z-score gives raw values that are joined to the day files.

Usage: python -I recover_rows.py RAW_DIR REFERENCE_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = [  # (block name, file stem, start, stop, sampled_from)
    ("02-14", "Wednesday-14-02-2018", 0, 50000, "random"),
    ("02-15", "Thursday-15-02-2018", 50000, 100000, "random"),
    ("02-21", "Wednesday-21-02-2018", 100000, 150000, "random"),
    ("02-22", "Thursday-22-02-2018", 150000, 150362, "web_in_order"),
    ("02-23", "Friday-23-02-2018", 150362, 150928, "web_in_order"),
    ("02-28", "Wednesday-28-02-2018", 150928, 200928, "random_dedup"),
    ("03-02", "Friday-02-03-2018", 200928, 250928, "random"),
]
LABELS = {"Benign": 0, "FTP-BruteForce": 1, "SSH-Bruteforce": 1, "DoS attacks-GoldenEye": 2,
          "DoS attacks-Slowloris": 2, "DDOS attack-LOIC-UDP": 3, "DDOS attack-HOIC": 3,
          "Brute Force -Web": 4, "Brute Force -XSS": 4, "SQL Injection": 4, "Bot": 5, "Infilteration": 6}


def load_day(raw_dir, stem, features):
    path = raw_dir / f"{stem}_TrafficForML_CICFlowMeter.csv"
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["file_row"] = np.arange(len(df))  # 0-based data row (line number - 2)
    df = df[df["Label"] != "Label"].copy()  # repeated header rows inside some files
    out = pd.DataFrame({"file_row": df["file_row"].to_numpy(), "Timestamp": df["Timestamp"].to_numpy(),
                        "Label": df["Label"].to_numpy()})
    out["Dst Port"] = pd.to_numeric(df["Dst Port"], errors="coerce").to_numpy()
    for c in features:
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
    return out


def main(raw_dir, ref_dir, out_dir):
    raw_dir, ref_dir, out_dir = Path(raw_dir), Path(ref_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = json.loads((ref_dir / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]
    z = np.load(ref_dir / "edge_feat_scaled.npy", allow_pickle=False).astype(np.float64)
    y = np.load(ref_dir / "label_mul.npy", allow_pickle=False).reshape(-1)
    assert z.shape == (250928, 77) and len(y) == len(z)

    days = {name: load_day(raw_dir, stem, features) for name, stem, *_ in DAYS}

    # 1) exact anchor: the 928 web rows, in file order
    anchor_x = []
    for name in ("02-22", "02-23"):
        d = days[name]
        anchor_x.append(d[d["Label"] != "Benign"][features].to_numpy())
    anchor_x = np.vstack(anchor_x)
    anchor_z = z[150000:150928]
    assert len(anchor_x) == 928 and (y[150000:150928] == 4).all()

    # Pandas z-score of the merged frame: inf was replaced in Flow Byts/s and Flow Pkts/s
    # (1e10 / 1e7), Flow Byts/s NaN -> 0; zero-variance columns became NaN -> 0.
    fill = {"Flow Byts/s": 1e10, "Flow Pkts/s": 1e7}
    affine = {}
    for j, c in enumerate(features):
        xa, za = anchor_x[:, j].copy(), anchor_z[:, j]
        if c in fill:
            xa[np.isposinf(xa)] = fill[c]
        if c == "Flow Byts/s":
            xa[np.isnan(xa)] = 0.0
        ok = np.isfinite(xa)
        if np.ptp(xa[ok]) == 0:
            affine[c] = None if np.ptp(z[:, j]) > 0 else ("constant", float(xa[ok][0]))
            continue
        slope, intercept = np.polyfit(xa[ok], za[ok], 1)
        resid = za[ok] - (slope * xa[ok] + intercept)
        affine[c] = ("fit", slope, intercept, float(np.abs(resid).max()))

    unresolved = [c for c, v in affine.items() if v is None]
    xhat = np.full_like(z, np.nan)
    for j, c in enumerate(features):
        v = affine[c]
        if v is None:
            continue
        xhat[:, j] = v[1] if v[0] == "constant" else (z[:, j] - v[2]) / v[1]

    # 2) join keys: columns whose inverted values are integral everywhere (exact recovery)
    resolved = [c for c in features if affine[c] is not None]
    key_cols = []
    for c in resolved:
        j = features.index(c)
        col = xhat[:, j]
        if np.isfinite(col).all() and np.abs(col - np.round(col)).max() < 0.05 and np.abs(col).max() < 2**31:
            key_cols.append(c)
    print("unresolved (no anchor variance):", unresolved, flush=True)
    print("integral join keys:", len(key_cols), key_cols, flush=True)

    from sklearn.neighbors import KDTree
    cont = [c for c in ["Flow Duration", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Header Len", "Flow IAT Mean",
                        "Flow IAT Max", "Flow IAT Min", "Fwd IAT Tot", "Bwd IAT Tot", "Fwd Pkt Len Mean",
                        "Bwd Pkt Len Mean", "Idle Max", "Active Max"] if c in resolved and c not in key_cols]
    jr = [features.index(c) for c in resolved]
    tol = 1e-5
    records, matched_raw = [], []
    for name, stem, start, stop, how in DAYS:
        d = days[name]
        if how == "web_in_order":
            pool = d[d["Label"] != "Benign"].reset_index(drop=True)
        elif how == "random_dedup":
            # notebook: drop_duplicates() over all 80 columns, dropna(), header row dropped
            pool = d.drop_duplicates(subset=["Dst Port", "Timestamp", *features, "Label"]).dropna(
                subset=["Dst Port", *features]).reset_index(drop=True)
        else:
            pool = d.reset_index(drop=True)
        praw = pool[features].to_numpy(dtype=np.float64)
        pz = praw.copy()
        for c, fv in fill.items():
            k = features.index(c)
            pz[np.isposinf(pz[:, k]), k] = fv
            if c == "Flow Byts/s":
                pz[np.isnan(pz[:, k]), k] = 0.0
        for j, c in enumerate(features):
            v = affine[c]
            if v is not None and v[0] == "fit":
                pz[:, j] = v[1] * pz[:, j] + v[2]
        qz = z[start:stop]
        jk = [features.index(c) for c in key_cols]
        # int64 keys: float hashes would separate -0.0 from 0.0
        left_keys = pd.DataFrame(np.round(xhat[start:stop][:, jk]).astype(np.int64), columns=key_cols)
        right_keys = pd.DataFrame(np.round(np.nan_to_num(praw[:, jk], nan=-2**62)).astype(np.int64), columns=key_cols)
        both = pd.concat([left_keys, right_keys], ignore_index=True)
        gid = pd.util.hash_pandas_object(both, index=False).to_numpy()
        gid = pd.factorize(gid)[0].astype(np.float64)
        jc = [features.index(c) for c in cont]
        qcoord = np.column_stack([gid[:len(qz)] * 1e7, qz[:, jc]])
        pcoord = np.column_stack([gid[len(qz):] * 1e7, np.nan_to_num(pz[:, jc], nan=1e9)])
        tree = KDTree(pcoord, leaf_size=64)
        kq = min(32, len(pcoord))
        dist, ind = tree.query(qcoord, k=kq)
        plabel = pool["Label"].map(LABELS).fillna(15).to_numpy(dtype=np.int64)
        yq = y[start:stop]
        # full verification on every resolved column; among rows within tolerance prefer the
        # E-ResGAT label (the day files list some flows twice, once Benign and once attack)
        best_row = np.full(len(qz), -1)
        best_score = np.full(len(qz), np.inf)
        best_err = np.full(len(qz), np.inf)
        for t in range(kq):
            cand = ind[:, t]
            err = np.abs(np.nan_to_num(pz[cand][:, jr], nan=1e9) - qz[:, jr]) / np.maximum(1.0, np.abs(qz[:, jr]))
            e = err.max(axis=1)
            score = e + (plabel[cand] != yq)
            better = score < best_score
            best_score[better] = score[better]
            best_err[better] = e[better]
            best_row[better] = cand[better]
        ok = best_err <= tol
        # indistinguishable pool rows: identical raw 77-feature vectors and label; E-ResGAT drew
        # without replacement, so spread its rows over distinct members with a fixed seed
        fid = pd.factorize(pd.util.hash_pandas_object(pool[features].fillna(-1.2345e300), index=False).to_numpy())[0]
        gkey = fid.astype(np.int64) * 16 + plabel
        order = np.argsort(gkey, kind="stable")
        uniq, first, count = np.unique(gkey[order], return_index=True, return_counts=True)
        pos = np.searchsorted(uniq, gkey[best_row])
        rng = np.random.default_rng(20180214 + start)
        assigned = best_row.copy()
        reused = np.zeros(len(qz), dtype=bool)
        qidx = np.flatnonzero(ok)
        for gpos in np.unique(pos[qidx]):
            rows = qidx[pos[qidx] == gpos] if count[gpos] > 1 else None
            if rows is None:
                continue
            members = order[first[gpos]:first[gpos] + count[gpos]]
            pick = rng.permutation(members)
            if len(rows) > len(pick):
                reused[rows] = True
                pick = rng.choice(members, len(rows))
            assigned[rows] = pick[:len(rows)]
        g = pd.DataFrame({"gkey": gkey, "dport": pool["Dst Port"].to_numpy(), "ts": pool["Timestamp"].to_numpy()})
        n_same = count[pos]
        n_dport = g.groupby("gkey")["dport"].transform("nunique").to_numpy()[assigned]
        n_ts = g.groupby("gkey")["ts"].transform("nunique").to_numpy()[assigned]
        for i, r in enumerate(range(start, stop)):
            b = assigned[i]
            records.append({"eres_row": r, "day": name, "file_row": int(pool["file_row"].iat[b]) if ok[i] else -1,
                            "err": float(best_err[i]), "n_same_features": int(n_same[i]) if ok[i] else 0,
                            "n_dport_same_features": int(n_dport[i]) if ok[i] else 0,
                            "n_ts_same_features": int(n_ts[i]) if ok[i] else 0, "reused": bool(reused[i]),
                            "label_file": pool["Label"].iat[b] if ok[i] else None, "label_eres": int(y[r])})
        sel = pool.iloc[assigned[ok]].copy()
        sel.insert(0, "eres_row", np.arange(start, stop)[ok])
        sel.insert(1, "day", name)
        matched_raw.append(sel)
        print(name, "verified", int(ok.sum()), "of", stop - start, "max err ok", float(best_err[ok].max()) if ok.any() else None, flush=True)

    pd.concat(matched_raw, ignore_index=True).to_parquet(out_dir / "matched_raw.parquet", index=False)
    rec = pd.DataFrame(records)
    rec["label_ok"] = rec["label_file"].map(LABELS) == rec["label_eres"]
    rec.to_csv(out_dir / "row_map.csv", index=False)
    summary = {
        "rows": len(rec), "matched": int((rec.file_row >= 0).sum()),

        "label_agree": int(rec.label_ok.sum()),
        "unique_feature_vector": int((rec.n_same_features == 1).sum()),
        "dup_features_same_dport": int(((rec.n_same_features > 1) & (rec.n_dport_same_features == 1)).sum()),
        "dup_features_dport_differs": int((rec.n_dport_same_features > 1).sum()),
        "dup_features_ts_differs": int((rec.n_ts_same_features > 1).sum()),
        "reused_pool_rows": int(rec.reused.sum()),
        "per_day": rec.groupby("day").agg(n=("eres_row", "size"), matched=("file_row", lambda s: int((s >= 0).sum())),
                                           unique=("n_same_features", lambda s: int((s == 1).sum())),
                                           dport_amb=("n_dport_same_features", lambda s: int((s > 1).sum()))).to_dict("index"),
        "key_columns": key_cols, "unresolved_columns": unresolved,
        "affine": {c: (None if v is None else list(v)) for c, v in affine.items()},
    }
    (out_dir / "row_map_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("affine",)}, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:4])
