"""Attach observed endpoints (Src IP, Src Port, Dst IP) to the recovered AWS rows by
joining them with the DistriNet re-extraction of the same pcaps.

Tiers (each AWS row takes its best tier; every DistriNet flow is used once):
  1 exact : Dst Port, Protocol, start second, Flow Duration, Tot Fwd/Bwd Pkts, TotLen Fwd/Bwd Pkts
  2 relax : Dst Port, Protocol, start second +-1, Tot Fwd/Bwd Pkts, TotLen Fwd/Bwd Pkts,
            |duration difference| <= max(1 ms, 1 %)
  3 loose : Dst Port, Protocol, start second +-2, Tot Fwd/Bwd Pkts,
            |duration difference| <= max(10 ms, 5 %)
The AWS timestamp is a 12-hour clock without AM/PM in AST (UTC-4); both readings are tried.

Usage: python -I join_endpoints.py MATCHED_RAW_PARQUET DISTRINET_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AWS = {"dport": "Dst Port", "proto": "Protocol", "dur": "Flow Duration", "fwd": "Tot Fwd Pkts",
       "bwd": "Tot Bwd Pkts", "lfwd": "TotLen Fwd Pkts", "lbwd": "TotLen Bwd Pkts"}
DN = {"dport": "Dst Port", "proto": "Protocol", "dur": "Flow Duration", "fwd": "Total Fwd Packet",
      "bwd": "Total Bwd packets", "lfwd": "Total Length of Fwd Packet", "lbwd": "Total Length of Bwd Packet"}
TIERS = [
    ("exact", ["dport", "proto", "dur", "fwd", "bwd", "lfwd", "lbwd"], [0], 0.0, 0.0),
    ("relax", ["dport", "proto", "fwd", "bwd", "lfwd", "lbwd"], [0, -1, 1], 1000.0, 0.01),
    ("loose", ["dport", "proto", "fwd", "bwd"], [0, -1, 1, -2, 2], 10000.0, 0.05),
]


def aws_frame(rows):
    t = pd.to_datetime(rows["Timestamp"], format="%d/%m/%Y %H:%M:%S")
    out = pd.DataFrame({"file_row": rows["file_row"].to_numpy()})
    for k, c in AWS.items():
        out[k] = pd.to_numeric(rows[c]).to_numpy().astype(np.int64)
    out["sec_am"] = ((t + pd.Timedelta(hours=4)).astype("int64") // 10**9).to_numpy()
    out["sec_pm"] = ((t + pd.Timedelta(hours=16)).astype("int64") // 10**9).to_numpy()
    return out.drop_duplicates("file_row").reset_index(drop=True)


def dn_frame(path):
    d = pd.read_parquet(path)
    out = pd.DataFrame({"dn_row": np.arange(len(d))})
    for k, c in DN.items():
        out[k] = pd.to_numeric(d[c]).to_numpy().astype(np.int64)
    ts = pd.to_datetime(d["Timestamp"], format="ISO8601")
    out["sec"] = (ts.astype("int64") // 10**9).to_numpy()
    return out, d


def candidates(a, dn, keys, deltas, abs_tol, rel_tol):
    parts = []
    for reading in ("sec_am", "sec_pm"):
        for delta in deltas:
            cols = ["file_row", *keys, *([] if "dur" in keys else ["dur"]), reading]
            left = a[cols].rename(columns={reading: "sec"})
            left = left.assign(sec=left["sec"] + delta)
            if "dur" in keys:
                m = left.merge(dn[["dn_row", "sec", *keys]], on=["sec", *keys])
                m["ddur"] = 0
            else:
                m = left.merge(dn[["dn_row", "sec", "dur", *keys]], on=["sec", *keys], suffixes=("", "_dn"))
                m["ddur"] = (m["dur"] - m["dur_dn"]).abs()
                m = m[m["ddur"] <= np.maximum(abs_tol, rel_tol * m["dur"].abs())]
            m["dsec"] = abs(delta)
            parts.append(m[["file_row", "dn_row", "ddur", "dsec"]])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["file_row", "dn_row"])


def main(matched_parquet, dn_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matched = pd.read_parquet(matched_parquet)
    results, stats = [], {}
    for day, rows in matched.groupby("day", sort=True):
        a = aws_frame(rows)
        dn, draw = dn_frame(Path(dn_dir) / f"{day}.parquet")
        assigned_dn = np.zeros(len(dn), dtype=bool)
        choice = {}
        tier_count = {}
        for rank, (tier, keys, deltas, abs_tol, rel_tol) in enumerate(TIERS):
            todo = a[~a["file_row"].isin(choice.keys())]
            if todo.empty:
                break
            pairs = candidates(todo, dn, keys, deltas, abs_tol, rel_tol)
            pairs = pairs[~assigned_dn[pairs["dn_row"].to_numpy()]]
            n_cand = pairs.groupby("file_row").size()
            # spread identical candidates deterministically; prefer closest time then duration
            rng = np.random.default_rng(42 + rank)
            pairs = pairs.assign(tie=rng.random(len(pairs))).sort_values(["dsec", "ddur", "tie"])
            used_rows = set()
            for fr, dr in zip(pairs["file_row"].to_numpy(), pairs["dn_row"].to_numpy()):
                if fr in used_rows or assigned_dn[dr]:
                    continue
                used_rows.add(fr)
                assigned_dn[dr] = True
                choice[fr] = (dr, tier, int(n_cand.get(fr, 0)))
            tier_count[tier] = len(used_rows)
        sel = pd.DataFrame([(fr, *v) for fr, v in choice.items()], columns=["file_row", "dn_row", "tier", "n_cand"])
        pick = draw.iloc[sel["dn_row"].to_numpy()].reset_index(drop=True)
        sel = pd.concat([sel.reset_index(drop=True),
                         pick[["id", "Src IP", "Src Port", "Dst IP", "Timestamp", "Label", "Attempted Category"]]
                         .rename(columns={"id": "dn_id", "Timestamp": "dn_timestamp", "Label": "dn_label",
                                          "Attempted Category": "dn_attempted",
                                          "Src IP": "Src IP", "Src Port": "Src Port", "Dst IP": "Dst IP"}),
                         pick[["Dst Port"]].rename(columns={"Dst Port": "dn_dst_port"})], axis=1)
        joined = rows.merge(sel, on="file_row", how="left")
        results.append(joined)
        stats[day] = {"eres_rows": len(rows), "aws_rows": len(a), "distrinet_candidates": len(dn),
                      "tiers": tier_count, "eres_with_endpoints": int(joined["Src IP"].notna().sum())}
        print(day, stats[day], flush=True)
    full = pd.concat(results, ignore_index=True).sort_values("eres_row").reset_index(drop=True)
    full.to_parquet(out_dir / "joined.parquet", index=False)
    (out_dir / "join_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
