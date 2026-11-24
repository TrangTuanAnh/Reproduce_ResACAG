"""One endpoint per recovered AWS row, then per E-ResGAT row.

Primary source: pcap re-split candidate pairs (work/pcap/<day>.pairs.csv). Each pcap flow is
used once (closest duration first, seeded tie-break); AWS rows still unassigned may reuse a
flow, which happens when the AWS CSV lists the same flow twice (02-28 host 172.31.69.9 block).
Fallback: exact-tier DistriNet matches (work/join/joined.parquet).

Usage: python -I assign_endpoints.py MATCHED_RAW_PARQUET PCAP_DIR DISTRINET_JOINED OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def assign(pairs, seed):
    pairs = pairs.copy()
    pairs["flow"] = pd.factorize(pairs["member"] + "|" + pairs["src"] + "|" + pairs["sport"].astype(str) + "|" +
                                 pairs["dst"] + "|" + pairs["dport"].astype(str) + "|" + pairs["start"].astype(str))[0]
    n_cand = pairs.groupby("file_row").size()
    n_ippairs = pairs.groupby("file_row")[["src", "dst"]].apply(lambda g: len(g.drop_duplicates()))
    rng = np.random.default_rng(seed)
    pairs = pairs.assign(tie=rng.random(len(pairs))).sort_values(["ddur", "tie"])
    used_flow, chosen = set(), {}
    for fr, fl, idx in zip(pairs["file_row"].to_numpy(), pairs["flow"].to_numpy(), pairs.index.to_numpy()):
        if fr in chosen or fl in used_flow:
            continue
        chosen[fr] = (idx, "pcap")
        used_flow.add(fl)
    for fr, idx in zip(pairs["file_row"].to_numpy(), pairs.index.to_numpy()):
        if fr not in chosen:
            chosen[fr] = (idx, "pcap_reused_flow")
    rows = pd.DataFrame([(fr, idx, tier) for fr, (idx, tier) in chosen.items()], columns=["file_row", "idx", "tier"])
    sel = pairs.loc[rows["idx"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame({"file_row": rows["file_row"].to_numpy(), "tier": rows["tier"].to_numpy(),
                        "Src IP": sel["src"].to_numpy(), "Src Port": sel["sport"].to_numpy(),
                        "Dst IP": sel["dst"].to_numpy(), "ep_dst_port": sel["dport"].to_numpy(),
                        "start_us": sel["start"].to_numpy(), "duration_diff_us": sel["ddur"].to_numpy(),
                        "pcap_member": sel["member"].to_numpy()})
    out["n_cand"] = out["file_row"].map(n_cand).to_numpy()
    out["n_ip_pairs"] = out["file_row"].map(n_ippairs).to_numpy()
    return out


def main(matched_parquet, pcap_dir, dn_joined, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matched = pd.read_parquet(matched_parquet)
    dn = pd.read_parquet(dn_joined)
    results, stats = [], {}
    for day, rows in matched.groupby("day", sort=True):
        path = Path(pcap_dir) / f"{day}.pairs.csv"
        pairs = pd.read_csv(path, dtype={"src": str, "dst": str}) if path.exists() else pd.DataFrame()
        pairs = pairs.drop_duplicates().reset_index(drop=True)  # a member may be re-run after an interrupted job
        # Each AWS CSV is a concatenation of per-host blocks; keep a flow only when its capture host has a
        # dense block (pcap_flows.py block_lo/block_hi) around the AWS row, which drops coincidental matches
        # from hosts that are not part of the CSV.
        hosts = Path(pcap_dir) / f"{day}.hosts.jsonl"
        if len(pairs) and hosts.exists():
            blocks = {}
            for line in hosts.read_text(encoding="utf-8").splitlines():
                info = json.loads(line)
                if info.get("block_lo", -1) >= 0:
                    blocks[info["member"]] = (info["block_lo"] - 1000, info["block_hi"] + 1000)
            lo = pairs["member"].map(lambda m: blocks.get(m, (1, 0))[0])
            hi = pairs["member"].map(lambda m: blocks.get(m, (1, 0))[1])
            inside = (pairs["file_row"] >= lo) & (pairs["file_row"] <= hi)
            stats_dropped = int((~inside).sum())
            pairs = pairs[inside].reset_index(drop=True)
        else:
            stats_dropped = 0
        ep = assign(pairs, 42) if len(pairs) else pd.DataFrame(
            columns=["file_row", "tier", "Src IP", "Src Port", "Dst IP", "ep_dst_port", "start_us",
                     "duration_diff_us", "pcap_member", "n_cand", "n_ip_pairs"])
        joined = rows.merge(ep, on="file_row", how="left")
        # fallback: DistriNet exact matches for rows the pcap pass did not cover
        d = dn[(dn["day"] == day) & (dn["tier"] == "exact")][["eres_row", "Src IP", "Src Port", "Dst IP", "dn_id"]]
        miss = joined["Src IP"].isna()
        if miss.any() and len(d):
            fb = joined.loc[miss, ["eres_row"]].merge(d, on="eres_row", how="inner")
            fb_idx = joined.index[miss][joined.loc[miss, "eres_row"].isin(fb["eres_row"]).to_numpy()]
            fbm = fb.set_index("eres_row").loc[joined.loc[fb_idx, "eres_row"]]
            joined.loc[fb_idx, "Src IP"] = fbm["Src IP"].to_numpy()
            joined.loc[fb_idx, "Src Port"] = pd.to_numeric(fbm["Src Port"]).to_numpy()
            joined.loc[fb_idx, "Dst IP"] = fbm["Dst IP"].to_numpy()
            joined.loc[fb_idx, "tier"] = "distrinet_exact"
        assert (joined["ep_dst_port"].isna() | (joined["ep_dst_port"] == pd.to_numeric(joined["Dst Port"]))).all()
        results.append(joined)
        tiers = joined["tier"].value_counts().to_dict()
        stats[day] = {"eres_rows": len(rows), "with_endpoints": int(joined["Src IP"].notna().sum()), "tiers": tiers,
                      "rows_with_several_ip_pairs": int((joined["n_ip_pairs"] > 1).sum()),
                      "rows_with_several_candidates": int((joined["n_cand"] > 1).sum()),
                      "pairs_outside_host_block_dropped": stats_dropped}
        print(day, stats[day], flush=True)
    full = pd.concat(results, ignore_index=True).sort_values("eres_row").reset_index(drop=True)
    full.to_parquet(out_dir / "joined.parquet", index=False)
    (out_dir / "assign_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:5])
