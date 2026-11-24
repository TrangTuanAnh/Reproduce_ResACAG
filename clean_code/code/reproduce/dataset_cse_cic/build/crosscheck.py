"""Agreement between the two independent endpoint sources on rows both could match:
pcap re-split (primary) vs DistriNet exact-tier join.

Usage: python -I crosscheck.py PCAP_ASSIGN_DIR DISTRINET_JOIN_DIR [PCAP_PAIRS_DIR]
"""
import json
import sys
from pathlib import Path

import pandas as pd


def main(assign_dir, dn_dir, pairs_dir=None):
    a = pd.read_parquet(Path(assign_dir) / "joined.parquet")
    d = pd.read_parquet(Path(dn_dir) / "joined.parquet")
    a = a[a["tier"].isin(["pcap", "pcap_reused_flow"])][["eres_row", "day", "Src IP", "Src Port", "Dst IP",
                                                          "n_ip_pairs", "label_eres" if "label_eres" in a else "Label"]]
    d = d[d["tier"] == "exact"][["eres_row", "Src IP", "Src Port", "Dst IP"]]
    m = a.merge(d, on="eres_row", suffixes=("_pcap", "_dn"))
    m["Src Port_pcap"] = pd.to_numeric(m["Src Port_pcap"])
    m["Src Port_dn"] = pd.to_numeric(m["Src Port_dn"])
    unambiguous = m["n_ip_pairs"] == 1
    out = {"rows_both": len(m),
           "same_ip_pair": float(((m["Src IP_pcap"] == m["Src IP_dn"]) & (m["Dst IP_pcap"] == m["Dst IP_dn"])).mean()),
           "same_ip_pair_unambiguous": float(((m["Src IP_pcap"] == m["Src IP_dn"]) &
                                               (m["Dst IP_pcap"] == m["Dst IP_dn"]))[unambiguous].mean()),
           "same_4tuple": float(((m["Src IP_pcap"] == m["Src IP_dn"]) & (m["Dst IP_pcap"] == m["Dst IP_dn"]) &
                                 (m["Src Port_pcap"] == m["Src Port_dn"])).mean()),
           "by_day": {}}
    for day, g in m.groupby("day"):
        out["by_day"][day] = {"rows": len(g),
                              "same_ip_pair": float(((g["Src IP_pcap"] == g["Src IP_dn"]) &
                                                      (g["Dst IP_pcap"] == g["Dst IP_dn"])).mean()),
                              "same_4tuple": float(((g["Src IP_pcap"] == g["Src IP_dn"]) &
                                                    (g["Dst IP_pcap"] == g["Dst IP_dn"]) &
                                                    (g["Src Port_pcap"] == g["Src Port_dn"])).mean())}
    Path(assign_dir, "crosscheck.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
