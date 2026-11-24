"""Assemble cse_cic.parquet: E-ResGAT rows and labels, raw CICFlowMeter-V3 features from the
AWS day CSVs, endpoints of the matching pcap flows; rows without an observed endpoint are dropped.

The SHA-256 of the result is compared with build/expected_dataset.json (the packaged file).

Usage: python -I build_dataset.py JOINED_PARQUET ERESGAT_DIR ROW_MAP_DIR ENDPOINTS_DIR OUT_DIR
  JOINED_PARQUET  output of assign_endpoints.py
  ERESGAT_DIR     output of fetch_eresgat.py (label_mul.npy, feature_names.json)
  ROW_MAP_DIR     output of recover_rows.py (row_map_summary.json)
  ENDPOINTS_DIR   folder with assign_stats.json and crosscheck.json
  OUT_DIR         dataset folder (writes cse_cic.parquet, provenance.json)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Benign", "Brute-force", "DoS", "DDoS", "Web Attack", "Botnet", "Infiltration"]


def main(joined_path, eresgat_dir, row_map_dir, endpoints_dir, out_dir):
    eresgat_dir, out_dir, endpoints_dir = Path(eresgat_dir), Path(out_dir), Path(endpoints_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = json.loads((eresgat_dir / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]
    y = np.load(eresgat_dir / "label_mul.npy", allow_pickle=False).reshape(-1).astype(np.int64)
    joined = pd.read_parquet(joined_path)
    frame = joined[joined["Src IP"].notna()].copy()
    out = pd.DataFrame({"eres_row": frame["eres_row"].to_numpy(), "day": frame["day"].to_numpy(),
                        "aws_file_row": frame["file_row"].to_numpy(), "Timestamp": frame["Timestamp"].to_numpy(),
                        "label": y[frame["eres_row"].to_numpy()], "aws_label": frame["Label"].to_numpy(),
                        "join_tier": frame["tier"].to_numpy(), "join_candidates": frame["n_cand"].to_numpy(),
                        "join_ip_pairs": frame["n_ip_pairs"].to_numpy(), "pcap_member": frame["pcap_member"].to_numpy(),
                        "start_us": frame["start_us"].to_numpy()})
    out["Src IP"] = frame["Src IP"].astype(str).to_numpy()
    out["Src Port"] = pd.to_numeric(frame["Src Port"]).astype(np.int64).to_numpy()
    out["Dst IP"] = frame["Dst IP"].astype(str).to_numpy()
    out["Dst Port"] = pd.to_numeric(frame["Dst Port"]).astype(np.int64).to_numpy()
    for c in features:
        out[c] = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=np.float64)
    out = out.sort_values("eres_row").reset_index(drop=True)
    path = out_dir / "cse_cic.parquet"
    out.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    counts_all = np.bincount(y, minlength=7)
    counts_kept = np.bincount(out["label"], minlength=7)
    summary = json.loads((Path(row_map_dir) / "row_map_summary.json").read_text(encoding="utf-8"))
    cross = json.loads((endpoints_dir / "crosscheck.json").read_text(encoding="utf-8"))
    provenance = {
        "dataset": "CSE-CIC-IDS2018 subset published by E-ResGAT (size and class mix match ResACAG Table 1), with observed endpoints",
        "file": "cse_cic.parquet", "sha256": digest, "rows": int(len(out)), "rows_reference_subset": int(len(y)),
        "class_counts": dict(zip(CLASSES, counts_kept.tolist())),
        "class_counts_reference_subset": dict(zip(CLASSES, counts_all.tolist())),
        "class_coverage": dict(zip(CLASSES, (counts_kept / np.maximum(counts_all, 1)).round(6).tolist())),
        "endpoint_source_tiers": out["join_tier"].value_counts().to_dict(),
        "per_day": json.loads((endpoints_dir / "assign_stats.json").read_text(encoding="utf-8")),
        "columns": {"label": "E-ResGAT label_mul (0 Benign, 1 Brute-force, 2 DoS, 3 DDoS, 4 Web Attack, 5 Botnet, 6 Infiltration)",
                    "Src IP/Src Port/Dst IP/Dst Port": "endpoints of the matching flow re-split from the original pcap",
                    "77 features": "raw AWS CSV values (inf/NaN kept; the train-only scaler imputes them)",
                    "audit": "eres_row, day, aws_file_row, Timestamp, aws_label, join_tier, join_candidates, join_ip_pairs, pcap_member, start_us"},
        "topology": "observed_ip_port_role_bipartite",
        "crosscheck_distrinet": {k: cross[k] for k in ("rows_both", "same_ip_pair", "same_4tuple")},
        "row_recovery": {k: summary[k] for k in ("rows", "matched", "label_agree", "unique_feature_vector",
                                                   "dup_features_same_dport", "dup_features_dport_differs")},
        "known_limitations": "README.md section 'Giới hạn'",
        "sources": {
            "eresgat_inputs": "github.com/George730/E-ResGAT @ 3e0ceb8f73e146eb60041ff122449efacb539769 (build/eresgat_inputs.json)",
            "aws_processed_csv": "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed Traffic Data for ML Algorithms/<Day>_TrafficForML_CICFlowMeter.csv",
            "pcap": "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Original Network Traffic and Log data/<Day>/pcap.zip (byte-identical mirror huggingface.co/datasets/bencorn/CICIDS2018)",
            "fallback_and_crosscheck": "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip (Liu, Engelen, Lynar, Essam, Joosen, IEEE CNS 2022)"}}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: provenance[k] for k in ("rows", "sha256", "class_counts")}, indent=1))
    expected_path = Path(__file__).resolve().parent / "expected_dataset.json"
    if expected_path.exists():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))["sha256"]
        print("MATCH packaged cse_cic.parquet" if digest == expected else
              f"DIFFERENT from packaged cse_cic.parquet (expected {expected[:12]}...); do not replace the packaged file blindly")


if __name__ == "__main__":
    main(*sys.argv[1:6])
