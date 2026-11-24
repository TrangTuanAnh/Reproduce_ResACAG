"""Numeric, non-pickle data loading with explicit provenance and split boundaries."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DARKNET_CLASSES = ["Benign", "Audio Stream", "Browsing", "Chat", "Email", "P2P", "Transfer", "Video Stream", "VOIP"]
CSE_CLASSES = ["Benign", "Brute-force", "DoS", "DDoS", "Web Attack", "Botnet", "Infiltration"]
ENDPOINTS = ["Src IP", "Src Port", "Dst IP", "Dst Port"]
# Audit columns of cse_cic.parquet: row order, capture time and join metadata must never become features.
CSE_NON_FEATURES = set(ENDPOINTS) | {"Timestamp", "label", "eres_row", "day", "aws_file_row", "aws_label", "join_tier",
                                    "join_candidates", "join_ip_pairs", "pcap_member", "start_us"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    endpoints: np.ndarray
    classes: list[str]
    feature_names: list[str]
    metadata: dict


def endpoint_ids(frame: pd.DataFrame) -> np.ndarray:
    missing = set(ENDPOINTS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing real graph endpoints: {sorted(missing)}. No inferred or invented IPs are allowed.")
    if frame[ENDPOINTS].isna().any().any():
        raise ValueError("Null graph endpoints")
    # Separate source/destination namespaces guarantees a bipartite graph without relabeling IPs.
    keys = [frame[ip].astype(str) + ":" + frame[port].astype(str)
            for ip, port in [("Src IP", "Src Port"), ("Dst IP", "Dst Port")]]
    src, source_names = pd.factorize(keys[0], sort=True)
    dst, _ = pd.factorize(keys[1], sort=True)
    return np.column_stack([src, dst + len(source_names)]).astype(np.int64)


def darknet_labels(frame):
    subtype = frame["Label.1"].astype(str).str.lower().str.replace("-", "", regex=False).str.replace(" ", "", regex=False)
    lookup = {"audiostreaming": 1, "browsing": 2, "chat": 3, "email": 4, "p2p": 5,
              "filetransfer": 6, "videostreaming": 7, "voip": 8}
    benign = frame["Label"].isin(["Non-Tor", "NonVPN"])
    labels = subtype.map(lookup)
    labels[benign] = 0
    if labels.isna().any() or not frame["Label"].isin(["Non-Tor", "NonVPN", "Tor", "VPN"]).all():
        raise ValueError("Unknown Darknet label; do not silently map it to Benign")
    return labels.to_numpy(dtype=np.int64)


def load_dataset(name: str, data_root: Path) -> Dataset:
    if name == "darknet":
        path = data_root / "darknet_original" / "Darknet.csv"
        df = pd.read_csv(path, low_memory=False)
        excluded = set(ENDPOINTS + ["Flow ID", "Timestamp", "Label", "Label.1"])
        features = [c for c in df.columns if c not in excluded]
        x = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        return Dataset(x, darknet_labels(df), endpoint_ids(df), DARKNET_CLASSES, features,
                       {"name": name, "path": str(path.resolve()), "sha256": sha256(path),
                        "topology": "observed_ip_port_role_bipartite", "rows": len(df),
                        "provenance": json.loads(path.with_name("provenance.json").read_text(encoding="utf-8"))})
    if name == "cse_reference":
        root = data_root / "reference_cse"
        x = np.load(root / "edge_feat_scaled.npy", allow_pickle=False).astype(np.float32)
        y = np.load(root / "label_mul.npy", allow_pickle=False).reshape(-1).astype(np.int64)
        endpoints = np.load(root / "adj_random.npy", allow_pickle=False).astype(np.int64)
        # Obtain the 77 documented column names from the original CSV header, without loading it.
        columns = pd.read_csv(data_root / "02-14-2018.csv", nrows=0).columns
        features = [c for c in columns if c not in ["Dst Port", "Timestamp", "Label"]]
        if x.ndim != 2 or x.shape[1] != len(features) or len(y) != len(x) or endpoints.shape != (len(x), 2):
            raise ValueError("Inconsistent reference numeric artifacts")
        return Dataset(x, y, endpoints, CSE_CLASSES, features,
                       {"name": name, "rows": len(y), "topology": "published_synthetic_random",
                        "warning": "Not observed network topology; reference [24], not verified ResACAG data. Upstream full-data standardization/imputation.",
                        "provenance": json.loads((root / "provenance.json").read_text(encoding="utf-8")),
                        "sha256": sha256(root / "edge_feat_scaled.npy")})
    if name == "cse_cic":
        # Same E-ResGAT CSE-CIC rows with raw CICFlowMeter-V3 features and endpoints re-observed in the
        # original pcaps (lab/5); code/reproduce/dataset_cse_cic/README.md documents the build.
        root = data_root / "cse_cic"
        path = root / "cse_cic.parquet"
        frame = pd.read_parquet(path)
        schema = json.loads((root / "feature_names.json").read_text(encoding="utf-8"))
        features = schema["feature_names"]
        if not isinstance(features, list) or not all(isinstance(c, str) for c in features):
            raise ValueError("Invalid CSE feature-name schema")
        if len(set(features)) != len(features) or set(features) & CSE_NON_FEATURES:
            raise ValueError("CSE features must be unique and exclude endpoints, timestamp, label and audit columns")
        x = frame[features].to_numpy(dtype=np.float32)
        y = frame["label"].to_numpy(dtype=np.int64)
        if y.min() < 0 or y.max() >= len(CSE_CLASSES):
            raise ValueError("Unknown CSE label id")
        return Dataset(x, y, endpoint_ids(frame), CSE_CLASSES, features,
                       {"name": name, "path": str(path.resolve()), "sha256": sha256(path), "rows": len(frame),
                        "topology": "observed_ip_port_role_bipartite",
                        "provenance": json.loads((root / "provenance.json").read_text(encoding="utf-8"))})
    raise ValueError(f"Unknown dataset: {name}")


def split_indices(y, mode="stratified", seed=42):
    idx = np.arange(len(y))
    if mode == "sequential":
        train, test, val = np.split(idx, [int(0.5 * len(y)), int(0.8 * len(y))])
    elif mode == "stratified":
        train, rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
        test, val = train_test_split(rest, test_size=0.4, random_state=seed, stratify=y[rest])
        train, test, val = map(np.sort, [train, test, val])
    else:
        raise ValueError(mode)
    return {"train": train, "test": test, "validation": val}


class TrainMinMax:
    """Fit finite medians and ranges on train only; retain all 77 features.

    Constant columns map to zero, non-finite inputs to train medians. Transform
    clips to [0,1], a declared choice absent from the paper.
    """
    def fit(self, x):
        values = np.where(np.isfinite(x), x, np.nan).astype(np.float64)
        medians = []
        for column in values.T:
            finite = column[np.isfinite(column)]
            medians.append(float(np.median(finite)) if len(finite) else 0.0)
        self.median = np.array(medians)
        values = np.where(np.isfinite(values), values, self.median)
        self.minimum = values.min(axis=0)
        self.span = values.max(axis=0) - self.minimum
        self.constant = self.span == 0
        self.span[self.constant] = 1
        return self

    def transform(self, x):
        values = np.where(np.isfinite(x), x, self.median).astype(np.float64)
        scaled = np.clip((values - self.minimum) / self.span, 0, 1)
        scaled[:, self.constant] = 0
        return scaled.astype(np.float32)

    def save(self, path):
        np.savez(path, median=self.median, minimum=self.minimum, span=self.span, constant=self.constant)


def class_counts(y, classes):
    return dict(zip(classes, np.bincount(y, minlength=len(classes)).astype(int).tolist()))
