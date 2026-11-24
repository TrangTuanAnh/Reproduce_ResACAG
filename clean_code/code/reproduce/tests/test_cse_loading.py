import json

import numpy as np
import pandas as pd
import pytest

import verify_data
from resacag.data import load_dataset


def write_cse(root, features):
    root.mkdir()
    frame = pd.DataFrame({"label": [0, 3, 3], "Src IP": ["172.31.69.25", "18.219.5.43", "18.219.5.43"],
                          "Src Port": [51000, 60755, 60756], "Dst IP": ["8.8.8.8", "172.31.69.28", "172.31.69.28"],
                          "Dst Port": [53, 80, 80], "Timestamp": ["21/02/2018 10:11:05"] * 3})
    for i, name in enumerate(["Protocol", "Flow Duration", "Tot Fwd Pkts"]):
        frame[name] = np.arange(3, dtype=np.float64) + i
    frame.to_parquet(root / "cse_cic.parquet", index=False)
    (root / "feature_names.json").write_text(json.dumps({"feature_names": features}), encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps({"file": "cse_cic.parquet"}), encoding="utf-8")


def test_cse_uses_observed_endpoints_as_bipartite_topology(tmp_path):
    features = ["Protocol", "Flow Duration", "Tot Fwd Pkts"]
    write_cse(tmp_path / "cse_cic", features)
    dataset = load_dataset("cse_cic", tmp_path)
    assert dataset.feature_names == features and dataset.x.shape == (3, 3)
    np.testing.assert_array_equal(dataset.y, [0, 3, 3])
    src, dst = dataset.endpoints[:, 0], dataset.endpoints[:, 1]
    assert not set(src) & set(dst)
    # Both DDoS flows hit the same victim endpoint 172.31.69.28:80.
    assert dst[1] == dst[2] and dst[0] != dst[1]
    assert dataset.metadata["topology"] == "observed_ip_port_role_bipartite"


@pytest.mark.parametrize("leak", ["Dst Port", "Timestamp", "start_us"])
def test_cse_features_cannot_include_endpoints_or_audit_columns(tmp_path, leak):
    write_cse(tmp_path / "cse_cic", ["Protocol", leak])
    with pytest.raises(ValueError):
        load_dataset("cse_cic", tmp_path)


def test_verify_checks_cse_cic_against_committed_pins_without_darknet(tmp_path, monkeypatch):
    pins, root = verify_data.CSE_CIC_PINS, tmp_path / "cse_cic"
    root.mkdir()
    (root / "cse_cic.parquet").write_bytes(b"local build")
    with pytest.raises(ValueError, match="expected_dataset"):
        verify_data.verify(tmp_path, ["cse_cic"])
    pinned = json.loads((pins / "build" / "expected_dataset.json").read_text(encoding="utf-8"))["sha256"]
    monkeypatch.setattr(verify_data, "sha256", lambda path: pinned)
    for name in ["feature_names.json", "provenance.json"]:
        (root / name).write_bytes((pins / name).read_bytes())
    assert verify_data.verify(tmp_path, ["cse_cic"]) == ["cse_cic.parquet", "feature_names.json", "provenance.json"]
    # A reordered feature list would silently permute model inputs.
    schema = json.loads((pins / "feature_names.json").read_text(encoding="utf-8"))
    schema["feature_names"].reverse()
    (root / "feature_names.json").write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_names"):
        verify_data.verify(tmp_path, ["cse_cic"])
