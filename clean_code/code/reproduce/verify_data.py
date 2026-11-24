"""Verify immutable input artifacts against their recorded provenance hashes."""
import argparse
import json
from pathlib import Path

from resacag.data import sha256

# Committed pins for cse_cic: a provenance file copied from a local build cannot vouch for that build.
CSE_CIC_PINS = Path(__file__).resolve().parent / "dataset_cse_cic"


def verify(data_root, datasets=("darknet", "cse_reference")):
    checked = []
    if "darknet" in datasets:
        dark = data_root / "darknet_original"
        meta = json.loads((dark / "provenance.json").read_text(encoding="utf-8"))
        if sha256(dark / "Darknet.csv") != meta["csv_sha256"]:
            raise ValueError("Darknet CSV changed since provenance was recorded")
        checked.append("Darknet.csv")
    if "cse_reference" in datasets:  # rounds 1-4: E-ResGAT arrays with random adjacency
        cse = data_root / "reference_cse"
        meta = json.loads((cse / "provenance.json").read_text(encoding="utf-8"))
        for name, record in meta["files"].items():
            if sha256(cse / name) != record["sha256"]:
                raise ValueError(f"CSE reference artifact changed: {name}")
            checked.append(name)
    if "cse_cic" in datasets:  # lab/5: observed endpoints rebuilt from pcaps
        cic = data_root / "cse_cic"
        pin = json.loads((CSE_CIC_PINS / "build" / "expected_dataset.json").read_text(encoding="utf-8"))
        if sha256(cic / pin["file"]) != pin["sha256"]:
            raise ValueError("cse_cic.parquet differs from build/expected_dataset.json")
        checked.append(pin["file"])
        for name in ["feature_names.json", "provenance.json"]:
            local = json.loads((cic / name).read_text(encoding="utf-8"))
            if local != json.loads((CSE_CIC_PINS / name).read_text(encoding="utf-8")):
                raise ValueError(f"cse_cic {name} differs from code/reproduce/dataset_cse_cic/{name}")
            checked.append(name)
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--datasets", nargs="+", choices=["darknet", "cse_reference", "cse_cic"],
                        default=["darknet", "cse_reference"])
    args = parser.parse_args()
    print("Verified:", ", ".join(verify(args.data_root, args.datasets)))


if __name__ == "__main__":
    main()
