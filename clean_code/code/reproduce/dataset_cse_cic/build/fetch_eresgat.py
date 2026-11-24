"""Download the pinned E-ResGAT CSE-CIC arrays (build input only) and verify SHA-256.

They are only needed to map E-ResGAT rows back to the AWS CSV rows and to read the paper
labels; the training input is cse_cic.parquet. Never loads pickle or runs notebooks.

Usage: python -I fetch_eresgat.py OUT_DIR
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pinned = json.loads((HERE / "eresgat_inputs.json").read_text(encoding="utf-8"))
    session = requests.Session()
    for name, record in pinned["files"].items():
        target = out / name
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            partial = target.with_suffix(target.suffix + ".part")
            with session.get(record["url"], stream=True, timeout=120) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {name}")
            partial.replace(target)
        print(name, "verified", record["sha256"][:12], flush=True)
    # Column order of the 77 E-ResGAT features (AWS header without Dst Port, Timestamp, Label).
    shutil.copyfile(HERE.parent / "feature_names.json", out / "feature_names.json")


if __name__ == "__main__":
    main(sys.argv[1])
