"""Download the pinned public CIC-Darknet CSV mirror with provenance."""
import hashlib
import json
import zipfile
from pathlib import Path

import requests


def main():
    root = Path(__file__).resolve().parents[1] / "dataset" / "darknet_original"
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / "provenance.json"
    repo = "mateuscmarim/Darknet-traffic-classification"
    s = requests.Session()
    if provenance.exists():
        old = json.loads(provenance.read_text(encoding="utf-8"))
        commit = old["repository_commit"]
        target = root / "Darknet.csv"
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == old["csv_sha256"]:
            print("Verified existing Darknet.csv", old["csv_sha256"])
            return
    else:
        response = s.get(f"https://api.github.com/repos/{repo}/commits/HEAD", timeout=30)
        response.raise_for_status()
        commit = response.json()["sha"]
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/dataset/Darknet.zip"
    zip_path = root / "Darknet.zip"
    with s.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with zip_path.open("wb") as f:
            for block in response.iter_content(1024 * 1024):
                f.write(block)
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if m.filename.lower().endswith(".csv")]
        if len(members) != 1 or members[0].file_size > 256 * 1024 * 1024:
            raise ValueError("Unexpected dataset archive content")
        # Write only the expected payload to a fixed path, never extract arbitrary paths.
        csv_path = root / "Darknet.csv"
        csv_path.write_bytes(archive.read(members[0]))
    record = {"source": url, "repository_commit": commit,
              "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
              "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
              "official_dataset": "https://www.unb.ca/cic/datasets/darknet2020.html",
              "provenance": "Public research mirror; equality to ResACAG authors' graph/preprocessing not established"}
    provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(csv_path, record["csv_sha256"])


if __name__ == "__main__":
    main()
