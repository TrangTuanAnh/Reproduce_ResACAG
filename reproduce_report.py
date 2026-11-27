"""Run the experiments and aggregate analyses that accompany the ResACAG report."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "clean_code"
CORE = CODE / "code" / "reproduce"
ARTIFACTS = ROOT / "artifacts"
ANALYSIS = CODE / "report_analysis"
LAB6 = CODE / "lab" / "6" / "code"
LAB7 = CODE / "lab" / "7" / "code"
LAB9 = CODE / "lab" / "9" / "code"
SEEDS = ["42", "43", "44", "45", "46"]
CONTROLS = {"E1": "E6", "E2": "E0", "E3": "E2", "E4": "E7a",
            "E5": "E7b", "E6": "E8", "E7": "E9", "E8": "E10"}


def python(*arguments, cwd=CODE, capture=False):
    command = [sys.executable, "-X", "utf8", *(str(argument) for argument in arguments)]
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        encoding="utf-8",
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "6"},
    )
    return result.stdout


def verify_inputs(data_root, *datasets):
    python(CORE / "verify_data.py", "--data-root", data_root, "--datasets", *datasets, cwd=CORE)
    if "darknet" in datasets:
        pinned = ARTIFACTS / "inputs" / "darknet_provenance.json"
        expected = json.loads(pinned.read_text(encoding="utf-8"))["csv_sha256"]
        recorded = data_root / "darknet_original" / "provenance.json"
        actual = json.loads(recorded.read_text(encoding="utf-8"))["csv_sha256"]
        if actual != expected:
            raise ValueError("Darknet.csv does not match the hash used in the report.")


def verify_package():
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {record['path']}")
    print(f"Verified {len(manifest['files'])} packaged files.")


def run_control(experiment, seeds, data_root, destination, models=()):
    arguments = ["--exp", experiment, "--seeds", *seeds, "--out", destination,
                 "--data-root", data_root, "--epochs", "12", "--patience", "3",
                 "--node2vec-epochs", "2"]
    if models:
        arguments.extend(("--models", *models))
    python(LAB6 / "run_cse_cic_controls.py", *arguments)


def export_eda(data_root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    dark = ARTIFACTS / "reproduction" / "darknet" / "resacag"
    cse = ARTIFACTS / "reproduction" / "cse_cic" / "resacag"
    python(ANALYSIS / "summarize_datasets.py", data_root, destination / "dataset_summary.json")
    python(ANALYSIS / "inspect_feature_quality.py", data_root, dark, cse,
           destination / "feature_diagnostics.json")
    output = python(ANALYSIS / "evaluate_endpoint_lookup.py", data_root, capture=True)
    results = {}
    for line in output.splitlines():
        name, record = line.split(" ", 1)
        results[name] = json.loads(record)
    (destination / "endpoint_lookup.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def export_figures(eda, destination):
    destination.mkdir(parents=True, exist_ok=True)
    dark = ARTIFACTS / "reproduction" / "darknet"
    cse = ARTIFACTS / "reproduction" / "cse_cic"
    lab6 = ARTIFACTS / "lab6"
    python(ANALYSIS / "plot_reproduction_results.py", eda, dark / "resacag", cse / "resacag", destination)
    python(ANALYSIS / "plot_class_distributions.py", dark / "data_audit.json", cse / "data_audit.json",
           destination / "02_phan_bo_lop.pdf", destination / "02_phan_bo_lop.png")
    python(ANALYSIS / "plot_controlled_experiments.py", lab6 / "lab6_summary.csv",
           lab6 / "lab6_comparisons.csv", lab6 / "e10_reported_summary.csv", destination)
    python(ANALYSIS / "plot_source_ip.py", ARTIFACTS / "source_ip" / "summary.csv", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=[
        "verify", "darknet", "cse-cic", "controls", "analyze-controls", "node2vec-train",
        "source-ip", "appendix-gate", "appendix-history", "eda", "figures"
    ])
    parser.add_argument("--data-root", type=Path, default=CODE / "code" / "dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "runs", help="Directory for regenerated outputs.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for DarkNet and the appendix cosine gate.")
    parser.add_argument("--node2vec", type=Path, help="Optional seed-42 cache for the appendix cosine gate.")
    parser.add_argument("--experiment", choices=CONTROLS, help="Run a single Part C experiment by report ID.")
    args = parser.parse_args()
    if args.experiment and args.target != "controls":
        parser.error("--experiment is used with the controls target.")
    if args.node2vec and args.target != "appendix-gate":
        parser.error("--node2vec is used with the appendix-gate target.")
    data_root, out = args.data_root.resolve(), args.out.resolve()

    if args.target == "verify":
        verify_package()
    elif args.target == "darknet":
        verify_inputs(data_root, "darknet")
        python("-m", "resacag.experiment", "--data-root", data_root, "--datasets", "darknet",
               "--variants", "resacag", "--seeds", "42", "--epochs", "30", "--patience", "30",
               "--output", out / "darknet", "--device", args.device, cwd=CODE / "versions" / "darknet_lab2")
    elif args.target == "cse-cic":
        verify_inputs(data_root, "cse_cic")
        run_control("E6", ["42"], data_root, out / "cse_cic", ["resacag"])
    elif args.target == "controls":
        verify_inputs(data_root, "cse_cic")
        experiments = (CONTROLS[args.experiment],) if args.experiment else CONTROLS.values()
        for experiment in experiments:
            run_control(experiment, SEEDS, data_root, out / "lab6" / experiment)
    elif args.target == "analyze-controls":
        python(LAB6 / "analyze_cse_cic_controls.py", "--runs-root", out / "lab6",
               "--output", out / "lab6" / "summary")
    elif args.target == "node2vec-train":
        verify_inputs(data_root, "cse_cic")
        run_control("E5", SEEDS, data_root, out / "lab6" / "E5", ["resacag"])
    elif args.target == "source-ip":
        verify_inputs(data_root, "cse_cic")
        destination = out / "source_ip"
        python(LAB7 / "run_source_ip_experiments.py", "--data-root", data_root, "--splits", "RAND", "GRP",
               "--sets", "flow", "flow+id", "--models", "dt", "rf", "hgb", "--seeds", *SEEDS,
               "--frac", "1.0", "--out", destination)
        python(ANALYSIS / "summarize_source_ip.py", destination / "metrics.csv", destination / "summary.csv",
               "--contrasts", destination / "contrasts.csv")
    elif args.target in ("appendix-gate", "appendix-history"):
        verify_inputs(data_root, "cse_cic")
        experiment = args.target.removeprefix("appendix-")
        seeds = SEEDS[:3] if experiment == "gate" else SEEDS
        cache = ["--node2vec", args.node2vec.resolve()] if args.node2vec else []
        python(LAB9 / "run_improvements.py", experiment, "--data", data_root / "cse_cic" / "cse_cic.parquet",
               "--out", out / "appendix" / experiment, "--seeds", *seeds, "--device", args.device, *cache)
    elif args.target == "eda":
        verify_inputs(data_root, "darknet", "cse_cic")
        export_eda(data_root, out / "eda")
    elif args.target == "figures":
        regenerated = out / "eda" / "dataset_summary.json"
        eda = regenerated if regenerated.is_file() else ARTIFACTS / "eda" / "dataset_summary.json"
        export_figures(eda, out / "figures")


if __name__ == "__main__":
    main()
