"""Run the CSE-CIC controlled experiments from Lab 6."""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[3]          # gốc repo (IDPS-mine)
LAB = Path(__file__).resolve().parents[1]           # lab/6
REPRO = ROOT / "code" / "reproduce"
DATA_ROOT = ROOT / "code" / "dataset"
PARQUET = DATA_ROOT / "cse_cic" / "cse_cic.parquet"
sys.path.insert(0, str(REPRO))

import resacag.data as dd          # noqa: E402
import resacag.experiment as ex    # noqa: E402

C = 7  # số lớp của cse_cic
ALL_MODELS = ["resacag", "dt", "rf", "et", "hgb", "catboost"]
EXPS = {
    # E6: chia stratified, so sánh chính.
    "E6": dict(split="stratified", filter=None, transductive=True, models=ALL_MODELS, allow_missing=False),
    # E1: chia theo thời gian (start_us), test là 30% cuối.
    "E1": dict(split="timeforward", filter=None, transductive=True, models=ALL_MODELS, allow_missing=True),
    # E0: bỏ dòng ghép mơ hồ (join_candidates > 1).
    "E0": dict(split="stratified", filter="cand_le1", transductive=True, models=["resacag", "dt"], allow_missing=False),
    # E2: bỏ dòng test có vector 77 feature trùng với train.
    "E2": dict(split="stratified_nodup", filter=None, transductive=True, models=["resacag", "dt"], allow_missing=False),
    # E5: ResACAG với node2vec chỉ học trên train.
    "E5": dict(split="stratified", filter=None, transductive=False, models=["resacag"], allow_missing=False),
    # E7a (thăm dò, xem PROTOCOL_E7A.md): ResACAG với cross-entropy có trọng số lớp balanced, các thứ khác như E6.
    "E7a": dict(split="stratified", filter=None, transductive=True, models=["resacag"], allow_missing=False,
                loss="balanced"),
    # E7b (thăm dò, xem PROTOCOL_E7B.md): thay node2vec bằng đặc trưng node tổng hợp từ flow (không nhãn).
    "E7b": dict(split="stratified", filter=None, transductive=True, models=["resacag"], allow_missing=False,
                node_embedding="features"),
    # E7c (thăm dò, xem PROTOCOL_E7C.md): đặc trưng node chỉ từ flow train; node chưa thấy nhận vector 0 (như E5).
    "E7c": dict(split="stratified", filter=None, transductive=False, models=["resacag"], allow_missing=False,
                node_embedding="features", feature_scope="train"),
    # E8 (đối chứng, xem PROTOCOL_E8.md): MLP trên đặc trưng flow, không đồ thị, cùng pipeline ResACAG.
    "E8": dict(split="stratified", filter=None, transductive=False, models=["mlp"], allow_missing=False,
               node_embedding="none"),
    # E9 (đối chứng, xem PROTOCOL_E9.md): ResACAG với cột IP:port nguồn và đích được hoán vị giữa các flow.
    "E9": dict(split="stratified", filter=None, transductive=True, models=["resacag"], allow_missing=False,
               rewire=True),
    # E10 (thăm dò, xem PROTOCOL_E10.md): DT/HGB với đặc trưng flow + vector node nguồn/đích (graph) hoặc đã hoán vị (rewired).
    "E10": dict(split="stratified", filter=None, transductive=True, allow_missing=False,
                models=["dt_graph", "hgb_graph", "dt_rewired", "hgb_rewired"]),
}

STATE = {"filter": None, "cluster": None, "start_us": None, "hash": None, "dropped": 0, "need_time": False,
         "loss": "plain", "node_embedding": "node2vec", "x": None, "run_dir": None,
         "endpoints": None, "feature_scope": "all", "variant": "resacag", "rewire": False, "seed": 0}
_orig_load = dd.load_dataset
_orig_split = dd.split_indices


def balanced_weights(y_train):
    """Trọng số lớp kiểu sklearn 'balanced': n / (số lớp có mặt * n_c). Lớp không có trong train nhận 0."""
    counts = np.bincount(y_train, minlength=C).astype(float)
    present = counts > 0
    w = np.zeros(C)
    w[present] = len(y_train) / (present.sum() * counts[present])
    return torch.tensor(w, dtype=torch.float32)


class _WeightedF:
    """Thay resacag.experiment.F. cross_entropy có trọng số lớp; các hàm khác chuyển tiếp sang torch.nn.functional.

    Lưu ý: trọng số tác động đến huấn luyện (có chủ đích), nên macro-F1 của E7a khác E6 là điều mong đợi.
    Riêng val_loss và test_loss được ghi log bằng loss có trọng số (evaluate dùng cùng F), nên không so trực tiếp
    với E6. Tie-break chọn checkpoint theo −val_loss cũng dùng loss có trọng số.
    """
    def __init__(self, weight):
        self._w = weight

    def cross_entropy(self, logits, target, *a, **k):
        return torch.nn.functional.cross_entropy(logits, target, weight=self._w.to(logits.device))

    def __getattr__(self, name):
        return getattr(torch.nn.functional, name)


def feature_node_embedding(graph, dim, seed, device, epochs=None):
    """Thay ex.node2vec (E7b). Vector node = PCA(đặc trưng tổng hợp của các flow đi qua node). Không dùng nhãn.

    Đặc trưng: trung bình 77 feature đã chuẩn hóa (scaler train-only đã lưu trong run_dir/scaler.npz)
    trên mọi flow có node làm nguồn hoặc đích, cộng log(1 + số flow). Node không có flow nhận vector 0.
    Trả về (embedding float32, 0.0) để khớp chữ ký của ex.node2vec.
    """
    from sklearn.decomposition import PCA
    run_dir = Path(STATE["run_dir"])
    z = np.load(run_dir / "scaler.npz")
    sc = dd.TrainMinMax()
    sc.median, sc.minimum, sc.span, sc.constant = z["median"], z["minimum"], z["span"], z["constant"]
    ep_all = np.asarray(STATE["endpoints"], dtype=np.int64)
    xs_all = sc.transform(STATE["x"]).astype(np.float64)
    if STATE["feature_scope"] == "train":
        rows = np.load(run_dir / "splits.npz")["train"]   # chỉ flow train (E7c)
    else:
        rows = np.arange(len(ep_all))                      # mọi flow, không nhãn (E7b)
    ep, xs = ep_all[rows], xs_all[rows]
    n = int(ep_all.max()) + 1
    src, dst = ep[:, 0], ep[:, 1]
    s = np.zeros((n, xs.shape[1]))
    np.add.at(s, src, xs)
    np.add.at(s, dst, xs)
    cnt = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    mean = s / np.maximum(cnt, 1)[:, None]
    feat = np.column_stack([mean, np.log1p(cnt)])
    active = cnt > 0
    mu, sd = feat[active].mean(0), feat[active].std(0)
    sd[sd == 0] = 1
    zf = np.zeros_like(feat)
    zf[active] = (feat[active] - mu) / sd
    pca = PCA(n_components=dim, random_state=seed).fit(zf[active])
    emb = np.zeros((n, dim))
    emb[active] = pca.transform(zf[active])
    print(f"E7b: node features {feat.shape}, active nodes {int(active.sum())}, PCA dim {dim}, "
          f"variance kept {pca.explained_variance_ratio_.sum():.3f}", flush=True)
    return emb.astype(np.float32), 0.0


def load_cse(name, data_root):
    """Thay ex.load_dataset: đọc cse_cic rồi lọc theo thí nghiệm, đồng thời giữ metadata để bootstrap."""
    ds = _orig_load(name, data_root)
    meta = pd.read_parquet(PARQUET, columns=["Src IP", "join_candidates", "start_us"])
    if len(meta) != len(ds.y):
        raise RuntimeError(f"Số dòng parquet {len(meta)} khác dataset {len(ds.y)}")
    keep = np.ones(len(ds.y), dtype=bool)
    if STATE["filter"] == "cand_le1":
        keep = (meta["join_candidates"] <= 1).to_numpy()
    if STATE["need_time"]:
        # E1 only: 168 dòng distrinet_exact không có PCAP nên không có start_us. Bỏ chúng khỏi E1 (AMENDMENT_1 mục 8).
        timed = meta["start_us"].notna().to_numpy()
        dropped = keep & ~timed
        STATE["untimed_dropped"] = {"rows": int(dropped.sum()),
                                    "by_label": np.bincount(ds.y[dropped], minlength=len(ds.classes)).tolist()}
        print(f"E1: bỏ {int(dropped.sum())} dòng không có start_us, theo nhãn: {STATE['untimed_dropped']['by_label']}", flush=True)
        keep &= timed
    ds.x, ds.y, ds.endpoints = ds.x[keep], ds.y[keep], ds.endpoints[keep]
    STATE["x"] = ds.x  # cho biểu diễn đặc trưng node (E7b, E7c)
    if STATE["rewire"]:
        # E9: hoán vị độc lập cột nguồn và cột đích giữa các flow. Giữ bậc nút, phá quan hệ IP:port.
        rng = np.random.default_rng(STATE["seed"] + 777)
        ep = ds.endpoints.copy()
        ep[:, 0] = rng.permutation(ep[:, 0])
        ep[:, 1] = rng.permutation(ep[:, 1])
        ds.endpoints = ep
        ds.metadata = {**ds.metadata, "topology": "rewired_random_control (degree-preserving column permutation)"}
    STATE["endpoints"] = ds.endpoints
    # Đưa thông tin lọc vào khóa cache node2vec để không dùng nhầm embedding của thí nghiệm khác.
    ds.metadata = {**ds.metadata, "sha256": f"{ds.metadata['sha256']}|filter={STATE['filter']}",
                   "rows": int(keep.sum())}
    STATE["cluster"] = pd.factorize(meta["Src IP"].to_numpy()[keep])[0]
    STATE["start_us"] = meta["start_us"].to_numpy()[keep]
    STATE["hash"] = pd.util.hash_pandas_object(pd.DataFrame(ds.x), index=False).to_numpy()
    return ds


def split_patch(y, mode="stratified", seed=42):
    """Thay ex.split_indices: thêm hai chế độ 'timeforward' và 'stratified_nodup'."""
    if mode == "timeforward":
        order = np.argsort(STATE["start_us"], kind="stable")
        n = len(order)
        a, b = int(0.5 * n), int(0.7 * n)
        return {"train": np.sort(order[:a]), "validation": np.sort(order[a:b]), "test": np.sort(order[b:])}
    if mode == "stratified_nodup":
        out = _orig_split(y, "stratified", seed)
        dup = np.isin(STATE["hash"][out["test"]], STATE["hash"][out["train"]])
        STATE["dropped"] = int(dup.sum())
        out["test"] = out["test"][~dup]
        return out
    return _orig_split(y, mode, seed)


def tabular_model(name, seed):
    if name == "dt":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=2)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=2)
    if name == "hgb":  # tham số lấy từ phase3_study.py
        return HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1,
                                              l2_regularization=1, class_weight="balanced",
                                              early_stopping=False, random_state=seed)
    if name == "catboost":  # tham số lấy từ phase3_study.py
        return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.08, auto_class_weights="Balanced",
                                  loss_function="MultiClass", random_seed=seed, thread_count=2,
                                  verbose=False, allow_writing_files=False)
    raise ValueError(name)


def aligned_proba(model, x):
    proba = model.predict_proba(x)
    out = np.zeros((len(x), C))
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    return out


def save_preds(preds_dir, name, seed, y, pred, row_id, val_macro):
    np.savez_compressed(preds_dir / f"{name}_seed{seed}.npz",
                        y=np.asarray(y, dtype=np.int64), pred=np.asarray(pred, dtype=np.int64),
                        row_id=np.asarray(row_id, dtype=np.int64), val_macro=np.float64(val_macro))


def graph_augmented_x(ds, seed, split_mode, mode):
    """E10: nối vào 77 đặc trưng flow vector node (PCA 32) của nút nguồn và nút đích. Không dùng nhãn.

    Cách tính vector node giống feature_node_embedding (E7b). mode='rewired' hoán vị cột nguồn/đích
    giống E9 (seed + 777). Trả về ma trận float64 (n, 77 + 64).
    """
    from sklearn.decomposition import PCA
    s = ex.split_indices(ds.y, split_mode, seed)
    sc = dd.TrainMinMax().fit(ds.x[s["train"]])
    xs = sc.transform(ds.x).astype(np.float64)
    ep = np.asarray(ds.endpoints, dtype=np.int64).copy()
    if mode == "rewired":
        rng = np.random.default_rng(seed + 777)
        ep[:, 0] = rng.permutation(ep[:, 0])
        ep[:, 1] = rng.permutation(ep[:, 1])
    n = int(ep.max()) + 1
    src, dst = ep[:, 0], ep[:, 1]
    sm = np.zeros((n, xs.shape[1]))
    np.add.at(sm, src, xs)
    np.add.at(sm, dst, xs)
    cnt = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    feat = np.column_stack([sm / np.maximum(cnt, 1)[:, None], np.log1p(cnt)])
    active = cnt > 0
    mu, sd = feat[active].mean(0), feat[active].std(0)
    sd[sd == 0] = 1
    zf = np.zeros_like(feat)
    zf[active] = (feat[active] - mu) / sd
    pca = PCA(n_components=32, random_state=seed).fit(zf[active])
    emb = np.zeros((n, 32))
    emb[active] = pca.transform(zf[active])
    print(f"E10 {mode}: nodes {n}, active {int(active.sum())}, variance kept {pca.explained_variance_ratio_.sum():.3f}",
          flush=True)
    return np.hstack([ds.x.astype(np.float64), emb[src], emb[dst]])


def run_tabular(name, seed, ds, split_mode, preds_dir):
    base, _, aug_mode = name.partition("_")   # "dt_graph" -> ("dt", "graph"); "dt" -> ("dt", "")
    s = ex.split_indices(ds.y, split_mode, seed)
    tr, va, te = s["train"], s["validation"], s["test"]
    x = (graph_augmented_x(ds, seed, split_mode, aug_mode) if aug_mode else ds.x.astype(np.float64))
    x[~np.isfinite(x)] = np.nan
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[tr])
    xtr, xva, xte = imp.transform(x[tr]), imp.transform(x[va]), imp.transform(x[te])
    model = tabular_model(base, seed)
    t0 = time.perf_counter()
    # Không early stopping: số vòng cố định như phase3_study.py, để không chọn mô hình bằng nhãn validation.
    model.fit(xtr, ds.y[tr])
    seconds = time.perf_counter() - t0
    pva = aligned_proba(model, xva).argmax(1)
    val = f1_score(ds.y[va], pva, labels=list(range(C)), average="macro", zero_division=0)
    pte = aligned_proba(model, xte).argmax(1)
    save_preds(preds_dir, name, seed, ds.y[te], pte, te, val)
    return {"model": name, "seed": seed, "train_rows": int(len(tr)), "test_rows": int(len(te)),
            "validation_macro": float(val), "fit_seconds": seconds,
            "missing_train_classes": sorted(set(range(C)) - set(np.unique(ds.y[tr]).tolist()))}


def run_resacag(args, seed, run_dir, preds_dir):
    """Chạy một biến thể thuộc họ GNN/MLP của repo: 'resacag' (mặc định) hoặc 'mlp' (E8)."""
    variant = STATE["variant"]
    STATE["run_dir"] = str(run_dir)
    orig_n2v = ex.node2vec
    if STATE["node_embedding"] == "features":
        ex.node2vec = feature_node_embedding
    elif STATE["node_embedding"] == "none":
        def _zero_embedding(graph, dim, seed, device, epochs=None):
            n = int(np.asarray(STATE["endpoints"]).max()) + 1
            return np.zeros((n, dim), dtype=np.float32), 0.0
        ex.node2vec = _zero_embedding
    try:
        prepared = ex.prepare(args, "cse_cic", seed, run_dir, "cpu")
    finally:
        ex.node2vec = orig_n2v
    orig_F = ex.F
    if STATE["loss"] == "balanced":
        w = balanced_weights(prepared[3]["train"])
        (run_dir / "class_weights.json").write_text(json.dumps(w.tolist()), encoding="utf-8")
        ex.F = _WeightedF(w)
    try:
        ex.train_variant(args, "cse_cic", variant, seed, run_dir, *prepared)
    finally:
        ex.F = orig_F
    z = np.load(run_dir / variant / "predictions.npz")
    best = json.loads((run_dir / variant / "best_epoch.json").read_text(encoding="utf-8"))
    save_preds(preds_dir, variant, seed, z["y"], z["prediction"], z["original_row_ids"], best["validation_macro"])
    audit = json.loads((run_dir / "data_audit.json").read_text(encoding="utf-8"))
    return {"model": variant, "seed": seed, "validation_macro": best["validation_macro"],
            "epochs_completed": json.loads((run_dir / variant / "metrics.json").read_text(encoding="utf-8"))["epochs_completed"],
            "missing_train_classes": audit.get("missing_train_classes", [])}


def git_head():
    return "8cb42171d90642b3daa59669995e96f59c308f37"


def main():
    global PARQUET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=list(EXPS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--node2vec-epochs", type=int, default=2)
    ap.add_argument("--out", default=None, help="Output directory; defaults to lab/6/<EXP>.")
    a = ap.parse_args()
    PARQUET = a.data_root / "cse_cic" / "cse_cic.parquet"

    cfg = EXPS[a.exp]
    models = a.models or cfg["models"]
    bad = set(models) - set(cfg["models"])
    if bad:
        raise SystemExit(f"{a.exp} không dùng các mô hình: {sorted(bad)}")
    out = Path(a.out).resolve() if a.out else LAB / a.exp
    preds_dir = out / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    STATE["filter"] = cfg["filter"]
    STATE["need_time"] = cfg["split"] == "timeforward"
    STATE["loss"] = cfg.get("loss", "plain")
    STATE["node_embedding"] = cfg.get("node_embedding", "node2vec")
    STATE["feature_scope"] = cfg.get("feature_scope", "all")
    ex.load_dataset = load_cse
    ex.split_indices = split_patch
    torch.set_num_threads(2)  # giống resacag/experiment.py

    args = argparse.Namespace(
        data_root=str(a.data_root), datasets=["cse_cic"],
        transductive_node2vec=["cse_cic"] if cfg["transductive"] else [],
        epochs=a.epochs, patience=a.patience, hidden=32, batch_size=500, caps=[8, 4], fixed=[4, 2],
        node2vec_epochs=a.node2vec_epochs, max_rows=0, split=cfg["split"], edge_update="neighbor",
        device="cpu", profile_cpu=False, allow_missing_classes=cfg["allow_missing"])

    manifest = {
        "experiment": a.exp, "config": {k: v for k, v in cfg.items() if k != "models"}, "models": models,
        "seeds": a.seeds, "epochs": a.epochs, "patience": a.patience, "node2vec_epochs": a.node2vec_epochs,
        "git_head": git_head(), "resacag_code_sha256": ex.code_hash(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {p: md.version(p) for p in ["torch", "scikit-learn", "catboost", "pandas", "numpy"]},
        "excluded": {"tabm": "repo implementation hard-codes .cuda(); máy không có CUDA NVIDIA"},
        "runs": [],
    }
    for seed in a.seeds:
        t_seed = time.perf_counter()
        STATE["seed"] = seed
        STATE["rewire"] = cfg.get("rewire", False)
        ds = ex.load_dataset("cse_cic", a.data_root)
        if seed == a.seeds[0]:
            np.savez(out / "row_meta.npz", cluster=STATE["cluster"], start_us=STATE["start_us"])
        run_dir = out / "runs" / "cse_cic" / f"seed_{seed}"
        seed_info = {"seed": seed, "rows_after_filter": int(len(ds.y)), "models": []}
        for m in models:
            t0 = time.perf_counter()
            if m in ("resacag", "mlp"):
                STATE["variant"] = m
                info = run_resacag(args, seed, run_dir, preds_dir)
            else:
                info = run_tabular(m, seed, ds, cfg["split"], preds_dir)
            info["seconds_total"] = time.perf_counter() - t0
            info["dropped_duplicate_test_rows"] = STATE["dropped"] if cfg["split"] == "stratified_nodup" else 0
            seed_info["models"].append(info)
            print(f"[{a.exp}] seed={seed} {m} xong trong {info['seconds_total']:.0f}s", flush=True)
        seed_info["seconds_total"] = time.perf_counter() - t_seed
        manifest["runs"].append(seed_info)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    print(f"XONG {a.exp}. Kết quả: {preds_dir}", flush=True)


if __name__ == "__main__":
    main()
