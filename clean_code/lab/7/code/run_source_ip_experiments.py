"""Evaluate tree classifiers under random and source-IP splits."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, log_loss, recall_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import topology_features as F
import flow_splits as S

ROOT = Path(__file__).resolve().parents[3]
PARQUET = ROOT / "code/dataset/cse_cic/cse_cic.parquet"
C = 7
SETS = ["flow", "id", "topo_c", "flow+topo_c", "flow+topo_t", "flow+topo_r", "flow+id"]


def load(parquet=PARQUET):
    d = pd.read_parquet(parquet)
    names = json.load(open(parquet.parent / "feature_names.json", encoding="utf-8"))["feature_names"]
    dropped = d["start_us"].isna()
    info = {"dropped_untimed": int(dropped.sum()),
            "dropped_by_label": np.bincount(d.loc[dropped, "label"].astype(int), minlength=C).tolist()}
    d = d[~dropped].reset_index(drop=True)
    x = d[names].to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = np.nan  # inf -> NaN cho mọi mô hình
    ips = pd.concat([d["Src IP"], d["Dst IP"]]).astype(str)
    codes, _ = pd.factorize(ips)
    n = len(d)
    return dict(x=x, y=d["label"].to_numpy(int), src=codes[:n], dst=codes[n:],
                dport=d["Dst Port"].to_numpy(np.int64), sport=d["Src Port"].to_numpy(np.int64),
                t=d["start_us"].to_numpy(float), oct=np.hstack([F.ip_octets(d["Src IP"]), F.ip_octets(d["Dst IP"])]),
                srcip=d["Src IP"].astype(str).to_numpy()), info


def build_sets(D, seed, cache):
    """Trả về dict tên->ma trận. topo_c/topo_t không phụ thuộc seed (cache); topo_r phụ thuộc seed."""
    if "topo_c" not in cache:
        cache["topo_c"] = F.topo_causal(D["src"], D["dst"], D["dport"], D["t"])
        cache["topo_t"] = F.topo_transductive(D["src"], D["dst"], D["dport"])
    key = f"topo_r{seed}"
    if key not in cache:
        s2, d2 = F.rewire(D["src"], D["dst"], seed)
        cache[key] = F.topo_causal(s2, d2, D["dport"], D["t"])
    ident = np.hstack([D["oct"], D["sport"][:, None], D["dport"][:, None]]).astype(float)
    return {"flow": D["x"], "id": ident, "topo_c": cache["topo_c"],
            "flow+topo_c": np.hstack([D["x"], cache["topo_c"]]),
            "flow+topo_t": np.hstack([D["x"], cache["topo_t"]]),
            "flow+topo_r": np.hstack([D["x"], cache[key]]),
            "flow+id": np.hstack([D["x"], ident])}


def model(name, seed):
    if name == "dt":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=8)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1, l2_regularization=1,
                                              class_weight="balanced", early_stopping=False, random_state=seed)
    raise ValueError(name)


def get_split(kind, D, seed):
    y = D["y"]
    if kind == "RAND":
        tr, te = S.rand_split(y, seed); rep = None
    elif kind == "GRP":
        tr, te, rep = S.grp_split(y, D["src"], seed)
    elif kind == "TFC":
        tr, te = S.tfc_split(y, D["t"]); rep = None
    else:
        raise ValueError(kind)
    return tr, te, rep


def subsample(idx, y, frac, seed):
    """Pilot: giữ frac mỗi lớp (ít nhất 1 flow)."""
    if frac >= 1:
        return idx
    rng = np.random.default_rng(seed + 12345)
    keep = []
    for c in np.unique(y[idx]):
        m = idx[y[idx] == c]
        k = max(1, int(round(frac * len(m))))
        keep.append(rng.choice(m, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def metrics(y, pred, proba):
    present = np.unique(y)
    out = {"macro_f1": f1_score(y, pred, labels=present, average="macro", zero_division=0),
           "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
           "logloss": log_loss(y, np.clip(proba, 1e-15, 1), labels=list(range(C)))}
    rec = recall_score(y, pred, labels=list(range(C)), average=None, zero_division=0)
    for c in range(C):
        out[f"recall_{c}"] = rec[c] if c in present else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=PARQUET.parents[1])
    ap.add_argument("--splits", nargs="+", default=["GRP"]); ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--models", nargs="+", default=["dt"]); ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); (out / "preds").mkdir(parents=True, exist_ok=True)
    D, info = load(a.data_root / "cse_cic" / "cse_cic.parquet")
    head = "8cb42171d90642b3daa59669995e96f59c308f37"
    import sklearn, scipy, platform
    man = dict(args={key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}, commit=head, **info, rows=len(D["y"]),
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             sklearn=sklearn.__version__, scipy=scipy.__version__))
    cache = {}; rows = []; t0 = time.time()
    csv = out / "metrics.csv"
    done = set()
    if csv.exists():  # tiếp tục được khi bị ngắt
        old = pd.read_csv(csv); rows = old.to_dict("records")
        done = {(r["split"], r["seed"], r["model"], r["set"]) for r in rows}
    reports = {}
    for sp in a.splits:
        for seed in a.seeds:
            if sp == "TFC" and seed != a.seeds[0]:
                pass  # TFC xác định; seed vẫn đổi để đổi khởi tạo mô hình
            tr, te, rep = get_split(sp, D, seed)
            if rep: reports[f"{sp}_{seed}"] = rep
            tr = subsample(tr, D["y"], a.frac, seed); te = subsample(te, D["y"], a.frac, seed)
            assert not set(tr) & set(te)
            if sp == "GRP":
                assert not set(D["src"][tr]) & set(D["src"][te]), "IP nguồn rò giữa train và test"
            sets = build_sets(D, seed, cache)
            y = D["y"]
            for mname in a.models:
                for sname in a.sets:
                    if (sp, seed, mname, sname) in done:
                        continue
                    X = sets[sname]
                    m = model(mname, seed).fit(X[tr], y[tr])
                    proba = np.zeros((len(te), C)); proba[:, np.asarray(m.classes_, int)] = m.predict_proba(X[te])
                    pred = proba.argmax(1)
                    r = dict(split=sp, seed=seed, model=mname, set=sname, n_train=len(tr), n_test=len(te),
                             **metrics(y[te], pred, proba))
                    rows.append(r)
                    np.savez_compressed(out / "preds" / f"{sp}_{seed}_{mname}_{sname.replace('+','_')}.npz",
                                        y=y[te], pred=pred, proba=proba.astype(np.float32), te=te, grp=D["src"][te])
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    print(f"{time.time()-t0:7.0f}s {sp} s{seed} {mname} {sname}: macroF1={r['macro_f1']:.4f} ll={r['logloss']:.3f}", flush=True)
    man["split_reports"] = reports
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))


if __name__ == "__main__":
    main()
