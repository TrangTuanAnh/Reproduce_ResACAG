"""Aggregate Lab 6 scores and compute paired bootstrap comparisons."""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "report"
C = 7
N_BOOT = 2000
RNG_SEED = 20261115
BASELINES = ["dt", "rf", "et", "hgb", "catboost"]


def macro_w(y, p, w):
    """Macro-F1 có trọng số dòng, cùng công thức với sklearn (lớp không có mặt được tính 0)."""
    cm = np.bincount(y * C + p, weights=w, minlength=C * C).reshape(C, C)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(C), where=den > 0)
    return f1.mean()


def load_exp(exp):
    d = LAB / exp
    if not (d / "row_meta.npz").exists():
        return None, None
    cluster = np.load(d / "row_meta.npz")["cluster"]
    preds = {}
    for f in sorted((d / "preds").glob("*.npz")):
        name, seed = f.stem.rsplit("_seed", 1)
        z = np.load(f)
        o = np.argsort(z["row_id"], kind="stable")
        preds[(name, int(seed))] = dict(row=z["row_id"][o], y=z["y"][o].astype(np.int64),
                                        p=z["pred"][o].astype(np.int64), val=float(z["val_macro"]))
    return preds, cluster


def summarize(exp, preds):
    rows = []
    for m in sorted({k[0] for k in preds}):
        seeds = sorted(s for mm, s in preds if mm == m)
        mac, wf, val = [], [], []
        for s in seeds:
            r = preds[(m, s)]
            mac.append(f1_score(r["y"], r["p"], labels=list(range(C)), average="macro", zero_division=0))
            wf.append(f1_score(r["y"], r["p"], labels=list(range(C)), average="weighted", zero_division=0))
            val.append(r["val"])
        rows.append(dict(exp=exp, model=m, seeds=len(seeds),
                         macro_f1_mean=100 * np.mean(mac), macro_f1_sd=100 * np.std(mac, ddof=1) if len(mac) > 1 else np.nan,
                         weighted_f1_mean=100 * np.mean(wf), weighted_f1_sd=100 * np.std(wf, ddof=1) if len(wf) > 1 else np.nan,
                         validation_macro_mean=100 * np.mean(val)))
    return rows


def items_for(pa, pb, cluster, seeds):
    items = []
    for s in seeds:
        a, b = pa[s], pb[s]
        if not np.array_equal(a["row"], b["row"]):
            raise ValueError(f"Tập test khác nhau ở seed {s}; không ghép cặp được")
        _, ci = np.unique(cluster[a["row"]], return_inverse=True)
        items.append(dict(y=a["y"], pa=a["p"], pb=b["p"], ci=ci, K=int(ci.max()) + 1))
    return items


def paired_boot(items, rng):
    ones = [np.ones(len(it["y"])) for it in items]
    point = np.mean([macro_w(it["y"], it["pa"], w) - macro_w(it["y"], it["pb"], w) for it, w in zip(items, ones)])
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        d = 0.0
        for it in items:
            # Bootstrap theo cụm IP nguồn: rút K cụm có hoàn lại, mỗi dòng nhận trọng số = số lần cụm của nó được rút.
            cnt = np.bincount(rng.integers(0, it["K"], it["K"]), minlength=it["K"])
            w = cnt[it["ci"]].astype(np.float64)
            d += macro_w(it["y"], it["pa"], w) - macro_w(it["y"], it["pb"], w)
        boots[b] = d / len(items)
    return point, boots


def p_two_sided(boots):
    return max(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1 / N_BOOT)


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def compare(label, pa, pb, cluster, seeds_a, seeds_b, rng):
    seeds = sorted(set(seeds_a) & set(seeds_b))
    if not seeds:
        return None
    items = items_for(pa, pb, cluster, seeds)
    # Seed bootstrap riêng cho từng so sánh (theo tên), để thêm so sánh mới không làm đổi CI cũ.
    rng = np.random.default_rng(zlib.crc32(label.encode()) ^ RNG_SEED)
    point, boots = paired_boot(items, rng)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(comparison=label, seeds=len(seeds), delta_macro_pp=100 * point,
                ci_low_pp=100 * lo, ci_high_pp=100 * hi, p_raw=p_two_sided(boots))


def run_family(exp_label, preds, cluster, base_models, rng):
    seeds_r = sorted(s for m, s in preds if m == "resacag")
    rows = []
    for b in base_models:
        seeds_b = sorted(s for m, s in preds if m == b)
        pa = {s: preds[("resacag", s)] for s in seeds_r}
        pb = {s: preds[(b, s)] for s in seeds_b}
        r = compare(f"{exp_label}: resacag - {b}", pa, pb, cluster, seeds_r, seeds_b, rng)
        if r:
            rows.append(r)
    # Baseline tốt nhất theo macro-F1 TRÊN VALIDATION (không dùng test để chọn), theo PROTOCOL mục 7 E1/E6.
    val_mean = {b: np.mean([preds[(b, s)]["val"] for s in seeds_r if (b, s) in preds])
                for b in base_models if any((b, s) in preds for s in seeds_r)}
    if val_mean:
        best = max(val_mean, key=val_mean.get)
        pa = {s: preds[("resacag", s)] for s in seeds_r}
        pb = {s: preds[(best, s)] for s in seeds_r if (best, s) in preds}
        r = compare(f"{exp_label}: resacag - {best} (baseline tốt nhất theo validation)", pa, pb, cluster,
                    seeds_r, sorted(pb), rng)
        if r:
            r["best_by_validation"] = True
            rows.append(r)
    if rows:
        adj = holm([r["p_raw"] for r in rows])
        for r, a in zip(rows, adj):
            r["p_holm"] = a
            r["ci_excludes_0"] = bool(r["ci_low_pp"] > 0 or r["ci_high_pp"] < 0)
    return rows


def main():
    global LAB, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=LAB)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    LAB = args.runs_root.resolve()
    OUT = args.output.resolve() if args.output else LAB / "report"
    if not any((file.parent.parent / "row_meta.npz").is_file() for file in LAB.glob("*/preds/*.npz")):
        parser.error("No regenerated predictions found under --runs-root.")
    rng = np.random.default_rng(RNG_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    data = {e: load_exp(e) for e in ["E6", "E1", "E0", "E2", "E5", "E7a", "E7b", "E7c", "E8", "E9", "E10"]}

    summary, comps = [], []
    for e, (preds, _) in data.items():
        if preds is None:
            print(f"{e}: chưa có dữ liệu, bỏ qua")
            continue
        summary += summarize(e, preds)

    # Kiểm tra hàm macro có trọng số khớp sklearn trước khi tin các CI.
    any_exp = next(e for e, (p, _) in data.items() if p is not None)
    preds0, _ = data[any_exp]
    k = next(iter(preds0))
    r = preds0[k]
    ref = f1_score(r["y"], r["p"], labels=list(range(C)), average="macro", zero_division=0)
    assert abs(macro_w(r["y"], r["p"], np.ones(len(r["y"]))) - ref) < 1e-9, "macro_w lệch sklearn"
    print(f"Kiểm tra macro_w khớp sklearn trên {any_exp} {k}: OK")

    p6, c6 = data["E6"]
    if p6 is not None:
        comps += run_family("E6", p6, c6, BASELINES, rng)
    p1, _ = data["E1"]
    if p1 is not None:
        comps += run_family("E1", p1, data["E6"][1] if c6 is None else c6, BASELINES, rng)
    p0, c0 = data["E0"]
    if p0 is not None:
        comps += run_family("E0 (bỏ ghép mơ hồ)", p0, c0, ["dt"], rng)
    p2, c2 = data["E2"]
    if p2 is not None:
        comps += run_family("E2 (bỏ trùng lặp)", p2, c2, ["dt"], rng)
    p5, _ = data["E5"]
    if p5 is not None and p6 is not None:
        r5 = compare("E5 resacag (train-only) - E6 resacag (transductive)",
                     {s: p5[("resacag", s)] for s in sorted(s for m, s in p5 if m == "resacag")},
                     {s: p6[("resacag", s)] for s in sorted(s for m, s in p6 if m == "resacag")},
                     c6, sorted(s for m, s in p5 if m == "resacag"), sorted(s for m, s in p6 if m == "resacag"), rng)
        if r5:
            r5["p_holm"] = r5["p_raw"]
            r5["ci_excludes_0"] = bool(r5["ci_low_pp"] > 0 or r5["ci_high_pp"] < 0)
            comps.append(r5)

    # Ngưỡng E0 trong protocol: Δ(ResACAG − DT) thay đổi quá 2 điểm giữa A và B.
    e0_a = next((c for c in comps if c["comparison"] == "E6: resacag - dt"), None)
    e0_b = next((c for c in comps if c["comparison"] == "E0 (bỏ ghép mơ hồ): resacag - dt"), None)
    if e0_a and e0_b:
        shift = abs(e0_a["delta_macro_pp"] - e0_b["delta_macro_pp"])
        print(f"E0: Δ(A)={e0_a['delta_macro_pp']:.2f} pp, Δ(B)={e0_b['delta_macro_pp']:.2f} pp, "
              f"lệch={shift:.2f} pp -> {'VƯỢT ngưỡng 2 điểm' if shift > 2 else 'trong ngưỡng 2 điểm'}")

    # E7a (thăm dò): ResACAG có trọng số lớp so với E6 (cùng test set, cùng seed) và so với các baseline E6.
    # Họ riêng, Holm hiệu chỉnh riêng; ghi là THĂM DÒ trong báo cáo.
    p7, _ = data["E7a"]
    if p7 is not None and p6 is not None:
        seeds7 = sorted(s for m, s in p7 if m == "resacag")
        fam7 = []
        pa7 = {s: p7[("resacag", s)] for s in seeds7}
        r = compare("E7a (thăm dò): resacag trọng số lớp - E6 resacag",
                    pa7, {s: p6[("resacag", s)] for s in seeds7}, c6, seeds7, seeds7, rng)
        if r:
            fam7.append(r)
        for b in BASELINES:
            seeds_b = sorted(s for m, s in p6 if m == b)
            r = compare(f"E7a (thăm dò): resacag trọng số lớp - E6 {b}",
                        pa7, {s: p6[(b, s)] for s in seeds_b}, c6, seeds7, seeds_b, rng)
            if r:
                fam7.append(r)
        if fam7:
            adj7 = holm([r["p_raw"] for r in fam7])
            for r, a in zip(fam7, adj7):
                r["p_holm"] = a
                r["ci_excludes_0"] = bool(r["ci_low_pp"] > 0 or r["ci_high_pp"] < 0)
            comps += fam7

    # E7b (thăm dò): ResACAG với biểu diễn đặc trưng node so với E6 (cùng test set) và các baseline E6.
    p7b, _ = data["E7b"]
    if p7b is not None and p6 is not None:
        seeds7b = sorted(s for m, s in p7b if m == "resacag")
        fam7b = []
        pa7b = {s: p7b[("resacag", s)] for s in seeds7b}
        r = compare("E7b (thăm dò): resacag biểu diễn đặc trưng node - E6 resacag",
                    pa7b, {s: p6[("resacag", s)] for s in seeds7b}, c6, seeds7b, seeds7b, rng)
        if r:
            fam7b.append(r)
        for b in BASELINES:
            seeds_b = sorted(s for m, s in p6 if m == b)
            r = compare(f"E7b (thăm dò): resacag biểu diễn đặc trưng node - E6 {b}",
                        pa7b, {s: p6[(b, s)] for s in seeds_b}, c6, seeds7b, seeds_b, rng)
            if r:
                fam7b.append(r)
        if fam7b:
            adj7b = holm([r["p_raw"] for r in fam7b])
            for r, a in zip(fam7b, adj7b):
                r["p_holm"] = a
                r["ci_excludes_0"] = bool(r["ci_low_pp"] > 0 or r["ci_high_pp"] < 0)
            comps += fam7b

    # E7c (thăm dò): đặc trưng node chỉ từ flow train. So với E5 (cùng điều kiện train-only) và E7b.
    p7c, _ = data["E7c"]
    if p7c is not None:
        seeds7c = sorted(s for m, s in p7c if m == "resacag")
        fam7c = []
        pa7c = {s: p7c[("resacag", s)] for s in seeds7c}
        if p5 is not None:
            seeds5 = sorted(s for m, s in p5 if m == "resacag")
            r = compare("E7c (thăm dò): resacag đặc trưng train-only - E5 resacag (node2vec train-only)",
                        pa7c, {s: p5[("resacag", s)] for s in seeds5}, c6, seeds7c, seeds5, rng)
            if r:
                fam7c.append(r)
        if p7b is not None:
            r = compare("E7c (thăm dò): resacag đặc trưng train-only - E7b resacag (đặc trưng cả split)",
                        pa7c, {s: p7b[("resacag", s)] for s in seeds7b}, c6, seeds7c, seeds7b, rng)
            if r:
                fam7c.append(r)
        if p6 is not None:
            for b in ["dt"]:
                seeds_b = sorted(s for m, s in p6 if m == b)
                r = compare(f"E7c (thăm dò): resacag đặc trưng train-only - E6 {b}",
                            pa7c, {s: p6[(b, s)] for s in seeds_b}, c6, seeds7c, seeds_b, rng)
                if r:
                    fam7c.append(r)
        if fam7c:
            adj7c = holm([r["p_raw"] for r in fam7c])
            for r, a in zip(fam7c, adj7c):
                r["p_holm"] = a
                r["ci_excludes_0"] = bool(r["ci_low_pp"] > 0 or r["ci_high_pp"] < 0)
            comps += fam7c

    # E8 (đối chứng không đồ thị): tác động của đồ thị = ResACAG (E6) − MLP (E8), cùng test set, cùng seed.
    p8, _ = data["E8"]
    if p8 is not None and p6 is not None:
        seeds8 = sorted(s for m, s in p8 if m == "mlp")
        seeds6 = sorted(s for m, s in p6 if m == "resacag")
        fam8 = []
        r = compare("E8 (đối chứng): E6 resacag (có đồ thị) - E8 mlp (không đồ thị)",
                    {s: p6[("resacag", s)] for s in seeds6}, {s: p8[("mlp", s)] for s in seeds8},
                    c6, seeds6, seeds8, rng)
        if r:
            fam8.append(r)
        for b in BASELINES:
            seeds_b = sorted(s for m, s in p6 if m == b)
            r = compare(f"E8 (đối chứng): E8 mlp - E6 {b}",
                        {s: p8[("mlp", s)] for s in seeds8}, {s: p6[(b, s)] for s in seeds_b},
                        c6, seeds8, seeds_b, rng)
            if r:
                fam8.append(r)
        if fam8:
            adj8 = holm([r["p_raw"] for r in fam8])
            for r, a in zip(fam8, adj8):
                r["p_holm"] = a
                r["ci_excludes_0"] = bool(r["ci_low_pp"] > 0 or r["ci_high_pp"] < 0)
            comps += fam8

    # E9 (đối chứng topology ngẫu nhiên, giữ bậc): cấu trúc thật (E6 ResACAG) so với cấu trúc ngẫu nhiên (E9).
    p9, _ = data["E9"]
    if p9 is not None and p6 is not None:
        seeds9 = sorted(s for m, s in p9 if m == "resacag")
        seeds6 = sorted(s for m, s in p6 if m == "resacag")
        fam9 = []
        r = compare("E9 (đối chứng): E6 resacag (topology thật) - E9 resacag (topology ngẫu nhiên giữ bậc)",
                    {s: p6[("resacag", s)] for s in seeds6}, {s: p9[("resacag", s)] for s in seeds9},
                    c6, seeds6, seeds9, rng)
        if r:
            fam9.append(r)
        if p8 is not None:
            seeds8 = sorted(s for m, s in p8 if m == "mlp")
            r = compare("E9 (đối chứng): E9 resacag (topology ngẫu nhiên) - E8 mlp",
                        {s: p9[("resacag", s)] for s in seeds9}, {s: p8[("mlp", s)] for s in seeds8},
                        c6, seeds9, seeds8, rng)
            if r:
                fam9.append(r)
        if fam9:
            adj9 = holm([r["p_raw"] for r in fam9])
            for r, a in zip(fam9, adj9):
                r["p_holm"] = a
                r["ci_excludes_0"] = bool(r["ci_low_pp"] > 0 or r["ci_high_pp"] < 0)
            comps += fam9

    # E10 (thăm dò): cây quyết định + đặc trưng node (graph) / đã hoán vị (rewired), so với E6 và nhau.
    p10, _ = data["E10"]
    if p10 is not None and p6 is not None:
        def ss(P, m):
            return sorted(s for mm, s in P if mm == m)

        def pick(P, m):
            return {s: P[(m, s)] for s in ss(P, m)}

        pairs = [("E10: dt_graph - E6 dt", (p10, "dt_graph"), (p6, "dt")),
                 ("E10: hgb_graph - E6 hgb", (p10, "hgb_graph"), (p6, "hgb")),
                 ("E10: dt_graph - dt_rewired", (p10, "dt_graph"), (p10, "dt_rewired")),
                 ("E10: hgb_graph - hgb_rewired", (p10, "hgb_graph"), (p10, "hgb_rewired")),
                 ("E10: dt_rewired - E6 dt", (p10, "dt_rewired"), (p6, "dt")),
                 ("E10: dt_graph - E6 resacag", (p10, "dt_graph"), (p6, "resacag"))]
        fam10 = []
        for label, (PA, ma), (PB, mb) in pairs:
            if ss(PA, ma) and ss(PB, mb):
                r = compare(label, pick(PA, ma), pick(PB, mb), c6, ss(PA, ma), ss(PB, mb), rng)
                if r:
                    fam10.append(r)
        if fam10:
            adj10 = holm([r["p_raw"] for r in fam10])
            for r, a in zip(fam10, adj10):
                r["p_holm"] = a
                r["ci_excludes_0"] = bool(r["ci_low_pp"] > 0 or r["ci_high_pp"] < 0)
            comps += fam10

    s_df = pd.DataFrame(summary).round(3)
    c_df = pd.DataFrame(comps).round(4)
    s_df.to_csv(OUT / "lab6_summary.csv", index=False, encoding="utf-8")
    c_df.to_csv(OUT / "lab6_comparisons.csv", index=False, encoding="utf-8")
    print("\n== Điểm theo mô hình (F1 %, mean ± SD qua seed) ==")
    print(s_df.to_string(index=False))
    print("\n== Chênh lệch ghép cặp (pp; CI 95% cluster bootstrap; Holm) ==")
    print(c_df.to_string(index=False))
    print(f"\nĐã ghi: {OUT}")


if __name__ == "__main__":
    main()
