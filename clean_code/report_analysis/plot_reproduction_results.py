"""Vẽ các hình khảo sát dữ liệu (Phần A) và kết quả tái hiện (Phần B) cho CẢ HAI bộ dữ liệu.

Thứ tự panel luôn giống Hình 1: (a) CIC-DarkNet, (b) CSE-CIC-IDS2018.

python -I plot_reproduction_results.py EDA_JSON DARKNET_RUN_DIR CSE_RUN_DIR OUT_DIR
  EDA_JSON       kết quả của summarize_datasets.py
  DARKNET_RUN_DIR  thư mục có metrics.json, history.json, best_epoch.json của ResACAG trên darknet (seed 42)
  CSE_RUN_DIR      như trên, cho cse_cic (seed 42)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

eda_json, dark_run, cse_run, out_dir = sys.argv[1:5]
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
EDA = json.load(open(eda_json, encoding="utf-8"))
DARK, CSE = EDA["darknet"], EDA["cse_cic"]


def run(path):
    p = Path(path)
    return {"metrics": json.load(open(p / "metrics.json", encoding="utf-8")),
            "history": json.load(open(p / "history.json", encoding="utf-8")),
            "best": json.load(open(p / "best_epoch.json", encoding="utf-8"))}


RUN_D, RUN_C = run(dark_run), run(cse_run)

plt.rcParams["pdf.fonttype"] = 42
fam = [f for f in ("Arial", "Helvetica") if any(f == x.name for x in font_manager.fontManager.ttflist)]
plt.rcParams["font.family"] = fam + ["DejaVu Sans"]

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
TITLES = ("(a) CIC-DarkNet", "(b) CSE-CIC-IDS2018")


def vn_int(n):
    return "{:,}".format(int(round(n))).replace(",", ".")


def vn(x, d=2):
    return ("%." + str(d) + "f") % x if False else (("%." + str(d) + "f") % x).replace(".", ",")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(out_dir / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(out_dir / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("ok", name)


SLOTS = 9  # cả hai panel dùng 9 hàng để thanh có cùng độ dày

# ------------------------------------------------------------------ 08: trùng đặc trưng
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1))
for ax, title, d in zip(axes, TITLES, (DARK, CSE)):
    classes = d["classes"]
    for y, c in enumerate(classes):
        v = 100 * d["overlap_by_class"][y]
        o = 100 * d["overlap_other_label_by_class"][y]
        ax.barh(y, v, height=0.62, color=BLUE, zorder=3)
        if o > 0:
            ax.barh(y, o, height=0.30, color=ORANGE, zorder=4)
        ax.text(v + 1.5, y, vn(v) + "%", va="center", ha="left", fontsize=8)
        if o >= 10:
            ax.text(o / 2, y, vn(o) + "%", va="center", ha="center", fontsize=7, color="white", zorder=5)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_ylim(SLOTS - 0.5, -0.5)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel("Tỷ lệ trong tập kiểm tra của từng lớp (%)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", color="#e9e9e9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
fig.legend(handles=[Patch(color=BLUE, label="Trùng đặc trưng với ít nhất một mẫu train"),
                    Patch(color=ORANGE, label="Trong đó có mẫu train mang nhãn khác")],
           loc="lower center", ncol=2, frameon=False, fontsize=8.5, handlelength=1.4)
fig.tight_layout(w_pad=2.0, rect=[0, 0.07, 1, 1])
save(fig, "08_eda_trung_dac_trung")

# ------------------------------------------------------------------ 09: nhãn theo thời gian
dark_months = sorted(DARK["by_month"])
dark_cols = ["%d/%s" % (int(m[5:7]), m[:4]) for m in dark_months]
dark_mat = np.array([DARK["by_month"][m] for m in dark_months]).T  # lớp x tháng
cse_days = sorted(CSE["by_day"], key=lambda s: (s[:2] != "03", s))  # 02-xx trước 03-02
cse_days = sorted(CSE["by_day"], key=lambda s: (int(s[:2]), int(s[3:])))
cse_cols = ["%s/%s" % (s[3:], s[:2]) for s in cse_days]
cse_mat = np.array([CSE["by_day"][s] for s in cse_days]).T

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.8, 4.2), gridspec_kw={"width_ratios": [5, 7]}, layout="constrained")
im = None
for ax, title, mat, cols, d, xl in ((ax_a, TITLES[0], dark_mat, dark_cols, DARK, "Tháng ghi dữ liệu"),
                                    (ax_b, TITLES[1], cse_mat, cse_cols, CSE, "Ngày ghi dữ liệu trong năm 2018")):
    pct = 100.0 * mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, vn_int(mat[i, j]), ha="center", va="center", fontsize=7.2,
                    color="white" if pct[i, j] > 55 else ("#333333" if mat[i, j] else "#8a8a8a"))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8.3)
    ax.set_yticks(range(len(d["classes"])))
    ax.set_yticklabels(d["classes"], fontsize=8.8)
    ax.set_xlabel(xl, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
cb = fig.colorbar(im, ax=[ax_a, ax_b], fraction=0.035, pad=0.02)
cb.set_label("Tỷ lệ theo từng lớp (%)", fontsize=9)
cb.ax.tick_params(labelsize=8)
cb.outline.set_visible(False)
save(fig, "09_eda_nhan_theo_ngay")

# ------------------------------------------------------------------ 10a: lân cận cùng nhãn
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
for ax, title, d in zip(axes, TITLES, (DARK, CSE)):
    classes = d["classes"]
    for y, c in enumerate(classes):
        v = 100 * d["homophily_by_class"][y]
        ax.barh(y, v, height=0.62, color=BLUE, zorder=3)
        ax.text(v + 1.5, y, vn(v) + "%", va="center", ha="left", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_ylim(SLOTS - 0.5, -0.5)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel("Tỷ lệ luồng lân cận cùng nhãn (%)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", color="#e9e9e9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
fig.tight_layout(w_pad=2.0)
save(fig, "10_eda_lan_can_cung_nhan")

# ------------------------------------------------------------------ 10b: nút chưa thấy
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1))
for ax, title, d in zip(axes, TITLES, (DARK, CSE)):
    classes = d["classes"]
    u = d["unseen_by_class"]
    for y, c in enumerate(classes):
        left = 0.0
        for key, col in (("both_seen", BLUE), ("one_unseen", ORANGE), ("both_unseen", GREEN)):
            w = 100 * u[key][y]
            ax.barh(y, w, left=left, height=0.62, color=col, zorder=3)
            if w >= 11:
                ax.text(left + w / 2, y, vn(w, 1) + "%", ha="center", va="center", fontsize=7, color="white", zorder=5)
            left += w
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_ylim(SLOTS - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel("Tỷ lệ luồng kiểm tra của từng lớp (%)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    clean(ax)
fig.legend(handles=[Patch(color=BLUE, label="Cả hai nút đã thấy"), Patch(color=ORANGE, label="Một nút chưa thấy"),
                    Patch(color=GREEN, label="Cả hai nút chưa thấy")],
           loc="lower center", ncol=3, frameon=False, fontsize=8.5, handlelength=1.4)
fig.tight_layout(w_pad=2.0, rect=[0, 0.07, 1, 1])
save(fig, "10_eda_nut_chua_thay")

# ------------------------------------------------------------------ 05: diễn biến huấn luyện
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
for ax, title, r in zip(axes, TITLES, (RUN_D, RUN_C)):
    h = r["history"]
    ep = [e["epoch"] for e in h]
    fw = [100 * e["validation_f1_weighted"] for e in h]
    fm = [100 * e["validation_f1_macro"] for e in h]
    ax.plot(ep, fw, color=BLUE, marker="s", ms=4.5, lw=1.7, label="F1 weighted", zorder=3)
    ax.plot(ep, fm, color=ORANGE, marker="o", ms=4.5, lw=1.7, label="F1 macro", zorder=3)
    b = r["best"]["epoch"]
    ax.scatter([b], [fm[b - 1]], s=170, facecolors="none", edgecolors="#B8860B", linewidths=2, zorder=4)
    if len(ep) <= 6:
        for x, a_, m_ in zip(ep, fw, fm):
            ax.text(x, a_ + 2.5, vn(a_), ha="center", va="bottom", fontsize=7.5, color=BLUE)
            if x == b:
                ax.text(x + 0.12, m_ + 3.5, vn(m_), ha="left", va="bottom", fontsize=7.5, color=ORANGE)
            else:
                ax.text(x, m_ - 3, vn(m_), ha="center", va="top", fontsize=7.5, color=ORANGE)
        ax.set_xticks(ep)
    else:
        for x in (1, b, len(ep)):
            if x not in (1, b):
                ax.text(x, fm[x - 1] - 5, vn(fm[x - 1]), ha="center", va="top", fontsize=7.5, color=ORANGE)
            ax.text(x, fw[x - 1] + 2.5, vn(fw[x - 1]), ha="center", va="bottom", fontsize=7.5, color=BLUE)
        ax.text(b + 1.0, fm[b - 1] + 4, vn(fm[b - 1]), ha="left", va="bottom", fontsize=7.5, color=ORANGE)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.annotate("Mô hình được chọn: vòng %d" % b, xy=(b, fm[b - 1]),
                xytext=(9.5, 9) if len(ep) > 6 else (1.45, 52), textcoords="data", fontsize=8, color="#555555", ha="left",
                va="center", arrowprops=dict(arrowstyle="-", color="#777777", lw=0.9, shrinkA=2, shrinkB=9))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Vòng học", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(labelsize=8.5)
    ax.grid(color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
axes[0].set_ylabel("F1 trên tập chọn mô hình (%)", fontsize=9)
axes[0].legend(loc="upper right", frameon=False, fontsize=8.5)
fig.tight_layout(w_pad=1.5)
save(fig, "05_dien_bien_huan_luyen")

# ------------------------------------------------------------------ 06: ma trận nhầm lẫn
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.3), gridspec_kw={"width_ratios": [9, 7]}, layout="constrained")
for ax, title, r, d in zip(axs, TITLES, (RUN_D, RUN_C), (DARK, CSE)):
    cm = np.array(r["metrics"]["test"]["confusion_matrix"], dtype=float)
    pct = 100.0 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, vn_int(cm[i, j]), ha="center", va="center", fontsize=8.6,
                    color="white" if pct[i, j] > 55 else ("#333333" if cm[i, j] else "#9a9a9a"))
    names = d["classes"]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Nhãn dự đoán", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(length=0)
    ax.set_anchor("N")
    for s_ in ax.spines.values():
        s_.set_visible(False)
axs[0].set_ylabel("Nhãn thực tế", fontsize=9)
cb = fig.colorbar(im, ax=list(axs), fraction=0.025, pad=0.02, shrink=0.75)
cb.set_label("Tỷ lệ theo từng hàng (%)", fontsize=9)
cb.ax.tick_params(labelsize=8)
cb.outline.set_visible(False)
save(fig, "06_ma_tran_nham_lan")

# ------------------------------------------------------------------ 07: ảnh hưởng của việc thay đặc trưng
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
noise_rows = {}
for ax, title, r, key in zip(axes, TITLES, (RUN_D, RUN_C), ("darknet", "cse_cic")):
    nz = r["metrics"]["noise"]
    levels = sorted(nz, key=float)
    xs = [100 * float(k) for k in levels]
    fw = [100 * nz[k]["f1_weighted"] for k in levels]
    fm = [100 * nz[k]["f1_macro"] for k in levels]
    noise_rows[key] = list(zip(xs, fw, fm))
    ax.plot(xs, fw, color=BLUE, marker="s", ms=4.5, lw=1.7, label="F1 weighted", zorder=3)
    ax.plot(xs, fm, color=ORANGE, marker="o", ms=4.5, lw=1.7, label="F1 macro", zorder=3)
    for x, a_, m_ in zip(xs, fw, fm):
        ax.text(x, a_ + 2.5, vn(a_), ha="center", va="bottom", fontsize=7.5, color=BLUE)
        ax.text(x, m_ - 3, vn(m_), ha="center", va="top", fontsize=7.5, color=ORANGE)
    ax.set_xticks(xs)
    ax.set_xlim(-3, 33)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Tỷ lệ luồng kiểm tra bị thay đặc trưng (%)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(labelsize=8.5)
    ax.grid(color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
axes[0].set_ylabel("F1 trên tập kiểm tra (%)", fontsize=9)
axes[0].legend(loc="lower left", frameon=False, fontsize=8.5)
fig.tight_layout(w_pad=1.5)
save(fig, "07_anh_huong_nhieu")
json.dump(noise_rows, open(out_dir / "noise_rows.json", "w"), indent=1)
