"""Hình cho Phần C (đánh giá có kiểm soát trên cse_cic).

python -I plot_controlled_experiments.py LAB6_SUMMARY_CSV LAB6_COMPARISONS_CSV E10_SUMMARY_CSV OUT_DIR

  - fig_c_macro: macro-F1 (trung bình, độ lệch chuẩn qua 5 seed). (a) sáu mô hình của E6; (b) ResACAG và các đối chứng.
  - fig_c_forest: macro-F1 của ResACAG trừ Decision Tree qua các kiểm tra độ nhạy, kèm khoảng tin cậy 95%.
Các số tổng hợp của E10 được đọc từ tệp kết quả báo cáo đi kèm.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

summary_csv, comparisons_csv, e10_csv, out_dir = sys.argv[1:5]
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42
fam = [f for f in ("Arial", "Helvetica") if any(f == x.name for x in font_manager.fontManager.ttflist)]
plt.rcParams["font.family"] = fam + ["DejaVu Sans"]

BLUE, ORANGE = "#0072B2", "#D55E00"

rows = {}
for r in csv.DictReader(open(summary_csv, encoding="utf-8")):
    rows[(r["exp"], r["model"])] = (float(r["macro_f1_mean"]), float(r["macro_f1_sd"]))
comp = {r["comparison"]: r for r in csv.DictReader(open(comparisons_csv, encoding="utf-8"))}
with open(e10_csv, encoding="utf-8", newline="") as stream:
    e10 = {row["model"]: (float(row["macro_f1_mean"]), float(row["macro_f1_sd"]))
           for row in csv.DictReader(stream)}


def vn(x, d=1):
    return (("%." + str(d) + "f") % x).replace(".", ",")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ hình 1: macro-F1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2), gridspec_kw={"width_ratios": [6, 6.4]})

a = [("Decision\nTree", "dt"), ("Random\nForest", "rf"), ("Extra\nTrees", "et"),
     ("HistGradient-\nBoosting", "hgb"), ("ResACAG", "resacag"), ("CatBoost", "catboost")]
a = [(lab, *rows[("E6", key)], key == "resacag") for lab, key in a]
b = [
    # Nhãn trong báo cáo đã đánh số lại: E1 (E6 trong lab 6), E4 (E7a), E5 (E7b), E6 (E8), E7 (E9), E8 (E10).
    ("E1\nResACAG", *rows[("E6", "resacag")], True),
    ("E4\ntrọng số\nlớp", *rows[("E7a", "resacag")], False),
    ("E5\nnút từ đặc\ntrưng luồng", *rows[("E7b", "resacag")], False),
    ("E6\nMulti-Layer\nPerceptron", *rows[("E8", "mlp")], False),
    ("E7\ntopology\nngẫu nhiên", *rows[("E9", "resacag")], False),
    ("E8\nDecision Tree\n+ đồ thị thật", *e10["dt_graph"], False),
]


def draw(ax, data, title):
    for x, (label, m, sd, hi) in enumerate(data):
        ax.bar(x, m, color=ORANGE if hi else BLUE, width=0.7, zorder=3)
        ax.errorbar(x, m, yerr=sd, color="black", capsize=3, lw=1.2, capthick=1.2, zorder=4)
        ax.text(x, m + sd + 1.2, vn(m), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], fontsize=7.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", color="#e9e9e9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)


draw(ax1, a, "(a) E1: sáu mô hình")
draw(ax2, b, "(b) ResACAG và các đối chứng")
ax1.set_ylabel("Macro-F1 (%)", fontsize=9.5)
fig.tight_layout(w_pad=2.0)
fig.savefig(out_dir / "fig_c_macro.pdf", bbox_inches="tight")
fig.savefig(out_dir / "fig_c_macro.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ hình 2: ResACAG trừ Decision Tree
items = [
    ("E1: so sánh chính", "E6: resacag - dt"),
    ("E2: bỏ dòng ghép mơ hồ", "E0 (bỏ ghép mơ hồ): resacag - dt"),
    ("E3: bỏ dòng test trùng train", "E2 (bỏ trùng lặp): resacag - dt"),
    ("E4: trọng số lớp", "E7a (thăm dò): resacag trọng số lớp - E6 dt"),
    ("E5: nút từ đặc trưng luồng", "E7b (thăm dò): resacag biểu diễn đặc trưng node - E6 dt"),
]
fig, ax = plt.subplots(figsize=(8.0, 3.0))
for y, (label, key) in enumerate(items):
    r = comp[key]
    d, lo, hi = float(r["delta_macro_pp"]), float(r["ci_low_pp"]), float(r["ci_high_pp"])
    p = float(r["p_holm"])
    sig = p < 0.05
    col = ORANGE if sig else BLUE
    ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="butt", zorder=3)
    ax.plot([d], [y], marker="D", ms=6, color=col, zorder=4)
ax.axvline(0, color="#555555", lw=1.0, zorder=2)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([i[0] for i in items], fontsize=9)
ax.set_ylim(len(items) - 0.5, -0.5)
ax.set_xlim(-7, 1.2)
ax.set_xticks([-6, -4, -2, 0])
ax.set_xticklabels(["−6", "−4", "−2", "0"], fontsize=8.5)
ax.set_xlabel("Macro-F1 của ResACAG trừ Decision Tree (điểm phần trăm)", fontsize=9)
ax.grid(axis="x", color="#e9e9e9", lw=0.8, zorder=0)
ax.set_axisbelow(True)
clean(ax)
ax.legend(handles=[plt.Line2D([], [], color=BLUE, marker="D", lw=2.2, label="Holm p ≥ 0,05"),
                   plt.Line2D([], [], color=ORANGE, marker="D", lw=2.2, label="Holm p < 0,05")],
          loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(out_dir / "fig_c_forest.pdf", bbox_inches="tight")
fig.savefig(out_dir / "fig_c_forest.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("ok")
