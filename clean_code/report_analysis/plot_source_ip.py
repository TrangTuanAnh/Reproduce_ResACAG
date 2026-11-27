"""Hình E9: macro-F1 của ba mô hình cây khi chia ngẫu nhiên và khi chia theo IP nguồn.

python plot_source_ip.py SUMMARY_CSV OUT_DIR
  SUMMARY_CSV: artifacts/source_ip/summary.csv (cột split, model, set, macro_f1_mean, macro_f1_sd)
Kết quả: fig_c_ip.pdf và fig_c_ip.png trong OUT_DIR.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

summary_csv, out_dir = sys.argv[1:3]
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42
fam = [f for f in ("Arial", "Helvetica") if any(f == x.name for x in font_manager.fontManager.ttflist)]
plt.rcParams["font.family"] = fam + ["DejaVu Sans"]

BLUE, ORANGE = "#0072B2", "#D55E00"

rows = {}
with Path(summary_csv).open(encoding="utf-8", newline="") as source:
    for row in csv.DictReader(source):
        key = (row["split"], row["model"], row["set"])
        rows[key] = (100 * float(row["macro_f1_mean"]), 100 * float(row["macro_f1_sd"]))

MODELS = [("dt", "Decision\nTree"), ("rf", "Random\nForest"), ("hgb", "HistGradient-\nBoosting")]
PANELS = [("RAND", "(a) Chia ngẫu nhiên 70/30"), ("GRP", "(b) Chia theo IP nguồn")]
W = 0.36


def vn(x):
    return ("%.1f" % x).replace(".", ",")


fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), sharey=True)
for ax, (split, title) in zip(axes, PANELS):
    for i, (key, label) in enumerate(MODELS):
        for j, (feat, col) in enumerate((("flow", BLUE), ("flow+id", ORANGE))):
            m, sd = rows[(split, key, feat)]
            x = i + (j - 0.5) * W
            ax.bar(x, m, width=W * 0.92, color=col, zorder=3)
            ax.errorbar(x, m, yerr=sd, color="black", capsize=2.5, lw=1.1, capthick=1.1, zorder=4)
            ax.text(x, m + sd + 1.5, vn(m), ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[1] for m in MODELS], fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", color="#e9e9e9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Macro-F1 (%)", fontsize=9.5)
fig.legend(handles=[Patch(color=BLUE, label="Đặc trưng luồng"), Patch(color=ORANGE, label="Đặc trưng luồng và danh tính IP, cổng")],
           loc="lower center", ncol=2, frameon=False, fontsize=8.5, handlelength=1.4)
fig.tight_layout(w_pad=2.0, rect=[0, 0.07, 1, 1])
fig.savefig(out_dir / "fig_c_ip.pdf", bbox_inches="tight")
fig.savefig(out_dir / "fig_c_ip.png", dpi=170, bbox_inches="tight")
print("ok")
