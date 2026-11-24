import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

dark_audit, cse_audit, out_pdf, out_png = sys.argv[1:5]

plt.rcParams["pdf.fonttype"] = 42
fam = [f for f in ("Arial", "Helvetica") if any(f == x.name for x in font_manager.fontManager.ttflist)]
plt.rcParams["font.family"] = fam + ["DejaVu Sans"]


def load(path):
    a = json.load(open(path, encoding="utf-8"))
    classes = a["classes"]
    counts = [a["counts"][c] for c in classes]
    assert sum(counts) == a["rows"], (path, sum(counts), a["rows"])
    return classes, counts


def vn_int(n):
    return "{:,}".format(int(n)).replace(",", ".")


def vn_pct(x):
    return ("%.2f" % x).replace(".", ",")


BLUE, ORANGE = "#0072B2", "#D55E00"
SLOTS = 9  # cả hai panel dùng cùng số hàng để thanh có cùng độ dày

panels = [
    ("(a) CIC-DarkNet", load(dark_audit), 215000, [0, 50000, 100000, 150000]),
    ("(b) CSE-CIC-IDS2018", load(cse_audit), 310000, [0, 100000, 200000, 300000]),
]

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
for ax, (title, (classes, counts), xmax, ticks) in zip(axes, panels):
    total = sum(counts)
    for y, c, n in zip(range(len(classes)), classes, counts):
        ax.barh(y, n, height=0.62, color=BLUE if c == "Benign" else ORANGE, zorder=3)
        ax.text(n + xmax * 0.015, y, "%s (%s%%)" % (vn_int(n), vn_pct(100.0 * n / total)),
                va="center", ha="left", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_ylim(SLOTS - 0.5, -0.5)
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels([vn_int(t) for t in ticks], fontsize=8.5)
    ax.set_xlabel("Số luồng (tổng %s)" % vn_int(total), fontsize=9.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", color="#e9e9e9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(w_pad=2.0)
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=170, bbox_inches="tight")
print("ok", plt.rcParams["font.family"])
