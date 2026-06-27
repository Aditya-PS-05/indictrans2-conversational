#!/usr/bin/env python3
"""Generate the paper's figures from the real chrF2 results. Outputs vector PDF
(for LaTeX) + PNG (for web) into paper/figures/."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 140,
})
BLUE, GREY, RED = "#0073FF", "#B8C2CC", "#E0524A"
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# (lang, conv_base, conv_soup, flores_base, flores_soup)
DATA = [
    ("asm",61.9,69.4,44.2,44.1),("ben",68.7,73.6,58.1,58.1),("brx",62.1,68.0,45.9,46.0),
    ("doi",63.4,71.7,49.8,49.6),("gom",69.4,75.6,49.1,49.2),("guj",53.7,65.9,55.3,54.7),
    ("hin",53.9,56.8,60.0,59.6),("kan",52.4,57.1,59.0,58.7),("kas",45.8,51.3,39.1,38.8),
    ("mai",57.8,65.2,56.6,56.6),("mal",45.7,48.4,61.1,61.4),("mar",70.2,77.3,55.0,54.5),
    ("mni",56.4,62.8,48.2,48.3),("npi",68.8,72.1,59.4,59.0),("ory",56.4,65.7,52.0,51.3),
    ("pan",68.0,76.5,55.3,54.8),("san",54.5,63.6,36.3,36.5),("sat",50.7,56.2,30.7,30.3),
    ("tam",52.7,55.1,64.1,64.4),("tel",49.9,60.0,62.4,62.0),("urd",48.1,49.4,54.7,54.7),
]
# languages whose conversational test is hard subtitle data (the trustworthy gains)
HARD = {"hin","tam","mal","urd","npi"}

# ---- Figure 1: conversational chrF gain per language -----------------------------
rows = sorted(DATA, key=lambda r: r[2]-r[1])         # ascending so largest on top
labels = [r[0] for r in rows]
gains = [r[2]-r[1] for r in rows]
colors = [BLUE if r[0] in HARD else GREY for r in rows]
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(labels, gains, color=colors)
for y, g in enumerate(gains):
    ax.text(g+0.15, y, f"+{g:.1f}", va="center", fontsize=8.5)
ax.set_xlabel("conversational chrF2 gain  (soup − base)")
ax.set_title("Every Indic language improves on conversational chrF")
ax.set_xlim(0, max(gains)+1.6)
ax.legend(handles=[plt.Rectangle((0,0),1,1,color=BLUE), plt.Rectangle((0,0),1,1,color=GREY)],
          labels=["hard subtitle test (reliable)", "easy in-domain test"],
          loc="lower right", frameon=False, fontsize=9)
save(fig, "fig_gains")

# ---- Figure 2: Hindi forgetting and the fix -------------------------------------
models = ["base", "naive FT", "mixed FT", "soup"]
conv = [51.6, 57.0, 56.0, 54.1]
flores = [61.7, 57.8, 60.6, 61.8]
x = range(len(models)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.3))
b1 = ax.bar([i-w/2 for i in x], conv, w, label="conversational", color=BLUE)
b2 = ax.bar([i+w/2 for i in x], flores, w, label="FLORES (general)", color=GREY)
ax.axhline(61.7, ls="--", lw=1, color=RED)
ax.text(3.05, 61.9, "base FLORES", color=RED, fontsize=8, ha="right")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{b.get_height():.1f}",
            ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(models)
ax.set_ylabel("chrF2"); ax.set_ylim(48, 66)
ax.set_title("Hindi: naive fine-tuning forgets FLORES; replay + soup fixes it")
ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=9)
save(fig, "fig_forgetting")

# ---- Figure 3: gain vs. preservation scatter ------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.6))
dflores = [r[4]-r[3] for r in DATA]
dconv = [r[2]-r[1] for r in DATA]
ax.axvspan(-0.7, 0.7, color=GREY, alpha=0.25, label="FLORES within ±0.7 (tie)")
ax.axvline(0, color="black", lw=0.8)
ax.scatter(dflores, dconv, color=BLUE, s=30, zorder=3)
for r in DATA:
    if r[0] in HARD or (r[2]-r[1]) > 9:
        ax.annotate(r[0], (r[4]-r[3], r[2]-r[1]), fontsize=7.5,
                    xytext=(3,2), textcoords="offset points")
ax.set_xlabel("FLORES chrF2 change  (general quality)")
ax.set_ylabel("conversational chrF2 gain")
ax.set_title("Conversational gain without losing general quality")
ax.legend(loc="lower left", frameon=False, fontsize=9)
save(fig, "fig_tradeoff")

print("wrote:", sorted(os.listdir(OUT)))
