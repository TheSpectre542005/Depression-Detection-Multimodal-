"""
generate_cv_results.py
======================
Generates charts for the leakage-free 5-fold CV results and saves to 'results final/'.
Also adds a side-by-side comparison showing leaky vs honest numbers.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

OUT_DIR = "results final"
os.makedirs(OUT_DIR, exist_ok=True)

BG     = "#0F1117"
PANEL  = "#1A1D27"
TEXT_C = "#E0E0E0"
GRID_C = "#2A2D3A"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID_C, "axes.labelcolor": TEXT_C,
    "axes.titlecolor": TEXT_C, "xtick.color": TEXT_C,
    "ytick.color": TEXT_C, "text.color": TEXT_C,
    "grid.color": GRID_C, "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "DejaVu Sans", "font.size": 11,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID_C,
})

PALETTE = {
    "Text":        "#4C9BE8",
    "Audio":       "#5DBB8A",
    "Visual":      "#F4A261",
    "Late Fusion": "#9B59B6",
}

# ── Honest 5-fold OOF results ──────────────────────────────────────────────
honest = {
    "Text":        {"Accuracy":0.6256,"Balanced Acc":0.6093,"F1":0.4744,"AUC-ROC":0.6361,
                    "Precision":0.4066,"Sensitivity":0.5692,"Specificity":0.6494,
                    "AUC CI lo":0.5544,"AUC CI hi":0.7127},
    "Audio":       {"Accuracy":0.7306,"Balanced Acc":0.6128,"F1":0.4158,"AUC-ROC":0.6092,
                    "Precision":0.5833,"Sensitivity":0.3231,"Specificity":0.9026,
                    "AUC CI lo":0.5230,"AUC CI hi":0.6918},
    "Visual":      {"Accuracy":0.5434,"Balanced Acc":0.4975,"F1":0.3333,"AUC-ROC":0.4997,
                    "Precision":0.2941,"Sensitivity":0.3846,"Specificity":0.6104,
                    "AUC CI lo":0.4107,"AUC CI hi":0.5866},
    "Late Fusion": {"Accuracy":0.6712,"Balanced Acc":0.5973,"F1":0.4286,"AUC-ROC":0.6225,
                    "Precision":0.4426,"Sensitivity":0.4154,"Specificity":0.7792,
                    "AUC CI lo":0.5373,"AUC CI hi":0.6986},
}

# ── Leaky (train-set-included) results for comparison ─────────────────────
leaky = {
    "Text":        {"Accuracy":0.9498,"F1":0.9091,"AUC-ROC":0.9674},
    "Audio":       {"Accuracy":0.9406,"F1":0.8943,"AUC-ROC":0.9810},
    "Visual":      {"Accuracy":0.5936,"F1":0.5291,"AUC-ROC":0.6823},
    "Late Fusion": {"Accuracy":0.9361,"F1":0.8833,"AUC-ROC":0.9529},
}

df_honest = pd.DataFrame(honest).T
df_honest.index.name = "Model"
df_honest.to_csv(os.path.join(OUT_DIR, "cv_accuracy_matrix.csv"))
print("[SAVED] cv_accuracy_matrix.csv")

models = list(honest.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Leaky vs Honest side-by-side AUC comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Leaky (Train+Test) vs Honest (5-Fold OOF) — Key Metrics",
             fontsize=14, fontweight="bold", color=TEXT_C, y=1.01)

for ax_i, metric in enumerate(["Accuracy", "F1", "AUC-ROC"]):
    ax = axes[ax_i]
    ax.set_facecolor(PANEL)
    x = np.arange(len(models))
    w = 0.35
    leaky_vals  = [leaky[m][metric] for m in models]
    honest_vals = [honest[m][metric] for m in models]

    bars_l = ax.bar(x - w/2, leaky_vals,  w, color="#E74C3C", alpha=0.82, label="Leaky (reported)", zorder=3)
    bars_h = ax.bar(x + w/2, honest_vals, w, color="#2ECC71", alpha=0.82, label="Honest (5-fold OOF)", zorder=3)

    for bar, v in zip(bars_l, leaky_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                color="#E74C3C", fontweight="bold")
    for bar, v in zip(bars_h, honest_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                color="#2ECC71", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.axhline(0.7, color="#888", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(0.5, color="#555", linestyle=":", linewidth=1, alpha=0.5)

plt.tight_layout()
path8 = os.path.join(OUT_DIR, "8_leaky_vs_honest.png")
plt.savefig(path8, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path8}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: AUC with 95% Bootstrap CI error bars
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

x = np.arange(len(models))
aucs   = [honest[m]["AUC-ROC"]  for m in models]
lo_err = [honest[m]["AUC-ROC"] - honest[m]["AUC CI lo"] for m in models]
hi_err = [honest[m]["AUC CI hi"] - honest[m]["AUC-ROC"] for m in models]
colors = [PALETTE[m] for m in models]

bars = ax.bar(x, aucs, 0.5, color=colors, alpha=0.85, zorder=3)
ax.errorbar(x, aucs, yerr=[lo_err, hi_err],
            fmt="none", color="white", capsize=8, capthick=2,
            elinewidth=2, zorder=4)

for bar, v, lo, hi in zip(bars, aucs, lo_err, hi_err):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{v:.4f}", ha="center", va="bottom", fontsize=10,
            color=TEXT_C, fontweight="bold")
    ax.text(bar.get_x()+bar.get_width()/2, hi + bar.get_height() + 0.02,
            f"95% CI", ha="center", va="bottom", fontsize=8, color="#888")

ax.axhline(0.5, color="#E74C3C", linestyle="--", linewidth=1.5,
           alpha=0.7, label="Random chance (AUC=0.50)")
ax.axhline(0.7, color="#F4A261", linestyle=":", linewidth=1.2,
           alpha=0.6, label="Clinical screening threshold (0.70)")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title("AUC-ROC with 95% Bootstrap CI — 5-Fold OOF (Leakage-Free)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.35, zorder=0)

plt.tight_layout()
path9 = os.path.join(OUT_DIR, "9_auc_with_ci.png")
plt.savefig(path9, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path9}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Honest metrics heatmap
# ─────────────────────────────────────────────────────────────────────────────
heat_metrics = ["Accuracy","Balanced Acc","F1","AUC-ROC","Precision","Sensitivity","Specificity"]
heat_data    = np.array([[honest[m].get(met,0) for met in heat_metrics] for m in models])

cmap_heat = LinearSegmentedColormap.from_list("heat",
    [(0.1,0.13,0.18),(0.18,0.4,0.7),(0.3,0.76,0.56),(0.98,0.85,0.37)])

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

im = ax.imshow(heat_data, cmap=cmap_heat, aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(heat_metrics)))
ax.set_xticklabels(heat_metrics, fontsize=11, rotation=20, ha="right")
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=12, fontweight="bold")
ax.tick_params(length=0)

for i in range(len(models)):
    for j in range(len(heat_metrics)):
        v = heat_data[i, j]
        tc = "white" if v < 0.65 else "#0a0d14"
        ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=tc)

ax.set_title("Honest Accuracy Matrix — 5-Fold OOF (SMOTE inside folds, no leakage)",
             fontsize=13, fontweight="bold", pad=14)
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.ax.tick_params(colors=TEXT_C)
cbar.set_label("Score", color=TEXT_C)

plt.tight_layout()
path10 = os.path.join(OUT_DIR, "10_honest_heatmap.png")
plt.savefig(path10, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path10}")


# ─────────────────────────────────────────────────────────────────────────────
# Print final honest summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SENTIRA — Honest 5-Fold OOF Results (no leakage)")
print(f"{'='*70}")
COLS = ["Accuracy","Balanced Acc","F1","AUC-ROC","Precision","Sensitivity","Specificity"]
hdr = f"  {'Model':<16}" + "".join(f"{c:>13}" for c in COLS)
print(hdr)
print("  " + "-"*(len(hdr)-2))
for m in models:
    row  = honest[m]
    line = f"  {m:<16}" + "".join(f"{row.get(c,0):>13.4f}" for c in COLS)
    print(line)

print(f"\n  {'Model':<16}  {'AUC 95% CI'}")
print("  " + "-"*40)
for m in models:
    lo = honest[m]["AUC CI lo"]; hi = honest[m]["AUC CI hi"]
    print(f"  {m:<16}  [{lo:.4f}, {hi:.4f}]")

print(f"\n{'='*70}")
print(f"  Note: Text AUC CI includes 0.50 (lower bound 0.5544 is close to chance).")
print(f"  Audio CI [0.52, 0.69] — modest but real signal above chance.")
print(f"  Visual CI [0.41, 0.59] — overlaps 0.50 (no proven signal).")
print(f"  Late Fusion CI [0.54, 0.70] — consistent with published DAIC-WOZ baselines.")
print(f"{'='*70}\n")
print(f"[DONE] Outputs: {os.path.abspath(OUT_DIR)}")
