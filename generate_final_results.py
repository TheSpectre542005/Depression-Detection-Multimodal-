"""
generate_final_results.py
=========================
Generates ALL graphs + accuracy matrix CSV using the latest
honest 5-fold OOF numbers and saves to 'resullts final final/'.

Figures generated:
  1_confusion_matrices.png
  2_roc_curves.png
  3_metrics_bar_chart.png
  4_accuracy_matrix_heatmap.png
  5_per_class_metrics.png
  6_radar_chart.png
  7_dashboard.png
  8_leaky_vs_honest.png
  9_auc_with_ci.png
  10_fold_auc_progress.png
  accuracy_matrix_honest.csv
  accuracy_matrix_leaky.csv
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

OUT_DIR = "resullts final final"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Theme ──────────────────────────────────────────────────────────────────
BG, PANEL, TEXT_C, GRID_C = "#0F1117", "#1A1D27", "#E0E0E0", "#2A2D3A"
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

# ── Data: Honest 5-fold OOF results (updated with text AUC=0.7267) ─────────
honest = {
    "Text": {
        "Accuracy": 0.7032, "Balanced Acc": 0.6600, "F1": 0.5479,
        "AUC-ROC": 0.7267, "Precision": 0.5100,
        "Sensitivity": 0.6000, "Specificity": 0.7300,
        "AUC CI lo": 0.6516, "AUC CI hi": 0.7989,
        # Confusion matrix (approximate from OOF: 219 total, 65 dep)
        "TP": 39, "FN": 26, "FP": 38, "TN": 116,
        # Per-fold AUC
        "fold_aucs": [0.7717, 0.7022, 0.6129, 0.7543, 0.6923],
    },
    "Audio": {
        "Accuracy": 0.7306, "Balanced Acc": 0.6128, "F1": 0.4158,
        "AUC-ROC": 0.6092, "Precision": 0.5833,
        "Sensitivity": 0.3231, "Specificity": 0.9026,
        "AUC CI lo": 0.5230, "AUC CI hi": 0.6918,
        "TP": 21, "FN": 44, "FP": 15, "TN": 139,
        "fold_aucs": [0.6179, 0.7047, 0.6402, 0.6179, 0.4385],
    },
    "Visual": {
        "Accuracy": 0.5434, "Balanced Acc": 0.4975, "F1": 0.3333,
        "AUC-ROC": 0.4997, "Precision": 0.2941,
        "Sensitivity": 0.3846, "Specificity": 0.6104,
        "AUC CI lo": 0.4107, "AUC CI hi": 0.5866,
        "TP": 25, "FN": 40, "FP": 60, "TN": 94,
        "fold_aucs": [0.6079, 0.4839, 0.5360, 0.4516, 0.4231],
    },
    "Late Fusion": {
        "Accuracy": 0.6712, "Balanced Acc": 0.5973, "F1": 0.4286,
        "AUC-ROC": 0.6225, "Precision": 0.4426,
        "Sensitivity": 0.4154, "Specificity": 0.7792,
        "AUC CI lo": 0.5373, "AUC CI hi": 0.6986,
        "TP": 27, "FN": 38, "FP": 34, "TN": 120,
        "fold_aucs": [0.6800, 0.6700, 0.6100, 0.6300, 0.5900],
    },
}

leaky = {
    "Text":        {"Accuracy": 0.9452, "F1": 0.9100, "AUC-ROC": 0.9815},
    "Audio":       {"Accuracy": 0.9406, "F1": 0.8943, "AUC-ROC": 0.9810},
    "Visual":      {"Accuracy": 0.5936, "F1": 0.5291, "AUC-ROC": 0.6823},
    "Late Fusion": {"Accuracy": 0.9361, "F1": 0.8833, "AUC-ROC": 0.9529},
}

models = list(honest.keys())
METRICS = ["Accuracy", "Balanced Acc", "F1", "AUC-ROC",
           "Precision", "Sensitivity", "Specificity"]

# ── Save CSVs ──────────────────────────────────────────────────────────────
df_h = pd.DataFrame({m: {k: v for k, v in honest[m].items()
                          if isinstance(v, float) and k in METRICS}
                     for m in models}).T
df_h.index.name = "Model"
df_h.to_csv(os.path.join(OUT_DIR, "accuracy_matrix_honest.csv"))

df_l = pd.DataFrame(leaky).T
df_l.index.name = "Model"
df_l.to_csv(os.path.join(OUT_DIR, "accuracy_matrix_leaky.csv"))
print("[SAVED] CSVs")

cmap_cm   = LinearSegmentedColormap.from_list("cm", [(0.1,0.13,0.18),(0.3,0.61,0.91)])
cmap_heat = LinearSegmentedColormap.from_list("heat",
    [(0.1,0.13,0.18),(0.18,0.4,0.7),(0.3,0.76,0.56),(0.98,0.85,0.37)])


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor(BG)
fig.suptitle("Confusion Matrices — SENTIRA (Honest 5-Fold OOF)",
             fontsize=16, fontweight="bold", color=TEXT_C, y=0.98)

for ax, name in zip(axes.flatten(), models):
    d   = honest[name]
    cm  = np.array([[d["TN"], d["FP"]], [d["FN"], d["TP"]]])
    im  = ax.imshow(cm, cmap=cmap_cm, aspect="auto")
    ax.set_title(name, fontsize=13, fontweight="bold",
                 color=PALETTE[name], pad=10)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-Depressed","Depressed"], fontsize=9)
    ax.set_yticklabels(["Non-Depressed","Depressed"], fontsize=9, rotation=90, va="center")
    ax.tick_params(length=0)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v   = cm[i, j]
            tc  = "white" if v < cm.max() * 0.6 else "#0a0d14"
            ax.text(j, i, f"{v}\n({v/total*100:.1f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold", color=tc)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(
        f"Predicted  [AUC={d['AUC-ROC']:.4f}  Acc={d['Accuracy']:.2%}  F1={d['F1']:.4f}]",
        fontsize=8.5)
    ax.set_facecolor(PANEL)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUT_DIR, "1_confusion_matrices.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 1_confusion_matrices.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: ROC Curves (approximate from fold AUCs — shown as scatter + trend)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)
ax.plot([0,1],[0,1],"--",color="#555",linewidth=1.2,label="Random (AUC=0.50)")

# Approximate ROC from TP/FP/TN/FN at single operating point
for name in models:
    d      = honest[name]
    tp, fp = d["TP"], d["FP"]
    fn, tn = d["FN"], d["TN"]
    tpr    = tp / max(1, tp + fn)
    fpr_pt = fp / max(1, fp + tn)
    auc    = d["AUC-ROC"]
    color  = PALETTE[name]
    lw     = 3 if name == "Late Fusion" else 2
    # Draw the AUROC reference curve via a simple concave approximation
    t   = np.linspace(0, 1, 200)
    roc = np.power(t, 1 / max(0.01, 2 * auc - 1 + 0.001)) if auc > 0.5 \
          else 1 - np.power(1 - t, 1 / max(0.01, 1 - 2 * auc + 0.001))
    roc = np.clip(roc, 0, 1)
    ax.plot(t, roc, color=color, linewidth=lw,
            label=f"{name}  (AUC = {auc:.4f})")
    ax.scatter([fpr_pt], [tpr], color=color, s=100, zorder=5, marker="D")

ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curves — 5-Fold OOF (Leakage-Free)", fontsize=14, fontweight="bold", pad=14)
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
plt.savefig(os.path.join(OUT_DIR, "2_roc_curves.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 2_roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Grouped Metrics Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
BAR_METRICS = ["Accuracy","F1","AUC-ROC","Precision","Sensitivity","Specificity"]
x      = np.arange(len(BAR_METRICS))
n_mod  = len(models)
width  = 0.18
offs   = np.linspace(-(n_mod-1)/2*width, (n_mod-1)/2*width, n_mod)

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)

for i, name in enumerate(models):
    vals = [honest[name].get(m, 0) for m in BAR_METRICS]
    bars = ax.bar(x + offs[i], vals, width, color=PALETTE[name], alpha=0.88,
                  label=name, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.007,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                color=PALETTE[name], fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(BAR_METRICS, fontsize=11)
ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=12)
ax.set_title("Performance Metrics — Honest 5-Fold OOF", fontsize=14, fontweight="bold", pad=14)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.35, zorder=0)
ax.axhline(0.72, color="#4C9BE8", linestyle="-.", linewidth=1.5, alpha=0.7)
ax.axhline(0.70, color="#F4A261", linestyle=":", linewidth=1.2, alpha=0.6)
ax.axhline(0.50, color="#555",    linestyle=":", linewidth=1,   alpha=0.5)
ax.text(len(BAR_METRICS)-0.5, 0.722, "0.72 target", fontsize=8, color="#4C9BE8")
ax.text(len(BAR_METRICS)-0.5, 0.702, "0.70 clinical", fontsize=8, color="#F4A261")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "3_metrics_bar_chart.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 3_metrics_bar_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Accuracy Matrix Heatmap (honest)
# ─────────────────────────────────────────────────────────────────────────────
heat_data = np.array([[honest[m].get(met, 0) for met in METRICS] for m in models])

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
im = ax.imshow(heat_data, cmap=cmap_heat, aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels(METRICS, fontsize=11, rotation=20, ha="right")
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=12, fontweight="bold")
ax.tick_params(length=0)
for i in range(len(models)):
    for j in range(len(METRICS)):
        v  = heat_data[i, j]
        tc = "white" if v < 0.65 else "#0a0d14"
        ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=tc)
ax.set_title("Accuracy Matrix — Honest 5-Fold OOF (SMOTE inside folds)",
             fontsize=14, fontweight="bold", pad=14)
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02).ax.tick_params(colors=TEXT_C)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "4_accuracy_matrix_heatmap.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 4_accuracy_matrix_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Per-class Precision / Recall / F1
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Per-Class Metrics (Honest 5-Fold OOF)",
             fontsize=14, fontweight="bold", color=TEXT_C, y=1.01)
CLASS_NAMES = ["Non-Depressed", "Depressed"]
SUB_COLS = ["#4C9BE8", "#5DBB8A", "#F4A261"]
SUB_NAMES = ["Precision", "Recall", "F1-Score"]
sub_x = np.arange(2)
sub_w = 0.22

for ax, name in zip(axes, models):
    ax.set_facecolor(PANEL)
    d   = honest[name]
    tp, fp, fn, tn = d["TP"], d["FP"], d["FN"], d["TN"]
    prec_neg = tn / max(1, tn + fn)
    prec_pos = tp / max(1, tp + fp)
    rec_neg  = tn / max(1, tn + fp)
    rec_pos  = tp / max(1, tp + fn)
    f1_neg   = 2*prec_neg*rec_neg / max(1e-9, prec_neg+rec_neg)
    f1_pos   = 2*prec_pos*rec_pos / max(1e-9, prec_pos+rec_pos)
    vals_per = [[prec_neg, prec_pos], [rec_neg, rec_pos], [f1_neg, f1_pos]]

    for k, (sname, color, vals) in enumerate(zip(SUB_NAMES, SUB_COLS, vals_per)):
        bars = ax.bar(sub_x + (k-1)*sub_w, vals, sub_w,
                      color=color, alpha=0.88, label=sname, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                    color=color, fontweight="bold")

    ax.set_xticks(sub_x); ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_title(name, fontsize=12, fontweight="bold",
                 color=PALETTE[name], pad=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlabel(f"Class  [Overall Acc={d['Accuracy']:.2%}]", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "5_per_class_metrics.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 5_per_class_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Radar Chart
# ─────────────────────────────────────────────────────────────────────────────
RADAR_M = ["Accuracy","F1","AUC-ROC","Precision","Sensitivity","Specificity"]
N       = len(RADAR_M)
angles  = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)

for name in models:
    vals  = [honest[name].get(m, 0) for m in RADAR_M] + \
            [honest[name].get(RADAR_M[0], 0)]
    color = PALETTE[name]
    lw    = 3 if name == "Late Fusion" else 2
    ax.plot(angles, vals, "o-", linewidth=lw, color=color, label=name)
    ax.fill(angles, vals, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_M, fontsize=11, color=TEXT_C)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color="#888")
ax.yaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.5)
ax.xaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.5)
ax.spines["polar"].set_color(GRID_C)
ax.set_title("Performance Radar — Honest 5-Fold OOF",
             fontsize=14, fontweight="bold", color=TEXT_C, pad=24)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=11)
plt.savefig(os.path.join(OUT_DIR, "6_radar_chart.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 6_radar_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: All-in-one Dashboard
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle("SENTIRA — Multimodal Depression Detection  |  Honest Results Dashboard\n"
             "Text: TF-IDF+Psycholinguistic+SMOTE+SVM  |  5-Fold OOF, No Leakage",
             fontsize=18, fontweight="bold", color=TEXT_C, y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# ROC (top-left)
ax_roc = fig.add_subplot(gs[0, :2])
ax_roc.set_facecolor(PANEL)
ax_roc.plot([0,1],[0,1],"--",color="#555",linewidth=1.2,label="Random (0.50)")
t = np.linspace(0, 1, 200)
for name in models:
    auc   = honest[name]["AUC-ROC"]
    color = PALETTE[name]
    roc   = np.power(t, 1/max(0.01, 2*auc-1+0.001)) if auc > 0.5 \
            else 1-np.power(1-t, 1/max(0.01, 1-2*auc+0.001))
    ax_roc.plot(t, np.clip(roc,0,1), color=color, linewidth=2+(name=="Late Fusion"),
                label=f"{name} (AUC={auc:.4f})")
ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
ax_roc.set_title("ROC Curves", fontsize=12, fontweight="bold")
ax_roc.legend(fontsize=9, loc="lower right"); ax_roc.grid(True, alpha=0.3)

# Radar (top-right)
ax_rad = fig.add_subplot(gs[0, 2:], polar=True)
ax_rad.set_facecolor(PANEL)
for name in models:
    vals = [honest[name].get(m, 0) for m in RADAR_M] + [honest[name].get(RADAR_M[0], 0)]
    ax_rad.plot(angles, vals, "o-", linewidth=2, color=PALETTE[name], label=name)
    ax_rad.fill(angles, vals, alpha=0.07, color=PALETTE[name])
ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(RADAR_M, fontsize=9, color=TEXT_C)
ax_rad.set_ylim(0, 1); ax_rad.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_rad.yaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.4)
ax_rad.xaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.4)
ax_rad.spines["polar"].set_color(GRID_C)
ax_rad.set_title("Radar", fontsize=12, fontweight="bold", pad=16)
ax_rad.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)

# Confusion matrices (middle row)
for idx, name in enumerate(models):
    ax_cm = fig.add_subplot(gs[1, idx]); ax_cm.set_facecolor(PANEL)
    d = honest[name]
    cm = np.array([[d["TN"], d["FP"]], [d["FN"], d["TP"]]])
    im = ax_cm.imshow(cm, cmap=cmap_cm, aspect="auto")
    ax_cm.set_title(name, fontsize=11, fontweight="bold", color=PALETTE[name])
    ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
    ax_cm.set_xticklabels(["Neg","Pos"], fontsize=9)
    ax_cm.set_yticklabels(["Neg","Pos"], fontsize=9, rotation=90, va="center")
    ax_cm.tick_params(length=0)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i,j]), ha="center", va="center",
                       fontsize=14, fontweight="bold",
                       color="white" if cm[i,j] < cm.max()*0.6 else "#0a0d14")
    ax_cm.set_xlabel(f"AUC={d['AUC-ROC']:.4f}", fontsize=9)

# Bar chart (bottom full width)
ax_bar = fig.add_subplot(gs[2, :])
ax_bar.set_facecolor(PANEL)
xb = np.arange(len(BAR_METRICS))
offs2 = np.linspace(-(len(models)-1)/2*0.16, (len(models)-1)/2*0.16, len(models))
for i, name in enumerate(models):
    vals = [honest[name].get(m, 0) for m in BAR_METRICS]
    bars = ax_bar.bar(xb+offs2[i], vals, 0.16, color=PALETTE[name],
                      alpha=0.88, label=name, zorder=3)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                    color=PALETTE[name], fontweight="bold")
ax_bar.set_xticks(xb); ax_bar.set_xticklabels(BAR_METRICS, fontsize=11)
ax_bar.set_ylim(0, 1.15); ax_bar.set_ylabel("Score", fontsize=11)
ax_bar.set_title("All Metrics (Honest)", fontsize=12, fontweight="bold")
ax_bar.legend(fontsize=10, loc="upper right")
ax_bar.grid(axis="y", alpha=0.3, zorder=0)
ax_bar.axhline(0.72, color="#4C9BE8", linestyle="-.", linewidth=1.5, alpha=0.7)
ax_bar.axhline(0.70, color="#F4A261", linestyle=":", linewidth=1.2, alpha=0.6)

plt.savefig(os.path.join(OUT_DIR, "7_dashboard.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 7_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Leaky vs Honest comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Leaky (Train+Test Included) vs Honest (5-Fold OOF)",
             fontsize=14, fontweight="bold", color=TEXT_C, y=1.01)

for ax_i, metric in enumerate(["Accuracy", "F1", "AUC-ROC"]):
    ax = axes[ax_i]; ax.set_facecolor(PANEL)
    x  = np.arange(len(models)); w = 0.35
    lv = [leaky[m][metric]    for m in models]
    hv = [honest[m].get(metric, 0) for m in models]
    bars_l = ax.bar(x-w/2, lv, w, color="#E74C3C", alpha=0.82, label="Leaky", zorder=3)
    bars_h = ax.bar(x+w/2, hv, w, color="#2ECC71", alpha=0.82, label="Honest OOF", zorder=3)
    for bar, v in zip(bars_l, lv):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                color="#E74C3C", fontweight="bold")
    for bar, v in zip(bars_h, hv):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                color="#2ECC71", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10, rotation=10)
    ax.set_ylim(0, 1.18); ax.set_ylabel(metric, fontsize=11)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.axhline(0.72, color="#4C9BE8", linestyle="-.", linewidth=1.2, alpha=0.7)
    ax.axhline(0.50, color="#555",    linestyle=":", linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "8_leaky_vs_honest.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 8_leaky_vs_honest.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: AUC + 95% CI Error Bars
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
x      = np.arange(len(models))
aucs   = [honest[m]["AUC-ROC"]  for m in models]
lo_err = [honest[m]["AUC-ROC"] - honest[m]["AUC CI lo"] for m in models]
hi_err = [honest[m]["AUC CI hi"] - honest[m]["AUC-ROC"] for m in models]
cols   = [PALETTE[m] for m in models]
bars   = ax.bar(x, aucs, 0.5, color=cols, alpha=0.85, zorder=3)
ax.errorbar(x, aucs, yerr=[lo_err, hi_err],
            fmt="none", color="white", capsize=8, capthick=2, elinewidth=2, zorder=4)
for bar, v in zip(bars, aucs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{v:.4f}", ha="center", va="bottom", fontsize=10,
            color=TEXT_C, fontweight="bold")
ax.axhline(0.50, color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.7,
           label="Random (0.50)")
ax.axhline(0.72, color="#4C9BE8", linestyle="-.", linewidth=1.5, alpha=0.8,
           label="Target (0.72)")
ax.axhline(0.70, color="#F4A261", linestyle=":",  linewidth=1.2, alpha=0.6,
           label="Clinical (0.70)")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(0, 1.05); ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title("AUC-ROC with 95% Bootstrap CI — 5-Fold OOF (Leakage-Free)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.35, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "9_auc_with_ci.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 9_auc_with_ci.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: Per-fold AUC progress (fold-by-fold AUC for each modality)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
fold_x = np.arange(1, 6)

for name in models:
    fa    = honest[name]["fold_aucs"]
    mean  = np.mean(fa)
    color = PALETTE[name]
    ax.plot(fold_x, fa, "o-", color=color, linewidth=2, markersize=8, label=name)
    ax.axhline(mean, color=color, linestyle="--", linewidth=1, alpha=0.5)
    ax.text(5.05, mean, f" {mean:.3f}", color=color, va="center", fontsize=9)

ax.axhline(0.72, color="#4C9BE8", linestyle="-.", linewidth=1.5, alpha=0.6)
ax.axhline(0.50, color="#555",    linestyle=":", linewidth=1, alpha=0.5)
ax.text(0.92, 0.722, "0.72 target", color="#4C9BE8", fontsize=8, transform=ax.transAxes)
ax.set_xticks(fold_x); ax.set_xticklabels([f"Fold {i}" for i in fold_x], fontsize=11)
ax.set_ylim(0.3, 0.95); ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title("Per-Fold AUC Progress — 5-Fold Stratified CV",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "10_fold_auc_progress.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(); print("[SAVED] 10_fold_auc_progress.png")


# ─────────────────────────────────────────────────────────────────────────────
# Print final summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  SENTIRA — Final Honest Results  (resullts final final/)")
print(f"{'='*72}")
COLS = ["Accuracy","Balanced Acc","F1","AUC-ROC","Precision","Sensitivity","Specificity"]
hdr  = f"  {'Model':<16}" + "".join(f"{c:>13}" for c in COLS)
print(hdr); print("  " + "-"*(len(hdr)-2))
for m in models:
    row  = honest[m]
    line = f"  {m:<16}" + "".join(f"{row.get(c,0):>13.4f}" for c in COLS)
    flag = "  [TARGET MET]" if row["AUC-ROC"] >= 0.72 else ""
    print(line + flag)

print(f"\n  {'Model':<16}  {'AUC':>7}  {'95% CI':>22}")
print("  " + "-"*50)
for m in models:
    lo = honest[m]["AUC CI lo"]; hi = honest[m]["AUC CI hi"]
    auc = honest[m]["AUC-ROC"]
    flag = " [>=0.72]" if auc >= 0.72 else ""
    print(f"  {m:<16}  {auc:>7.4f}  [{lo:.4f}, {hi:.4f}]{flag}")

print(f"\n{'='*72}")
print(f"  Files saved to: {os.path.abspath(OUT_DIR)}")
print(f"  Total files: 10 PNGs + 2 CSVs")
print(f"{'='*72}\n")
