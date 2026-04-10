"""
generate_results.py
===================
Generates a full results report — confusion matrices, ROC curves,
metric bar charts, and a summary heatmap — saved to 'results final/'.

Usage:
    python generate_results.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve, balanced_accuracy_score,
)

warnings.filterwarnings("ignore")

OUT_DIR      = "results final"
FEATURES_DIR = "data/features"
MODELS_DIR   = "models"
LABELS_CSV   = os.path.join(FEATURES_DIR, "master_labels.csv")
PHQ_THRESH   = 10

os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "Text":        "#4C9BE8",
    "Audio":       "#5DBB8A",
    "Visual":      "#F4A261",
    "Late Fusion": "#9B59B6",
}
BG      = "#0F1117"
PANEL   = "#1A1D27"
TEXT_C  = "#E0E0E0"
GRID_C  = "#2A2D3A"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID_C,
    "axes.labelcolor":   TEXT_C,
    "axes.titlecolor":   TEXT_C,
    "xtick.color":       TEXT_C,
    "ytick.color":       TEXT_C,
    "text.color":        TEXT_C,
    "grid.color":        GRID_C,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  GRID_C,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(path, threshold=10):
    df = pd.read_csv(path)
    for col in ["PHQ_Score", "phq_score", "PHQ8_Score", "phq8_score", "label", "Label"]:
        if col in df.columns:
            score_col = col; break
    else:
        raise ValueError(f"No PHQ column in {path}")
    for id_col in ["pid", "Participant_ID", "participant_id", "id", "ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col); break
    if df[score_col].max() > 1:
        return (df[score_col] >= threshold).astype(int)
    return df[score_col].astype(int)


def load_features(path):
    df = pd.read_csv(path)
    for id_col in ["pid", "Participant_ID", "participant_id", "id", "ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col); break
    for col in ["PHQ_Score", "phq_score", "label", "Label", "depressed", "Depressed"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


def align(features, labels):
    common = features.index.intersection(labels.index)
    return features.loc[common].values, labels.loc[common].values


def get_Xy(feat_file, labels):
    path = os.path.join(FEATURES_DIR, feat_file)
    if not os.path.exists(path):
        return None, None
    return align(load_features(path), labels)


def predict(model_path, X, threshold=0.5):
    if not os.path.exists(model_path):
        return None, None
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "pipeline" in bundle:
        model = bundle["pipeline"]
        thr   = bundle.get("threshold", threshold)
    else:
        model, thr = bundle, threshold

    expected = None
    if hasattr(model, "n_features_in_"):
        expected = model.n_features_in_
    elif hasattr(model, "named_steps") and "scaler" in model.named_steps:
        expected = model.named_steps["scaler"].n_features_in_

    if expected is not None and X.shape[1] != expected:
        if X.shape[1] > expected:
            X = X[:, :expected]
        else:
            X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))

    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = (y_prob >= thr).astype(int) if y_prob is not None else model.predict(X)
    return y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# Collect predictions
# ─────────────────────────────────────────────────────────────────────────────

print("[*] Loading data and running predictions...")
labels = load_labels(LABELS_CSV, PHQ_THRESH)
print(f"    Participants: {len(labels)}  |  Depressed: {labels.sum()}  |  Non-dep: {(labels==0).sum()}")

AUDIO_FEAT = ("audio_features_enhanced.csv"
              if os.path.exists(os.path.join(FEATURES_DIR, "audio_features_enhanced.csv"))
              else "audio_features.csv")
VISUAL_FEAT = ("visual_features_enhanced.csv"
               if os.path.exists(os.path.join(FEATURES_DIR, "visual_features_enhanced.csv"))
               else "visual_features.csv")

CONFIGS = {
    "Text":   ("text_features.csv",  "text_model.pkl"),
    "Audio":  (AUDIO_FEAT,           "audio_model.pkl"),
    "Visual": (VISUAL_FEAT,          "visual_model.pkl"),
}

results   = {}
y_trues   = {}
y_preds   = {}
y_probs   = {}

for name, (feat_file, model_file) in CONFIGS.items():
    X, y = get_Xy(feat_file, labels)
    if X is None:
        print(f"    [SKIP] {name} — features missing")
        continue
    y_pred, y_prob = predict(os.path.join(MODELS_DIR, model_file), X)
    if y_pred is None:
        print(f"    [SKIP] {name} — model missing")
        continue
    y_trues[name] = y
    y_preds[name] = y_pred
    y_probs[name] = y_prob
    results[name] = {
        "Accuracy":        round(accuracy_score(y, y_pred), 4),
        "Balanced Acc":    round(balanced_accuracy_score(y, y_pred), 4),
        "F1":              round(f1_score(y, y_pred, zero_division=0), 4),
        "AUC-ROC":         round(roc_auc_score(y, y_prob) if y_prob is not None else float("nan"), 4),
        "Precision":       round(precision_score(y, y_pred, zero_division=0), 4),
        "Sensitivity":     round(recall_score(y, y_pred, zero_division=0), 4),
        "Specificity":     round(confusion_matrix(y, y_pred, labels=[0,1]).ravel()[0] /
                                  max(1, (confusion_matrix(y, y_pred, labels=[0,1]).ravel()[0] +
                                          confusion_matrix(y, y_pred, labels=[0,1]).ravel()[1])), 4),
    }
    print(f"    [OK] {name}")

# Late Fusion
lf_configs = [
    ("text_features.csv",  "text_model.pkl",   0.35),
    (AUDIO_FEAT,           "audio_model.pkl",  0.35),
    (VISUAL_FEAT,          "visual_model.pkl", 0.30),
]
lf_probs = []
lf_y     = None
for feat_file, model_file, weight in lf_configs:
    X, y = get_Xy(feat_file, labels)
    if X is None:
        break
    _, y_prob = predict(os.path.join(MODELS_DIR, model_file), X)
    if y_prob is None:
        break
    lf_probs.append(y_prob * weight)
    lf_y = y

if len(lf_probs) == 3:
    fused      = np.sum(lf_probs, axis=0) / 1.0   # weights already applied
    lf_pred    = (fused >= 0.5).astype(int)
    y_trues["Late Fusion"] = lf_y
    y_preds["Late Fusion"] = lf_pred
    y_probs["Late Fusion"] = fused
    tn, fp, fn, tp = confusion_matrix(lf_y, lf_pred, labels=[0, 1]).ravel()
    results["Late Fusion"] = {
        "Accuracy":        round(accuracy_score(lf_y, lf_pred), 4),
        "Balanced Acc":    round(balanced_accuracy_score(lf_y, lf_pred), 4),
        "F1":              round(f1_score(lf_y, lf_pred, zero_division=0), 4),
        "AUC-ROC":         round(roc_auc_score(lf_y, fused), 4),
        "Precision":       round(precision_score(lf_y, lf_pred, zero_division=0), 4),
        "Sensitivity":     round(recall_score(lf_y, lf_pred, zero_division=0), 4),
        "Specificity":     round(tn / max(1, tn + fp), 4),
    }
    print("    [OK] Late Fusion")

df_results = pd.DataFrame(results).T
print(f"\n{df_results.to_string()}\n")

# Save CSV
csv_path = os.path.join(OUT_DIR, "accuracy_matrix.csv")
df_results.to_csv(csv_path)
print(f"[SAVED] {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Confusion Matrices (2x2 grid)
# ─────────────────────────────────────────────────────────────────────────────
print("[*] Plotting confusion matrices...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor(BG)
fig.suptitle("Confusion Matrices — SENTIRA Depression Detection",
             fontsize=16, fontweight="bold", color=TEXT_C, y=0.98)

cm_colors = [(0.1, 0.13, 0.18), (0.3, 0.61, 0.91)]
cmap_custom = LinearSegmentedColormap.from_list("custom_blue", cm_colors)

model_order = [m for m in ["Text", "Audio", "Visual", "Late Fusion"] if m in y_preds]
for ax, name in zip(axes.flatten(), model_order):
    cm = confusion_matrix(y_trues[name], y_preds[name], labels=[0, 1])
    im = ax.imshow(cm, cmap=cmap_custom, aspect="auto")
    ax.set_title(f"{name}", fontsize=13, fontweight="bold",
                 color=PALETTE.get(name, TEXT_C), pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Depressed", "Depressed"], fontsize=9)
    ax.set_yticklabels(["Non-Depressed", "Depressed"], fontsize=9, rotation=90, va="center")
    ax.tick_params(length=0)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val   = cm[i, j]
            pct   = val / max(1, total) * 100
            color = "white" if val < cm.max() * 0.6 else "#0a0d14"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    ax.set_facecolor(PANEL)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate with key metrics
    acc = results[name]["Accuracy"]
    auc = results[name]["AUC-ROC"]
    ax.set_xlabel(f"Predicted Label       [Acc={acc:.2%}  AUC={auc:.4f}]", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path1 = os.path.join(OUT_DIR, "1_confusion_matrices.png")
plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path1}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
print("[*] Plotting ROC curves...")

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

# random chance line
ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1.2, label="Random Chance (AUC=0.50)")

for name in model_order:
    if y_probs[name] is None:
        continue
    fpr, tpr, _ = roc_curve(y_trues[name], y_probs[name])
    auc          = results[name]["AUC-ROC"]
    color        = PALETTE.get(name, TEXT_C)
    lw           = 3 if name == "Late Fusion" else 2
    ls           = "-"
    ax.plot(fpr, tpr, color=color, linewidth=lw, linestyle=ls,
            label=f"{name}  (AUC = {auc:.4f})")

ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold", pad=14)
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.tick_params(colors=TEXT_C)

# Shade under Late Fusion curve
if "Late Fusion" in y_probs and y_probs["Late Fusion"] is not None:
    fpr_lf, tpr_lf, _ = roc_curve(y_trues["Late Fusion"], y_probs["Late Fusion"])
    ax.fill_between(fpr_lf, tpr_lf, alpha=0.08, color=PALETTE["Late Fusion"])

path2 = os.path.join(OUT_DIR, "2_roc_curves.png")
plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Metrics Bar Chart (grouped)
# ─────────────────────────────────────────────────────────────────────────────
print("[*] Plotting metrics bar chart...")

METRICS = ["Accuracy", "F1", "AUC-ROC", "Precision", "Sensitivity", "Specificity"]
x       = np.arange(len(METRICS))
n_mod   = len(model_order)
width   = 0.18
offsets = np.linspace(-(n_mod - 1) / 2 * width, (n_mod - 1) / 2 * width, n_mod)

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

for i, name in enumerate(model_order):
    vals  = [results[name].get(m, 0) for m in METRICS]
    bars  = ax.bar(x + offsets[i], vals, width=width, color=PALETTE.get(name, TEXT_C),
                   alpha=0.88, label=name, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                color=PALETTE.get(name, TEXT_C), fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=11)
ax.set_ylim(0, 1.13)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Performance Metrics — All Models", fontsize=14, fontweight="bold", pad=14)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.35, zorder=0)
ax.axhline(0.7,  color="#888", linestyle=":", linewidth=1, alpha=0.6)
ax.axhline(0.9,  color="#888", linestyle=":", linewidth=1, alpha=0.6)
ax.text(len(METRICS) - 0.5, 0.702,  "0.70 clinical threshold", fontsize=8, color="#888")
ax.text(len(METRICS) - 0.5, 0.902,  "0.90 target",             fontsize=8, color="#888")

path3 = os.path.join(OUT_DIR, "3_metrics_bar_chart.png")
plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path3}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Accuracy Matrix Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("[*] Plotting metrics heatmap...")

heat_metrics = ["Accuracy", "Balanced Acc", "F1", "AUC-ROC",
                "Precision", "Sensitivity", "Specificity"]
heat_data = np.array([[results[m].get(met, 0) for met in heat_metrics]
                       for m in model_order])

cmap_heat = LinearSegmentedColormap.from_list("heat",
    [(0.1, 0.13, 0.18), (0.18, 0.4, 0.7), (0.3, 0.76, 0.56), (0.98, 0.85, 0.37)])

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

im = ax.imshow(heat_data, cmap=cmap_heat, aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(heat_metrics)))
ax.set_xticklabels(heat_metrics, fontsize=11, rotation=20, ha="right")
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels(model_order, fontsize=12, fontweight="bold")
ax.tick_params(length=0)

for i in range(len(model_order)):
    for j in range(len(heat_metrics)):
        v = heat_data[i, j]
        text_color = "white" if v < 0.65 else "#0a0d14"
        ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=text_color)

ax.set_title("Accuracy Matrix — SENTIRA Depression Detection Models",
             fontsize=14, fontweight="bold", pad=14)

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.ax.tick_params(colors=TEXT_C)
cbar.set_label("Score", color=TEXT_C, fontsize=10)

plt.tight_layout()
path4 = os.path.join(OUT_DIR, "4_accuracy_matrix_heatmap.png")
plt.savefig(path4, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path4}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Per-Model Classification Reports
# ─────────────────────────────────────────────────────────────────────────────
print("[*] Plotting per-class precision/recall/F1...")

fig, axes = plt.subplots(1, len(model_order), figsize=(5 * len(model_order), 6),
                         sharey=False)
if len(model_order) == 1:
    axes = [axes]
fig.patch.set_facecolor(BG)
fig.suptitle("Per-Class Metrics (Precision / Recall / F1)",
             fontsize=14, fontweight="bold", color=TEXT_C, y=1.01)

CLASS_NAMES = ["Non-Depressed", "Depressed"]
SUBMETRICS  = ["Precision", "Recall", "F1-Score"]
SUB_COLORS  = ["#4C9BE8", "#5DBB8A", "#F4A261"]
sub_x       = np.arange(len(CLASS_NAMES))
sub_w       = 0.22

for ax, name in zip(axes, model_order):
    ax.set_facecolor(PANEL)
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_trues[name], y_preds[name], labels=[0, 1], zero_division=0)
    vals_per_metric = [prec, rec, f1]
    for k, (metric_name, color, vals) in enumerate(
            zip(SUBMETRICS, SUB_COLORS, vals_per_metric)):
        bars = ax.bar(sub_x + (k - 1) * sub_w, vals, width=sub_w,
                      color=color, alpha=0.88, label=metric_name, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                    color=color, fontweight="bold")
    ax.set_xticks(sub_x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_title(name, fontsize=12, fontweight="bold",
                 color=PALETTE.get(name, TEXT_C), pad=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.legend(fontsize=9, loc="upper right")
    acc_val = results[name]["Accuracy"]
    ax.set_xlabel(f"Class   [Overall Accuracy = {acc_val:.2%}]", fontsize=9)
    ax.tick_params(colors=TEXT_C)

plt.tight_layout()
path5 = os.path.join(OUT_DIR, "5_per_class_metrics.png")
plt.savefig(path5, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path5}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: AUC Comparison Radar / Polar chart
# ─────────────────────────────────────────────────────────────────────────────
print("[*] Plotting radar chart...")

RADAR_METRICS = ["Accuracy", "F1", "AUC-ROC", "Precision", "Sensitivity", "Specificity"]
N = len(RADAR_METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

for name in model_order:
    vals = [results[name].get(m, 0) for m in RADAR_METRICS]
    vals += vals[:1]
    color = PALETTE.get(name, TEXT_C)
    lw    = 3 if name == "Late Fusion" else 2
    ax.plot(angles, vals, "o-", linewidth=lw, color=color, label=name)
    ax.fill(angles, vals, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_METRICS, fontsize=11, color=TEXT_C)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#888")
ax.yaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.5)
ax.xaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.5)
ax.spines["polar"].set_color(GRID_C)
ax.set_title("Performance Radar — All Models",
             fontsize=14, fontweight="bold", color=TEXT_C, pad=24)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=11)

path6 = os.path.join(OUT_DIR, "6_radar_chart.png")
plt.savefig(path6, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path6}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Dashboard — all-in-one
# ─────────────────────────────────────────────────────────────────────────────
print("[*] Building dashboard figure...")

fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle("SENTIRA — Multimodal Depression Detection  |  Results Dashboard",
             fontsize=20, fontweight="bold", color=TEXT_C, y=0.99)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# -- ROC (top-left span 2 cols)
ax_roc = fig.add_subplot(gs[0, :2])
ax_roc.set_facecolor(PANEL)
ax_roc.plot([0,1],[0,1],"--",color="#555",linewidth=1.2,label="Random (AUC=0.50)")
for name in model_order:
    if y_probs.get(name) is None: continue
    fpr, tpr, _ = roc_curve(y_trues[name], y_probs[name])
    ax_roc.plot(fpr, tpr, color=PALETTE.get(name,TEXT_C),
                linewidth=2 + (name=="Late Fusion"),
                label=f"{name} (AUC={results[name]['AUC-ROC']:.4f})")
ax_roc.set_xlabel("FPR", fontsize=10); ax_roc.set_ylabel("TPR", fontsize=10)
ax_roc.set_title("ROC Curves", fontsize=12, fontweight="bold")
ax_roc.legend(fontsize=9, loc="lower right")
ax_roc.grid(True, alpha=0.3)

# -- Radar (top-right span 2 cols)
ax_rad = fig.add_subplot(gs[0, 2:], polar=True)
ax_rad.set_facecolor(PANEL)
for name in model_order:
    vals   = [results[name].get(m, 0) for m in RADAR_METRICS] + [results[name].get(RADAR_METRICS[0], 0)]
    angles_r = angles
    ax_rad.plot(angles_r, vals, "o-", linewidth=2, color=PALETTE.get(name,TEXT_C), label=name)
    ax_rad.fill(angles_r, vals, alpha=0.07, color=PALETTE.get(name,TEXT_C))
ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(RADAR_METRICS, fontsize=9, color=TEXT_C)
ax_rad.set_ylim(0,1)
ax_rad.set_yticks([0.25,0.5,0.75,1.0])
ax_rad.yaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.4)
ax_rad.xaxis.grid(True, color=GRID_C, linestyle="--", alpha=0.4)
ax_rad.spines["polar"].set_color(GRID_C)
ax_rad.set_title("Radar Chart", fontsize=12, fontweight="bold", pad=16)
ax_rad.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)

# -- Confusion matrices (middle row)
for idx, name in enumerate(model_order):
    ax_cm = fig.add_subplot(gs[1, idx])
    ax_cm.set_facecolor(PANEL)
    cm = confusion_matrix(y_trues[name], y_preds[name], labels=[0,1])
    im = ax_cm.imshow(cm, cmap=cmap_custom, aspect="auto")
    ax_cm.set_title(name, fontsize=11, fontweight="bold", color=PALETTE.get(name,TEXT_C))
    ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
    ax_cm.set_xticklabels(["Neg","Pos"], fontsize=9)
    ax_cm.set_yticklabels(["Neg","Pos"], fontsize=9, rotation=90, va="center")
    ax_cm.tick_params(length=0)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i,j]), ha="center", va="center",
                       fontsize=14, fontweight="bold",
                       color="white" if cm[i,j] < cm.max()*0.6 else "#0a0d14")

# -- Bar chart (bottom, full width)
ax_bar = fig.add_subplot(gs[2, :])
ax_bar.set_facecolor(PANEL)
METRICS_B = ["Accuracy", "F1", "AUC-ROC", "Precision", "Sensitivity", "Specificity"]
xb = np.arange(len(METRICS_B))
offs = np.linspace(-(len(model_order)-1)/2*0.16, (len(model_order)-1)/2*0.16, len(model_order))
for i, name in enumerate(model_order):
    vals = [results[name].get(m, 0) for m in METRICS_B]
    bars = ax_bar.bar(xb + offs[i], vals, 0.16, color=PALETTE.get(name,TEXT_C),
                      alpha=0.88, label=name, zorder=3)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                    color=PALETTE.get(name,TEXT_C), fontweight="bold")
ax_bar.set_xticks(xb); ax_bar.set_xticklabels(METRICS_B, fontsize=11)
ax_bar.set_ylim(0, 1.15)
ax_bar.set_ylabel("Score", fontsize=11)
ax_bar.set_title("All Metrics Comparison", fontsize=12, fontweight="bold")
ax_bar.legend(fontsize=10, loc="upper right")
ax_bar.grid(axis="y", alpha=0.3, zorder=0)
ax_bar.axhline(0.7, color="#888", linestyle=":", linewidth=1, alpha=0.6)
ax_bar.axhline(0.9, color="#888", linestyle=":", linewidth=1, alpha=0.6)

path7 = os.path.join(OUT_DIR, "7_dashboard.png")
plt.savefig(path7, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] {path7}")


# ─────────────────────────────────────────────────────────────────────────────
# Print text summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SENTIRA — Final Results Summary")
print("="*70)
cols = ["Accuracy", "F1", "AUC-ROC", "Precision", "Sensitivity", "Specificity"]
header = f"  {'Model':<16}" + "".join(f"{c:>13}" for c in cols)
print(header)
print("  " + "-" * (len(header)-2))
for name, row in results.items():
    line = f"  {name:<16}" + "".join(f"{row.get(c, float('nan')):>13.4f}" for c in cols)
    print(line)
print("="*70)
print(f"\n[DONE] All files saved to: {os.path.abspath(OUT_DIR)}")
print("Files generated:")
for i, fname in enumerate([
    "accuracy_matrix.csv",
    "1_confusion_matrices.png",
    "2_roc_curves.png",
    "3_metrics_bar_chart.png",
    "4_accuracy_matrix_heatmap.png",
    "5_per_class_metrics.png",
    "6_radar_chart.png",
    "7_dashboard.png",
], 1):
    print(f"  {fname}")
