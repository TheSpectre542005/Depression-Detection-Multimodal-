"""
calculate_accuracies.py
=======================
Standalone accuracy calculator for the Depression Detection Multimodal project.
Loads trained .pkl models and feature CSVs, then computes a full metrics report
for every modality (Text, Audio, Visual, Early Fusion, Late Fusion).

Usage
-----
Run from the project root:
    python calculate_accuracies.py

Optional flags:
    --features_dir  Path to feature CSVs          (default: data/features)
    --models_dir    Path to .pkl model files       (default: models)
    --labels_csv    Path to PHQ-8 labels CSV       (default: data/features/labels.csv)
    --threshold     PHQ-8 binary threshold         (default: 10)
    --save          Save results table to CSV      (flag, off by default)
    --output        Output CSV path                (default: results/accuracies.csv)
    --plot          Show confusion matrices        (flag, off by default)

Requirements
------------
    pip install scikit-learn imbalanced-learn pandas numpy matplotlib joblib
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, label_binarize

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these names to match your actual file names if needed
# ─────────────────────────────────────────────────────────────────────────────

MODALITY_CONFIG = {
    "Text": {
        "features_file": "text_features.csv",
        "model_file":    "text_model.pkl",
    },
    "Audio": {
        "features_file": "audio_features_enhanced.csv" if os.path.exists("data/features/audio_features_enhanced.csv") else "audio_features.csv",
        "model_file":    "audio_model.pkl",
    },
    "Visual": {
        "features_file": "visual_features_enhanced.csv" if os.path.exists("data/features/visual_features_enhanced.csv") else "visual_features.csv",
        "model_file":    "visual_model.pkl",
    },
    "Early Fusion": {
        "features_file": "early_fusion_features.csv",
        "model_file":    "early_fusion_model.pkl",
    },
}

LATE_FUSION_CONFIG = {
    "model_files": ["text_model.pkl", "audio_model.pkl", "visual_model.pkl"],
    "feature_files": [
        "text_features.csv",
        "audio_features_enhanced.csv" if os.path.exists("data/features/audio_features_enhanced.csv") else "audio_features.csv",
        "visual_features_enhanced.csv" if os.path.exists("data/features/visual_features_enhanced.csv") else "visual_features.csv"
    ],
    # Weights from the project README (based on validation AUC)
    "weights": [0.35, 0.35, 0.30],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(labels_csv: str, threshold: int = 10) -> pd.Series:
    """Load PHQ-8 labels and binarise at `threshold`."""
    df = pd.read_csv(labels_csv)

    # Accept common column name variants
    for col in ["PHQ_Score", "phq_score", "PHQ8_Score", "phq8_score", "label", "Label"]:
        if col in df.columns:
            score_col = col
            break
    else:
        raise ValueError(
            f"Could not find a PHQ-8 score column in {labels_csv}.\n"
            f"Available columns: {list(df.columns)}"
        )

    for id_col in ["Participant_ID", "participant_id", "id", "ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col)
            break

    return (df[score_col] >= threshold).astype(int)


def load_features(features_path: str) -> pd.DataFrame:
    """Load a feature CSV. Assumes first column is participant ID."""
    df = pd.read_csv(features_path)

    for id_col in ["Participant_ID", "participant_id", "id", "ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col)
            break

    # Drop any leftover label columns that may have been saved alongside features
    for col in ["PHQ_Score", "phq_score", "label", "Label", "depressed", "Depressed"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


def align(features: pd.DataFrame, labels: pd.Series):
    """Return (X, y) aligned on their shared participant IDs."""
    common = features.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("No overlapping participant IDs between features and labels!")
    return features.loc[common].values, labels.loc[common].values


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Return a dict of all relevant classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall / true positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precision / positive predictive value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # negative predictive value

    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")

    return {
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "F1":          round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC":     round(auc, 4),
        "Precision":   round(ppv, 4),
        "Sensitivity": round(sensitivity, 4),   # recall / TPR
        "Specificity": round(specificity, 4),
        "NPV":         round(npv, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def print_table(results: dict):
    """Pretty-print results as a table."""
    main_cols = ["Accuracy", "F1", "AUC-ROC", "Precision", "Sensitivity", "Specificity", "NPV"]
    header = f"{'Model':<18}" + "".join(f"{c:>14}" for c in main_cols)
    sep = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)
    for model_name, metrics in results.items():
        row = f"{model_name:<18}" + "".join(
            f"{metrics.get(c, float('nan')):>14.4f}" for c in main_cols
        )
        print(row)
    print(sep + "\n")


def plot_confusion_matrices(results: dict, y_trues: dict, y_preds: dict):
    """Display confusion matrices for all models."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, _) in zip(axes, results.items()):
        cm = confusion_matrix(y_trues[name], y_preds[name])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Dep", "Dep"])
        ax.set_yticklabels(["Non-Dep", "Dep"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax)

    plt.suptitle("Confusion Matrices — Depression Detection Models", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_unimodal(name: str, cfg: dict, features_dir: str, models_dir: str,
                      labels: pd.Series):
    """Evaluate a single unimodal (or early-fusion) model."""
    feat_path  = os.path.join(features_dir, cfg["features_file"])
    model_path = os.path.join(models_dir,   cfg["model_file"])

    if not os.path.exists(feat_path):
        print(f"  [SKIP] {name}: features file not found → {feat_path}")
        return None, None, None

    if not os.path.exists(model_path):
        print(f"  [SKIP] {name}: model file not found → {model_path}")
        return None, None, None

    features = load_features(feat_path)
    X, y = align(features, labels)

    model_bundle = joblib.load(model_path)
    threshold = 0.5
    if isinstance(model_bundle, dict) and "pipeline" in model_bundle:
        model = model_bundle["pipeline"]
        threshold = model_bundle.get("threshold", 0.5)
    else:
        model = model_bundle

    # Handle dimension mismatch (e.g., if extraction updated but model was not retrained)
    if hasattr(model, "n_features_in_"):
        expected = model.n_features_in_
        if X.shape[1] > expected:
            X = X[:, :expected]
        elif X.shape[1] < expected:
            X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))
    elif hasattr(model, "named_steps") and "scaler" in model.named_steps:
        # If it's a pipeline, check the first step (usually scaler)
        expected = model.named_steps["scaler"].n_features_in_
        if X.shape[1] > expected:
            X = X[:, :expected]
        elif X.shape[1] < expected:
            X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))

    y_prob = (model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None)
    if y_prob is not None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X)

    return compute_metrics(y, y_pred, y_prob), y, y_pred


def evaluate_late_fusion(cfg: dict, features_dir: str, models_dir: str,
                         labels: pd.Series):
    """Weighted average late fusion across unimodal models."""
    loaded = []
    for feat_file, model_file in zip(cfg["feature_files"], cfg["model_files"]):
        feat_path  = os.path.join(features_dir, feat_file)
        model_path = os.path.join(models_dir,   model_file)

        if not os.path.exists(feat_path) or not os.path.exists(model_path):
            print(f"  [SKIP] Late Fusion: missing {feat_file} or {model_file}")
            return None, None, None

        features = load_features(feat_path)
        X, y = align(features, labels)
        model_bundle = joblib.load(model_path)
        if isinstance(model_bundle, dict) and "pipeline" in model_bundle:
            model = model_bundle["pipeline"]
        else:
            model = model_bundle
        if hasattr(model, "n_features_in_"):
            expected = model.n_features_in_
            if X.shape[1] > expected: X = X[:, :expected]
            elif X.shape[1] < expected: X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))
        elif hasattr(model, "named_steps") and "scaler" in model.named_steps:
            expected = model.named_steps["scaler"].n_features_in_
            if X.shape[1] > expected: X = X[:, :expected]
            elif X.shape[1] < expected: X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))
        
        loaded.append((X, y, model))

    # Use only the common participant subset
    y_ref = loaded[0][1]
    proba_stack = []
    weights = cfg["weights"]

    for (X, y, model), w in zip(loaded, weights):
        if not np.array_equal(y, y_ref):
            print("  [WARN] Late Fusion: participant ordering differs between modalities — skipping.")
            return None, None, None
        if hasattr(model, "predict_proba"):
            proba_stack.append(model.predict_proba(X)[:, 1] * w)
        else:
            proba_stack.append(model.predict(X).astype(float) * w)

    fused_prob = np.sum(proba_stack, axis=0) / sum(weights)
    y_pred = (fused_prob >= 0.5).astype(int)

    return compute_metrics(y_ref, y_pred, fused_prob), y_ref, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calculate accuracy metrics for all Depression Detection models."
    )
    parser.add_argument("--features_dir", default="data/features",
                        help="Directory containing feature CSV files")
    parser.add_argument("--models_dir",   default="models",
                        help="Directory containing .pkl model files")
    parser.add_argument("--labels_csv",   default=None,
                        help="Path to PHQ-8 labels CSV (auto-detected if omitted)")
    parser.add_argument("--threshold",    type=int, default=10,
                        help="PHQ-8 binary threshold (default 10)")
    parser.add_argument("--save",         action="store_true",
                        help="Save results to CSV")
    parser.add_argument("--output",       default="results/accuracies.csv",
                        help="Output CSV path (used with --save)")
    parser.add_argument("--plot",         action="store_true",
                        help="Show confusion matrix plots")
    args = parser.parse_args()

    # ── Locate labels CSV ──────────────────────────────────────────────────
    if args.labels_csv:
        labels_csv = args.labels_csv
    else:
        candidates = [
            os.path.join(args.features_dir, "labels.csv"),
            os.path.join(args.features_dir, "phq_labels.csv"),
            "data/labels.csv",
            "labels.csv",
        ]
        labels_csv = next((p for p in candidates if os.path.exists(p)), None)
        if labels_csv is None:
            print(
                "\n[ERROR] Could not find a labels CSV. "
                "Pass --labels_csv <path> explicitly.\n"
            )
            return

    print(f"\n{'='*60}")
    print(f"  Depression Detection — Model Accuracy Calculator")
    print(f"{'='*60}")
    print(f"  Features dir : {args.features_dir}")
    print(f"  Models dir   : {args.models_dir}")
    print(f"  Labels CSV   : {labels_csv}")
    print(f"  PHQ threshold: {args.threshold}")
    print(f"{'='*60}\n")

    labels = load_labels(labels_csv, threshold=args.threshold)
    print(f"  Loaded {len(labels)} participants  |  "
          f"Depressed: {labels.sum()}  |  Non-depressed: {(labels == 0).sum()}\n")

    results  = {}
    y_trues  = {}
    y_preds  = {}

    # ── Unimodal + Early Fusion ────────────────────────────────────────────
    for name, cfg in MODALITY_CONFIG.items():
        metrics, y_true, y_pred = evaluate_unimodal(
            name, cfg, args.features_dir, args.models_dir, labels
        )
        if metrics:
            results[name] = metrics
            y_trues[name] = y_true
            y_preds[name] = y_pred
            print(f"  ✓ {name}")

    # ── Late Fusion ────────────────────────────────────────────────────────
    metrics, y_true, y_pred = evaluate_late_fusion(
        LATE_FUSION_CONFIG, args.features_dir, args.models_dir, labels
    )
    if metrics:
        results["Late Fusion"] = metrics
        y_trues["Late Fusion"] = y_true
        y_preds["Late Fusion"] = y_pred
        print("  ✓ Late Fusion")

    if not results:
        print("\n[ERROR] No models could be evaluated. "
              "Check that --features_dir and --models_dir point to the right paths.\n")
        return

    # ── Print table ────────────────────────────────────────────────────────
    print_table(results)

    # ── Per-model classification reports ──────────────────────────────────
    for name in results:
        print(f"  ── {name} ──")
        print(classification_report(
            y_trues[name], y_preds[name],
            target_names=["Non-Depressed", "Depressed"],
            zero_division=0,
        ))

    # ── Save ───────────────────────────────────────────────────────────────
    if args.save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df_out = pd.DataFrame(results).T
        df_out.index.name = "Model"
        df_out.to_csv(args.output)
        print(f"  Results saved → {args.output}\n")

    # ── Plots ──────────────────────────────────────────────────────────────
    if args.plot:
        plot_confusion_matrices(results, y_trues, y_preds)


if __name__ == "__main__":
    main()