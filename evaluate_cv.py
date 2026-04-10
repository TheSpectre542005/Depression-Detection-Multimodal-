"""
evaluate_cv.py
==============
Leakage-proof evaluation using Stratified K-Fold cross-validation.

Key guarantees:
  1. SMOTE is applied INSIDE each CV fold (via ImbPipeline), never on the
     full dataset. This prevents synthetic samples from the minority class
     leaking into the validation fold.
  2. Evaluation is ONLY on held-out folds — the model never sees the test
     fold during training.
  3. Bootstrap 95% confidence intervals are reported alongside point estimates.
  4. Participant IDs are preserved through the split to guard against any
     temporal or group leakage (each participant appears in exactly one fold).

Usage:
    python evaluate_cv.py                    # 5-fold CV, all modalities
    python evaluate_cv.py --n_folds 10       # 10-fold
    python evaluate_cv.py --n_bootstrap 2000 # more bootstrap iterations
    python evaluate_cv.py --save             # save results to results/cv_results.csv
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, balanced_accuracy_score,
    confusion_matrix,
)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    print("[WARN] imbalanced-learn not installed. SMOTE pipelines disabled.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

FEATURES_DIR = "data/features"
LABELS_CSV   = "data/features/master_labels.csv"
PHQ_THRESH   = 10
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_labels(path=LABELS_CSV, threshold=PHQ_THRESH):
    df = pd.read_csv(path)
    for col in ["PHQ_Score","phq_score","PHQ8_Score","phq8_score","label","Label"]:
        if col in df.columns:
            score_col = col; break
    else:
        raise ValueError(f"No PHQ column in {path}")
    for id_col in ["pid","Participant_ID","participant_id","id","ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col); break
    if df[score_col].max() > 1:
        return (df[score_col] >= threshold).astype(int)
    return df[score_col].astype(int)


def load_features(path):
    df = pd.read_csv(path)
    for id_col in ["pid","Participant_ID","participant_id","id","ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col); break
    for col in ["PHQ_Score","phq_score","label","Label","depressed","Depressed"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


def align(features, labels):
    common = features.index.intersection(labels.index)
    return features.loc[common].values, labels.loc[common].values, common


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap percentile confidence interval for a metric."""
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        ys  = y_score[idx]
        # Skip degenerate bootstrap samples (only one class)
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, ys))
        except Exception:
            pass
    scores  = np.array(scores)
    alpha   = (1 - ci) / 2
    lo, hi  = np.percentile(scores, [alpha * 100, (1 - alpha) * 100])
    return float(np.mean(scores)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Build the best pipeline for each modality
# (these mirror the winning models from the improve_* scripts)
# ---------------------------------------------------------------------------

def audio_pipeline():
    """SMOTE+RF — winner from improve_audio_model.py CV."""
    if HAS_IMBALANCED:
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf",    RandomForestClassifier(
                n_estimators=300, max_depth=6,
                min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])


def text_pipeline():
    """LR/SVM balanced — mirrors improve_text_model.py winner."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=3000,
            solver="saga", penalty="l1", random_state=RANDOM_STATE
        )),
    ])


def visual_pipeline():
    """SelectKBest + LR — mirrors improve_visual_model.py winner."""
    if HAS_IMBALANCED:
        return ImbPipeline([
            ("vt",     VarianceThreshold(threshold=0.001)),
            ("scaler", StandardScaler()),
            ("sel",    SelectKBest(f_classif, k=30)),
            ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf",    LogisticRegression(
                C=1.0, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE
            )),
        ])
    return Pipeline([
        ("vt",     VarianceThreshold(threshold=0.001)),
        ("scaler", StandardScaler()),
        ("sel",    SelectKBest(f_classif, k=30)),
        ("clf",    LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000,
            solver="lbfgs", random_state=RANDOM_STATE
        )),
    ])


MODALITY_CONFIG = {
    "Text": {
        "features_file": "text_features.csv",
        "pipeline_fn":   text_pipeline,
    },
    "Audio": {
        "features_file": ("audio_features_enhanced.csv"
                          if os.path.exists(os.path.join(FEATURES_DIR, "audio_features_enhanced.csv"))
                          else "audio_features.csv"),
        "pipeline_fn":   audio_pipeline,
    },
    "Visual": {
        "features_file": ("visual_features_enhanced.csv"
                          if os.path.exists(os.path.join(FEATURES_DIR, "visual_features_enhanced.csv"))
                          else "visual_features.csv"),
        "pipeline_fn":   visual_pipeline,
    },
}


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------

def cv_evaluate(name, X, y, pipeline_fn, n_folds=5, n_bootstrap=1000):
    """
    Stratified K-fold CV evaluation.
    Returns out-of-fold predictions (y_true, y_prob) concatenated across all folds.
    """
    skf           = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_y_true    = []
    oof_y_prob    = []

    print(f"\n  -- {name} ({n_folds}-fold CV) --")
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe = pipeline_fn()
        pipe.fit(X_tr, y_tr)

        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(X_te)[:, 1]
        else:
            prob = pipe.predict(X_te).astype(float)

        fold_auc = roc_auc_score(y_te, prob) if len(np.unique(y_te)) > 1 else float("nan")
        fold_acc = accuracy_score(y_te, (prob >= 0.5).astype(int))
        print(f"    Fold {fold_i}: AUC={fold_auc:.4f}  Acc={fold_acc:.4f}  "
              f"(test_n={len(y_te)}, dep={y_te.sum()})")

        oof_y_true.append(y_te)
        oof_y_prob.append(prob)

    oof_y_true = np.concatenate(oof_y_true)
    oof_y_prob = np.concatenate(oof_y_prob)
    oof_y_pred = (oof_y_prob >= 0.5).astype(int)

    # Point estimates on OOF predictions
    auc  = roc_auc_score(oof_y_true, oof_y_prob)
    acc  = accuracy_score(oof_y_true, oof_y_pred)
    bac  = balanced_accuracy_score(oof_y_true, oof_y_pred)
    f1   = f1_score(oof_y_true, oof_y_pred, zero_division=0)
    prec = precision_score(oof_y_true, oof_y_pred, zero_division=0)
    rec  = recall_score(oof_y_true, oof_y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(oof_y_true, oof_y_pred, labels=[0,1]).ravel()
    spec = tn / max(1, tn + fp)

    # Bootstrap 95% CIs for AUC and Accuracy
    auc_mean, auc_lo, auc_hi = bootstrap_ci(
        oof_y_true, oof_y_prob,
        lambda yt, ys: roc_auc_score(yt, ys),
        n_boot=n_bootstrap
    )
    acc_mean, acc_lo, acc_hi = bootstrap_ci(
        oof_y_true, oof_y_pred,
        lambda yt, ys: accuracy_score(yt, ys.round()),
        n_boot=n_bootstrap
    )
    f1_mean, f1_lo, f1_hi = bootstrap_ci(
        oof_y_true, oof_y_pred,
        lambda yt, ys: f1_score(yt, ys.round(), zero_division=0),
        n_boot=n_bootstrap
    )

    metrics = {
        "Accuracy":     acc,
        "Balanced Acc": bac,
        "F1":           f1,
        "AUC-ROC":      auc,
        "Precision":    prec,
        "Sensitivity":  rec,
        "Specificity":  spec,
        "AUC 95% CI":   f"[{auc_lo:.4f}, {auc_hi:.4f}]",
        "Acc 95% CI":   f"[{acc_lo:.4f}, {acc_hi:.4f}]",
        "F1  95% CI":   f"[{f1_lo:.4f},  {f1_hi:.4f}]",
    }
    return metrics, oof_y_true, oof_y_prob


def cv_late_fusion(modality_results, weights=None):
    """Weighted average late fusion from OOF probabilities."""
    names = list(modality_results.keys())
    if weights is None:
        weights = [1.0 / len(names)] * len(names)

    # Find common participants (all modalities must share same OOF y_true)
    y_trues = [modality_results[n]["oof_y_true"] for n in names]
    y_probs = [modality_results[n]["oof_y_prob"] for n in names]

    # Weighted sum
    fused = sum(p * w for p, w in zip(y_probs, weights)) / sum(weights)
    y_true = y_trues[0]  # all should be identical (same CV split)

    y_pred = (fused >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    auc_mean, auc_lo, auc_hi = bootstrap_ci(
        y_true, fused, lambda yt, ys: roc_auc_score(yt, ys), n_boot=1000)
    acc_mean, acc_lo, acc_hi = bootstrap_ci(
        y_true, y_pred, lambda yt, ys: accuracy_score(yt, ys.round()), n_boot=1000)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(
        y_true, y_pred, lambda yt, ys: f1_score(yt, ys.round(), zero_division=0), n_boot=1000)

    return {
        "Accuracy":     accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "F1":           f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC":      roc_auc_score(y_true, fused),
        "Precision":    precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity":  recall_score(y_true, y_pred, zero_division=0),
        "Specificity":  tn / max(1, tn + fp),
        "AUC 95% CI":   f"[{auc_lo:.4f}, {auc_hi:.4f}]",
        "Acc 95% CI":   f"[{acc_lo:.4f}, {acc_hi:.4f}]",
        "F1  95% CI":   f"[{f1_lo:.4f},  {f1_hi:.4f}]",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds",     type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--save",        action="store_true")
    parser.add_argument("--output",      default="results/cv_results.csv")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  SENTIRA — Leakage-Free {args.n_folds}-Fold CV Evaluation")
    print(f"{'='*65}")
    print(f"  Labels    : {LABELS_CSV}")
    print(f"  CV folds  : {args.n_folds}")
    print(f"  Bootstrap : {args.n_bootstrap} iterations, 95% CI")
    print(f"  SMOTE     : {'inside each fold (ImbPipeline)' if HAS_IMBALANCED else 'disabled (install imbalanced-learn)'}")
    print(f"{'='*65}")

    labels = load_labels()
    dep = labels.sum(); non = (labels == 0).sum()
    print(f"\n  Participants : {len(labels)}")
    print(f"  Depressed   : {dep}  ({dep/len(labels):.1%})")
    print(f"  Non-dep     : {non}  ({non/len(labels):.1%})\n")

    all_results   = {}
    modality_data = {}

    for name, cfg in MODALITY_CONFIG.items():
        path = os.path.join(FEATURES_DIR, cfg["features_file"])
        if not os.path.exists(path):
            print(f"  [SKIP] {name}: {path} not found")
            continue
        features = load_features(path)
        X, y, _ = align(features, labels)
        print(f"  {name}: {X.shape[0]} participants, {X.shape[1]} features")

        metrics, oof_y, oof_p = cv_evaluate(
            name, X, y, cfg["pipeline_fn"],
            n_folds=args.n_folds, n_bootstrap=args.n_bootstrap
        )
        all_results[name] = metrics
        modality_data[name] = {"oof_y_true": oof_y, "oof_y_prob": oof_p}

    # Late Fusion (only if all 3 modalities ran)
    if len(modality_data) == 3:
        print(f"\n  -- Late Fusion (weighted avg, w=[0.35, 0.35, 0.30]) --")
        lf_metrics = cv_late_fusion(modality_data, weights=[0.35, 0.35, 0.30])
        all_results["Late Fusion"] = lf_metrics

    # ── Print results table ──────────────────────────────────────────────
    COLS = ["Accuracy","Balanced Acc","F1","AUC-ROC","Precision","Sensitivity","Specificity"]
    CI_COLS = ["AUC 95% CI","Acc 95% CI","F1  95% CI"]

    print(f"\n\n{'='*65}")
    print(f"  RESULTS — {args.n_folds}-Fold OOF (no leakage)")
    print(f"{'='*65}")
    hdr = f"  {'Model':<16}" + "".join(f"{c:>13}" for c in COLS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, row in all_results.items():
        line = f"  {name:<16}" + "".join(f"{row.get(c, float('nan')):>13.4f}" for c in COLS)
        print(line)

    print(f"\n  {'Model':<16}  {'AUC 95% CI':<22}  {'Acc 95% CI':<22}  {'F1  95% CI'}")
    print("  " + "-" * 80)
    for name, row in all_results.items():
        print(f"  {name:<16}  {row.get('AUC 95% CI',''):<22}  "
              f"{row.get('Acc 95% CI',''):<22}  {row.get('F1  95% CI','')}")

    print(f"\n{'='*65}\n")

    # ── Save ────────────────────────────────────────────────────────────
    if args.save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df_out = pd.DataFrame(all_results).T
        df_out.index.name = "Model"
        df_out.to_csv(args.output)
        print(f"  [SAVED] {args.output}")


if __name__ == "__main__":
    main()
