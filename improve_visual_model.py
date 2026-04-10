"""
improve_visual_model.py
=======================
Retrains the visual model -- mirrors the approach used in improve_audio_model.py.

Root cause of bad visual accuracy (predicts all-non-depressed):
  - The old visual_model.pkl was trained on only 20 features.
  - visual_features.csv now has 214 features.
  - The dimension mismatch caused the model to truncate to 20 stale features,
    collapsing discrimination power.

Fixes applied:
  1. Multi-candidate classifier search (LR, SVM, RF, GB, XGB)
  2. SMOTE + class-weight balancing
  3. PCA dimensionality reduction (handles wide feature matrix)
  4. Threshold tuning on held-out validation set
  5. Saved as {pipeline, threshold} bundle -- compatible with calculate_accuracies.py

Usage
-----
    python improve_visual_model.py
    python improve_visual_model.py --labels_csv data/features/master_labels.csv
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
)
from sklearn.calibration import CalibratedClassifierCV

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    print("[WARN] imbalanced-learn not found. Install with: pip install imbalanced-learn")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VISUAL_FEATURES_FILE = (
    "data/features/visual_features_enhanced.csv"
    if os.path.exists("data/features/visual_features_enhanced.csv")
    else "data/features/visual_features.csv"
)
LABELS_CANDIDATES = [
    "data/features/master_labels.csv",
    "data/features/labels.csv",
    "data/features/phq_labels.csv",
    "data/labels.csv",
    "labels.csv",
]
OUTPUT_MODEL_PATH = "models/visual_model.pkl"
PHQ_THRESHOLD     = 10
RANDOM_STATE      = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_labels(path, threshold=10):
    df = pd.read_csv(path)
    for col in ["PHQ_Score", "phq_score", "PHQ8_Score", "phq8_score", "label", "Label"]:
        if col in df.columns:
            score_col = col
            break
    else:
        raise ValueError(f"No PHQ score column found in {path}. Columns: {list(df.columns)}")
    for id_col in ["pid", "Participant_ID", "participant_id", "id", "ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col)
            break
    if df[score_col].max() > 1:
        return (df[score_col] >= threshold).astype(int)
    return df[score_col].astype(int)


def load_features(path):
    df = pd.read_csv(path)
    for id_col in ["pid", "Participant_ID", "participant_id", "id", "ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col)
            break
    for col in ["PHQ_Score", "phq_score", "label", "Label", "depressed", "Depressed"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


def align(features, labels):
    common = features.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError("No overlapping IDs between features and labels!")
    print(f"  Aligned on {len(common)} participants")
    return features.loc[common].values, labels.loc[common].values


def tune_threshold(model, X_val, y_val):
    """
    Find threshold maximising balanced accuracy while keeping both classes
    represented (prevents the 'predict everything positive' trap on small val sets).
    """
    from sklearn.metrics import balanced_accuracy_score
    proba = model.predict_proba(X_val)[:, 1]
    n = len(y_val)
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (proba >= t).astype(int)
        pos_pct = preds.mean()
        # Require at least 5% and at most 95% predicted positive
        if pos_pct < 0.05 or pos_pct > 0.95:
            continue
        score = balanced_accuracy_score(y_val, preds)
        if score > best_score:
            best_score, best_t = score, t
    # Fallback: use 0.5 if no threshold passed the guard
    if best_score < 0:
        best_t = 0.5
    return best_t, best_score


def print_metrics(name, y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    print(f"\n  {'Model':<32} Acc     F1      Prec    Recall  AUC")
    print(f"  {'-'*72}")
    print(f"  {name:<32} {acc:.4f}  {f1:.4f}  {prec:.4f}  {rec:.4f}  {auc:.4f}")


# ---------------------------------------------------------------------------
# Candidate pipelines
# ---------------------------------------------------------------------------
def build_pipelines(n_pca):
    pipes = {}
    vt = ("vt", VarianceThreshold(threshold=0.001))  # drop near-zero variance cols
    k  = min(30, n_pca * 2)  # SelectKBest k

    pipes["LR_balanced"] = Pipeline([
        vt,
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("clf",    LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=3000,
            solver="saga", penalty="l1", random_state=RANDOM_STATE
        )),
    ])

    pipes["LR_kbest"] = Pipeline([
        vt,
        ("scaler", StandardScaler()),
        ("sel",    SelectKBest(f_classif, k=k)),
        ("clf",    LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000,
            solver="lbfgs", random_state=RANDOM_STATE
        )),
    ])

    pipes["SVM_rbf"] = Pipeline([
        vt,
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("clf",    CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=RANDOM_STATE),
            cv=3
        )),
    ])

    pipes["SVM_kbest"] = Pipeline([
        vt,
        ("scaler", StandardScaler()),
        ("sel",    SelectKBest(f_classif, k=k)),
        ("clf",    CalibratedClassifierCV(
            SVC(kernel="rbf", C=2.0, gamma="scale",
                class_weight="balanced", random_state=RANDOM_STATE),
            cv=3
        )),
    ])

    pipes["RandomForest"] = Pipeline([
        vt,
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    pipes["GradientBoosting"] = Pipeline([
        vt,
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=RANDOM_STATE
        )),
    ])

    if HAS_XGB:
        pipes["XGBoost"] = Pipeline([
            vt,
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=n_pca, random_state=RANDOM_STATE)),
            ("clf",    xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                scale_pos_weight=2,
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_STATE, verbosity=0
            )),
        ])

    return pipes


def build_smote_pipelines(n_pca):
    if not HAS_IMBALANCED:
        return {}
    pipes = {}
    vt = ("vt", VarianceThreshold(threshold=0.001))
    k  = min(30, n_pca * 2)

    pipes["SMOTE+LR"] = ImbPipeline([
        vt,
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        ("clf",    LogisticRegression(
            C=0.5, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE
        )),
    ])

    pipes["SMOTE+LR_kbest"] = ImbPipeline([
        vt,
        ("scaler", StandardScaler()),
        ("sel",    SelectKBest(f_classif, k=k)),
        ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        ("clf",    LogisticRegression(
            C=1.0, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE
        )),
    ])

    pipes["SMOTE+RF"] = ImbPipeline([
        vt,
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=6,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    if HAS_XGB:
        pipes["SMOTE+XGB"] = ImbPipeline([
            vt,
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=n_pca, random_state=RANDOM_STATE)),
            ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf",    xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_STATE, verbosity=0
            )),
        ])

    return pipes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual_features", default=VISUAL_FEATURES_FILE)
    parser.add_argument("--labels_csv",       default=None)
    parser.add_argument("--threshold",         type=int, default=PHQ_THRESHOLD)
    parser.add_argument("--output",            default=OUTPUT_MODEL_PATH)
    args = parser.parse_args()

    # -- Find labels
    labels_csv = args.labels_csv or next(
        (p for p in LABELS_CANDIDATES if os.path.exists(p)), None
    )
    if labels_csv is None:
        print("[ERROR] Labels CSV not found. Use --labels_csv <path>")
        return

    print(f"\n{'='*62}")
    print("  Visual Model Improvement Script")
    print(f"{'='*62}")
    print(f"  Visual features : {args.visual_features}")
    print(f"  Labels CSV      : {labels_csv}")
    print(f"  PHQ threshold   : {args.threshold}")
    print(f"  Output model    : {args.output}")
    print(f"{'='*62}\n")

    # -- Load data
    labels   = load_labels(labels_csv, args.threshold)
    features = load_features(args.visual_features)
    X, y     = align(features, labels)

    n_feat = X.shape[1]
    pos = y.sum(); neg = len(y) - pos
    print(f"  Dataset: {len(y)} samples | Depressed: {pos} | Non-depressed: {neg}")
    print(f"  Class ratio: 1 : {neg/max(pos,1):.1f}")
    print(f"  Feature dims: {n_feat}\n")

    # PCA components: cap at 40 or half of features, whichever is smaller
    n_pca = min(40, n_feat // 2, 30)
    print(f"  PCA components for dimensionality reduction: {n_pca}\n")

    # -- Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # CV scored on F1 (better metric for imbalanced data)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_pipes = {**build_pipelines(n_pca), **build_smote_pipelines(n_pca)}

    print("  Running 5-fold cross-validation on all candidates...\n")
    print(f"  {'Classifier':<25} CV F1 (mean +/- std)")
    print(f"  {'-'*50}")

    cv_scores = {}
    for name, pipe in all_pipes.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="f1", n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"  {name:<25} {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\n  [OK] Best CV model: {best_name}  ({cv_scores[best_name]:.4f})\n")

    # -- Retrain best model on full train set
    best_pipe = all_pipes[best_name]
    best_pipe.fit(X_train, y_train)

    # -- Threshold tuning on validation set
    best_threshold, _ = tune_threshold(best_pipe, X_val, y_val)
    print(f"  Optimal decision threshold (val set): {best_threshold:.2f}\n")

    # -- Final evaluation on validation set
    y_prob_val = best_pipe.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val >= best_threshold).astype(int)

    print("  -- Validation Set Results --")
    print_metrics(f"{best_name} (t={best_threshold:.2f})",
                  y_val, y_pred_val, y_prob_val)
    print("\n" + classification_report(
        y_val, y_pred_val,
        target_names=["Non-Depressed", "Depressed"],
        zero_division=0
    ))

    # -- Save model
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model_bundle = {"pipeline": best_pipe, "threshold": best_threshold}
    joblib.dump(model_bundle, args.output)

    plain_path = args.output.replace(".pkl", "_plain.pkl")
    joblib.dump(best_pipe, plain_path)

    print(f"\n  [OK] Improved model saved -> {args.output}")
    print(f"  [OK] Plain pipeline saved -> {plain_path}")

    final_acc = accuracy_score(y_val, y_pred_val)
    if final_acc < 0.65:
        print("\n  [NOTE] Accuracy below 0.65.")
        print("    * Try: python improve_visual_model.py --visual_features data/features/visual_features_enhanced.csv")
        print("    * Or regenerate visual features: python main.py --extract_visual")
    else:
        print(f"\n  [TARGET] Target reached! Visual accuracy = {final_acc:.4f}")


if __name__ == "__main__":
    main()
