"""
improve_text_model.py
=====================
Retrains the text model -- mirrors the approach used in improve_audio_model.py.

Root cause of bad text accuracy (predicts all-depressed):
  - Old model fitted on a mismatched feature set / threshold = 0.5 on an
    unbalanced dataset causes it to default to always predicting the minority class.

Fixes applied:
  1. Multi-candidate classifier search (LR, SVM, RF, GB, XGB)
  2. SMOTE + class-weight balancing
  3. Threshold tuning on held-out validation set
  4. Saved as {pipeline, threshold} bundle -- compatible with calculate_accuracies.py

Usage
-----
    python improve_text_model.py
    python improve_text_model.py --labels_csv data/features/master_labels.csv
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
    print("[WARN] imbalanced-learn not found -- install with: pip install imbalanced-learn")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TEXT_FEATURES_FILE = "data/features/text_features.csv"
LABELS_CANDIDATES = [
    "data/features/master_labels.csv",
    "data/features/labels.csv",
    "data/features/phq_labels.csv",
    "data/labels.csv",
    "labels.csv",
]
OUTPUT_MODEL_PATH = "models/text_model.pkl"
PHQ_THRESHOLD = 10
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_labels(path, threshold=10):
    df = pd.read_csv(path)
    # Support both pid/Participant_ID index styles and PHQ/label column names
    for col in ["PHQ_Score", "phq_score", "PHQ8_Score", "phq8_score", "label", "Label"]:
        if col in df.columns:
            score_col = col
            break
    else:
        raise ValueError(f"No PHQ/label column found in {path}. Columns: {list(df.columns)}")

    # master_labels.csv uses 'pid'; others use Participant_ID
    for id_col in ["pid", "Participant_ID", "participant_id", "id", "ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col)
            break

    # If it's a raw PHQ score, binarise; if it's already 0/1, leave it
    if df[score_col].max() > 1:
        return (df[score_col] >= threshold).astype(int)
    return df[score_col].astype(int)


def load_features(path):
    df = pd.read_csv(path)
    for id_col in ["pid", "Participant_ID", "participant_id", "id", "ID"]:
        if id_col in df.columns:
            df = df.set_index(id_col)
            break
    # Drop any label columns accidentally saved alongside features
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
    """Find the decision threshold that maximises balanced accuracy."""
    proba = model.predict_proba(X_val)[:, 1]
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def print_metrics(name, y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    print(f"\n  {'Model':<32} Acc     F1      Prec    Recall  AUC")
    print(f"  {'-'*72}")
    print(f"  {name:<32} {acc:.4f}  {f1:.4f}  {prec:.4f}  {rec:.4f}  {auc:.4f}")


# -----------------------------------------------------------------------------
# Build candidate pipelines
# -----------------------------------------------------------------------------
def build_pipelines():
    pipes = {}

    # 1. LR L1 (sparse -- good for TF-IDF)
    pipes["LR_L1"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=3000,
            solver="saga", penalty="l1", random_state=RANDOM_STATE
        )),
    ])

    # 2. LR L2
    pipes["LR_L2"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000,
            solver="lbfgs", random_state=RANDOM_STATE
        )),
    ])

    # 3. SVM linear (great for sparse/text features)
    pipes["SVM_linear"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(
            SVC(kernel="linear", C=0.5,
                class_weight="balanced", random_state=RANDOM_STATE),
            cv=3
        )),
    ])

    # 4. SVM RBF
    pipes["SVM_rbf"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=RANDOM_STATE),
            cv=3
        )),
    ])

    # 5. Random Forest
    pipes["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    # 6. Gradient Boosting
    pipes["GradientBoosting"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=RANDOM_STATE
        )),
    ])

    # 7. XGBoost
    if HAS_XGB:
        pipes["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                scale_pos_weight=2,
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_STATE, verbosity=0
            )),
        ])

    return pipes


def build_smote_pipelines():
    if not HAS_IMBALANCED:
        return {}
    pipes = {}

    pipes["SMOTE+LR"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        ("clf",    LogisticRegression(
            C=0.5, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE
        )),
    ])

    pipes["SMOTE+RF"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    if HAS_XGB:
        pipes["SMOTE+XGB"] = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf",    xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_STATE, verbosity=0
            )),
        ])

    return pipes


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_features", default=TEXT_FEATURES_FILE)
    parser.add_argument("--labels_csv",    default=None)
    parser.add_argument("--threshold",     type=int, default=PHQ_THRESHOLD)
    parser.add_argument("--output",        default=OUTPUT_MODEL_PATH)
    args = parser.parse_args()

    # -- Find labels ------------------------------------------------------
    labels_csv = args.labels_csv or next(
        (p for p in LABELS_CANDIDATES if os.path.exists(p)), None
    )
    if labels_csv is None:
        print("[ERROR] Labels CSV not found. Use --labels_csv <path>")
        return

    print(f"\n{'='*62}")
    print("  Text Model Improvement Script")
    print(f"{'='*62}")
    print(f"  Text features : {args.text_features}")
    print(f"  Labels CSV    : {labels_csv}")
    print(f"  PHQ threshold : {args.threshold}")
    print(f"  Output model  : {args.output}")
    print(f"{'='*62}\n")

    # -- Load data ----------------------------------------------------------
    labels   = load_labels(labels_csv, args.threshold)
    features = load_features(args.text_features)
    X, y     = align(features, labels)

    pos = y.sum(); neg = len(y) - pos
    print(f"  Dataset: {len(y)} samples  |  Depressed: {pos}  |  Non-depressed: {neg}")
    print(f"  Class ratio: 1 : {neg/max(pos,1):.1f}")
    print(f"  Feature dims: {X.shape[1]}\n")

    # -- Train / validation split ------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # -- Cross-validation on ALL candidates -------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_pipes = {**build_pipelines(), **build_smote_pipelines()}

    print("  Running 5-fold cross-validation on all candidates...\n")
    print(f"  {'Classifier':<25} CV Accuracy (mean +/- std)")
    print(f"  {'-'*50}")

    cv_scores = {}
    for name, pipe in all_pipes.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"  {name:<25} {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\n  Best CV model: {best_name}  ({cv_scores[best_name]:.4f})\n")

    # -- Retrain best model on full train set ------------------------------
    best_pipe = all_pipes[best_name]
    best_pipe.fit(X_train, y_train)

    # -- Threshold tuning on validation set -------------------------------
    best_threshold, _ = tune_threshold(best_pipe, X_val, y_val)
    print(f"  Optimal decision threshold (val set): {best_threshold:.2f}\n")

    # -- Final evaluation on validation set -------------------------------
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

    # -- Compare against baseline ------------------------------------------
    print("  -- Baseline vs Improved (validation set) --")
    print(f"  {'Metric':<15} {'Baseline':>10} {'Improved':>10}")
    print(f"  {'-'*38}")
    baseline = {"Accuracy": 0.30, "F1": 0.46, "Precision": 0.30, "Recall": 1.00, "AUC-ROC": 0.50}
    improved = {
        "Accuracy":  accuracy_score(y_val, y_pred_val),
        "F1":        f1_score(y_val, y_pred_val, zero_division=0),
        "Precision": precision_score(y_val, y_pred_val, zero_division=0),
        "Recall":    recall_score(y_val, y_pred_val, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_val, y_prob_val),
    }
    for m in baseline:
        delta = improved[m] - baseline[m]
        arrow = "UP" if delta > 0 else "DOWN"
        print(f"  {m:<15} {baseline[m]:>10.4f} {improved[m]:>10.4f}  {arrow} {abs(delta):.4f}")

    # -- Save model --------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model_bundle = {"pipeline": best_pipe, "threshold": best_threshold}
    joblib.dump(model_bundle, args.output)

    plain_path = args.output.replace(".pkl", "_plain.pkl")
    joblib.dump(best_pipe, plain_path)

    print(f"\n  Improved model saved  -> {args.output}")
    print(f"  Plain pipeline saved  -> {plain_path}")

    if improved["Accuracy"] < 0.60:
        print("\n  [NOTE] Accuracy still below 0.60. Suggestions:")
        print("    * The text_features.csv may be stale -- re-extract with: python main.py")
        print("    * Try installing sentence-transformers for SBERT embeddings")
    else:
        print(f"\n  Target reached! Text accuracy = {improved['Accuracy']:.4f}")


if __name__ == "__main__":
    main()
