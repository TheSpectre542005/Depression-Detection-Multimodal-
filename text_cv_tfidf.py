"""
text_cv_tfidf.py
================
Leakage-free 5-fold CV combining:
  1. Psycholinguistic features (from text_features_enhanced.csv)
  2. Within-fold TF-IDF on raw transcript text (fit on TRAIN fold only)

Approach:
  - Each fold: TfidfVectorizer is fit only on train participants' text.
  - TF-IDF + psycholinguistic features are hstacked and fed to classifier.
  - SMOTE is applied AFTER hstacking (inside ImbPipeline).
  - Reports OOF AUC + bootstrap 95% CI.
  - Saves best retrained model to models/text_model.pkl

Usage:
    python text_cv_tfidf.py
    python text_cv_tfidf.py --n_folds 10 --n_bootstrap 2000
"""

import os, warnings, argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    balanced_accuracy_score, precision_score, recall_score,
)
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

DATA_ROOT    = os.path.join(os.path.expanduser("~"), "Downloads", "E-DAIC", "data")
FEATURES_DIR = "data/features"
LABELS_CSV   = "data/features/master_labels.csv"
ENHANCED_CSV = "data/features/text_features_enhanced.csv"
PHQ_THRESH   = 10
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Read raw transcript text per participant
# ─────────────────────────────────────────────────────────────────────────────

def read_raw_texts(pids, data_root=DATA_ROOT):
    """Returns dict pid -> concatenated participant text (lowercased)."""
    texts = {}
    for pid in pids:
        path = os.path.join(data_root, f"{pid}_P", f"{pid}_Transcript.csv")
        if not os.path.exists(path):
            texts[pid] = ""
            continue
        try:
            df = pd.read_csv(path)
            if "Speaker" in df.columns:
                df = df[df["Speaker"].str.upper().isin(
                    ["PARTICIPANT", "P", "SUBJECT"])]
            if "Text" in df.columns:
                # Weight by confidence if available
                if "Confidence" in df.columns:
                    high_conf = df[df["Confidence"] >= 0.6]["Text"].fillna("")
                    texts[pid] = " ".join(high_conf.astype(str)).lower().strip()
                else:
                    texts[pid] = " ".join(df["Text"].fillna("").astype(str)).lower().strip()
            else:
                texts[pid] = ""
        except Exception:
            texts[pid] = ""
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    # Labels
    labels = pd.read_csv(LABELS_CSV)
    for id_col in ["pid","Participant_ID","participant_id","id","ID"]:
        if id_col in labels.columns:
            labels = labels.set_index(id_col); break
    y_series = (labels["PHQ_Score"] >= PHQ_THRESH).astype(int)

    # Psycholinguistic features (no leakage — each feature is per-participant rate)
    df_psych = pd.read_csv(ENHANCED_CSV)
    for id_col in ["pid","Participant_ID","participant_id","id","ID"]:
        if id_col in df_psych.columns:
            df_psych = df_psych.set_index(id_col); break
    for col in ["PHQ_Score","phq_score","label","Label","depressed","Depressed"]:
        if col in df_psych.columns:
            df_psych = df_psych.drop(columns=[col])
    df_psych = df_psych.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Align
    common   = df_psych.index.intersection(y_series.index)
    pids     = list(common)
    X_psych  = df_psych.loc[common].values
    y        = y_series.loc[common].values

    # Raw texts
    print(f"  Reading raw transcripts for {len(pids)} participants...")
    texts_dict = read_raw_texts([str(p) for p in pids])
    raw_texts  = [texts_dict.get(str(p), "") for p in pids]

    return pids, X_psych, raw_texts, y


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Within-fold TF-IDF combination
# ─────────────────────────────────────────────────────────────────────────────

def make_fold_features(
    texts_train, texts_test,
    X_psych_train, X_psych_test,
    n_tfidf=80,
):
    """
    Fit TF-IDF on train texts only, then combine with psycholinguistic features.
    Returns (X_train_combined, X_test_combined).
    """
    # TF-IDF with LSA (TruncatedSVD)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,            # must appear in at least 3 participants
        max_df=0.90,         # ignore near-universal words
        max_features=2000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf_train = tfidf.fit_transform(texts_train)
    tfidf_test  = tfidf.transform(texts_test)

    # Dimensionality reduction with LSA
    n_components = min(n_tfidf, tfidf_train.shape[1] - 1, tfidf_train.shape[0] - 1)
    lsa = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    lsa_train = lsa.fit_transform(tfidf_train)
    lsa_test  = lsa.transform(tfidf_test)

    # Scale psycholinguistic features
    sc = StandardScaler()
    psych_tr = sc.fit_transform(X_psych_train)
    psych_te = sc.transform(X_psych_test)

    # Hstack: LSA + psycholinguistic
    X_train_c = np.hstack([lsa_train, psych_tr])
    X_test_c  = np.hstack([lsa_test,  psych_te])

    return X_train_c, X_test_c


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Classifier candidates
# ─────────────────────────────────────────────────────────────────────────────

def get_classifiers(n_feats):
    clfs = {}
    k = min(60, n_feats)

    clfs["LR_L2"] = Pipeline([
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=3000, solver="lbfgs")),
    ])
    clfs["LR_L1"] = Pipeline([
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", LogisticRegression(C=0.5, class_weight="balanced",
                                   max_iter=3000, solver="saga", penalty="l1")),
    ])
    clfs["SVM"] = Pipeline([
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", CalibratedClassifierCV(
            SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced"), cv=3)),
    ])
    clfs["RF"] = RandomForestClassifier(
        n_estimators=400, max_depth=10, class_weight="balanced",
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
    )
    clfs["GB"] = Pipeline([
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=RANDOM_STATE)),
    ])
    if HAS_XGB:
        clfs["XGB"] = Pipeline([
            ("sel", SelectKBest(f_classif, k=k)),
            ("clf", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                scale_pos_weight=2.5,
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_STATE, verbosity=0)),
        ])
    if HAS_SMOTE:
        clfs["SMOTE+LR"] = ImbPipeline([
            ("sel",   SelectKBest(f_classif, k=k)),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf",   LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")),
        ])
        clfs["SMOTE+SVM"] = ImbPipeline([
            ("sel",   SelectKBest(f_classif, k=k)),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf",   CalibratedClassifierCV(
                SVC(kernel="rbf", C=2.0, gamma="scale"), cv=3)),
        ])
        clfs["SMOTE+RF"] = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf",   RandomForestClassifier(
                n_estimators=400, max_depth=10,
                min_samples_leaf=2, random_state=RANDOM_STATE)),
        ])

    return clfs


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Honest CV evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(n_folds=5, n_tfidf=80, n_bootstrap=1000):
    print(f"\n{'='*62}")
    print(f"  Text Model -- Leakage-Free {n_folds}-Fold CV + TF-IDF")
    print(f"  TF-IDF dims: {n_tfidf}  Bootstrap: {n_bootstrap}")
    print(f"{'='*62}")

    pids, X_psych, raw_texts, y = load_data()
    n = len(y)
    dep = y.sum()
    print(f"  Samples: {n}  |  Depressed: {dep}  |  Non-dep: {n - dep}\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    # Run one fold to determine feature size
    tr0, te0 = next(iter(skf.split(X_psych, y)))
    texts_tr  = [raw_texts[i] for i in tr0]
    texts_te  = [raw_texts[i] for i in te0]
    X_tr0, X_te0 = make_fold_features(texts_tr, texts_te, X_psych[tr0], X_psych[te0], n_tfidf)
    n_combined = X_tr0.shape[1]
    print(f"  Combined feature dims (LSA {n_tfidf} + psych): {n_combined}\n")

    classifiers  = get_classifiers(n_combined)
    clf_oof_probs = {name: [] for name in classifiers}
    clf_oof_y     = []

    print(f"  {'Fold':<6} {'Dep/Tot':<10}" + "".join(f"{n[:12]:>13}" for n in classifiers))
    print("  " + "-" * (16 + 13 * len(classifiers)))

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_psych, y), 1):
        texts_tr = [raw_texts[i] for i in tr_idx]
        texts_te = [raw_texts[i] for i in te_idx]
        X_tr_c, X_te_c = make_fold_features(
            texts_tr, texts_te, X_psych[tr_idx], X_psych[te_idx], n_tfidf)

        y_tr = y[tr_idx]; y_te = y[te_idx]
        clf_oof_y.append(y_te)

        row = f"  {fold_i:<6} {y_te.sum()}/{len(y_te):<6}"
        for name, clf in classifiers.items():
            from sklearn.base import clone
            c = clone(clf)
            c.fit(X_tr_c, y_tr)
            prob = c.predict_proba(X_te_c)[:, 1]
            clf_oof_probs[name].append(prob)
            auc = roc_auc_score(y_te, prob) if len(np.unique(y_te)) > 1 else float("nan")
            row += f"{auc:>13.4f}"
        print(row)

    oof_y = np.concatenate(clf_oof_y)
    print(f"\n  {'Classifier':<18} {'OOF AUC':>10}  {'95% CI':>20}  {'Acc':>8}  {'F1':>8}")
    print("  " + "-" * 72)

    results = {}
    for name in classifiers:
        oof_p     = np.concatenate(clf_oof_probs[name])
        oof_pred  = (oof_p >= 0.5).astype(int)
        auc       = roc_auc_score(oof_y, oof_p)
        acc       = accuracy_score(oof_y, oof_pred)
        f1        = f1_score(oof_y, oof_pred, zero_division=0)
        # Bootstrap CI
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(oof_y), len(oof_y))
            if len(np.unique(oof_y[idx])) < 2: continue
            boot.append(roc_auc_score(oof_y[idx], oof_p[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        results[name] = {"auc": auc, "lo": lo, "hi": hi, "acc": acc, "f1": f1,
                         "oof_p": oof_p}
        marker = " <-- BEST" if auc == max(r["auc"] for r in results.values()) else ""
        print(f"  {name:<18} {auc:>10.4f}  [{lo:.4f}, {hi:.4f}]  {acc:>8.4f}  {f1:>8.4f}{marker}")

    best_name = max(results, key=lambda n: results[n]["auc"])
    best      = results[best_name]
    print(f"\n  Best: {best_name}")
    print(f"  AUC  : {best['auc']:.4f}  95% CI [{best['lo']:.4f}, {best['hi']:.4f}]")
    print(f"  Acc  : {best['acc']:.4f}")
    print(f"  F1   : {best['f1']:.4f}")

    if best["auc"] >= 0.72:
        print(f"\n  [TARGET MET] AUC >= 0.72 achieved!")
    else:
        gap = 0.72 - best["auc"]
        print(f"\n  [NOTE] AUC = {best['auc']:.4f}, gap to 0.72 = {gap:.4f}")

    # ── Retrain best classifier on FULL dataset ──────────────────────────
    print(f"\n  Retraining {best_name} on full dataset ({n} samples)...")

    # Build final TF-IDF on all texts
    final_tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.90,
        max_features=2000, sublinear_tf=True, strip_accents="unicode",
    )
    tfidf_all = final_tfidf.fit_transform(raw_texts)
    n_comp    = min(n_tfidf, tfidf_all.shape[1] - 1, n - 1)
    final_lsa = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    lsa_all   = final_lsa.fit_transform(tfidf_all)
    final_sc  = StandardScaler()
    psych_all = final_sc.fit_transform(X_psych)
    X_all = np.hstack([lsa_all, psych_all])

    from sklearn.base import clone
    clf_final = clone(classifiers[best_name])
    clf_final.fit(X_all, y)

    # Save as a wrapped bundle that includes the TF-IDF + LSA + scaler steps
    bundle = {
        "clf":          clf_final,
        "tfidf":        final_tfidf,
        "lsa":          final_lsa,
        "psych_scaler": final_sc,
        "threshold":    0.5,
        "feature_file": ENHANCED_CSV,
        "model_type":   "tfidf+psycholinguistic",
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, "models/text_model_tfidf.pkl")
    print(f"  Saved -> models/text_model_tfidf.pkl")
    print(f"{'='*62}\n")

    return best["auc"], best["lo"], best["hi"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds",     type=int, default=5)
    parser.add_argument("--n_tfidf",     type=int, default=80,
                        help="LSA components from TF-IDF (default 80)")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    auc, lo, hi = run_cv(
        n_folds=args.n_folds,
        n_tfidf=args.n_tfidf,
        n_bootstrap=args.n_bootstrap,
    )
