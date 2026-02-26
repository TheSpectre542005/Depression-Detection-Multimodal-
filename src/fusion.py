# src/fusion.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib, os, warnings

warnings.filterwarnings('ignore', category=UserWarning)
os.makedirs('models', exist_ok=True)


def train_unimodal(X, y, name):
    print(f"\n  Training [{name}] model...")

    # ── Scale ──────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Hyperparameter search (proper CV with SMOTE inside) ────
    # Try multiple regularization strengths to find the best one
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_c, best_f1 = 0.1, -1
    for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        cv_pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote',  SMOTE(random_state=42, k_neighbors=3)),
            ('model',  LogisticRegression(
                penalty='l1', C=C, solver='saga',
                class_weight='balanced', max_iter=5000, random_state=42)),
        ])
        scores = cross_val_score(cv_pipe, X, y, cv=cv, scoring='f1', n_jobs=1)
        mean_f1 = scores.mean()
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_c  = C

    # ── Final CV with best C to report scores ──────────────────
    cv_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote',  SMOTE(random_state=42, k_neighbors=3)),
        ('model',  LogisticRegression(
            penalty='l1', C=best_c, solver='saga',
            class_weight='balanced', max_iter=5000, random_state=42)),
    ])
    f1_scores  = cross_val_score(cv_pipe, X, y, cv=cv, scoring='f1', n_jobs=1)
    acc_scores = cross_val_score(cv_pipe, X, y, cv=cv, scoring='accuracy', n_jobs=1)

    print(f"    Best C={best_c}")
    print(f"    CV Accuracy : {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"    CV F1-Score : {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # ── Train final model on all training data ─────────────────
    model = LogisticRegression(
        penalty='l1', C=best_c, solver='saga',
        class_weight='balanced', max_iter=5000, random_state=42,
    )
    model.fit(X_scaled, y)

    joblib.dump(model,  f'models/{name}_model.pkl')
    joblib.dump(scaler, f'models/{name}_scaler.pkl')
    print(f"    ✅ Saved → models/{name}_model.pkl")

    return model, scaler


def find_best_threshold(model, scaler, X_val, y_val):
    """
    Find the probability threshold that maximises F1 on validation data.
    Constrain threshold to [0.25, 0.65] to prevent degenerate predictions.
    """
    from sklearn.metrics import f1_score
    X_sc  = scaler.transform(X_val)
    probs = model.predict_proba(X_sc)[:, 1]

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.25, 0.65, 0.01):
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    print(f"    Best threshold: {best_t:.2f}  (F1={best_f1:.3f})")
    return best_t


def late_fusion_predict(models_scalers, X_dict,
                        weights=None, threshold=0.40):
    """
    Late fusion: weighted average of predicted probabilities.

    models_scalers : {'text': (model, scaler), ...}
    X_dict         : {'text': X_text, ...}
    weights        : {'text': 0.5, 'audio': 0.2, 'visual': 0.3}
    threshold      : decision boundary
    """
    modalities = list(models_scalers.keys())

    if weights is None:
        w       = 1 / len(modalities)
        weights = {m: w for m in modalities}

    # Normalize weights to sum to 1
    total_w = sum(weights[m] for m in modalities)
    weights = {m: weights[m] / total_w for m in modalities}

    n        = len(next(iter(X_dict.values())))
    prob_sum = np.zeros(n)

    for m in modalities:
        model, scaler = models_scalers[m]
        X_sc   = scaler.transform(X_dict[m])
        probs  = model.predict_proba(X_sc)[:, 1]
        prob_sum += probs * weights[m]

    predictions = (prob_sum >= threshold).astype(int)
    return predictions, prob_sum