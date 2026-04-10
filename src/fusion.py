# src/fusion.py
import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import warnings

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (MODELS_DIR, SMOTE_K_NEIGHBORS, CV_SPLITS, RANDOM_STATE,
                    C_GRID, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP,
                    AUGMENT_MIXUP, AUGMENT_NOISE, MIXUP_ALPHA, NOISE_STD, AUGMENT_FACTOR)

warnings.filterwarnings('ignore', category=UserWarning)
os.makedirs(MODELS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def mixup_augment(X, y, alpha=0.2, n_augment=None, random_state=42):
    """
    Mixup data augmentation — creates synthetic samples by blending pairs.
    Particularly effective for small tabular datasets.

    Args:
        X: feature matrix
        y: labels
        alpha: Beta distribution parameter (smaller = closer to originals)
        n_augment: number of augmented samples (default: len(minority_class))
    """
    rng = np.random.RandomState(random_state)
    minority_mask = y == 1  # Depression is minority class
    X_min = X[minority_mask]
    y_min = y[minority_mask]

    if len(X_min) < 2:
        return X, y

    if n_augment is None:
        n_augment = len(X_min)

    X_aug, y_aug = [], []
    for _ in range(n_augment):
        i, j = rng.choice(len(X_min), 2, replace=False)
        lam = rng.beta(alpha, alpha)
        x_new = lam * X_min[i] + (1 - lam) * X_min[j]
        X_aug.append(x_new)
        y_aug.append(1)

    X_combined = np.vstack([X, np.array(X_aug)])
    y_combined = np.concatenate([y, np.array(y_aug)])
    return X_combined, y_combined


def noise_augment(X, y, noise_std=0.05, n_augment=None, random_state=42):
    """
    Gaussian noise augmentation — adds small random perturbations to minority class.

    Args:
        X: feature matrix
        y: labels
        noise_std: standard deviation of noise (relative to feature std)
        n_augment: number of augmented samples
    """
    rng = np.random.RandomState(random_state)
    minority_mask = y == 1
    X_min = X[minority_mask]

    if len(X_min) < 1:
        return X, y

    if n_augment is None:
        n_augment = len(X_min)

    # Scale noise by feature standard deviation
    feature_stds = np.std(X, axis=0) + 1e-10

    X_aug, y_aug = [], []
    for _ in range(n_augment):
        idx = rng.choice(len(X_min))
        noise = rng.normal(0, noise_std, size=X_min.shape[1]) * feature_stds
        x_new = X_min[idx] + noise
        X_aug.append(x_new)
        y_aug.append(1)

    X_combined = np.vstack([X, np.array(X_aug)])
    y_combined = np.concatenate([y, np.array(y_aug)])
    return X_combined, y_combined


def select_features(X, y, feature_names=None, k=30):
    """Select top k features using mutual information."""
    if X.shape[1] <= k:
        return X, slice(None), None

    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)

    if feature_names:
        selected_mask = selector.get_support()
        selected_names = [n for n, sel in zip(feature_names, selected_mask) if sel]
        logger.info(f"    Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
        logger.debug(f"    Selected: {selected_names[:10]}...")

    return X_selected, selector.get_support(), selector


def get_model_candidates():
    """Return model candidates for ensemble training."""
    models = {
        'lr_l1': LogisticRegression(
            penalty='l1', C=0.1, solver='saga',
            class_weight='balanced', max_iter=10000, random_state=RANDOM_STATE
        ),
        'lr_l2': LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs',
            class_weight='balanced', max_iter=10000, random_state=RANDOM_STATE
        ),
        'svm_rbf': SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'svm_linear': SVC(
            kernel='linear', C=0.5, probability=True,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
    }

    # Random Forest only if we have enough samples
    models['rf'] = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )

    models['gb'] = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )

    # XGBoost — excellent for structured audio features
    if HAS_XGBOOST:
        models['xgb'] = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=2.5,  # Handle class imbalance
            eval_metric='logloss', use_label_encoder=False,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )

    return models


def train_with_cv(model, X, y, cv, name="model"):
    """Train model with cross-validation and return scores."""
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=min(SMOTE_K_NEIGHBORS, len(y)//4 + 1))),
        ('model', model),
    ])

    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=1)
    roc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=1)

    return {
        'model': model,
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std(),
        'roc_mean': roc_scores.mean(),
        'roc_std': roc_scores.std(),
        'name': name
    }


def calibrate_model(model, X_val, y_val):
    """Apply Platt scaling for better probability estimates."""
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated.fit(X_val, y_val)
    return calibrated


def train_unimodal(X, y, name):
    """
    Enhanced unimodal training with model selection and ensemble approach.
    """
    logger.info(f"Training [{name}] model...")
    logger.info(f"    Input shape: {X.shape}, Class distribution: {np.bincount(y)}")

    # Feature selection for high-dimensional features
    if X.shape[1] > 50:
        k_features = min(60, X.shape[1])
        X, feature_mask, selector = select_features(X, y, k=k_features)
        logger.info(f"    Selected top {k_features} features")
    else:
        feature_mask = slice(None)
        selector = None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=min(CV_SPLITS, min(np.bincount(y))), shuffle=True, random_state=RANDOM_STATE)

    # Try multiple models and pick the best
    candidates = get_model_candidates()
    results = []

    for model_name, model in candidates.items():
        try:
            result = train_with_cv(model, X, y, cv, model_name)
            results.append(result)
            logger.info(f"    [{model_name}] CV F1: {result['f1_mean']:.3f} (+/- {result['f1_std']:.3f}), "
                       f"ROC-AUC: {result['roc_mean']:.3f}")
        except Exception as e:
            logger.warning(f"    [{model_name}] Failed: {e}")

    if not results:
        logger.error("    All models failed, falling back to basic Logistic Regression")
        best_model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)
    else:
        # Pick model with best F1 score
        best_result = max(results, key=lambda x: x['f1_mean'])
        logger.info(f"    Best model: {best_result['name']} (F1={best_result['f1_mean']:.3f})")
        best_model = best_result['model']

    # Final training with best model + augmentation
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(SMOTE_K_NEIGHBORS, sum(y==0)//2 + 1))
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    logger.info(f"    SMOTE: {len(y)} -> {len(y_resampled)} samples")

    # Apply additional augmentation
    if AUGMENT_MIXUP:
        n_aug = int(sum(y == 1) * AUGMENT_FACTOR)
        X_resampled, y_resampled = mixup_augment(
            X_resampled, y_resampled, alpha=MIXUP_ALPHA, n_augment=n_aug)
        logger.info(f"    Mixup: +{n_aug} samples → {len(y_resampled)} total")

    if AUGMENT_NOISE:
        n_aug = int(sum(y == 1) * AUGMENT_FACTOR)
        X_resampled, y_resampled = noise_augment(
            X_resampled, y_resampled, noise_std=NOISE_STD, n_augment=n_aug)
        logger.info(f"    Noise: +{n_aug} samples → {len(y_resampled)} total")

    best_model.fit(X_resampled, y_resampled)

    # Calibrate probabilities
    X_cal, y_cal = X_resampled[:min(len(y_resampled)//3, 30)], y_resampled[:min(len(y_resampled)//3, 30)]
    try:
        model_calibrated = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
        model_calibrated.fit(X_cal, y_cal)
        best_model = model_calibrated
        logger.info("    Applied probability calibration")
    except Exception:
        logger.info("    Using uncalibrated probabilities")

    # Save model and artifacts
    model_path = os.path.join(MODELS_DIR, f'{name}_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'{name}_scaler.pkl')

    if selector is not None:
        selector_path = os.path.join(MODELS_DIR, f'{name}_selector.pkl')
        joblib.dump(selector, selector_path)

    joblib.dump(best_model, model_path)
    joblib.dump({'scaler': scaler, 'feature_mask': feature_mask}, scaler_path)

    logger.info(f"    Saved -> {model_path}")

    return best_model, {'scaler': scaler, 'feature_mask': feature_mask, 'selector': selector}


def transform_with_selector(X, artifacts):
    """Apply feature selection if available."""
    selector = artifacts.get('selector')
    mask = artifacts.get('feature_mask')

    if selector is not None:
        return selector.transform(X)
    elif mask is not None and mask is not slice(None):
        return X[:, mask]
    return X


def find_best_threshold(model, artifacts, X_val, y_val, metric='f1'):
    """
    Find optimal threshold using F-beta score (beta=2 favors recall).
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

    scaler = artifacts['scaler']
    X_selected = transform_with_selector(X_val, artifacts)
    X_sc = scaler.transform(X_selected)

    probs = model.predict_proba(X_sc)[:, 1]

    best_t, best_score = 0.5, 0.0
    beta = 2.0  # Favor recall (depression detection is safety-critical)

    for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP):
        preds = (probs >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_val, preds, zero_division=0)
        elif metric == 'fbeta':
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            score = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec) if (prec + rec) > 0 else 0
        elif metric == 'youden':
            # Youden's J = sensitivity + specificity - 1
            tpr = recall_score(y_val, preds, zero_division=0)
            fpr = np.mean((preds == 1) & (y_val == 0))
            score = tpr - fpr
        else:
            score = f1_score(y_val, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_t = t

    logger.info(f"    Best threshold: {best_t:.2f} (score={best_score:.3f})")
    return best_t


def find_cost_sensitive_threshold(model, artifacts, X_val, y_val, fn_cost=5, fp_cost=1):
    """
    Find threshold that minimizes weighted cost of errors.
    False negatives (missing depression) are weighted more heavily.

    Args:
        fn_cost: Cost of false negative (default 5x more expensive)
        fp_cost: Cost of false positive (default 1x)
    """
    from sklearn.metrics import confusion_matrix

    scaler = artifacts['scaler']
    X_selected = transform_with_selector(X_val, artifacts)
    X_sc = scaler.transform(X_selected)
    probs = model.predict_proba(X_sc)[:, 1]

    best_t, best_cost = 0.5, float('inf')

    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        cost = fn * fn_cost + fp * fp_cost
        if cost < best_cost:
            best_cost = cost
            best_t = t

    # Calculate metrics at best threshold
    preds = (probs >= best_t).astype(int)
    cm = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    logger.info(f"    Cost-sensitive threshold: {best_t:.2f} (cost={best_cost}, sensitivity={sensitivity:.3f})")
    return best_t


def late_fusion_predict(models_scalers, X_dict, weights=None, threshold=0.40, y_val=None):
    """
    Enhanced late fusion with performance-based weighting.

    If y_val is provided, weights are computed from validation AUC.
    Otherwise, provided weights or equal weighting is used.
    """
    from sklearn.metrics import roc_auc_score

    modalities = list(models_scalers.keys())

    if weights is None and y_val is not None:
        # Compute weights from validation performance
        weights = {}
        for m in modalities:
            if m not in X_dict:
                continue
            model, artifacts = models_scalers[m]
            scaler = artifacts['scaler']
            X_selected = transform_with_selector(X_dict[m], artifacts)
            X_sc = scaler.transform(X_selected)
            probs = model.predict_proba(X_sc)[:, 1]
            try:
                auc = roc_auc_score(y_val, probs)
                # Weight by AUC above random chance (0.5)
                weights[m] = max(0.01, auc - 0.5)
            except:
                weights[m] = 0.01
    elif weights is None:
        weights = {m: 1.0 / len(modalities) for m in modalities}

    # Normalize weights
    total_w = sum(weights.get(m, 0) for m in modalities)
    if total_w > 0:
        weights = {m: weights.get(m, 0) / total_w for m in modalities}
    else:
        weights = {m: 1.0 / len(modalities) for m in modalities}

    logger.info(f"    Fusion weights: " + ", ".join(f"{m}={weights.get(m, 0):.2f}" for m in modalities))

    n = len(next(iter(X_dict.values())))
    prob_sum = np.zeros(n)

    for m in modalities:
        if weights.get(m, 0) <= 0.01:  # Skip very low weight modalities
            continue

        model, artifacts = models_scalers[m]
        scaler = artifacts['scaler']

        X_selected = transform_with_selector(X_dict[m], artifacts)
        X_sc = scaler.transform(X_selected)
        probs = model.predict_proba(X_sc)[:, 1]

        prob_sum += probs * weights[m]

    predictions = (prob_sum >= threshold).astype(int)
    return predictions, prob_sum


def train_meta_learner(models_scalers, X_dict, y, cv_splits=3):
    """
    Train a meta-learner for stacking fusion.
    """
    from sklearn.linear_model import LogisticRegression

    # Build meta-features
    meta_features = []
    for m in ['text', 'audio', 'visual']:
        if m in models_scalers:
            model, artifacts = models_scalers[m]
            scaler = artifacts['scaler']
            X_selected = transform_with_selector(X_dict[m], artifacts)
            X_sc = scaler.transform(X_selected)
            probs = model.predict_proba(X_sc)[:, 1]
            meta_features.append(probs)

    meta_X = np.column_stack(meta_features)

    # Train meta-learner with regularization
    meta_model = LogisticRegression(
        C=0.5, penalty='l2', class_weight='balanced',
        max_iter=5000, random_state=RANDOM_STATE
    )
    meta_model.fit(meta_X, y)

    logger.info(f"    Meta-learner coefficients: {meta_model.coef_[0]}")

    return meta_model
