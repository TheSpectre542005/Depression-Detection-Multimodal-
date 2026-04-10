# src/visual_browser_model.py
"""
Browser-compatible visual model for depression detection.

Problem: Training uses OpenFace AUs + CNN features (PCA-reduced).
         The browser uses face-api.js which outputs 7 expression probabilities.
         These are completely different feature spaces.

Solution: Map OpenFace AUs → 7 basic expressions, then train on expression
           features so the model matches what face-api.js provides at inference.

AU→Expression mapping (Ekman's FACS):
  Happy     = AU06 (Cheek Raise) + AU12 (Lip Corner Pull)
  Sad       = AU01 (Inner Brow Raise) + AU04 (Brow Lower) + AU15 (Lip Depress)
  Angry     = AU04 (Brow Lower) + AU05 (Upper Lid) + AU07 (Lid Tight) + AU23 (Lip Tight)
  Surprised = AU01 (Inner Brow Raise) + AU02 (Outer Brow Raise) + AU05 + AU26 (Jaw Drop)
  Fearful   = AU01 + AU02 + AU04 + AU05 + AU20 (Lip Stretch)
  Disgusted = AU09 (Nose Wrinkle) + AU15 + AU17 (Chin Raise)
  Neutral   = all AUs low
"""
import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_ROOT, FEATURES_DIR, MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

SAVE_PATH = os.path.join(FEATURES_DIR, "visual_browser_features.csv")

# ── AU → Expression Mapping (FACS-based) ──────────────────────────

# AU column names in OpenFace CSV (intensity, 0–5 scale)
AU_NAMES = {
    'AU01_r': 'inner_brow_raise',
    'AU02_r': 'outer_brow_raise',
    'AU04_r': 'brow_lowerer',
    'AU05_r': 'upper_lid_raise',
    'AU06_r': 'cheek_raise',
    'AU07_r': 'lid_tightener',
    'AU09_r': 'nose_wrinkler',
    'AU10_r': 'upper_lip_raise',
    'AU12_r': 'lip_corner_pull',
    'AU14_r': 'dimpler',
    'AU15_r': 'lip_corner_depress',
    'AU17_r': 'chin_raise',
    'AU20_r': 'lip_stretch',
    'AU23_r': 'lip_tightener',
    'AU25_r': 'lips_part',
    'AU26_r': 'jaw_drop',
    'AU45_r': 'blink',
}


def au_to_expressions(au_dict):
    """
    Convert OpenFace AU intensities (0–5) to expression probabilities (0–1).

    Uses FACS-based mappings with sigmoid normalization.
    Returns dict of 7 expression probabilities.
    """
    def get_au(name, default=0.0):
        return float(au_dict.get(name, default))

    def sigmoid(x, center=1.5, steepness=2.0):
        """Map AU intensity (0-5) to probability (0-1)."""
        return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

    # Compute raw expression scores from AUs
    happy_raw = (get_au('AU06_r') + get_au('AU12_r')) / 2
    sad_raw = (get_au('AU01_r') + get_au('AU04_r') + get_au('AU15_r')) / 3
    angry_raw = (get_au('AU04_r') + get_au('AU05_r') + get_au('AU07_r') + get_au('AU23_r')) / 4
    surprised_raw = (get_au('AU01_r') + get_au('AU02_r') + get_au('AU05_r') + get_au('AU26_r')) / 4
    fearful_raw = (get_au('AU01_r') + get_au('AU02_r') + get_au('AU04_r') + get_au('AU20_r')) / 4
    disgusted_raw = (get_au('AU09_r') + get_au('AU15_r') + get_au('AU17_r')) / 3

    # Apply sigmoid to convert to probabilities
    expressions = {
        'happy': sigmoid(happy_raw),
        'sad': sigmoid(sad_raw),
        'angry': sigmoid(angry_raw),
        'surprised': sigmoid(surprised_raw),
        'fearful': sigmoid(fearful_raw),
        'disgusted': sigmoid(disgusted_raw),
    }

    # Neutral = when no strong expression is detected
    max_expr = max(expressions.values())
    expressions['neutral'] = 1.0 - max_expr

    # Softmax-like normalization
    total = sum(expressions.values())
    if total > 0:
        expressions = {k: v / total for k, v in expressions.items()}

    return expressions


def extract_browser_visual_features(pid, data_root):
    """
    Extract expression-based features that match face-api.js output.

    For each of the 7 expressions, compute temporal statistics:
    - mean, std, max, trend → 4 features per expression = 28 features
    Plus depression-specific derived features = ~6 features
    Total: ~34 features
    """
    path = os.path.join(data_root, f"{pid}_P", "features",
                        f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")

    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)

        # Quality filter
        if 'confidence' in df.columns:
            df = df[df['confidence'] >= 0.80]
        if 'success' in df.columns:
            df = df[df['success'] == 1]

        if df.empty or len(df) < 5:
            return None

        # Get available AU columns
        au_cols = [c for c in df.columns if c.endswith('_r') and 'AU' in c]
        if not au_cols:
            return None

        # Convert each frame's AUs to expression probabilities
        expressions_over_time = []
        for _, row in df.iterrows():
            au_dict = {col: row.get(col, 0) for col in au_cols}
            expr = au_to_expressions(au_dict)
            expressions_over_time.append(expr)

        expr_df = pd.DataFrame(expressions_over_time)
        features = {}

        # ── Per-expression temporal statistics ──
        for expr_name in ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']:
            if expr_name not in expr_df.columns:
                continue
            vals = expr_df[expr_name].values

            features[f'expr_{expr_name}_mean'] = float(np.mean(vals))
            features[f'expr_{expr_name}_std'] = float(np.std(vals))
            features[f'expr_{expr_name}_max'] = float(np.max(vals))

            # Trend (linear fit slope)
            if len(vals) > 3:
                try:
                    t = np.arange(len(vals))
                    features[f'expr_{expr_name}_trend'] = float(np.polyfit(t, vals, 1)[0])
                except (np.linalg.LinAlgError, ValueError):
                    features[f'expr_{expr_name}_trend'] = 0.0

        # ── Depression-specific derived features ──

        # Flat affect: low variance across all expressions (monotone face)
        expr_vars = [expr_df[e].var() for e in ['happy', 'sad', 'neutral']
                     if e in expr_df.columns]
        features['flat_affect'] = 1.0 - float(np.mean(expr_vars)) if expr_vars else 0.0

        # Sad-to-happy ratio (elevated in depression)
        if 'sad' in expr_df.columns and 'happy' in expr_df.columns:
            happy_mean = expr_df['happy'].mean()
            sad_mean = expr_df['sad'].mean()
            features['sad_happy_ratio'] = sad_mean / max(happy_mean, 0.01)

        # Negative expression dominance
        neg_cols = [c for c in ['sad', 'angry', 'fearful', 'disgusted'] if c in expr_df.columns]
        pos_cols = [c for c in ['happy', 'surprised'] if c in expr_df.columns]
        if neg_cols and pos_cols:
            neg_mean = expr_df[neg_cols].mean(axis=1).mean()
            pos_mean = expr_df[pos_cols].mean(axis=1).mean()
            features['neg_pos_expr_ratio'] = neg_mean / max(pos_mean, 0.01)

        # Smile frequency (how often happy > 0.3)
        if 'happy' in expr_df.columns:
            features['smile_frequency'] = float((expr_df['happy'] > 0.3).mean())

        # Expression transition rate (how often dominant expression changes)
        if len(expr_df) > 1:
            dominant = expr_df.idxmax(axis=1)
            transitions = (dominant != dominant.shift()).sum()
            features['expression_transition_rate'] = transitions / len(expr_df)

        return features

    except Exception as e:
        logger.warning(f"Error extracting browser visual features for {pid}: {e}")
        return None


def build_browser_visual_features(participant_ids):
    """
    Build expression-based visual features compatible with face-api.js.
    """
    logger.info("Extracting BROWSER-COMPATIBLE visual features...")
    logger.info("  Mapping OpenFace AUs → face-api.js expressions")
    logger.info("  Using temporal statistics on 7 basic expressions")

    records = []
    missing = 0

    for pid in participant_ids:
        feats = extract_browser_visual_features(pid, DATA_ROOT)
        if feats is not None:
            feats['pid'] = pid
            records.append(feats)
        else:
            missing += 1

    if missing:
        logger.info(f"  Missing browser visual features: {missing} participants")

    df = pd.DataFrame(records).fillna(0)

    # Remove constant features
    feat_cols = [c for c in df.columns if c != 'pid']
    constant = [c for c in feat_cols if df[c].nunique() <= 1]
    if constant:
        logger.info(f"  Removing {len(constant)} constant features")
        df = df.drop(columns=constant)

    # Replace infinities
    df = df.replace([np.inf, -np.inf], 0)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    logger.info(f"  ✅ Browser visual features saved → {SAVE_PATH}")
    logger.info(f"  Shape: {df.shape}")
    return df


def train_browser_visual_model(X, y, save=True):
    """
    Train a lightweight visual model on expression features.
    This model can be loaded in app.py and used with face-api.js data.
    """
    from imblearn.over_sampling import SMOTE

    logger.info("Training browser-compatible visual model...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE for class imbalance
    try:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(3, sum(y == 1) - 1))
        X_res, y_res = smote.fit_resample(X_scaled, y)
        logger.info(f"  SMOTE: {len(y)} → {len(y_res)} samples")
    except Exception:
        X_res, y_res = X_scaled, y

    model = LogisticRegression(
        C=0.5, penalty='l2', class_weight='balanced',
        max_iter=5000, random_state=RANDOM_STATE
    )
    model.fit(X_res, y_res)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, 'visual_browser_model.pkl'))
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'visual_browser_scaler.pkl'))
        logger.info(f"  ✅ Browser visual model saved → {MODELS_DIR}/visual_browser_model.pkl")

    return model, scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    labels = pd.read_csv(os.path.join(FEATURES_DIR, 'master_labels.csv'))
    df = build_browser_visual_features(labels['pid'].tolist())

    # Train browser model
    feat_cols = [c for c in df.columns if c != 'pid']
    merged = df.merge(labels[['pid', 'label']], on='pid')
    X = merged[feat_cols].values
    y = merged['label'].values
    train_browser_visual_model(X, y)
