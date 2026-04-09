# src/visual_features_enhanced.py
"""
Enhanced visual feature extraction using CNN deep learning features
from ResNet, VGG16, and DenseNet201 pre-extracted from video frames.
"""
import pandas as pd
import numpy as np
import os
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_ROOT, FEATURES_DIR

logger = logging.getLogger(__name__)

SAVE_PATH = os.path.join(FEATURES_DIR, "visual_features_enhanced.csv")


def aggregate_temporal_features(df, feature_prefix, aggregate_cols=True):
    """
    Aggregate frame-level CNN features using multiple strategies.
    Returns statistical summaries that capture temporal patterns.
    """
    # Get feature columns (exclude name, timeStamp)
    feature_cols = [c for c in df.columns if c not in ['name', 'timeStamp']]

    if not feature_cols:
        return None

    features = df[feature_cols].select_dtypes(include=[np.number])

    if features.empty:
        return None

    record = {}

    # For each feature dimension, compute temporal statistics
    for col in features.columns:
        vals = features[col].dropna()
        if len(vals) == 0:
            continue

        # Basic statistics
        record[f'{col}_mean'] = vals.mean()
        record[f'{col}_std'] = vals.std() if len(vals) > 1 else 0
        record[f'{col}_min'] = vals.min()
        record[f'{col}_max'] = vals.max()
        record[f'{col}_median'] = vals.median()

        # Temporal dynamics (rate of change)
        if len(vals) > 1:
            diffs = vals.diff().dropna()
            record[f'{col}_diff_mean'] = diffs.mean()
            record[f'{col}_diff_std'] = diffs.std() if len(diffs) > 1 else 0

    return record


def extract_cnn_features(pid, data_root):
    """
    Extract CNN features from multiple architectures.
    Uses ResNet, VGG16, and DenseNet201 features.
    """
    feat_dir = os.path.join(data_root, f"{pid}_P", "features")

    cnn_files = {
        'densenet': f"{pid}_densenet201.csv",
        'vgg16': f"{pid}_vgg16.csv",
        'vgg': f"{pid}_CNN_VGG.mat.csv" if os.path.exists(os.path.join(feat_dir, f"{pid}_CNN_VGG.mat.csv")) else f"{pid}_CNN_ResNet.mat.csv",
        'resnet': f"{pid}_CNN_ResNet.mat.csv",
    }

    all_features = {}

    for cnn_name, filename in cnn_files.items():
        path = os.path.join(feat_dir, filename)
        if not os.path.exists(path):
            continue

        try:
            # Handle both .csv and .mat.csv files
            if filename.endswith('.mat.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path)

            if df.empty:
                continue

            # Aggregate temporal features
            agg = aggregate_temporal_features(df, cnn_name)
            if agg:
                # Prefix with cnn type
                for k, v in agg.items():
                    all_features[f'{cnn_name}_{k}'] = v

        except Exception as e:
            logger.debug(f"Error loading {cnn_name} for {pid}: {e}")

    return all_features


def extract_openface_enhanced(pid, data_root):
    """
    Enhanced OpenFace feature extraction with more sophisticated aggregation.
    """
    path = os.path.join(data_root, f"{pid}_P", "features",
                        f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")

    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)

        # High confidence filtering
        if 'confidence' in df.columns:
            df = df[df['confidence'] >= 0.85]
        if 'success' in df.columns:
            df = df[df['success'] == 1]

        if df.empty:
            return {}

        # Feature groups
        au_r_cols = [c for c in df.columns if 'AU' in c and '_r' in c]
        au_c_cols = [c for c in df.columns if 'AU' in c and '_c' in c]
        pose_cols = [c for c in df.columns if 'pose_' in c]
        gaze_cols = [c for c in df.columns if 'gaze_' in c]

        features = {}

        # Action Units - enhanced statistics
        for col in au_r_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                features[f'{col}_mean'] = vals.mean()
                features[f'{col}_std'] = vals.std() if len(vals) > 1 else 0
                features[f'{col}_max'] = vals.max()
                # Percentage of time AU is active (> threshold)
                features[f'{col}_active_pct'] = (vals > 0.5).mean()

        # AU presence (binary)
        for col in au_c_cols:
            if col in df.columns:
                features[f'{col}_presence'] = df[col].mean()

        # Pose - head movement patterns
        for col in pose_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                features[f'{col}_mean'] = vals.mean()
                features[f'{col}_std'] = vals.std() if len(vals) > 1 else 0
                features[f'{col}_range'] = vals.max() - vals.min()

        # Gaze - eye movement patterns
        for col in gaze_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                features[f'{col}_mean'] = vals.mean()
                features[f'{col}_std'] = vals.std() if len(vals) > 1 else 0

        # Derived depression-relevant features
        if 'AU12_r' in df.columns and 'AU06_r' in df.columns:
            # Smile symmetry (relevant for depression)
            features['smile_intensity'] = df[['AU12_r', 'AU06_r']].mean(axis=1).mean()

        if 'AU01_r' in df.columns and 'AU04_r' in df.columns:
            # Inner brow raise (sadness indicator)
            features['sadness_indicator'] = (df['AU01_r'] + df['AU04_r']).mean()

        if 'pose_Rx' in df.columns and 'pose_Ry' in df.columns:
            # Head movement variability
            features['head_movement_var'] = np.sqrt(df['pose_Rx']**2 + df['pose_Ry']**2).std()

        return features

    except Exception as e:
        logger.warning(f"Error extracting OpenFace features for {pid}: {e}")
        return {}


def build_visual_features_enhanced(participant_ids):
    """
    Build enhanced visual features using CNN + OpenFace features.
    """
    logger.info("Extracting ENHANCED visual features...")
    logger.info("  Using CNN features: DenseNet201, VGG16, ResNet")
    logger.info("  Using enhanced OpenFace features")

    records = []
    missing = []

    for pid in participant_ids:
        record = {'pid': pid}

        # Extract CNN features
        cnn_feats = extract_cnn_features(pid, DATA_ROOT)
        if cnn_feats:
            record.update(cnn_feats)
        else:
            missing.append(f"{pid}_cnn")

        # Extract enhanced OpenFace features
        of_feats = extract_openface_enhanced(pid, DATA_ROOT)
        if of_feats:
            record.update(of_feats)
        else:
            missing.append(f"{pid}_openface")

        if len(record) > 1:  # More than just pid
            records.append(record)
        else:
            missing.append(pid)

    if missing:
        logger.info(f"Missing visual features for {len(missing)} participants")

    df = pd.DataFrame(records).fillna(0)

    # Remove constant features
    constant_cols = [c for c in df.columns if c != 'pid' and df[c].nunique() <= 1]
    if constant_cols:
        logger.info(f"  Removing {len(constant_cols)} constant features")
        df = df.drop(columns=constant_cols)

    # Remove highly correlated features (>0.99)
    corr_matrix = df.drop('pid', axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
    if to_drop:
        logger.info(f"  Removing {len(to_drop)} highly correlated features")
        df = df.drop(columns=to_drop)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    logger.info(f"  ✅ Enhanced visual features saved → {SAVE_PATH}")
    logger.info(f"  Shape: {df.shape}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    labels = pd.read_csv(os.path.join(FEATURES_DIR, 'master_labels.csv'))
    build_visual_features_enhanced(labels['pid'].tolist())
