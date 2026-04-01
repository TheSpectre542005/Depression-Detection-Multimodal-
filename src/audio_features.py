# src/audio_features.py
import pandas as pd
import numpy as np
import os, logging

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_ROOT, FEATURES_DIR

logger = logging.getLogger(__name__)

SAVE_PATH = os.path.join(FEATURES_DIR, "audio_features.csv")

def load_semicolon_csv(path):
    """Load openSMILE files which use semicolon separator."""
    try:
        df = pd.read_csv(path, sep=';')

        # Drop name and frameTime columns
        drop_cols = [c for c in df.columns
                     if c in ['name', 'frameTime'] or 'unknown' in str(c).lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])
        return df
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return None

def aggregate(df):
    """Convert frame-level features → single vector (mean + std + min + max)."""
    if df is None or df.empty:
        return None
    means = df.mean().values
    stds  = df.std().fillna(0).values
    mins  = df.min().values
    maxs  = df.max().values
    return np.concatenate([means, stds, mins, maxs])

def build_audio_features(participant_ids):
    logger.info("Extracting audio features...")
    records = []
    missing = []

    for pid in participant_ids:
        feat_dir  = os.path.join(DATA_ROOT, f"{pid}_P", "features")
        mfcc_path = os.path.join(feat_dir, f"{pid}_OpenSMILE2.3.0_mfcc.csv")
        egem_path = os.path.join(feat_dir, f"{pid}_OpenSMILE2.3.0_egemaps.csv")

        mfcc_df = load_semicolon_csv(mfcc_path)
        egem_df = load_semicolon_csv(egem_path)

        if mfcc_df is None and egem_df is None:
            missing.append(pid)
            continue

        record = {'pid': pid}

        # MFCC aggregated features
        mfcc_vec = aggregate(mfcc_df)
        if mfcc_vec is not None:
            for i, v in enumerate(mfcc_vec):
                record[f'mfcc_{i}'] = v

        # eGeMAPS aggregated features
        egem_vec = aggregate(egem_df)
        if egem_vec is not None:
            for i, v in enumerate(egem_vec):
                record[f'egemap_{i}'] = v

        records.append(record)

    if missing:
        logger.info(f"Missing audio features: {len(missing)} participants")

    df = pd.DataFrame(records).fillna(0)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    logger.info(f"  ✅ Audio features saved → {SAVE_PATH}")
    logger.info(f"  Shape: {df.shape}")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    labels = pd.read_csv(os.path.join(FEATURES_DIR, 'master_labels.csv'))
    build_audio_features(labels['pid'].tolist())