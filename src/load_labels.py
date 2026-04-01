# src/load_labels.py
import pandas as pd
import os, logging

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import LABELS_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)

def load_labels():
    train = pd.read_csv(os.path.join(LABELS_DIR, "train_split.csv"))
    dev   = pd.read_csv(os.path.join(LABELS_DIR, "dev_split.csv"))

    logger.info("=== TRAIN SPLIT ===")
    logger.info(f"Columns: {train.columns.tolist()}")
    logger.debug(f"\n{train.head(3)}")

    logger.info("=== DEV SPLIT ===")
    logger.info(f"Columns: {dev.columns.tolist()}")
    logger.debug(f"\n{dev.head(3)}")

    return train, dev

def build_master_labels(save=True):
    train, dev = load_labels()
    df = pd.concat([train, dev], ignore_index=True)

    # ── Auto-detect column names ──────────────────────────
    pid_col = None
    for candidate in ['Participant_ID','participant_id','ID','id','SubjectID']:
        if candidate in df.columns:
            pid_col = candidate
            break

    score_col = None
    for candidate in ['PHQ_Score','PHQ8_Score','phq_score','PHQ8','phq8','Score']:
        if candidate in df.columns:
            score_col = candidate
            break

    if pid_col is None or score_col is None:
        logger.error(f"Could not auto-detect columns! All columns: {df.columns.tolist()}")
        return None

    logger.info(f"Using: pid_col='{pid_col}', score_col='{score_col}'")

    # Binary label: PHQ >= 10 → Depressed
    df['label'] = (df[score_col] >= 10).astype(int)
    df = df[[pid_col, score_col, 'label']].rename(columns={pid_col: 'pid'})
    df['pid'] = df['pid'].astype(int)

    logger.info(f"Class Distribution:\n{df['label'].value_counts()}")
    logger.info(f"Total participants: {len(df)}")
    logger.info(f"Depressed (1): {df['label'].sum()}")
    logger.info(f"Not Depressed (0): {(df['label']==0).sum()}")
    logger.info(f"Imbalance ratio: {df['label'].mean():.2%} positive")

    if save:
        os.makedirs(FEATURES_DIR, exist_ok=True)
        df.to_csv(os.path.join(FEATURES_DIR, 'master_labels.csv'), index=False)
        logger.info(f"✅ Saved → {FEATURES_DIR}/master_labels.csv")

    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_master_labels()