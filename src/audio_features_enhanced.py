# src/audio_features_enhanced.py
"""
Enhanced audio feature extraction for depression detection.

Focused on clinically-validated acoustic biomarkers with controlled
feature dimensionality to prevent overfitting (target: 150-250 features
for N≈219 samples).

Architecture:
  Tier 1 — Prosodic biomarkers (F0, jitter, shimmer, HNR, loudness, pauses)
  Tier 2 — Compact MFCC statistics (mean + std + range + skew for 13 coefficients)
  Tier 3 — Compact eGeMAPS statistics (mean + std only, deduplicated)

Previous version produced 1219 features → AUC 0.5478 (overfitting).
This version targets ~150-250 features → better generalization.
"""
import pandas as pd
import numpy as np
import os
import logging
from scipy import stats as sp_stats

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_ROOT, FEATURES_DIR

logger = logging.getLogger(__name__)

SAVE_PATH = os.path.join(FEATURES_DIR, "audio_features_enhanced.csv")

# ── Data quality tracking ──────────────────────────────────────────
data_quality_log = {
    'missing_files': [],
    'empty_files': [],
    'load_errors': [],
    'good_participants': 0,
    'poor_participants': 0,
}

# ── Depression-relevant eGeMAPS column keywords ────────────────────
# These are the features most predictive of depression in literature
DEPRESSION_EGEMAP_KEYWORDS = [
    'f0',  'pitch',                        # Fundamental frequency
    'loudness', 'energy', 'rms',           # Energy / loudness
    'jitter',                              # Pitch perturbation
    'shimmer',                             # Amplitude perturbation
    'hnr', 'HNR',                          # Harmonics-to-noise ratio
    'spectral', 'slope', 'flux',           # Spectral shape
    'alpha',                               # Alpha ratio (spectral)
    'hammarberg',                          # Hammarberg index
    'formant', 'F1', 'F2', 'F3',           # Formant frequencies
    'mfcc',                                # MFCCs in eGeMAPS
]


def load_csv_robust(path, pid=""):
    """
    Load CSV handling semicolon (OpenSMILE default) and comma delimiters.
    Returns numeric-only DataFrame with metadata columns dropped,
    or None on failure.
    """
    if not os.path.exists(path):
        if pid:
            data_quality_log['missing_files'].append(f"{pid}:{os.path.basename(path)}")
        return None

    try:
        size = os.path.getsize(path)
        if size == 0:
            if pid:
                data_quality_log['empty_files'].append(f"{pid}:{os.path.basename(path)}")
            return None

        # Try semicolon first (OpenSMILE default)
        df = pd.read_csv(path, sep=';')
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=',')
        if df.shape[1] <= 1:
            logger.warning(f"[{pid}] Single column after both delimiters: {path}")
            return None

        # Drop metadata columns
        drop_cols = [c for c in df.columns
                     if c in ['name', 'frameTime'] or 'unknown' in str(c).lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        # Keep only numeric
        df = df.select_dtypes(include=[np.number])
        if df.empty:
            return None

        return df

    except pd.errors.EmptyDataError:
        if pid:
            data_quality_log['empty_files'].append(f"{pid}:{os.path.basename(path)}")
        return None
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        if pid:
            data_quality_log['load_errors'].append(f"{pid}:{str(e)[:60]}")
        return None


def compact_stats(values, prefix):
    """
    Compute a compact set of 6 summary statistics for a time series.
    Avoids the previous 18-stat explosion that caused overfitting.
    """
    feats = {}
    if len(values) == 0:
        return feats

    feats[f'{prefix}_mean'] = np.mean(values)
    feats[f'{prefix}_std'] = np.std(values) if len(values) > 1 else 0
    feats[f'{prefix}_range'] = np.ptp(values)  # max - min
    feats[f'{prefix}_median'] = np.median(values)

    if len(values) > 3:
        feats[f'{prefix}_skew'] = float(sp_stats.skew(values))
        feats[f'{prefix}_iqr'] = float(np.percentile(values, 75) - np.percentile(values, 25))

    return feats


def compute_delta_stats(values, prefix):
    """Compute delta (velocity) statistics — just mean and std."""
    if len(values) < 4:
        return {}
    delta = np.gradient(values)
    return {
        f'{prefix}_delta_mean': np.mean(np.abs(delta)),
        f'{prefix}_delta_std': np.std(delta),
    }


# ── TIER 1: Clinical Prosodic Biomarkers (~30 features) ───────────

def extract_prosodic_biomarkers(pid, data_root):
    """
    Extract depression-specific prosodic features from eGeMAPS.

    Clinical evidence:
    - Reduced F0 variability → depression indicator
    - Increased jitter/shimmer → voice instability
    - Lower HNR → breathier voice quality
    - Reduced loudness range → flat affect
    - More pauses, slower speech → psychomotor retardation
    """
    feat_dir = os.path.join(data_root, f"{pid}_P", "features")
    path = os.path.join(feat_dir, f"{pid}_OpenSMILE2.3.0_egemaps.csv")
    features = {}

    df = load_csv_robust(path, pid)
    if df is None or df.empty:
        return features

    # ── F0 / Pitch ──
    pitch_cols = [c for c in df.columns
                  if any(x in str(c).lower() for x in ['f0', 'pitch'])]
    if pitch_cols:
        pitch = df[pitch_cols[0]].dropna().values
        if len(pitch) > 0:
            features.update(compact_stats(pitch, 'prosody_f0'))
            features.update(compute_delta_stats(pitch, 'prosody_f0'))

            # Voiced ratio (F0 > 0 = voiced)
            voiced = pitch[pitch > 0]
            features['prosody_voiced_ratio'] = len(voiced) / max(len(pitch), 1)
            if len(voiced) > 1:
                features['prosody_voiced_f0_mean'] = np.mean(voiced)
                features['prosody_voiced_f0_std'] = np.std(voiced)
                if np.mean(voiced) > 1e-10:
                    features['prosody_f0_cv'] = np.std(voiced) / np.mean(voiced)

    # ── Loudness / Energy ──
    energy_cols = [c for c in df.columns
                   if any(x in str(c).lower() for x in ['loudness', 'energy', 'rms'])]
    if energy_cols:
        energy = df[energy_cols[0]].dropna().values
        if len(energy) > 0:
            features.update(compact_stats(energy, 'prosody_energy'))
            features.update(compute_delta_stats(energy, 'prosody_energy'))

    # ── Jitter (pitch perturbation — elevated in depression) ──
    jitter_cols = [c for c in df.columns if 'jitter' in str(c).lower()]
    for col in jitter_cols[:2]:  # Limit to 2 jitter types
        data = df[col].dropna().values
        if len(data) > 0:
            col_clean = str(col).replace(' ', '_')[:20]
            features[f'prosody_jitter_{col_clean}_mean'] = np.mean(data)
            features[f'prosody_jitter_{col_clean}_std'] = np.std(data) if len(data) > 1 else 0

    # ── Shimmer (amplitude perturbation — elevated in depression) ──
    shimmer_cols = [c for c in df.columns if 'shimmer' in str(c).lower()]
    for col in shimmer_cols[:2]:
        data = df[col].dropna().values
        if len(data) > 0:
            col_clean = str(col).replace(' ', '_')[:20]
            features[f'prosody_shimmer_{col_clean}_mean'] = np.mean(data)
            features[f'prosody_shimmer_{col_clean}_std'] = np.std(data) if len(data) > 1 else 0

    # ── HNR (harmonics-to-noise — lower in depression) ──
    hnr_cols = [c for c in df.columns
                if 'hnr' in str(c).lower() or 'HNR' in str(c)]
    if hnr_cols:
        hnr = df[hnr_cols[0]].dropna().values
        if len(hnr) > 0:
            features.update(compact_stats(hnr, 'prosody_hnr'))

    return features


# ── TIER 2: Compact MFCC Features (~65 features) ─────────────────

def extract_mfcc_compact(pid, data_root):
    """
    Extract compact MFCC features — 6 stats per coefficient.
    MFCCs 1-13 only (standard), with delta stats for first 5.
    Previous version: ~300+ MFCC features. Now: ~65.
    """
    feat_dir = os.path.join(data_root, f"{pid}_P", "features")
    path = os.path.join(feat_dir, f"{pid}_OpenSMILE2.3.0_mfcc.csv")
    features = {}

    df = load_csv_robust(path, pid)
    if df is None or df.empty:
        return features

    n_cols = min(df.shape[1], 13)  # Standard 13 MFCCs

    for i in range(n_cols):
        col_data = df.iloc[:, i].dropna().values
        if len(col_data) == 0 or np.std(col_data) < 1e-10:
            continue

        features.update(compact_stats(col_data, f'mfcc_{i}'))

        # Delta stats only for first 5 MFCCs (most informative)
        if i < 5:
            features.update(compute_delta_stats(col_data, f'mfcc_{i}'))

    # ── Global MFCC dynamics ──
    if df.shape[0] > 1 and df.shape[1] > 0:
        # Energy proxy (first MFCC coefficient)
        energy = df.iloc[:, 0].dropna().values
        if len(energy) > 3:
            # Energy trend (declining energy → fatigue/depression)
            t = np.arange(len(energy))
            try:
                features['mfcc_energy_trend'] = np.polyfit(t, energy, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                pass

        # Overall MFCC variability (low variability → monotone voice)
        cv_vals = []
        for i in range(n_cols):
            col_data = df.iloc[:, i].dropna().values
            if len(col_data) > 1 and abs(np.mean(col_data)) > 1e-10:
                cv_vals.append(np.std(col_data) / abs(np.mean(col_data)))
        if cv_vals:
            features['mfcc_cv_mean'] = np.mean(cv_vals)

    return features


# ── TIER 3: Compact eGeMAPS Features (~80-120 features) ───────────

def extract_egemaps_compact(pid, data_root):
    """
    Extract compact eGeMAPS features — focused on depression-relevant columns.
    Only mean + std per column (instead of 18 stats per column).
    Previous version: ~800+ eGeMAPS features. Now: ~80-120.
    """
    feat_dir = os.path.join(data_root, f"{pid}_P", "features")
    path = os.path.join(feat_dir, f"{pid}_OpenSMILE2.3.0_egemaps.csv")
    features = {}

    df = load_csv_robust(path, pid)
    if df is None or df.empty:
        return features

    # Filter to depression-relevant columns only
    relevant_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in DEPRESSION_EGEMAP_KEYWORDS):
            relevant_cols.append(col)

    # If no keyword matches, use all columns but limit count
    if not relevant_cols:
        relevant_cols = list(df.columns)[:30]
    else:
        relevant_cols = relevant_cols[:60]  # Cap at 60 columns

    for col in relevant_cols:
        data = df[col].dropna().values
        if len(data) == 0 or np.std(data) < 1e-10:
            continue

        col_clean = str(col).replace(' ', '_').replace('-', '_')[:25]

        # Just mean + std + range (3 features per column instead of 18)
        features[f'egemap_{col_clean}_mean'] = np.mean(data)
        features[f'egemap_{col_clean}_std'] = np.std(data) if len(data) > 1 else 0
        features[f'egemap_{col_clean}_range'] = np.ptp(data)

    # ── Speech dynamics ──
    if df.shape[0] > 1:
        # Speech activity proxy
        activity = (df.abs() > 0.01).any(axis=1).astype(int)
        features['egemap_speech_ratio'] = activity.mean()

        # Pause detection
        pauses = activity.diff().fillna(0)
        n_pauses = min((pauses == -1).sum(), (pauses == 1).sum())
        features['egemap_pause_count'] = n_pauses
        features['egemap_pause_rate'] = n_pauses / max(len(activity), 1)

    return features


# ── MAIN BUILDER ──────────────────────────────────────────────────

def build_audio_features_enhanced(participant_ids):
    """
    Build enhanced audio features with controlled dimensionality.

    Target: 150-250 features (vs previous 1219).
    Architecture:
      - Tier 1: Prosodic biomarkers (~30 features)
      - Tier 2: Compact MFCC (~65 features)
      - Tier 3: Compact eGeMAPS (~80-120 features)
    """
    logger.info("Extracting ENHANCED audio features (v2 — compact)...")
    logger.info("  Tier 1: Prosodic biomarkers (F0, jitter, shimmer, HNR)")
    logger.info("  Tier 2: Compact MFCC statistics (13 coefficients)")
    logger.info("  Tier 3: Compact eGeMAPS statistics (depression-relevant)")
    logger.info("  BoAW: SKIPPED (too noisy for N=%d)", len(participant_ids))

    records = []
    missing = []

    for pid in participant_ids:
        record = {'pid': pid}
        n_features_before = len(record)

        # Tier 1: Clinical prosodic biomarkers
        prosody = extract_prosodic_biomarkers(pid, DATA_ROOT)
        record.update(prosody)

        # Tier 2: Compact MFCC features
        mfcc = extract_mfcc_compact(pid, DATA_ROOT)
        record.update(mfcc)

        # Tier 3: Compact eGeMAPS features
        egemaps = extract_egemaps_compact(pid, DATA_ROOT)
        record.update(egemaps)

        if len(record) > 1:  # has features beyond 'pid'
            records.append(record)
            data_quality_log['good_participants'] += 1
        else:
            missing.append(pid)
            data_quality_log['poor_participants'] += 1

    if missing:
        logger.info(f"  Missing audio data: {len(missing)} participants")

    df = pd.DataFrame(records).fillna(0)

    # ── Post-processing: remove useless features ──
    feature_cols = [c for c in df.columns if c != 'pid']
    n_before = len(feature_cols)

    # 1. Remove constant features (variance == 0)
    constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
    if constant_cols:
        logger.info(f"  Removing {len(constant_cols)} constant features")
        df = df.drop(columns=constant_cols)

    # 2. Remove near-zero-variance features
    feature_cols = [c for c in df.columns if c != 'pid']
    low_var_cols = [c for c in feature_cols if df[c].var() < 1e-8]
    if low_var_cols:
        logger.info(f"  Removing {len(low_var_cols)} near-zero-variance features")
        df = df.drop(columns=low_var_cols)

    # 3. Remove highly correlated features (threshold 0.85 — stricter than before)
    feature_cols = [c for c in df.columns if c != 'pid']
    if len(feature_cols) > 1:
        try:
            corr_matrix = df[feature_cols].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
            if to_drop:
                logger.info(f"  Removing {len(to_drop)} highly correlated features (r>0.85)")
                df = df.drop(columns=to_drop)
        except Exception as e:
            logger.warning(f"  Correlation filtering failed: {e}")

    # 4. Replace infinities
    df = df.replace([np.inf, -np.inf], 0)

    n_after = len([c for c in df.columns if c != 'pid'])
    logger.info(f"  Feature reduction: {n_before} → {n_after}")

    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    logger.info(f"  ✅ Enhanced audio features saved → {SAVE_PATH}")
    logger.info(f"  Shape: {df.shape}")
    return df


def print_data_quality_report():
    """Print a summary of data quality issues encountered."""
    print("\n" + "=" * 60)
    print("Audio Feature Extraction — Data Quality Report (v2)")
    print("=" * 60)

    total_issues = (len(data_quality_log['missing_files']) +
                    len(data_quality_log['empty_files']) +
                    len(data_quality_log['load_errors']))

    print(f"\n  Good participants: {data_quality_log['good_participants']}")
    print(f"  Poor participants: {data_quality_log['poor_participants']}")

    if total_issues == 0:
        print("  ✅ No data quality issues detected")
        return

    print(f"\n  Total issues: {total_issues}")
    print(f"    Missing files: {len(data_quality_log['missing_files'])}")
    print(f"    Empty files:   {len(data_quality_log['empty_files'])}")
    print(f"    Load errors:   {len(data_quality_log['load_errors'])}")

    if data_quality_log['missing_files']:
        print("\n  Sample missing files:")
        for f in data_quality_log['missing_files'][:5]:
            print(f"    - {f}")

    if data_quality_log['load_errors']:
        print("\n  Sample load errors:")
        for e in data_quality_log['load_errors'][:5]:
            print(f"    - {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    labels = pd.read_csv(os.path.join(FEATURES_DIR, 'master_labels.csv'))
    build_audio_features_enhanced(labels['pid'].tolist())
    print_data_quality_report()
