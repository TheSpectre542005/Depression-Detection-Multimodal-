#!/usr/bin/env python3
"""
Diagnostic script for Depression Detection data quality.
Run this to check if E-DAIC data is properly loaded and identify issues.
"""
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import DATA_ROOT, LABELS_DIR, FEATURES_DIR


def check_labels():
    """Check if labels can be loaded."""
    print("=" * 60)
    print("Checking Labels")
    print("=" * 60)

    try:
        from src.load_labels import build_master_labels
        labels = build_master_labels()
        print(f"✅ Labels loaded: {len(labels)} participants")
        print(f"   Depressed (PHQ-8 >= 10): {labels['label'].sum()}")
        print(f"   Not Depressed: {(labels['label'] == 0).sum()}")
        print(f"   Depression rate: {labels['label'].mean():.1%}")
        return labels['pid'].tolist()
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return []


def check_transcripts(pids):
    """Check transcript availability."""
    print("\n" + "=" * 60)
    print("Checking Transcripts")
    print("=" * 60)

    found = 0
    missing = []

    for pid in pids[:10]:  # Check first 10
        path = os.path.join(DATA_ROOT, f"{pid}_P", f"{pid}_Transcript.csv")
        if os.path.exists(path):
            found += 1
            try:
                df = pd.read_csv(path)
                print(f"✅ {pid}: {len(df)} rows, {df.shape[1]} cols")
            except Exception as e:
                print(f"⚠️  {pid}: File exists but error reading: {e}")
        else:
            missing.append(pid)

    if missing:
        print(f"\n❌ Missing transcripts for: {missing}")
    print(f"\nSummary: {found}/{min(10, len(pids))} transcripts found")


def check_audio_features(pids):
    """Check audio feature files availability and quality."""
    print("\n" + "=" * 60)
    print("Checking Audio Features (Detailed)")
    print("=" * 60)

    files_to_check = {
        'MFCC': 'OpenSMILE2.3.0_mfcc.csv',
        'eGeMAPS': 'OpenSMILE2.3.0_egemaps.csv',
        'BoAW_MFCC': 'BoAW_openSMILE_2.3.0_MFCC.csv',
        'BoAW_eGeMAPS': 'BoAW_openSMILE_2.3.0_eGeMAPS.csv',
    }

    stats = {k: {'found': 0, 'missing': 0, 'empty': 0, 'error': 0} for k in files_to_check}

    for pid in pids[:20]:  # Sample first 20
        feat_dir = os.path.join(DATA_ROOT, f"{pid}_P", "features")

        for feature_name, filename_template in files_to_check.items():
            path = os.path.join(feat_dir, f"{pid}_{filename_template}")

            if not os.path.exists(path):
                stats[feature_name]['missing'] += 1
                continue

            try:
                # Try to load
                if 'BoAW' in feature_name:
                    df = pd.read_csv(path, header=None)
                else:
                    df = pd.read_csv(path)

                if df.empty:
                    stats[feature_name]['empty'] += 1
                else:
                    stats[feature_name]['found'] += 1
                    # Print details for first found file
                    if stats[feature_name]['found'] == 1:
                        print(f"\n📄 {feature_name} example ({pid}):")
                        print(f"   Path: {path}")
                        print(f"   Shape: {df.shape}")
                        print(f"   Columns (first 5): {list(df.columns)[:5] if hasattr(df, 'columns') else 'N/A'}")
            except Exception as e:
                stats[feature_name]['error'] += 1

    print("\n" + "-" * 60)
    print("Audio Feature Summary (sample of 20):")
    print("-" * 60)
    for feature_name, stat in stats.items():
        total = stat['found'] + stat['missing'] + stat['empty'] + stat['error']
        if total > 0:
            print(f"{feature_name:15} | Found: {stat['found']:3} | Missing: {stat['missing']:3} | "
                  f"Empty: {stat['empty']:3} | Error: {stat['error']:3}")

    # Diagnosis
    print("\n" + "=" * 60)
    print("Diagnosis")
    print("=" * 60)
    mfcc_ok = stats['MFCC']['found'] > 10
    egemaps_ok = stats['eGeMAPS']['found'] > 10

    if not mfcc_ok and not egemaps_ok:
        print("❌ CRITICAL: Audio feature files not found or unreadable")
        print(f"   Expected data root: {DATA_ROOT}")
        print("   Check that E-DAIC data is downloaded and path is correct")
    elif stats['MFCC']['missing'] > 15:
        print("⚠️  WARNING: Most MFCC files missing - audio model will perform poorly")
    else:
        print("✅ Audio files appear to be present")

    return stats


def check_visual_features(pids):
    """Check visual feature files."""
    print("\n" + "=" * 60)
    print("Checking Visual Features")
    print("=" * 60)

    files_to_check = {
        'OpenFace': 'OpenFace2.1.0_Pose_gaze_AUs.csv',
        'DenseNet': 'densenet201.csv',
        'VGG16': 'vgg16.csv',
        'ResNet': 'CNN_ResNet.mat.csv',
    }

    stats = {k: {'found': 0, 'missing': 0} for k in files_to_check}

    for pid in pids[:10]:
        feat_dir = os.path.join(DATA_ROOT, f"{pid}_P", "features")

        for feature_name, filename in files_to_check.items():
            path = os.path.join(feat_dir, f"{pid}_{filename}")
            if os.path.exists(path):
                stats[feature_name]['found'] += 1
            else:
                stats[feature_name]['missing'] += 1

    for feature_name, stat in stats.items():
        total = stat['found'] + stat['missing']
        if total > 0:
            pct = stat['found'] / total * 100
            status = "✅" if pct > 50 else "⚠️"
            print(f"{status} {feature_name:12} | Found: {stat['found']}/{total} ({pct:.0f}%)")


def test_feature_extraction(pids):
    """Test actual feature extraction on a sample."""
    print("\n" + "=" * 60)
    print("Testing Feature Extraction")
    print("=" * 60)

    sample_pids = pids[:3] if len(pids) >= 3 else pids

    # Test text features
    print("\n1. Text Features:")
    try:
        from src.text_features import extract_text_features
        df = extract_text_features(sample_pids)
        print(f"   ✅ Extracted: {df.shape}")
        if not df.empty:
            print(f"   Columns: {list(df.columns)[:5]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test audio features
    print("\n2. Audio Features:")
    try:
        from src.audio_features_enhanced import build_audio_features_enhanced
        df = build_audio_features_enhanced(sample_pids)
        print(f"   ✅ Extracted: {df.shape}")
        if df.shape[1] < 10:
            print(f"   ⚠️  WARNING: Very few features extracted ({df.shape[1]})")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test visual features
    print("\n3. Visual Features:")
    try:
        from src.visual_features_enhanced import build_visual_features_enhanced
        df = build_visual_features_enhanced(sample_pids)
        print(f"   ✅ Extracted: {df.shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def main():
    print("\n" + "=" * 60)
    print("MindScan Depression Detection - Data Diagnostics")
    print("=" * 60)
    print(f"Data Root: {DATA_ROOT}")
    print(f"Labels Dir: {LABELS_DIR}")

    # Check if paths exist
    if not os.path.exists(DATA_ROOT):
        print(f"\n❌ CRITICAL: Data root does not exist!")
        print(f"   Create it or set EDAIC_DATA_ROOT environment variable")
        print(f"\n   Current path: {DATA_ROOT}")
        print(f"\n   To fix:")
        print(f"   1. Download E-DAIC dataset")
        print(f"   2. Extract to ~/Downloads/E-DAIC/")
        print(f"   3. Or set EDAIC_DATA_ROOT env var to correct path")
        return 1

    # Run diagnostics
    pids = check_labels()
    if not pids:
        print("\n❌ Cannot continue without labels")
        return 1

    check_transcripts(pids)
    audio_stats = check_audio_features(pids)
    check_visual_features(pids)
    test_feature_extraction(pids)

    # Summary
    print("\n" + "=" * 60)
    print("Summary & Recommendations")
    print("=" * 60)

    issues = []
    if audio_stats['MFCC']['found'] < 5:
        issues.append("Audio MFCC files missing - audio model will fail")
    if audio_stats['eGeMAPS']['found'] < 5:
        issues.append("Audio eGeMAPS files missing")

    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nRecommendations:")
        print("   1. Verify E-DAIC dataset is complete")
        print("   2. Check DATA_ROOT path in config.py")
        print("   3. Consider disabling audio modality if data unavailable")
    else:
        print("✅ Data appears to be available")
        print("\nNext steps:")
        print("   1. Run: python main.py")
        print("   2. Check results/ directory for outputs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
