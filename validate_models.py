#!/usr/bin/env python3
"""
Quick validation script to check if trained models meet clinical thresholds.
Run after training to verify model performance.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, recall_score, precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import MODELS_DIR


def check_model_files():
    """Check if model files exist."""
    print("=" * 60)
    print("Checking Model Files")
    print("=" * 60)

    required_files = [
        'text_model.pkl',
        'text_scaler.pkl',
        'text_tfidf.pkl',
    ]

    optional_files = [
        'audio_model.pkl',
        'audio_scaler.pkl',
        'visual_model.pkl',
        'visual_scaler.pkl',
    ]

    all_ok = True
    for f in required_files:
        path = os.path.join(MODELS_DIR, f)
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {f}")
        if not exists:
            all_ok = False

    print("\nOptional files:")
    for f in optional_files:
        path = os.path.join(MODELS_DIR, f)
        exists = os.path.exists(path)
        status = "✅" if exists else "⚠️"
        print(f"{status} {f}")

    return all_ok


def validate_model_performance():
    """Validate models meet minimum clinical thresholds."""
    print("\n" + "=" * 60)
    print("Validating Model Performance")
    print("=" * 60)

    # Load results if they exist
    results_path = 'results/all_results.csv'
    if not os.path.exists(results_path):
        print("❌ No results file found. Run main.py first.")
        return False

    results = pd.read_csv(results_path)
    print(f"\nLoaded results for {len(results)} models:\n")
    print(results.to_string(index=False))

    # Clinical thresholds
    MIN_AUC = 0.70
    MIN_SENSITIVITY = 0.70
    MIN_F1 = 0.60

    print("\n" + "-" * 60)
    print("Clinical Threshold Check")
    print("-" * 60)
    print(f"Minimum AUC-ROC:      {MIN_AUC}")
    print(f"Minimum Sensitivity:  {MIN_SENSITIVITY}")
    print(f"Minimum F1-Score:     {MIN_F1}")
    print("-" * 60)

    all_pass = True
    for _, row in results.iterrows():
        model_name = row['Model']
        auc = row.get('AUC-ROC', 0)
        sens = row.get('Sensitivity', row.get('Recall', 0))
        f1 = row.get('F1', 0)

        checks = []
        if auc < MIN_AUC:
            checks.append(f"AUC {auc:.3f} < {MIN_AUC}")
        if sens < MIN_SENSITIVITY:
            checks.append(f"Sens {sens:.3f} < {MIN_SENSITIVITY}")
        if f1 < MIN_F1:
            checks.append(f"F1 {f1:.3f} < {MIN_F1}")

        if checks:
            print(f"❌ {model_name:20} | FAILED: {', '.join(checks)}")
            all_pass = False
        else:
            print(f"✅ {model_name:20} | PASSED all thresholds")

    return all_pass


def generate_recommendations():
    """Generate recommendations based on results."""
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)

    results_path = 'results/all_results.csv'
    if not os.path.exists(results_path):
        print("No results to analyze.")
        return

    results = pd.read_csv(results_path)

    # Find best model
    best_auc = results.loc[results['AUC-ROC'].idxmax()]
    best_f1 = results.loc[results['F1'].idxmax()]

    print(f"\nBest AUC-ROC: {best_auc['Model']} ({best_auc['AUC-ROC']:.3f})")
    print(f"Best F1:      {best_f1['Model']} ({best_f1['F1']:.3f})")

    # Check for common issues
    print("\nAnalysis:")

    # Check if fusion helps
    fusion_models = results[results['Model'].str.contains('Fusion')]
    if not fusion_models.empty:
        best_fusion = fusion_models.loc[fusion_models['AUC-ROC'].idxmax()]
        unimodal = results[~results['Model'].str.contains('Fusion')]
        best_unimodal = unimodal.loc[unimodal['AUC-ROC'].idxmax()]

        if best_fusion['AUC-ROC'] <= best_unimodal['AUC-ROC']:
            print("⚠️  Fusion is not improving over best unimodal model")
            print("   Recommendation: Debug fusion weights or disable audio")

    # Check audio performance
    audio_row = results[results['Model'] == 'Audio Only']
    if not audio_row.empty:
        audio_auc = audio_row.iloc[0]['AUC-ROC']
        if audio_auc < 0.55:
            print(f"⚠️  Audio model performing poorly (AUC={audio_auc:.3f})")
            print("   Recommendation: Check audio data quality or exclude from fusion")

    # Check for low recall
    low_recall = results[results['Recall'] < 0.50]
    if not low_recall.empty:
        print("⚠️  Some models have low recall (may miss depressed patients)")
        print("   Recommendation: Use cost-sensitive thresholding (FN weight=5)")


def main():
    print("\n" + "=" * 60)
    print("MindScan Model Validation")
    print("=" * 60)

    files_ok = check_model_files()
    if not files_ok:
        print("\n❌ Required model files missing. Run main.py to train models.")
        return 1

    performance_ok = validate_model_performance()
    generate_recommendations()

    print("\n" + "=" * 60)
    if performance_ok:
        print("✅ All models meet clinical thresholds")
    else:
        print("⚠️  Some models below clinical thresholds")
        print("   Review ACTION_PLAN.md for improvement steps")
    print("=" * 60)

    return 0 if performance_ok else 1


if __name__ == "__main__":
    sys.exit(main())
