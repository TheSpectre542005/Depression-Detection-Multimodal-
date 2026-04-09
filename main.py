# main.py
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score as sk_f1

from config import THRESHOLD_MIN, THRESHOLD_MAX

from src.load_labels     import build_master_labels
from src.text_features   import build_text_features
from src.audio_features_enhanced  import build_audio_features_enhanced
from src.visual_features_enhanced import build_visual_features_enhanced
from src.fusion          import (train_unimodal, late_fusion_predict,
                                  find_best_threshold, find_cost_sensitive_threshold,
                                  transform_with_selector, train_meta_learner)
from src.evaluate        import (evaluate, plot_roc_curves, plot_precision_recall_curves,
                                  plot_model_comparison, save_results_table,
                                  generate_summary_report, plot_calibration_curve)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

os.makedirs('data/features', exist_ok=True)
os.makedirs('models',        exist_ok=True)
os.makedirs('results',       exist_ok=True)

print("=" * 60)
print("   DEPRESSION DETECTION — ENHANCED MULTIMODAL PIPELINE")
print("=" * 60)

# ── 1. LABELS ──────────────────────────────────────────────────
logger.info("\n📋 Loading labels...")
labels = pd.read_csv('data/features/master_labels.csv')
pids   = labels['pid'].tolist()
logger.info(f"  Total: {len(pids)} | "
            f"Depressed: {labels['label'].sum()} | "
            f"Not Depressed: {(labels['label']==0).sum()}")

# ── 2. LOAD/EXTRACT FEATURES ───────────────────────────────────
logger.info("\n📦 Loading features...")
text_df   = pd.read_csv('data/features/text_features.csv')

# Use enhanced features — always re-extract to ensure latest feature engineering is used
if os.path.exists('data/features/audio_features_enhanced.csv'):
    logger.info("  Re-extracting enhanced audio features (to apply latest improvements)...")
os.makedirs('data/features', exist_ok=True)
audio_df = build_audio_features_enhanced(labels['pid'].tolist())

if os.path.exists('data/features/visual_features_enhanced.csv'):
    visual_df = pd.read_csv('data/features/visual_features_enhanced.csv')
else:
    logger.info("  Extracting enhanced visual features...")
    visual_df = build_visual_features_enhanced(labels['pid'].tolist())

logger.info(f"  Text   : {text_df.shape}")
logger.info(f"  Audio  : {audio_df.shape} (Enhanced with BoAW + Prosody)")
logger.info(f"  Visual : {visual_df.shape} (Enhanced with CNN)")

# ── 3. MERGE ───────────────────────────────────────────────────
logger.info("\n🔗 Merging all features...")
merged = labels.copy()
merged = merged.merge(text_df,   on='pid', how='inner')
merged = merged.merge(audio_df,  on='pid', how='inner')
merged = merged.merge(visual_df, on='pid', how='inner')
logger.info(f"  Merged shape: {merged.shape}")

# ── 4. FEATURE MATRICES ────────────────────────────────────────
# Text features (same as before)
text_cols   = [c for c in merged.columns
               if c.startswith(('sent_','word_','unique_','lexical_','avg_','tfidf_'))]

# Enhanced audio features (includes BoAW, prosody, and traditional)
audio_cols  = [c for c in merged.columns
               if c not in ['pid', 'label'] and c not in text_cols and c not in
               [col for col in merged.columns if any(x in col for x in ['densenet', 'vgg', 'resnet', 'AU', 'pose_', 'gaze_'])]]

# Enhanced visual features (includes CNN + OpenFace)
visual_cols = [c for c in merged.columns
               if any(x in c for x in ['densenet', 'vgg', 'resnet', 'AU', 'pose_', 'gaze_'])]

X_text   = merged[text_cols].fillna(0).values
X_audio  = merged[audio_cols].fillna(0).values
X_visual = merged[visual_cols].fillna(0).values
y        = merged['label'].values

logger.info(f"\n  Feature dimensions (raw):")
logger.info(f"    Text   : {X_text.shape}")
logger.info(f"    Audio  : {X_audio.shape}")
logger.info(f"    Visual : {X_visual.shape}")

# ── 5. TRAIN / VALIDATION / TEST SPLIT ────────────────────────
# 70% train | 15% validation (threshold tuning) | 15% test
idx = np.arange(len(y))
np.random.seed(42)

tr_val, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
tr, val    = train_test_split(tr_val, test_size=0.176, stratify=y[tr_val], random_state=42)
# 0.176 of 85% ≈ 15% of total

logger.info(f"\n  Train: {len(tr)} | Val: {len(val)} | Test: {len(te)}")
logger.info(f"  Test  → depressed: {y[te].sum()} | not depressed: {(y[te]==0).sum()}")

# ── 5b. PCA DIMENSIONALITY REDUCTION ─────────────────────────
# Fit PCA on training data only to prevent data leakage.
N_COMPONENTS_AUDIO  = min(40, X_audio.shape[1], len(tr) - 1)
N_COMPONENTS_VISUAL = min(25, X_visual.shape[1], len(tr) - 1)

pca_audio  = PCA(n_components=N_COMPONENTS_AUDIO,  random_state=42)
pca_visual = PCA(n_components=N_COMPONENTS_VISUAL, random_state=42)

# Fit on TRAIN only
pca_audio.fit(X_audio[tr])
pca_visual.fit(X_visual[tr])

# Transform all splits
X_audio_pca  = pca_audio.transform(X_audio)
X_visual_pca = pca_visual.transform(X_visual)

logger.info(f"\n  After PCA:")
logger.info(f"    Audio  : {X_audio.shape} → {X_audio_pca.shape}  "
            f"(explained var: {pca_audio.explained_variance_ratio_.sum():.1%})")
logger.info(f"    Visual : {X_visual.shape} → {X_visual_pca.shape}  "
            f"(explained var: {pca_visual.explained_variance_ratio_.sum():.1%})")

# Use PCA-reduced versions for audio and visual
X_audio  = X_audio_pca
X_visual = X_visual_pca

# ── 6. TRAIN UNIMODAL MODELS ──────────────────────────────────
logger.info("\n🤖 Training unimodal models (with model selection)...")

text_model,   text_artifacts   = train_unimodal(X_text[tr],   y[tr], 'text')
audio_model,  audio_artifacts  = train_unimodal(X_audio[tr],  y[tr], 'audio')
visual_model, visual_artifacts = train_unimodal(X_visual[tr], y[tr], 'visual')

# ── 7. FIND BEST THRESHOLD PER MODALITY (on validation set) ───
logger.info("\n🎯 Finding best decision thresholds on validation set...")
# Use cost-sensitive thresholding: FN costs 5x more than FP (missing depression is worse)
t_text   = find_cost_sensitive_threshold(text_model,   text_artifacts,   X_text[val],   y[val], fn_cost=5, fp_cost=1)
t_audio  = find_cost_sensitive_threshold(audio_model,  audio_artifacts,  X_audio[val],  y[val], fn_cost=5, fp_cost=1)
t_visual = find_cost_sensitive_threshold(visual_model, visual_artifacts, X_visual[val], y[val], fn_cost=5, fp_cost=1)

# ── 8. EVALUATE UNIMODAL ON TEST SET ──────────────────────────
logger.info("\n📊 Evaluating on test set...")
results  = []
roc_data = []
pr_data = []
val_aucs = {}

for name, model, artifacts, X, threshold, mod_key in [
    ('Text Only',   text_model,   text_artifacts,   X_text,   t_text,   'text'),
    ('Audio Only',  audio_model,  audio_artifacts,  X_audio,  t_audio,  'audio'),
    ('Visual Only', visual_model, visual_artifacts, X_visual, t_visual, 'visual'),
]:
    # Compute validation AUC for fusion weights
    X_val_selected = transform_with_selector(X[val], artifacts)
    X_val_sc = artifacts['scaler'].transform(X_val_selected)
    val_probs = model.predict_proba(X_val_sc)[:, 1]
    val_aucs[mod_key] = roc_auc_score(y[val], val_probs)

    # Test set evaluation
    X_te_selected = transform_with_selector(X[te], artifacts)
    X_te_sc = artifacts['scaler'].transform(X_te_selected)
    probs   = model.predict_proba(X_te_sc)[:, 1]
    preds   = (probs >= threshold).astype(int)
    logger.info(f"\n  [{name}] threshold={threshold:.2f} | "
                f"predicted depressed: {preds.sum()}/{len(preds)}")
    results.append(evaluate(y[te], preds, probs, name))
    roc_data.append({'name': name, 'y_true': y[te], 'y_prob': probs})
    pr_data.append({'name': name, 'y_true': y[te], 'y_prob': probs})

    # Calibration plot
    plot_calibration_curve(y[te], probs, name)

# ── 9. LATE FUSION WITH SMART WEIGHTS ─────────────────────────
logger.info("\n🔀 Running late fusion...")

models_scalers = {
    'text'  : (text_model,   text_artifacts),
    'audio' : (audio_model,  audio_artifacts),
    'visual': (visual_model, visual_artifacts),
}

X_val_dict = {'text': X_text[val], 'audio': X_audio[val], 'visual': X_visual[val]}
X_te_dict  = {'text': X_text[te],  'audio': X_audio[te],  'visual': X_visual[te]}

# Exclude modalities with very poor validation performance
MIN_AUC = 0.55
included = {m: auc for m, auc in val_aucs.items() if auc > MIN_AUC}
excluded = {m: auc for m, auc in val_aucs.items() if auc <= MIN_AUC}

if excluded:
    for m, auc in excluded.items():
        logger.info(f"  ⚠️  Excluding [{m}] from late fusion (val AUC={auc:.3f} ≤ {MIN_AUC})")

if not included:
    logger.info("  ⚠️  No modality above threshold — using all with equal weights")
    included = val_aucs.copy()

# Use validation AUC for weighting (AUC above random chance)
adjusted = {m: max(auc - 0.5, 0.01) for m, auc in included.items()}
total_adj = sum(adjusted.values())
weights = {m: a / total_adj for m, a in adjusted.items()}
for m in {'text', 'audio', 'visual'} - set(included.keys()):
    weights[m] = 0.0

logger.info(f"  Weights: " + ", ".join(
    f"{m}={weights[m]:.2f}" for m in ['text', 'audio', 'visual']))

# Find best threshold for fusion
best_ft, best_ff1 = 0.40, 0.0
for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX, 0.01):
    p, _ = late_fusion_predict(models_scalers, X_val_dict,
                               weights=weights, threshold=t)
    f1 = sk_f1(y[val], p, zero_division=0)
    if f1 > best_ff1:
        best_ff1 = f1
        best_ft  = t

logger.info(f"  Best threshold: {best_ft:.2f}  (Val F1={best_ff1:.3f})")

preds, probs = late_fusion_predict(
    models_scalers, X_te_dict, weights=weights, threshold=best_ft)
logger.info(f"  Predicted depressed: {preds.sum()}/{len(preds)}")
results.append(evaluate(y[te], preds, probs, 'Late Fusion'))
roc_data.append({'name': 'Late Fusion', 'y_true': y[te], 'y_prob': probs})
pr_data.append({'name': 'Late Fusion', 'y_true': y[te], 'y_prob': probs})

# ── 10. STACKING META-LEARNER FUSION ──────────────────────────
logger.info("\n🧠 Running stacking meta-learner fusion...")

# Build meta-features: predicted probabilities from each modality
def build_meta_features(models_scalers, X_dict):
    """Get predicted probabilities from each modality as meta-features."""
    meta = []
    for m in ['text', 'audio', 'visual']:
        if m in models_scalers:
            model, artifacts = models_scalers[m]
            X_selected = transform_with_selector(X_dict[m], artifacts)
            X_sc = artifacts['scaler'].transform(X_selected)
            meta.append(model.predict_proba(X_sc)[:, 1])
    return np.column_stack(meta)

X_tr_dict = {'text': X_text[tr], 'audio': X_audio[tr], 'visual': X_visual[tr]}
meta_train = build_meta_features(models_scalers, X_tr_dict)
meta_val   = build_meta_features(models_scalers, X_val_dict)
meta_test  = build_meta_features(models_scalers, X_te_dict)

# Train meta-learner
meta_model = train_meta_learner(models_scalers, X_tr_dict, y[tr])

# Find best threshold for stacking
stack_probs_val = meta_model.predict_proba(meta_val)[:, 1]
best_st, best_sf1 = 0.50, 0.0
for t in np.arange(0.25, 0.65, 0.01):
    p = (stack_probs_val >= t).astype(int)
    f1 = sk_f1(y[val], p, zero_division=0)
    if f1 > best_sf1:
        best_sf1 = f1
        best_st  = t

logger.info(f"  Best threshold: {best_st:.2f}  (Val F1={best_sf1:.3f})")

stack_probs = meta_model.predict_proba(meta_test)[:, 1]
stack_preds = (stack_probs >= best_st).astype(int)
logger.info(f"  Predicted depressed: {stack_preds.sum()}/{len(stack_preds)}")
results.append(evaluate(y[te], stack_preds, stack_probs, 'Stacking Fusion'))
roc_data.append({'name': 'Stacking Fusion', 'y_true': y[te], 'y_prob': stack_probs})
pr_data.append({'name': 'Stacking Fusion', 'y_true': y[te], 'y_prob': stack_probs})

# ── 11. EARLY FUSION (concatenate all features) ───────────────
logger.info("\n🔗 Running early fusion (all features concatenated)...")

X_early_train = np.hstack([X_text[tr],  X_audio[tr],  X_visual[tr]])
X_early_val   = np.hstack([X_text[val], X_audio[val], X_visual[val]])
X_early_test  = np.hstack([X_text[te],  X_audio[te],  X_visual[te]])

early_model, early_artifacts = train_unimodal(X_early_train, y[tr], 'early_fusion')
t_early = find_best_threshold(early_model, early_artifacts, X_early_val, y[val], metric='fbeta')

X_early_te_selected = transform_with_selector(X_early_test, early_artifacts)
X_ete_sc = early_artifacts['scaler'].transform(X_early_te_selected)
early_probs = early_model.predict_proba(X_ete_sc)[:, 1]
early_preds = (early_probs >= t_early).astype(int)
logger.info(f"  Predicted depressed: {early_preds.sum()}/{len(early_preds)}")
results.append(evaluate(y[te], early_preds, early_probs, 'Early Fusion'))
roc_data.append({'name': 'Early Fusion', 'y_true': y[te], 'y_prob': early_probs})
pr_data.append({'name': 'Early Fusion', 'y_true': y[te], 'y_prob': early_probs})

# ── 12. PLOTS & SAVE ──────────────────────────────────────────
logger.info("\n📈 Generating plots...")
plot_roc_curves(roc_data)
plot_precision_recall_curves(pr_data)
plot_model_comparison(results)
save_results_table(results)
generate_summary_report(results)

# Save feature importance info
logger.info("\n📊 Model Training Complete!")
logger.info("=" * 60)

# Print final summary
df_results = pd.DataFrame(results)
logger.info("\n🏆 FINAL RESULTS SUMMARY:")
logger.info("-" * 60)
for _, row in df_results.iterrows():
    logger.info(f"  {row['Model']:<20} | AUC: {row['AUC-ROC']:.3f} | F1: {row['F1']:.3f} | "
                f"Prec: {row['Precision']:.3f} | Rec: {row['Recall']:.3f}")
