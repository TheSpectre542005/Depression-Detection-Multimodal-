# main.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score as sk_f1

from src.load_labels     import build_master_labels
from src.text_features   import build_text_features
from src.audio_features  import build_audio_features
from src.visual_features import build_visual_features
from src.fusion          import (train_unimodal, late_fusion_predict,
                                  find_best_threshold)
from src.evaluate        import (evaluate, plot_roc_curves,
                                  plot_model_comparison, save_results_table)

os.makedirs('data/features', exist_ok=True)
os.makedirs('models',        exist_ok=True)
os.makedirs('results',       exist_ok=True)

print("=" * 55)
print("   DEPRESSION DETECTION — MULTIMODAL PIPELINE")
print("=" * 55)

# ── 1. LABELS ──────────────────────────────────────────────────
print("\n📋 Loading labels...")
labels = pd.read_csv('data/features/master_labels.csv')
pids   = labels['pid'].tolist()
print(f"  Total: {len(pids)} | "
      f"Depressed: {labels['label'].sum()} | "
      f"Not Depressed: {(labels['label']==0).sum()}")

# ── 2. LOAD FEATURES ───────────────────────────────────────────
print("\n📦 Loading features...")
text_df   = pd.read_csv('data/features/text_features.csv')
audio_df  = pd.read_csv('data/features/audio_features.csv')
visual_df = pd.read_csv('data/features/visual_features.csv')
print(f"  Text   : {text_df.shape}")
print(f"  Audio  : {audio_df.shape}")
print(f"  Visual : {visual_df.shape}")

# ── 3. MERGE ───────────────────────────────────────────────────
print("\n🔗 Merging all features...")
merged = labels.copy()
merged = merged.merge(text_df,   on='pid', how='inner')
merged = merged.merge(audio_df,  on='pid', how='inner')
merged = merged.merge(visual_df, on='pid', how='inner')
print(f"  Merged shape: {merged.shape}")

# ── 4. FEATURE MATRICES ────────────────────────────────────────
text_cols   = [c for c in merged.columns
               if c.startswith(('sent_','word_','unique_','lexical_','avg_','tfidf_'))]
audio_cols  = [c for c in merged.columns
               if c.startswith(('mfcc_','egemap_'))]
visual_cols = [c for c in merged.columns
               if any(x in c for x in ['AU','pose_','gaze_'])]

X_text   = merged[text_cols].fillna(0).values
X_audio  = merged[audio_cols].fillna(0).values
X_visual = merged[visual_cols].fillna(0).values
y        = merged['label'].values

print(f"\n  Feature dimensions (raw):")
print(f"    Text   : {X_text.shape}")
print(f"    Audio  : {X_audio.shape}")
print(f"    Visual : {X_visual.shape}")

# ── 5. TRAIN / VALIDATION / TEST SPLIT ────────────────────────
# 70% train | 15% validation (threshold tuning) | 15% test
idx = np.arange(len(y))
tr_val, te = train_test_split(idx, test_size=0.15,
                               stratify=y, random_state=42)
tr, val    = train_test_split(tr_val, test_size=0.176,
                               stratify=y[tr_val], random_state=42)
# 0.176 of 85% ≈ 15% of total

print(f"\n  Train: {len(tr)} | Val: {len(val)} | Test: {len(te)}")
print(f"  Test  → depressed: {y[te].sum()} | "
      f"not depressed: {(y[te]==0).sum()}")

# ── 5b. PCA DIMENSIONALITY REDUCTION ─────────────────────────
# Fit PCA on training data only to prevent data leakage.
# Audio (248 features) and Visual (214 features) are way too high
# for 153 training samples → reduce to manageable dimensions.
N_COMPONENTS_AUDIO  = min(20, X_audio.shape[1], len(tr) - 1)
N_COMPONENTS_VISUAL = min(20, X_visual.shape[1], len(tr) - 1)

pca_audio  = PCA(n_components=N_COMPONENTS_AUDIO,  random_state=42)
pca_visual = PCA(n_components=N_COMPONENTS_VISUAL, random_state=42)

# Fit on TRAIN only
pca_audio.fit(X_audio[tr])
pca_visual.fit(X_visual[tr])

# Transform all splits
X_audio_pca  = pca_audio.transform(X_audio)
X_visual_pca = pca_visual.transform(X_visual)

print(f"\n  After PCA:")
print(f"    Audio  : {X_audio.shape} → {X_audio_pca.shape}  "
      f"(explained var: {pca_audio.explained_variance_ratio_.sum():.1%})")
print(f"    Visual : {X_visual.shape} → {X_visual_pca.shape}  "
      f"(explained var: {pca_visual.explained_variance_ratio_.sum():.1%})")

# Use PCA-reduced versions for audio and visual
X_audio  = X_audio_pca
X_visual = X_visual_pca

# ── 6. TRAIN UNIMODAL MODELS ──────────────────────────────────
print("\n🤖 Training unimodal models...")
text_model,   text_scaler   = train_unimodal(X_text[tr],   y[tr], 'text')
audio_model,  audio_scaler  = train_unimodal(X_audio[tr],  y[tr], 'audio')
visual_model, visual_scaler = train_unimodal(X_visual[tr], y[tr], 'visual')

# ── 7. FIND BEST THRESHOLD PER MODALITY (on validation set) ───
print("\n🎯 Finding best decision thresholds on validation set...")
t_text   = find_best_threshold(text_model,   text_scaler,   X_text[val],   y[val])
t_audio  = find_best_threshold(audio_model,  audio_scaler,  X_audio[val],  y[val])
t_visual = find_best_threshold(visual_model, visual_scaler, X_visual[val], y[val])

# ── 8. EVALUATE UNIMODAL ON TEST SET ──────────────────────────
print("\n📊 Evaluating on test set...")
results  = []
roc_data = []
val_aucs = {}

for name, model, scaler, X, threshold, mod_key in [
    ('Text Only',   text_model,   text_scaler,   X_text,   t_text,   'text'),
    ('Audio Only',  audio_model,  audio_scaler,  X_audio,  t_audio,  'audio'),
    ('Visual Only', visual_model, visual_scaler, X_visual, t_visual, 'visual'),
]:
    # Compute validation AUC for fusion weights
    X_val_sc = scaler.transform(X[val])
    val_probs = model.predict_proba(X_val_sc)[:, 1]
    val_aucs[mod_key] = roc_auc_score(y[val], val_probs)

    # Test set evaluation
    X_te_sc = scaler.transform(X[te])
    probs   = model.predict_proba(X_te_sc)[:, 1]
    preds   = (probs >= threshold).astype(int)
    print(f"\n  [{name}] threshold={threshold:.2f} | "
          f"predicted depressed: {preds.sum()}/{len(preds)}")
    results.append(evaluate(y[te], preds, probs, name))
    roc_data.append({'name': name, 'y_true': y[te], 'y_prob': probs})

# ── 9. LATE FUSION WITH SMART WEIGHTS ─────────────────────────
print("\n🔀 Running late fusion...")

models_scalers = {
    'text'  : (text_model,   text_scaler),
    'audio' : (audio_model,  audio_scaler),
    'visual': (visual_model, visual_scaler),
}
X_val_dict = {'text': X_text[val], 'audio': X_audio[val], 'visual': X_visual[val]}
X_te_dict  = {'text': X_text[te],  'audio': X_audio[te],  'visual': X_visual[te]}

# Exclude modalities with val AUC ≤ 0.52 (no better than random)
MIN_AUC = 0.52
included = {m: auc for m, auc in val_aucs.items() if auc > MIN_AUC}
excluded = {m: auc for m, auc in val_aucs.items() if auc <= MIN_AUC}

if excluded:
    for m, auc in excluded.items():
        print(f"  ⚠️  Excluding [{m}] from late fusion (val AUC={auc:.3f} ≤ {MIN_AUC})")

if not included:
    print("  ⚠️  No modality above random — using all with equal weights")
    included = val_aucs.copy()

adjusted = {m: max(auc - 0.5, 0.01) for m, auc in included.items()}
total_adj = sum(adjusted.values())
weights = {m: a / total_adj for m, a in adjusted.items()}
for m in {'text', 'audio', 'visual'} - set(included.keys()):
    weights[m] = 0.0

print(f"  Weights: " + ", ".join(
    f"{m}={weights[m]:.2f}" for m in ['text', 'audio', 'visual']))

best_ft, best_ff1 = 0.40, 0.0
for t in np.arange(0.25, 0.65, 0.01):
    p, _ = late_fusion_predict(models_scalers, X_val_dict,
                               weights=weights, threshold=t)
    f1 = sk_f1(y[val], p, zero_division=0)
    if f1 > best_ff1:
        best_ff1 = f1
        best_ft  = t

print(f"  Best threshold: {best_ft:.2f}  (Val F1={best_ff1:.3f})")
preds, probs = late_fusion_predict(
    models_scalers, X_te_dict, weights=weights, threshold=best_ft)
print(f"  Predicted depressed: {preds.sum()}/{len(preds)}")
results.append(evaluate(y[te], preds, probs, 'Late Fusion'))
roc_data.append({'name': 'Late Fusion', 'y_true': y[te], 'y_prob': probs})

# ── 10. STACKING META-LEARNER FUSION ──────────────────────────
print("\n🧠 Running stacking meta-learner fusion...")
from sklearn.linear_model import LogisticRegression as LR

# Build meta-features: predicted probabilities from each modality
def build_meta_features(models_scalers, X_dict):
    """Get predicted probabilities from each modality as meta-features."""
    meta = []
    for m in ['text', 'audio', 'visual']:
        model, scaler = models_scalers[m]
        X_sc = scaler.transform(X_dict[m])
        meta.append(model.predict_proba(X_sc)[:, 1])
    return np.column_stack(meta)

X_tr_dict = {'text': X_text[tr], 'audio': X_audio[tr], 'visual': X_visual[tr]}
meta_train = build_meta_features(models_scalers, X_tr_dict)
meta_val   = build_meta_features(models_scalers, X_val_dict)
meta_test  = build_meta_features(models_scalers, X_te_dict)

# Train meta-learner on training probabilities
meta_model = LR(class_weight='balanced', C=1.0, max_iter=5000, random_state=42)
meta_model.fit(meta_train, y[tr])
print(f"  Meta-learner coefficients: text={meta_model.coef_[0][0]:.3f}, "
      f"audio={meta_model.coef_[0][1]:.3f}, visual={meta_model.coef_[0][2]:.3f}")

# Find best threshold for stacking
stack_probs_val = meta_model.predict_proba(meta_val)[:, 1]
best_st, best_sf1 = 0.50, 0.0
for t in np.arange(0.25, 0.65, 0.01):
    p = (stack_probs_val >= t).astype(int)
    f1 = sk_f1(y[val], p, zero_division=0)
    if f1 > best_sf1:
        best_sf1 = f1
        best_st  = t

print(f"  Best threshold: {best_st:.2f}  (Val F1={best_sf1:.3f})")

stack_probs = meta_model.predict_proba(meta_test)[:, 1]
stack_preds = (stack_probs >= best_st).astype(int)
print(f"  Predicted depressed: {stack_preds.sum()}/{len(stack_preds)}")
results.append(evaluate(y[te], stack_preds, stack_probs, 'Stacking Fusion'))
roc_data.append({'name': 'Stacking Fusion', 'y_true': y[te], 'y_prob': stack_probs})

# ── 11. EARLY FUSION (concatenate all features) ───────────────
print("\n🔗 Running early fusion (all features concatenated)...")
X_early_train = np.hstack([X_text[tr],  X_audio[tr],  X_visual[tr]])
X_early_val   = np.hstack([X_text[val], X_audio[val], X_visual[val]])
X_early_test  = np.hstack([X_text[te],  X_audio[te],  X_visual[te]])

early_model, early_scaler = train_unimodal(X_early_train, y[tr], 'early_fusion')
t_early = find_best_threshold(early_model, early_scaler, X_early_val, y[val])

X_ete_sc = early_scaler.transform(X_early_test)
early_probs = early_model.predict_proba(X_ete_sc)[:, 1]
early_preds = (early_probs >= t_early).astype(int)
print(f"  Predicted depressed: {early_preds.sum()}/{len(early_preds)}")
results.append(evaluate(y[te], early_preds, early_probs, 'Early Fusion'))
roc_data.append({'name': 'Early Fusion', 'y_true': y[te], 'y_prob': early_probs})

# ── 12. PLOTS & SAVE ──────────────────────────────────────────
print("\n📈 Generating plots...")
plot_roc_curves(roc_data)
plot_model_comparison(results)
save_results_table(results)

print("\n" + "="*55)
print("  ✅ PIPELINE COMPLETE")
print("  📁 Check results/ folder for all outputs")
print("="*55)
