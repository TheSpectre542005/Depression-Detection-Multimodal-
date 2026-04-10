# main.py
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score as sk_f1

from config import (THRESHOLD_MIN, THRESHOLD_MAX, CV_SPLITS, CV_REPEATS,
                    MIN_CLINICAL_AUC, RANDOM_STATE)

from src.load_labels     import build_master_labels
from src.text_features   import build_text_features
from src.audio_features_enhanced  import build_audio_features_enhanced
from src.visual_features_enhanced import build_visual_features_enhanced
from src.fusion          import (train_unimodal, late_fusion_predict,
                                  find_best_threshold, find_cost_sensitive_threshold,
                                  transform_with_selector, train_meta_learner)
from src.evaluate        import (evaluate, plot_roc_curves, plot_precision_recall_curves,
                                  plot_model_comparison, save_results_table,
                                  generate_summary_report, plot_calibration_curve,
                                  bootstrap_ci, evaluate_cv)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    os.makedirs('data/features', exist_ok=True)
    os.makedirs('models',        exist_ok=True)
    os.makedirs('results',       exist_ok=True)

    print("=" * 60)
    print("   DEPRESSION DETECTION — ROBUST PIPELINE (CV + CIs)")
    print("=" * 60)

    # ── 1. LABELS ──────────────────────────────────────────────────
    logger.info("\n📋 Loading labels...")
    labels = pd.read_csv('data/features/master_labels.csv')
    pids   = labels['pid'].tolist()
    logger.info(f"  Total: {len(pids)} | Depressed: {labels['label'].sum()} | Not Depressed: {(labels['label']==0).sum()}")

    # ── 2. LOAD/EXTRACT FEATURES ───────────────────────────────────
    logger.info("\n📦 Loading features...")

    if os.path.exists('data/features/text_features.csv'):
        text_df = pd.read_csv('data/features/text_features.csv')
    else:
        logger.info("  Extracting text features...")
        text_df = build_text_features(pids)

    if os.path.exists('data/features/audio_features_enhanced.csv'):
        audio_df = pd.read_csv('data/features/audio_features_enhanced.csv')
    else:
        logger.info("  Extracting enhanced audio features...")
        audio_df = build_audio_features_enhanced(pids)

    if os.path.exists('data/features/visual_features_enhanced.csv'):
        visual_df = pd.read_csv('data/features/visual_features_enhanced.csv')
    else:
        logger.info("  Extracting enhanced visual features...")
        visual_df = build_visual_features_enhanced(pids)

    # ── 3. MERGE ───────────────────────────────────────────────────
    logger.info("\n🔗 Merging features...")
    merged = labels.copy()
    merged = merged.merge(text_df,   on='pid', how='inner')
    merged = merged.merge(audio_df,  on='pid', how='inner')
    merged = merged.merge(visual_df, on='pid', how='inner')

    # Feature definitions
    text_cols   = [c for c in merged.columns if c.startswith(('sent_','word_','unique_','lexical_','avg_','tfidf_','sbert_')) or c in [
        'dep_lexicon_count', 'dep_lexicon_ratio', 'dep_lexicon_unique', 'fps_ratio', 'fpp_ratio', 'tp_ratio',
        'absolutist_count', 'absolutist_ratio', 'negation_count', 'negation_ratio', 'sent_variance', 'sent_range',
        'mean_sent_len', 'response_brevity', 'question_ratio', 'hedging_count', 'hedging_ratio'
    ]]
    audio_cols  = [c for c in merged.columns if c not in ['pid', 'label'] and c not in text_cols and not any(x in c for x in ['densenet', 'vgg', 'resnet', 'AU', 'pose_', 'gaze_'])]
    visual_cols = [c for c in merged.columns if any(x in c for x in ['densenet', 'vgg', 'resnet', 'AU', 'pose_', 'gaze_'])]

    X_text   = merged[text_cols].fillna(0).values
    X_audio  = merged[audio_cols].fillna(0).values
    X_visual = merged[visual_cols].fillna(0).values
    y        = merged['label'].values

    logger.info(f"  Text: {X_text.shape} | Audio: {X_audio.shape} | Visual: {X_visual.shape}")

    # ── 4. HOLDOUT SPLIT ──────────────────────────────────────────
    # 85% trainval (for CV) | 15% holdout test
    idx = np.arange(len(y))
    tr_val, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    logger.info(f"\n  Main Train/Val: {len(tr_val)} | Holdout Test: {len(te)}")

    # Variables to accumulate CV results
    cv_results_text = []
    cv_results_audio = []
    cv_results_visual = []
    cv_results_late = []

    rskf = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)

    # ── 5. REPEATED STRATIFIED K-FOLD CV ──────────────────────────
    logger.info(f"\n🔄 Running {CV_SPLITS}-Fold CV ({CV_REPEATS} repeats) on {len(tr_val)} samples...")

    fold = 1
    for train_idx, val_idx in rskf.split(tr_val, y[tr_val]):
        tr_f = tr_val[train_idx]
        val_f = tr_val[val_idx]

        # --- PCA Inside Fold (Leakage Prevention) ---
        pca_audio = PCA(n_components=min(40, X_audio.shape[1], len(tr_f)-1), random_state=RANDOM_STATE)
        pca_visual = PCA(n_components=min(25, X_visual.shape[1], len(tr_f)-1), random_state=RANDOM_STATE)
        
        Xa_tr = pca_audio.fit_transform(X_audio[tr_f])
        Xv_tr = pca_visual.fit_transform(X_visual[tr_f])
        
        Xa_val = pca_audio.transform(X_audio[val_f])
        Xv_val = pca_visual.transform(X_visual[val_f])

        # --- Train Unimodal ---
        # Temporarily hide logs during CV loop
        logging.getLogger('src.fusion').setLevel(logging.WARNING)
        logging.getLogger('src.evaluate').setLevel(logging.WARNING)
        
        # We explicitly turn off save here because we save models in the finale loop to avoid concurrency artifacts
        tm, ta = train_unimodal(X_text[tr_f], y[tr_f], f'cv_t_{fold}', save=False)
        am, aa = train_unimodal(Xa_tr, y[tr_f], f'cv_a_{fold}', save=False)
        vm, va = train_unimodal(Xv_tr, y[tr_f], f'cv_v_{fold}', save=False)

        # Thresholding & Eval
        tt = find_cost_sensitive_threshold(tm, ta, X_text[val_f], y[val_f])
        at = find_cost_sensitive_threshold(am, aa, Xa_val, y[val_f])
        vt = find_cost_sensitive_threshold(vm, va, Xv_val, y[val_f])

        # Text
        p_t = tm.predict_proba(ta['scaler'].transform(transform_with_selector(X_text[val_f], ta)))[:, 1]
        cv_results_text.append(evaluate(y[val_f], (p_t >= tt).astype(int), p_t, 'Text Only', verbose=False))
        
        # Audio
        p_a = am.predict_proba(aa['scaler'].transform(transform_with_selector(Xa_val, aa)))[:, 1]
        cv_results_audio.append(evaluate(y[val_f], (p_a >= at).astype(int), p_a, 'Audio Only', verbose=False))
        
        # Visual
        p_v = vm.predict_proba(va['scaler'].transform(transform_with_selector(Xv_val, va)))[:, 1]
        cv_results_visual.append(evaluate(y[val_f], (p_v >= vt).astype(int), p_v, 'Visual Only', verbose=False))
        
        # Late Fusion
        models_sc = {'text': (tm, ta), 'audio': (am, aa), 'visual': (vm, va)}
        Xv_dict = {'text': X_text[val_f], 'audio': Xa_val, 'visual': Xv_val}
        # Simple equal weighting for CV loop
        w = {'text': 0.5, 'audio': 0.25, 'visual': 0.25}
        best_ft, best_ff1 = 0.40, 0.0
        for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX, 0.05):
            p, _ = late_fusion_predict(models_sc, Xv_dict, weights=w, threshold=t)
            if sk_f1(y[val_f], p, zero_division=0) > best_ff1:
                best_ff1 = sk_f1(y[val_f], p, zero_division=0)
                best_ft = t
        preds, probs = late_fusion_predict(models_sc, Xv_dict, weights=w, threshold=best_ft)
        cv_results_late.append(evaluate(y[val_f], preds, probs, 'Late Fusion', verbose=False))

        fold += 1

    logging.getLogger('src.fusion').setLevel(logging.INFO)
    logging.getLogger('src.evaluate').setLevel(logging.INFO)

    # Aggregate CV results
    logger.info("\n📊 K-Fold Aggregated Validation Results (Mean ± Std):")
    for cv_res in [cv_results_text, cv_results_audio, cv_results_visual, cv_results_late]:
        agg = evaluate_cv(cv_res)
        logger.info(f"  {agg['Model']:<12}: AUC = {agg.get('AUC-ROC_mean', 0.0):.3f} ± {agg.get('AUC-ROC_std', 0.0):.3f} | F1 = {agg.get('F1_mean', 0.0):.3f} ± {agg.get('F1_std', 0.0):.3f}")

    # ── 6. FINAL MODEL TRAINING (on full trainval) ────────────────
    logger.info("\n🚀 Training final models on full trainval set (85%)...")

    pca_audio = PCA(n_components=min(40, X_audio.shape[1], len(tr_val)-1), random_state=RANDOM_STATE)
    pca_visual = PCA(n_components=min(25, X_visual.shape[1], len(tr_val)-1), random_state=RANDOM_STATE)

    X_audio_pca = pca_audio.fit_transform(X_audio)
    X_visual_pca = pca_visual.fit_transform(X_visual)

    Xa_tr_full = X_audio_pca[tr_val]
    Xv_tr_full = X_visual_pca[tr_val]

    text_model, text_artifacts = train_unimodal(X_text[tr_val], y[tr_val], 'text')
    audio_model, audio_artifacts = train_unimodal(Xa_tr_full, y[tr_val], 'audio')
    visual_model, visual_artifacts = train_unimodal(Xv_tr_full, y[tr_val], 'visual')

    # Save PCAs
    import joblib
    joblib.dump(pca_audio, 'models/audio_pca.pkl')
    joblib.dump(pca_visual, 'models/visual_pca.pkl')

    t_text = find_cost_sensitive_threshold(text_model, text_artifacts, X_text[tr_val], y[tr_val])
    t_audio = find_cost_sensitive_threshold(audio_model, audio_artifacts, Xa_tr_full, y[tr_val])
    t_visual = find_cost_sensitive_threshold(visual_model, visual_artifacts, Xv_tr_full, y[tr_val])

    # ── 7. HOLDOUT TEST SET EVALUATION WITH BOOSTSTRAP CIs ────────
    logger.info("\n🏆 Evaluating Final Models on Holdout Test Set (15%)...")
    results = []
    roc_data = []
    pr_data = []

    Xa_te = X_audio_pca[te]
    Xv_te = X_visual_pca[te]

    models_scalers = {
        'text': (text_model, text_artifacts),
        'audio': (audio_model, audio_artifacts),
        'visual': (visual_model, visual_artifacts)
    }
    X_te_dict = {'text': X_text[te], 'audio': Xa_te, 'visual': Xv_te}

    w = {'text': 0.5, 'audio': 0.25, 'visual': 0.25}
    best_late_t = 0.45
    late_preds, late_probs = late_fusion_predict(models_scalers, X_te_dict, weights=w, threshold=best_late_t)

    # Collect predictions
    eval_targets = [
        ('Text Only', text_model, text_artifacts, X_text[te], t_text),
        ('Audio Only', audio_model, audio_artifacts, Xa_te, t_audio),
        ('Visual Only', visual_model, visual_artifacts, Xv_te, t_visual)
    ]

    for name, model, art, x_feats, thresh in eval_targets:
        p = model.predict_proba(art['scaler'].transform(transform_with_selector(x_feats, art)))[:, 1]
        preds = (p >= thresh).astype(int)
        
        # Point evaluation
        res = evaluate(y[te], preds, p, name)
        
        # Bootstrap CIs for AUC
        _, lb, ub = bootstrap_ci(y[te], p, roc_auc_score)
        res['AUC-ROC_CI_Lower'] = lb
        res['AUC-ROC_CI_Upper'] = ub
        
        results.append(res)
        roc_data.append({'name': name, 'y_true': y[te], 'y_prob': p})
        pr_data.append({'name': name, 'y_true': y[te], 'y_prob': p})
        
        if name == 'Late Fusion':
            plot_calibration_curve(y[te], p, name)

    # Late Fusion CIs
    res_late = evaluate(y[te], late_preds, late_probs, 'Late Fusion')
    _, lb_l, ub_l = bootstrap_ci(y[te], late_probs, roc_auc_score)
    res_late['AUC-ROC_CI_Lower'] = lb_l
    res_late['AUC-ROC_CI_Upper'] = ub_l
    results.append(res_late)
    roc_data.append({'name': 'Late Fusion', 'y_true': y[te], 'y_prob': late_probs})
    pr_data.append({'name': 'Late Fusion', 'y_true': y[te], 'y_prob': late_probs})

    logger.info("\n📊 Generating plots and reports...")
    plot_roc_curves(roc_data)
    plot_precision_recall_curves(pr_data)
    plot_model_comparison(results)
    save_results_table(results)
    generate_summary_report(results)

    df_results = pd.DataFrame(results)
    logger.info("\n🏆 FINAL HOLDOUT RESULTS SUMMARY (with 95% CIs):")
    logger.info("-" * 80)
    for _, row in df_results.iterrows():
        auc = row['AUC-ROC']
        lb = row.get('AUC-ROC_CI_Lower', auc)
        ub = row.get('AUC-ROC_CI_Upper', auc)
        
        status = "✅" if auc >= MIN_CLINICAL_AUC else "⚠️"
        logger.info(f"  {status} {row['Model']:<12} | AUC: {auc:.3f} [{lb:.3f}-{ub:.3f}] | F1: {row['F1']:.3f} | Prec: {row['Precision']:.3f} | Rec: {row['Recall']:.3f}")

    logger.info("=" * 80)

if __name__ == "__main__":
    main()
