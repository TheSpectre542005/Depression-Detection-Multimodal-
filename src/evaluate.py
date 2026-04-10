# src/evaluate.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    balanced_accuracy_score,fbeta_score
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
logger = logging.getLogger(__name__)


def bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=1000, ci_level=0.95, **kwargs):
    """
    Compute bootstrap confidence interval for any metric function.

    Args:
        y_true: true labels
        y_prob: predicted probabilities (or predictions for non-prob metrics)
        metric_fn: function(y_true, y_prob, **kwargs) → float
        n_bootstrap: number of bootstrap samples
        ci_level: confidence level (default 0.95)

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(42)
    n = len(y_true)
    scores = []

    point_estimate = metric_fn(y_true, y_prob, **kwargs)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_prob[idx], **kwargs)
            scores.append(score)
        except (ValueError, ZeroDivisionError):
            continue

    if not scores:
        return point_estimate, point_estimate, point_estimate

    alpha = (1 - ci_level) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))
    return point_estimate, lower, upper


def evaluate_cv(cv_results_list):
    """
    Aggregate results across cross-validation folds.

    Args:
        cv_results_list: list of dicts from evaluate() for each fold

    Returns:
        dict with mean ± std for each metric
    """
    if not cv_results_list:
        return {}

    df = pd.DataFrame(cv_results_list)
    metrics = ['Accuracy', 'Balanced_Acc', 'F1', 'F2', 'Precision', 'Recall',
               'AUC-ROC', 'Avg_Precision', 'Sensitivity', 'Specificity']

    summary = {'Model': df['Model'].iloc[0]}
    for m in metrics:
        if m in df.columns:
            vals = df[m].values
            summary[f'{m}_mean'] = round(float(np.mean(vals)), 4)
            summary[f'{m}_std'] = round(float(np.std(vals)), 4)

    return summary


def clinical_metrics(y_true, y_pred):
    """Compute clinically relevant metrics (sensitivity, specificity, etc.)."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity (Recall) - ability to detect depressed patients
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Specificity - ability to detect non-depressed patients
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # PPV (Precision) - probability that positive prediction is correct
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    # NPV - probability that negative prediction is correct
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
    }


def evaluate(y_true, y_pred, y_prob, name="Model"):
    """Comprehensive model evaluation with clinical metrics."""
    logger.info(f"\n{'='*50}")
    logger.info(f"  📊 {name}")
    logger.info(f"{'='*50}")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # F-beta with beta=2 (favors recall - critical for depression detection)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    # Clinical metrics
    clinical = clinical_metrics(y_true, y_pred)

    logger.info(f"  Accuracy        : {acc:.4f}")
    logger.info(f"  Balanced Acc    : {bal_acc:.4f}")
    logger.info(f"  F1-Score        : {f1:.4f}")
    logger.info(f"  F2-Score        : {f2:.4f}  (favors recall)")
    logger.info(f"  Precision       : {prec:.4f}")
    logger.info(f"  Recall          : {rec:.4f}")
    logger.info(f"  AUC-ROC         : {auc:.4f}")
    logger.info(f"  Avg Precision   : {avg_prec:.4f}")
    logger.info(f"  ─────────────────────────────────")
    logger.info(f"  Sensitivity     : {clinical['sensitivity']:.4f}  (detect depressed)")
    logger.info(f"  Specificity     : {clinical['specificity']:.4f}  (detect non-depressed)")
    logger.info(f"  PPV             : {clinical['ppv']:.4f}  (precision)")
    logger.info(f"  NPV             : {clinical['npv']:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['Not Depressed','Depressed'], zero_division=0)}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Dep.', 'Depressed'],
                yticklabels=['Not Dep.', 'Depressed'],
                ax=axes[0])
    axes[0].set_title(f'{name} — Confusion Matrix (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=['Not Dep.', 'Depressed'],
                yticklabels=['Not Dep.', 'Depressed'],
                ax=axes[1])
    axes[1].set_title(f'{name} — Confusion Matrix (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    safe_name = name.replace(" ", "_")
    plt.savefig(os.path.join(RESULTS_DIR, f'{safe_name}_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'Model': name,
        'Accuracy': round(acc, 4),
        'Balanced_Acc': round(bal_acc, 4),
        'F1': round(f1, 4),
        'F2': round(f2, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'AUC-ROC': round(auc, 4),
        'Avg_Precision': round(avg_prec, 4),
        'Sensitivity': round(clinical['sensitivity'], 4),
        'Specificity': round(clinical['specificity'], 4),
        'PPV': round(clinical['ppv'], 4),
        'NPV': round(clinical['npv'], 4),
        'TN': int(cm[0, 0]),
        'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]),
        'TP': int(cm[1, 1]),
    }


def plot_roc_curves(roc_data, filename='roc_curves.png'):
    """Plot ROC curves with AUC scores and confidence intervals."""
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i, d in enumerate(roc_data):
        fpr, tpr, _ = roc_curve(d['y_true'], d['y_prob'])
        auc = roc_auc_score(d['y_true'], d['y_prob'])
        color = colors[i % len(colors)]

        plt.plot(fpr, tpr, color=color, lw=2.5,
                 label=f"{d['name']} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')

    # Add shaded region for good performance
    plt.fill_between([0, 1], [0, 0.8], [0.2, 1], alpha=0.1, color='green', label='Good Performance Zone')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved → {RESULTS_DIR}/{filename}")


def plot_precision_recall_curves(pr_data, filename='pr_curves.png'):
    """Plot Precision-Recall curves."""
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i, d in enumerate(pr_data):
        precision, recall, _ = precision_recall_curve(d['y_true'], d['y_prob'])
        avg_prec = average_precision_score(d['y_true'], d['y_prob'])
        color = colors[i % len(colors)]

        plt.plot(recall, precision, color=color, lw=2.5,
                 label=f"{d['name']} (AP = {avg_prec:.3f})")

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved → {RESULTS_DIR}/{filename}")


def plot_model_comparison(results_list, filename='model_comparison.png'):
    """Enhanced model comparison with multiple metrics."""
    df = pd.DataFrame(results_list).set_index('Model')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Main metrics
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC-ROC']
    df[metrics].plot(kind='bar', ax=axes[0, 0], colormap='Set2', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('Model Performance — Core Metrics', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(loc='lower right', fontsize=8)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Confusion matrix components
    if 'TP' in df.columns:
        cm_metrics = ['TP', 'TN', 'FP', 'FN']
        df[cm_metrics].plot(kind='bar', ax=axes[0, 1], colormap='RdYlGn', edgecolor='black', linewidth=0.5)
        axes[0, 1].set_title('Confusion Matrix Components', fontweight='bold')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].tick_params(axis='x', rotation=45)

    # Balanced Accuracy vs AUC
    if 'Balanced_Acc' in df.columns:
        axes[1, 0].scatter(df['AUC-ROC'], df['Balanced_Acc'], s=100, c=range(len(df)), cmap='viridis', edgecolors='black')
        for i, model in enumerate(df.index):
            axes[1, 0].annotate(model, (df['AUC-ROC'].iloc[i], df['Balanced_Acc'].iloc[i]),
                               fontsize=8, ha='center', va='bottom')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('AUC-ROC')
        axes[1, 0].set_ylabel('Balanced Accuracy')
        axes[1, 0].set_title('Balanced Acc vs AUC-ROC', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

    # Precision vs Recall
    axes[1, 1].scatter(df['Recall'], df['Precision'], s=100, c=range(len(df)), cmap='plasma', edgecolors='black')
    for i, model in enumerate(df.index):
        axes[1, 1].annotate(model, (df['Recall'].iloc[i], df['Precision'].iloc[i]),
                           fontsize=8, ha='center', va='bottom')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision vs Recall', fontweight='bold')
    axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved → {RESULTS_DIR}/{filename}")


def plot_calibration_curve(y_true, y_prob, name, filename=None):
    """Plot calibration curve for probability reliability."""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label=name, linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Calibration Curve — {name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename is None:
        filename = f'{name.replace(" ", "_")}_calibration.png'

    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def save_results_table(results_list, filename='all_results.csv'):
    """Save comprehensive results table."""
    df = pd.DataFrame(results_list)

    # Reorder columns for better readability
    cols = ['Model', 'Accuracy', 'Balanced_Acc', 'F1', 'Precision', 'Recall',
            'AUC-ROC', 'Avg_Precision', 'TP', 'TN', 'FP', 'FN']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
    logger.info(f"\n  Results table saved → {RESULTS_DIR}/{filename}")
    logger.info(f"\n{df.to_string(index=False)}")


def generate_summary_report(results_list, output_file='summary_report.txt'):
    """Generate a text summary report."""
    filepath = os.path.join(RESULTS_DIR, output_file)

    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  DEPRESSION DETECTION MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        best_f1 = max(results_list, key=lambda x: x.get('F1', 0))
        best_auc = max(results_list, key=lambda x: x.get('AUC-ROC', 0))
        best_recall = max(results_list, key=lambda x: x.get('Recall', 0))

        f.write(f"Best F1-Score:      {best_f1['Model']} ({best_f1['F1']:.4f})\n")
        f.write(f"Best AUC-ROC:       {best_auc['Model']} ({best_auc['AUC-ROC']:.4f})\n")
        f.write(f"Best Recall:        {best_recall['Model']} ({best_recall['Recall']:.4f})\n")
        f.write("\n" + "-" * 60 + "\n\n")

        for r in results_list:
            f.write(f"\n{r['Model']}:\n")
            for k, v in r.items():
                if k != 'Model':
                    f.write(f"  {k}: {v}\n")

    logger.info(f"  Summary report saved → {filepath}")
