# src/evaluate.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

os.makedirs('results', exist_ok=True)

def evaluate(y_true, y_pred, y_prob, name="Model"):
    print(f"\n{'='*50}")
    print(f"  📊 {name}")
    print(f"{'='*50}")
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Not Depressed','Depressed'])}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Dep.', 'Depressed'],
                yticklabels=['Not Dep.', 'Depressed'])
    plt.title(f'{name} — Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    safe_name = name.replace(" ", "_")
    plt.savefig(f'results/{safe_name}_confusion_matrix.png', dpi=150)
    plt.close()

    return {
        'Model'    : name,
        'Accuracy' : round(acc,  4),
        'F1'       : round(f1,   4),
        'Precision': round(prec, 4),
        'Recall'   : round(rec,  4),
        'AUC-ROC'  : round(auc,  4),
    }


def plot_roc_curves(roc_data):
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, d in enumerate(roc_data):
        fpr, tpr, _ = roc_curve(d['y_true'], d['y_prob'])
        auc = roc_auc_score(d['y_true'], d['y_prob'])
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 lw=2, label=f"{d['name']} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — All Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/roc_curves.png', dpi=150)
    plt.close()
    print("  ✅ Saved → results/roc_curves.png")


def plot_model_comparison(results_list):
    df = pd.DataFrame(results_list).set_index('Model')
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC-ROC']
    ax = df[metrics].plot(kind='bar', figsize=(12, 5),
                          colormap='Set2', edgecolor='black', linewidth=0.5)
    plt.title('Model Comparison — All Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150)
    plt.close()
    print("  ✅ Saved → results/model_comparison.png")


def save_results_table(results_list):
    df = pd.DataFrame(results_list)
    df.to_csv('results/all_results.csv', index=False)
    print("\n  ✅ Results table saved → results/all_results.csv")
    print(df.to_string(index=False))