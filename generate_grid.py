import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Data based on final_metrics.csv and tmp_cm3.py (Threshold 0.38)
metrics_data = {
    'Model': ['Text', 'Audio', 'Visual', 'Fusion'],
    'AUC': [0.6601, 0.6203, 0.5842, 0.6551],
    'Accuracy': [0.6164, 0.7169, 0.4840, 0.6256],
    'F1': [0.5435, 0.3922, 0.5022, 0.5060],
    'Precision': [0.4202, 0.5405, 0.3519, 0.4158],
    'Recall': [0.7692, 0.3077, 0.8769, 0.6462],
    'Threshold': [0.28, 0.55, 0.32, 0.38]
}

# Confusion Matrix for Fusion at 0.38
cm_data = np.array([[115, 39], [23, 42]])

df = pd.DataFrame(metrics_data)

# Ensure results dir exists
os.makedirs('results', exist_ok=True)

# Delete existing contents of results folder
for file in os.listdir('results'):
    file_path = os.path.join('results', file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Save new values
df.to_csv('results/performance_metrics.csv', index=False)

# Create 2x2 Grid Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SENTIRA Model Performance Dashboard', fontsize=20, fontweight='bold', y=0.95)

# 1. Accuracy Matrix (Top Left)
sns.heatmap(df.set_index('Model')[['AUC', 'Accuracy', 'F1', 'Precision', 'Recall']], 
            annot=True, cmap='Blues', fmt='.3f', ax=axes[0, 0])
axes[0, 0].set_title('Performance Matrix by Modality', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('')

# 2. Performance Metrics Bar Chart (Top Right)
df_melted = df.melt(id_vars='Model', value_vars=['Accuracy', 'F1', 'AUC'], 
                    var_name='Metric', value_name='Score')
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_ylim(0, 1.0)
axes[0, 1].legend(loc='lower right')

# 3. Confusion Matrix (Bottom Left)
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0], 
            xticklabels=['Not Depressed', 'Depressed'], 
            yticklabels=['Not Depressed', 'Depressed'],
            annot_kws={'size': 16})
axes[1, 0].set_title('Fusion Confusion Matrix (Threshold=0.38)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('True Label', fontsize=12)
axes[1, 0].set_xlabel('Predicted Label', fontsize=12)

# 4. Modality Importance / Fusion Weights (Bottom Right)
weights = {'Text': 0.478, 'Audio': 0.315, 'Visual': 0.207}
axes[1, 1].pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%', 
               colors=['#4daf4a', '#377eb8', '#e41a1c'], startangle=90,
               textprops={'fontsize': 14})
axes[1, 1].set_title('Fusion Modality Weights', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.savefig('results/performance_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Saved performance_dashboard.png and performance_metrics.csv to results/")
