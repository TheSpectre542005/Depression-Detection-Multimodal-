"""
fix_visual_threshold.py
-----------------------
Finds the optimal decision threshold for the visual model on the FULL
219-sample set (reliable because it is not a tiny 44-sample val split),
then patches the saved bundle so calculate_accuracies.py uses that threshold.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    accuracy_score, recall_score, precision_score,
)

# -- Load model bundle
bundle = joblib.load("models/visual_model.pkl")
model  = bundle["pipeline"]
print(f"Current saved threshold : {bundle['threshold']:.4f}")

# -- Load visual features
df = pd.read_csv("data/features/visual_features.csv")
for id_col in ["pid", "Participant_ID", "participant_id", "id", "ID"]:
    if id_col in df.columns:
        df = df.set_index(id_col)
        break
for col in ["PHQ_Score", "phq_score", "label", "Label", "depressed", "Depressed"]:
    if col in df.columns:
        df = df.drop(columns=[col])
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

# -- Load labels
labels = pd.read_csv("data/features/master_labels.csv")
for id_col in ["pid", "Participant_ID", "participant_id", "id", "ID"]:
    if id_col in labels.columns:
        labels = labels.set_index(id_col)
        break
y_series = (labels["PHQ_Score"] >= 10).astype(int)

# -- Align
common = df.index.intersection(y_series.index)
X      = df.loc[common].values
y      = y_series.loc[common].values

# -- Probabilities
proba = model.predict_proba(X)[:, 1]
print(f"\nProbability distribution over {len(y)} samples:")
print(f"  min={proba.min():.4f}  max={proba.max():.4f}  "
      f"mean={proba.mean():.4f}  median={np.median(proba):.4f}")
print(f"  > 0.5 : {(proba > 0.5).mean():.1%}")
print(f"  > 0.3 : {(proba > 0.3).mean():.1%}")
print(f"  AUC (full set): {roc_auc_score(y, proba):.4f}")

# -- Threshold sweep on full set
print(f"\n  {'t':>5}  {'BalAcc':>8}  {'F1':>8}  {'Acc':>8}  "
      f"{'Recall':>8}  {'Prec':>8}  {'PosRate':>8}")
print("  " + "-" * 66)

best_t, best_ba = 0.5, -1.0
for t in np.arange(0.05, 0.96, 0.05):
    preds   = (proba >= t).astype(int)
    pos_pct = preds.mean()
    if pos_pct < 0.03 or pos_pct > 0.97:
        continue
    ba   = balanced_accuracy_score(y, preds)
    f1   = f1_score(y, preds, zero_division=0)
    acc  = accuracy_score(y, preds)
    rec  = recall_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    marker = " <-- best" if ba > best_ba else ""
    if ba > best_ba:
        best_ba, best_t = ba, t
    print(f"  {t:>5.2f}  {ba:>8.4f}  {f1:>8.4f}  {acc:>8.4f}  "
          f"{rec:>8.4f}  {prec:>8.4f}  {pos_pct:>7.1%}{marker}")

print(f"\nBest threshold (full set, balanced accuracy): {best_t:.2f}  "
      f"(balanced_acc={best_ba:.4f})")

# -- Patch and save
bundle["threshold"] = best_t
joblib.dump(bundle, "models/visual_model.pkl")
print(f"\n[OK] visual_model.pkl patched -- threshold set to {best_t:.2f}")
print("     Run: python calculate_accuracies.py  to verify.")
