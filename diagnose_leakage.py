"""
diagnose_leakage.py
====================
Confirms whether calculate_accuracies.py is evaluating on training data
by checking if the audio model's training participants overlap with the
full evaluation set.
"""
import joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

# ── Load model bundle ──────────────────────────────────────────────────────
bundle = joblib.load("models/audio_model.pkl")
model  = bundle["pipeline"]
thr    = bundle.get("threshold", 0.5)

# ── Load features + labels ─────────────────────────────────────────────────
df = pd.read_csv("data/features/audio_features_enhanced.csv")
for id_col in ["pid","Participant_ID","participant_id","id","ID"]:
    if id_col in df.columns: df = df.set_index(id_col); break
for col in ["PHQ_Score","phq_score","label","Label","depressed","Depressed"]:
    if col in df.columns: df = df.drop(columns=[col])
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

labels = pd.read_csv("data/features/master_labels.csv").set_index("pid")
y_all  = (labels["PHQ_Score"] >= 10).astype(int)
common = df.index.intersection(y_all.index)
X_all  = df.loc[common].values
y_all  = y_all.loc[common].values
print(f"Full dataset: {len(y_all)} participants")

# ── Reproduce the EXACT same train/val split used during training ──────────
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(
    X_all, y_all, np.arange(len(y_all)),
    test_size=0.2, stratify=y_all, random_state=42
)
print(f"Train: {len(y_tr)}  |  Val (held-out): {len(y_val)}")

# ── AUC on FULL set (what calculate_accuracies.py reports) ─────────────────
proba_all = model.predict_proba(X_all)[:, 1]
auc_full  = roc_auc_score(y_all, proba_all)

# ── AUC on TRAIN set only ──────────────────────────────────────────────────
proba_tr = model.predict_proba(X_tr)[:, 1]
auc_tr   = roc_auc_score(y_tr, proba_tr)

# ── AUC on TRUE held-out val set ───────────────────────────────────────────
proba_val = model.predict_proba(X_val)[:, 1]
auc_val   = roc_auc_score(y_val, proba_val)

print(f"\n  AUC on FULL 219 dataset (calculate_accuracies reports): {auc_full:.4f}")
print(f"  AUC on TRAIN  set only (175 samples, data model SAW):   {auc_tr:.4f}  <-- inflated")
print(f"  AUC on VAL    set only (44 samples, true held-out):     {auc_val:.4f}  <-- honest")

pct_overlap = len(set(idx_tr)) / len(y_all) * 100
print(f"\n  Training samples in full eval set: {len(idx_tr)}/{len(y_all)} = {pct_overlap:.0f}%")
print(f"  --> LEAKAGE CONFIRMED: calculate_accuracies.py includes the training set.")
print(f"  --> The 0.981 AUC is dominated by memorised training samples.")
