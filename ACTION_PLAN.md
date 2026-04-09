# MindScan Depression Detection — Action Plan

## Priority Matrix

| Priority | Issue | Impact | Effort | File(s) |
|----------|-------|--------|--------|---------|
| **P0 (Critical)** | Audio modality broken | High | Medium | `audio_features_enhanced.py` |
| **P0 (Critical)** | Overall AUC < 0.70 | High | High | `fusion.py`, `main.py` |
| **P1 (High)** | Missing requirements.txt | Medium | Low | Create new file |
| **P1 (High)** | Hardcoded paths | Medium | Low | `config.py` |
| **P1 (High)** | Fusion ineffective | High | Medium | `fusion.py` |
| **P2 (Medium)** | Web app only uses text | High | Medium | `app.py` |
| **P2 (Medium)** | Low F1 scores | High | Medium | All model files |
| **P3 (Low)** | Error handling | Low | Low | Feature extraction files |

---

## Immediate Actions (Do First)

### 1. Debug Audio Features
**Problem**: Audio model accuracy 0.394 (worse than random)

**Diagnostic Script**:
```python
# Run this to diagnose audio data issues
def diagnose_audio_data(pid):
    feat_dir = f"{DATA_ROOT}/{pid}_P/features"
    files = [
        f"{pid}_OpenSMILE2.3.0_mfcc.csv",
        f"{pid}_OpenSMILE2.3.0_egemaps.csv",
        f"{pid}_BoAW_openSMILE_2.3.0_MFCC.csv"
    ]
    for f in files:
        path = os.path.join(feat_dir, f)
        print(f"{f}: exists={os.path.exists(path)}")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Shape: {df.shape}, NaN: {df.isna().sum().sum()}")
```

**Expected Fix Time**: 2-4 hours

---

### 2. Create requirements.txt
**Create file** at `depression_project/requirements.txt`:
```
flask>=2.0.0,<3.0.0
numpy>=1.21.0,<2.0.0
pandas>=1.3.0,<2.0.0
scikit-learn>=1.0.0,<2.0.0
imbalanced-learn>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
nltk>=3.6.0
vaderSentiment>=3.3.0
scipy>=1.7.0
joblib>=1.0.0
pytest>=6.2.0
```

**Expected Fix Time**: 10 minutes

---

### 3. Fix Config Paths
**Edit** `config.py`:
```python
import os

DATA_ROOT = os.environ.get(
    'EDAIC_DATA_ROOT',
    os.path.join(os.path.expanduser('~'), 'Downloads', 'E-DAIC', 'data')
)
LABELS_DIR = os.environ.get(
    'EDAIC_LABELS_DIR',
    os.path.join(os.path.expanduser('~'), 'Downloads', 'E-DAIC', 'labels')
)
```

**Expected Fix Time**: 15 minutes

---

## Model Performance Improvements

### 4. Implement Cost-Sensitive Learning
**Edit** `src/fusion.py`, replace threshold finding:

```python
def find_cost_sensitive_threshold(model, artifacts, X_val, y_val):
    """Find threshold that minimizes cost of false negatives."""
    scaler = artifacts['scaler']
    X_selected = transform_with_selector(X_val, artifacts)
    X_sc = scaler.transform(X_selected)
    probs = model.predict_proba(X_sc)[:, 1]
    
    best_t, best_cost = 0.5, float('inf')
    FN_COST = 5  # Missing depression is 5x worse than false alarm
    FP_COST = 1
    
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        cost = fn * FN_COST + fp * FP_COST
        if cost < best_cost:
            best_cost = cost
            best_t = t
    
    return best_t
```

**Expected Impact**: +5-10% recall

---

### 5. Improve Fusion Strategy
**Edit** `src/fusion.py`, replace late fusion:

```python
def attention_based_fusion(models_scalers, X_dict, y_val=None):
    """Dynamic fusion using attention weights."""
    from sklearn.metrics import roc_auc_score
    
    # Compute per-modality validation performance
    weights = {}
    if y_val is not None:
        for m, (model, artifacts) in models_scalers.items():
            X_sel = transform_with_selector(X_dict[m], artifacts)
            X_sc = artifacts['scaler'].transform(X_sel)
            probs = model.predict_proba(X_sc)[:, 1]
            auc = roc_auc_score(y_val, probs)
            # Weight by AUC above random chance
            weights[m] = max(0, auc - 0.5)
    else:
        weights = {m: 1.0 for m in models_scalers}
    
    # Normalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    # Weighted combination
    n = len(next(iter(X_dict.values())))
    combined = np.zeros(n)
    for m, w in weights.items():
        model, artifacts = models_scalers[m]
        X_sel = transform_with_selector(X_dict[m], artifacts)
        X_sc = artifacts['scaler'].transform(X_sel)
        probs = model.predict_proba(X_sc)[:, 1]
        combined += w * probs
    
    return combined, weights
```

**Expected Impact**: +2-5% AUC

---

### 6. Add Temporal Features
**Edit** `src/visual_features_enhanced.py` and `src/audio_features_enhanced.py`:

```python
def extract_temporal_features(time_series):
    """Enhanced temporal statistics."""
    from scipy import stats
    import pandas as pd
    
    features = {
        'mean': np.mean(time_series),
        'std': np.std(time_series),
        'min': np.min(time_series),
        'max': np.max(time_series),
        'range': np.max(time_series) - np.min(time_series),
        'trend': np.polyfit(range(len(time_series)), time_series, 1)[0],
        'autocorr_1': pd.Series(time_series).autocorr(lag=1) or 0,
        'autocorr_2': pd.Series(time_series).autocorr(lag=2) or 0,
        'skewness': stats.skew(time_series),
        'kurtosis': stats.kurtosis(time_series),
    }
    return features
```

**Expected Impact**: +3-7% AUC

---

## Web Application Integration

### 7. Add Audio/Visual Model Loading
**Edit** `app.py`:

```python
# After line 46, add:
try:
    audio_model = joblib.load(os.path.join(MODELS_DIR, 'audio_model.pkl'))
    audio_scaler = joblib.load(os.path.join(MODELS_DIR, 'audio_scaler.pkl'))
    HAS_AUDIO = True
except FileNotFoundError:
    HAS_AUDIO = False

try:
    visual_model = joblib.load(os.path.join(MODELS_DIR, 'visual_model.pkl'))
    visual_scaler = joblib.load(os.path.join(MODELS_DIR, 'visual_scaler.pkl'))
    HAS_VISUAL = True
except FileNotFoundError:
    HAS_VISUAL = False
```

---

## Testing & Validation

### 8. Add Model Validation Tests
**Create** `tests/test_models.py`:

```python
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score

def test_model_performance():
    """Test that models meet minimum clinical thresholds."""
    # Load test data
    # ... load models and test data ...
    
    for model_name, model in [('text', text_model), ('visual', visual_model)]:
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        
        # Minimum clinical threshold
        assert auc > 0.70, f"{model_name} AUC {auc:.3f} below 0.70 threshold"
        
        # Check recall at 50% threshold
        preds = (probs >= 0.5).astype(int)
        recall = recall_score(y_test, preds)
        assert recall > 0.70, f"{model_name} recall {recall:.3f} below 0.70 threshold"
```

---

## Monitoring & Maintenance

### 9. Add Performance Logging
**Edit** `src/evaluate.py`:

```python
def log_model_performance(model_name, metrics, output_file='model_history.csv'):
    """Log model performance over time."""
    import datetime
    import csv
    
    timestamp = datetime.datetime.now().isoformat()
    row = [timestamp, model_name] + list(metrics.values())
    
    file_exists = os.path.exists(output_file)
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'model'] + list(metrics.keys()))
        writer.writerow(row)
```

---

## Success Metrics

### Target Performance Goals

| Metric | Current Best | Target | Minimum Acceptable |
|--------|-------------|--------|-------------------|
| AUC-ROC | 0.657 | 0.80 | 0.70 |
| Sensitivity | ~0.50 | 0.85 | 0.80 |
| Specificity | ~0.70 | 0.80 | 0.70 |
| F1-Score | 0.526 | 0.75 | 0.60 |

### Completion Criteria

- [ ] Audio modality debugged and fixed OR removed
- [ ] At least one model achieves AUC > 0.70
- [ ] requirements.txt created and tested
- [ ] Config paths made portable
- [ ] Fusion strategy improved
- [ ] Web app integrates all available modalities
- [ ] Clinical metrics (sensitivity/specificity) documented

---

## Time Estimates

| Task | Estimated Time | Priority |
|------|---------------|----------|
| Debug audio features | 2-4 hours | P0 |
| Create requirements.txt | 10 min | P1 |
| Fix config paths | 15 min | P1 |
| Implement cost-sensitive learning | 1-2 hours | P2 |
| Improve fusion | 2-3 hours | P1 |
| Add temporal features | 2-3 hours | P2 |
| Web app integration | 3-4 hours | P2 |
| Add clinical metrics | 1-2 hours | P2 |
| Testing & validation | 2-3 hours | P2 |

**Total Estimated Time**: 14-23 hours

---

*Action Plan Version: 1.0*  
*Last Updated: 2026-04-06*
