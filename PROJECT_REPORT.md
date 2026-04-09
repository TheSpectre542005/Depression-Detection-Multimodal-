# MindScan Depression Detection Project — Comprehensive Report

## Executive Summary

This document provides a detailed analysis of the MindScan (formerly Sentira) Depression Detection project, a multimodal machine learning system designed to detect depression by analyzing text, audio, and visual features from clinical interviews. The system is built on the E-DAIC (Extended DAIC-WOZ) dataset and includes both a training pipeline and a Flask web application for real-time screening.

---

## 1. Project Overview

### 1.1 Purpose
MindScan is an AI-powered mental health screening tool that aims to:
- Detect depression through multimodal analysis (text, speech, facial expressions)
- Provide early screening support using validated clinical instruments (PHQ-8)
- Offer a privacy-focused, session-only assessment experience

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Acquisition Layer                        │
│         Text │ Audio │ Visual (E-DAIC Dataset)                  │
└──────┬───────────┬───────────┬──────────────────────────────────┘
       │           │           │
┌──────▼───┐  ┌────▼────┐ ┌────▼──────┐
│  Text    │  │  Audio  │ │  Visual   │   Feature Extraction
│ Features │  │Features │ │ Features  │
└──────┬───┘  └────┬────┘ └─────┬─────┘
       │           │            │
       └───────────┼────────────┘
                   │
         ┌─────────▼──────────┐
         │   Late Fusion       │   Multimodal Fusion
         │  (Weighted Avg)   │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Depression Risk   │   Classification
         │  Score (0-1)       │
         └────────────────────┘
```

### 1.3 Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| ML Framework | scikit-learn, imbalanced-learn |
| NLP | NLTK, VADER Sentiment, TF-IDF |
| Audio | OpenSMILE (pre-extracted MFCC, eGeMAPS, BoAW) |
| Visual | OpenFace (AUs, pose, gaze), CNN features (DenseNet, VGG, ResNet) |
| Web | Flask, face-api.js |
| Frontend | HTML/CSS/JavaScript (dark glassmorphism theme) |

### 1.4 Dataset: E-DAIC (Extended DAIC-WOZ)
- **Size**: 275 clinical interview recordings
- **Modalities**: Audio, video, and text transcripts per participant
- **Labels**: PHQ-8 depression severity scores
- **Binary Classification**: PHQ-8 >= 10 (Depressed) vs < 10 (Not Depressed)

---

## 2. Feature Engineering Analysis

### 2.1 Text Features (59 features)
| Feature Category | Description | Count |
|-----------------|-------------|-------|
| VADER Sentiment | Negative, neutral, positive, compound scores | 4 |
| Linguistic | Word count, unique words, lexical diversity | 3 |
| Statistical | Average word length, average confidence | 2 |
| TF-IDF | Top 50 term frequency-inverse document frequency | 50 |

**Technique**: L1-regularized Logistic Regression with SMOTE for class balancing

### 2.2 Audio Features (Enhanced)
| Feature Category | Description | Source |
|-----------------|-------------|--------|
| BoAW | Bag-of-Audio-Words histograms | OpenSMILE 2.3.0 |
| MFCC | Mel-Frequency Cepstral Coefficients | OpenSMILE |
| eGeMAPS | Extended Geneva Minimalistic Acoustic Parameter Set | OpenSMILE |
| Prosodic | Pitch, energy, jitter, shimmer, HNR | Depression-relevant biomarkers |
| Temporal | Speech ratio, pause frequency, energy trends | Custom extraction |

**Dimensionality**: Raw ~500+ features → PCA to 25 components

### 2.3 Visual Features (Enhanced)
| Feature Category | Description | Source |
|-----------------|-------------|--------|
| CNN Features | DenseNet201, VGG16, ResNet deep features | Pre-extracted frames |
| Action Units | 17 AU intensities + presence indicators | OpenFace 2.1.0 |
| Head Pose | 3D pose (Rx, Ry, Rz) with temporal stats | OpenFace |
| Gaze | Eye gaze direction and movement | OpenFace |
| Derived | Smile intensity, sadness indicator, head movement variance | Custom |

**Dimensionality**: Raw ~1000+ features → PCA to 25 components

---

## 3. Model Architecture & Fusion Strategies

### 3.1 Unimodal Models
Each modality trains independently using:
- **Model Selection**: Logistic Regression (L1/L2), SVM (RBF/Linear), Random Forest, Gradient Boosting
- **Feature Selection**: SelectKBest with mutual information (top 30-40 features)
- **Balancing**: SMOTE with k=3 (adjusted for class distribution)
- **Calibration**: Platt scaling for probability calibration
- **Threshold Tuning**: F-beta optimization (beta=2 favors recall)

### 3.2 Fusion Strategies Implemented

| Strategy | Method | Description |
|----------|--------|-------------|
| **Late Fusion** | Weighted average of unimodal probabilities | Weights based on validation AUC; excludes modalities below 0.55 AUC |
| **Stacking Fusion** | Meta-learner (Logistic Regression) on unimodal outputs | Learns optimal combination from base predictions |
| **Early Fusion** | Concatenate all features before training | Single model on combined feature space |

### 3.3 Threshold Optimization
- **Range**: 0.25 to 0.65 (constrained to prevent degenerate predictions)
- **Metric**: F-beta score with beta=2 (favors recall for safety-critical depression detection)

---

## 4. Model Performance & Accuracy Analysis

### 4.1 Current Results (from README)

| Model | Accuracy | F1 | AUC-ROC |
|-------|----------|-----|---------|
| **Text Only** | 0.758 | 0.429 | 0.591 |
| **Audio Only** | 0.394 | 0.474 | 0.548 |
| **Visual Only** | 0.455 | 0.526 | 0.657 |
| **Late Fusion** | 0.758 | 0.429 | 0.591 |
| **Early Fusion** | 0.576 | 0.462 | 0.635 |

### 4.2 Performance Analysis

#### Strengths:
1. **Visual Modality** shows highest AUC-ROC (0.657), suggesting facial expressions contain the most discriminative signals for depression
2. **Text Modality** achieves good accuracy (0.758) but suffers from low F1 (0.429), indicating class imbalance issues
3. **Fusion approaches** demonstrate attempts to combine strengths

#### Critical Weaknesses:
1. **Overall AUC-ROC scores are below 0.7**, which is generally considered the minimum threshold for clinically useful screening tools
2. **Low F1 scores across all models** (0.429-0.526) indicate poor balance between precision and recall
3. **Audio modality performs poorly** (accuracy 0.394, worse than random)
4. **Late Fusion shows no improvement** over Text-only (same accuracy/F1), suggesting fusion logic needs revision
5. **Class imbalance is poorly handled** despite SMOTE application

### 4.3 Clinical Relevance Thresholds

For depression screening, the following clinical thresholds are recommended:

| Metric | Minimum Acceptable | Good | Excellent |
|--------|-------------------|------|-----------|
| Sensitivity (Recall) | > 0.80 | > 0.85 | > 0.90 |
| Specificity | > 0.70 | > 0.80 | > 0.90 |
| AUC-ROC | > 0.70 | > 0.80 | > 0.90 |
| F1-Score | > 0.60 | > 0.70 | > 0.80 |

**Current Status**: None of the models meet the "Minimum Acceptable" threshold for clinical deployment.

---

## 5. Strengths of the Project

### 5.1 Technical Strengths
1. **Comprehensive Feature Engineering**
   - Multi-layered feature extraction (BoAW, CNN, prosodic)
   - Temporal dynamics captured (velocity, rate of change)
   - Domain-informed features (depression-relevant AUs)

2. **Robust Pipeline Design**
   - Proper train/validation/test splits (70/15/15)
   - Cross-validation with stratification
   - Data leakage prevention (TF-IDF fit only on train)

3. **Model Selection Framework**
   - Multiple candidate models evaluated
   - Automated selection based on CV F1
   - Probability calibration implemented

4. **Fusion Strategy Variety**
   - Late fusion with smart weighting
   - Stacking with meta-learner
   - Early fusion comparison

### 5.2 Engineering Strengths
1. **Modular Code Architecture** — Well-organized src/ directory with clear separation
2. **Configuration Management** — Centralized config.py for all hyperparameters
3. **Logging & Monitoring** — Comprehensive logging throughout pipeline
4. **Web Application** — Full Flask app with API endpoints
5. **Testing** — Unit tests for core functionality
6. **Documentation** — Detailed README and design documents

### 5.3 Design Strengths
1. **User Experience** — Dark glassmorphism UI design
2. **Privacy-First** — Session-only processing, no data storage
3. **Accessibility** — Keyboard shortcuts, screen reader support
4. **Crisis Resources** — Support numbers included

---

## 6. Weaknesses & Critical Issues

### 6.1 Critical Performance Issues

#### Issue 1: Poor Overall Accuracy
- **Problem**: Best AUC-ROC is only 0.657 (Visual), far below clinical usefulness
- **Impact**: High false positive and false negative rates
- **Risk**: Could cause harm by missing depressed individuals or causing unnecessary alarm

#### Issue 2: Audio Modality Failure
- **Problem**: Audio model performs worse than random (accuracy 0.394)
- **Root Cause**: Likely data quality issues or inappropriate feature extraction
- **Impact**: Corrupts fusion results, adds noise rather than signal

#### Issue 3: Ineffective Fusion
- **Problem**: Late Fusion achieves identical scores to Text-only
- **Root Cause**: Text modality dominates; other modalities likely have low confidence
- **Impact**: No benefit from multimodal approach

#### Issue 4: Low F1 Scores
- **Problem**: Best F1 is only 0.526 (Visual)
- **Impact**: Poor balance between catching true cases (recall) and avoiding false alarms (precision)
- **Clinical Risk**: Missing depressed individuals (low recall) or over-flagging healthy individuals (low precision)

### 6.2 Data Issues

#### Issue 5: Dataset Size
- **Problem**: Only 275 samples with likely class imbalance
- **Impact**: Insufficient data for deep learning or complex models
- **Evidence**: SMOTE required, models struggle to generalize

#### Issue 6: Class Imbalance
- **Problem**: Depression prevalence in dataset may not reflect real-world rates
- **Current Handling**: SMOTE may be creating synthetic samples that don't preserve meaningful patterns
- **Impact**: Models biased toward majority class

### 6.3 Technical Debt

#### Issue 7: Hardcoded Paths
- **File**: `config.py` lines 5-12
- **Problem**: Absolute Windows paths hardcoded
- **Impact**: Not portable across environments

#### Issue 8: Missing Requirements File
- **Problem**: No requirements.txt found in the project
- **Impact**: Difficult to reproduce environment

#### Issue 9: Error Handling
- **Problem**: Silent failures in feature extraction (try/except with only debug logging)
- **File**: `audio_features_enhanced.py`, `visual_features_enhanced.py`
- **Impact**: Difficult to diagnose data loading issues

#### Issue 10: Incomplete Web App Integration
- **Problem**: Web app only uses text model; audio/visual models not integrated
- **File**: `app.py` lines 32-46
- **Impact**: Web app can't leverage multimodal capabilities

### 6.4 Algorithmic Issues

#### Issue 11: Threshold Constraint
- **Problem**: Threshold search limited to 0.25-0.65
- **Impact**: May miss optimal thresholds outside this range
- **File**: `config.py` lines 29-32

#### Issue 12: PCA Explained Variance
- **Problem**: PCA reduces to fixed 25 components without checking explained variance
- **Impact**: May lose important information or keep noise
- **Suggested Fix**: Use cumulative explained variance threshold (e.g., 95%)

#### Issue 13: Feature Selection Instability
- **Problem**: SelectKBest with mutual information may select different features on different runs
- **Impact**: Model reproducibility issues

---

## 7. Recommendations for Improvement

### 7.1 Immediate Fixes (High Priority)

#### Fix 1: Audio Feature Debugging
```python
# Add detailed logging to identify why audio performs poorly
# In audio_features_enhanced.py:
def diagnose_audio_features(pid, data_root):
    """Diagnostic function to check data quality."""
    feat_dir = os.path.join(data_root, f"{pid}_P", "features")
    
    files_to_check = [
        f"{pid}_OpenSMILE2.3.0_mfcc.csv",
        f"{pid}_OpenSMILE2.3.0_egemaps.csv",
        f"{pid}_BoAW_openSMILE_2.3.0_MFCC.csv",
    ]
    
    for f in files_to_check:
        path = os.path.join(feat_dir, f)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"{f}: exists={exists}, size={size} bytes")
```

#### Fix 2: Environment Portability
```python
# In config.py, replace hardcoded paths:
import os

DATA_ROOT = os.environ.get(
    'EDAIC_DATA_ROOT',
    os.path.join(os.path.expanduser('~'), 'Downloads', 'E-DAIC', 'data')
)
```

#### Fix 3: Create requirements.txt
```
flask>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
nltk>=3.6
vaderSentiment>=3.3.0
scipy>=1.7.0
joblib>=1.0.0
pytest>=6.2.0
```

### 7.2 Model Performance Improvements (High Priority)

#### Fix 4: Address Class Imbalance Properly
```python
# Instead of SMOTE, try:
from sklearn.utils.class_weight import compute_class_weight

# Use class_weight='balanced' in models (already done) AND
# adjust the decision threshold based on cost-sensitive learning
# False negatives (missing depression) are more costly than false positives

def find_cost_sensitive_threshold(y_val, probs, fn_cost=5, fp_cost=1):
    """Find threshold that minimizes weighted cost of errors."""
    best_t, best_cost = 0.5, float('inf')
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        cost = fn * fn_cost + fp * fp_cost
        if cost < best_cost:
            best_cost = cost
            best_t = t
    return best_t
```

#### Fix 5: Temporal Modeling
```python
# Current approach aggregates temporal features statically
# Consider using:
from sklearn.ensemble import HistGradientBoostingClassifier
# or sequence models (LSTM/Transformer) for temporal patterns

# For now, extract more temporal statistics:
def extract_temporal_features(time_series):
    """Enhanced temporal feature extraction."""
    features = {
        'mean': np.mean(time_series),
        'std': np.std(time_series),
        'trend': np.polyfit(range(len(time_series)), time_series, 1)[0],
        'autocorr_1': pd.Series(time_series).autocorr(lag=1),
        'autocorr_2': pd.Series(time_series).autocorr(lag=2),
        'zero_crossings': len(np.where(np.diff(np.sign(time_series)))[0]),
        'entropy': stats.entropy(np.histogram(time_series, bins=10)[0] + 1e-10),
    }
    return features
```

#### Fix 6: Cross-Modal Attention
```python
# Current fusion treats modalities independently
# Implement attention-based fusion:

class AttentionFusion:
    """Learn attention weights for each modality dynamically."""
    
    def __init__(self, n_modalities=3):
        self.attention_weights = None
        self.meta_learner = LogisticRegression(class_weight='balanced')
    
    def fit(self, X_dict, y):
        # X_dict: {'text': X_text, 'audio': X_audio, 'visual': X_visual}
        # Learn attention based on validation performance
        self.attention_weights = {}
        for mod, X in X_dict.items():
            # Compute validation AUC for weighting
            scores = cross_val_score(
                LogisticRegression(class_weight='balanced'), 
                X, y, cv=3, scoring='roc_auc'
            )
            self.attention_weights[mod] = max(0, scores.mean() - 0.5)
        
        # Normalize weights
        total = sum(self.attention_weights.values())
        if total > 0:
            self.attention_weights = {k: v/total for k, v in self.attention_weights.items()}
        else:
            self.attention_weights = {k: 1/len(X_dict) for k in X_dict}
```

### 7.3 Architecture Improvements (Medium Priority)

#### Fix 7: Implement Proper Web App Integration
```python
# In app.py, add audio/visual model loading and prediction:

try:
    audio_model = joblib.load(os.path.join(MODELS_DIR, 'audio_model.pkl'))
    audio_scaler = joblib.load(os.path.join(MODELS_DIR, 'audio_scaler.pkl'))
    HAS_AUDIO_MODEL = True
except FileNotFoundError:
    HAS_AUDIO_MODEL = False
    logger.warning("Audio model not found")

try:
    visual_model = joblib.load(os.path.join(MODELS_DIR, 'visual_model.pkl'))
    visual_scaler = joblib.load(os.path.join(MODELS_DIR, 'visual_scaler.pkl'))
    HAS_VISUAL_MODEL = True
except FileNotFoundError:
    HAS_VISUAL_MODEL = False
    logger.warning("Visual model not found")

# Add endpoints for real-time audio/visual feature extraction
@app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio():
    """Analyze audio features from recorded interview."""
    # Implement Web Audio API integration
    pass

@app.route('/api/analyze-visual', methods=['POST'])
def analyze_visual():
    """Analyze visual features from webcam stream."""
    # Implement face-api.js backend processing
    pass
```

#### Fix 8: Add Model Monitoring & Drift Detection
```python
# Add to evaluate.py:

def detect_feature_drift(X_train, X_test, threshold=0.05):
    """Detect if test features have drifted from training."""
    from scipy.stats import ks_2samp
    
    drift_detected = []
    for col in range(X_train.shape[1]):
        stat, p_value = ks_2samp(X_train[:, col], X_test[:, col])
        if p_value < threshold:
            drift_detected.append(col)
    
    return drift_detected

def monitor_prediction_drift(y_pred_history, window_size=100):
    """Monitor if prediction distribution changes over time."""
    if len(y_pred_history) < window_size * 2:
        return False
    
    recent = y_pred_history[-window_size:]
    older = y_pred_history[-(window_size*2):-window_size]
    
    # Chi-square test for distribution change
    from scipy.stats import chi2_contingency
    contingency = [[sum(recent), len(recent) - sum(recent)],
                   [sum(older), len(older) - sum(older)]]
    _, p_value, _, _ = chi2_contingency(contingency)
    
    return p_value < 0.05
```

### 7.4 Clinical Validation (High Priority)

#### Fix 9: Implement Clinically Meaningful Metrics
```python
# Add to evaluate.py:

def clinical_metrics(y_true, y_pred, y_prob):
    """Compute clinically relevant metrics."""
    from sklearn.metrics import (
        sensitivity_score, specificity_score,
        ppv, npv  # Positive/Negative Predictive Value
    )
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Likelihood ratios
    lr_positive = sensitivity / (1 - specificity) if specificity < 1 else float('inf')
    lr_negative = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'lr_positive': lr_positive,
        'lr_negative': lr_negative,
        'diagnostic_odds_ratio': lr_positive / lr_negative if lr_negative > 0 else float('inf')
    }
```

#### Fix 10: Add Confidence Intervals
```python
from sklearn.utils import resample

def bootstrap_ci(y_true, y_prob, n_bootstrap=1000, ci=0.95):
    """Compute confidence intervals for AUC using bootstrap."""
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    
    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    lower_idx = int((1 - ci) / 2 * len(sorted_scores))
    upper_idx = int((1 + ci) / 2 * len(sorted_scores))
    
    return sorted_scores[lower_idx], sorted_scores[upper_idx]
```

---

## 8. Risk Assessment

### 8.1 Clinical Risk
| Risk | Severity | Mitigation |
|------|----------|------------|
| False negatives (missing depression) | **HIGH** | Add explicit disclaimer; never use as sole diagnostic tool |
| False positives (unnecessary alarm) | **MEDIUM** | Provide context in results; suggest professional consultation |
| Algorithmic bias | **HIGH** | Test on diverse populations; document training data demographics |
| Data privacy | **MEDIUM** | Keep session-only processing; add encryption in transit |

### 8.2 Technical Risk
| Risk | Severity | Mitigation |
|------|----------|------------|
| Model drift | **MEDIUM** | Implement monitoring; schedule periodic retraining |
| Dependency vulnerabilities | **LOW** | Pin versions; run security audits |
| Scalability | **LOW** | Models are lightweight; consider caching |

---

## 9. Deployment Checklist

### Before Production Deployment:

- [ ] **Fix audio modality** — Debug and fix or remove
- [ ] **Achieve minimum clinical thresholds** — AUC > 0.70, Sensitivity > 0.80
- [ ] **Implement comprehensive testing** — Unit, integration, end-to-end
- [ ] **Add model monitoring** — Drift detection, performance tracking
- [ ] **Security audit** — Input validation, dependency scanning
- [ ] **Clinical validation** — Test on held-out clinical dataset
- [ ] **IRB approval** — If used in research/clinical settings
- [ ] **Legal review** — Disclaimer adequacy, liability assessment

---

## 10. Conclusion

The MindScan Depression Detection project demonstrates strong engineering practices with a modular architecture, comprehensive feature extraction, and thoughtful UI design. However, **the current model performance is insufficient for clinical deployment**.

### Key Takeaways:
1. **Visual modality shows most promise** (AUC 0.657) and should be prioritized
2. **Audio modality needs debugging** — currently degrades performance
3. **Fusion strategy needs redesign** — current approach provides no benefit
4. **Class imbalance handling** requires more sophisticated approaches
5. **Clinical validation** is essential before any real-world use

### Recommended Path Forward:
1. **Short-term**: Fix technical debt, debug audio features, improve class balancing
2. **Medium-term**: Collect more data, implement temporal models, clinical validation
3. **Long-term**: Deploy as research tool only, gather feedback, iterate

---

## Appendix A: File Structure

```
Depression_Detection/
├── MindScan_Design_Document.md    # UI/UX design specifications
├── setup_project.py               # Project initialization
└── depression_project/
    ├── README.md                  # Project overview
    ├── main.py                    # ML training pipeline
    ├── app.py                     # Flask web application
    ├── config.py                  # Centralized configuration
    ├── requirements.txt           # MISSING — needs to be created
    ├── src/
    │   ├── __init__.py
    │   ├── text_features.py       # Text feature extraction
    │   ├── audio_features_enhanced.py  # Audio features (BoAW + prosody)
    │   ├── visual_features_enhanced.py # Visual features (CNN + OpenFace)
    │   ├── fusion.py              # Model training & fusion
    │   ├── evaluate.py            # Evaluation metrics & plots
    │   └── load_labels.py         # Label loading from E-DAIC
    ├── tests/
    │   └── test_app.py            # Unit tests
    ├── data/features/             # Extracted feature CSVs
    ├── models/                    # Trained model files (.pkl)
    ├── results/                   # Evaluation outputs
    ├── templates/                 # HTML templates
    └── static/                    # CSS, JS, assets
```

## Appendix B: References

1. E-DAIC Dataset: [DAIC-WOZ Database](https://dcapswoz.ict.usc.edu/)
2. PHQ-8 Questionnaire: [Patient Health Questionnaire](https://patient.info/doctor/patient-health-questionnaire-phq-9)
3. OpenSMILE: [Feature Extraction Tool](https://audeering.github.io/opensmile/)
4. OpenFace: [Facial Behavior Analysis](https://github.com/TadasBaltrusaitis/OpenFace)

---

*Report generated: 2026-04-06*  
*Project version: Enhanced Multimodal Pipeline*  
*Author: Claude Code Analysis*
