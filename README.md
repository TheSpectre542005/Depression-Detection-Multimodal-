# 🧠 SENTIRA — Multimodal Depression Detection

A multimodal machine learning system that detects depression by analyzing **text**, **speech**, and **facial expressions** from clinical interviews. Built on the E-DAIC (Extended DAIC-WOZ) dataset with a Flask web application for real-time screening.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

Depression is a major mental health disorder often underdiagnosed due to reliance on subjective self-reporting. SENTIRA is an automated screening tool that combines three communication channels with attention-based fusion:

| Modality | Features Extracted | Method |
|----------|-------------------|--------|
| **Text** | Sentiment (VADER), TF-IDF, clinical NLP markers, SBERT embeddings | NLP |
| **Audio** | Prosodic biomarkers, compact MFCC, eGeMAPS features | OpenSMILE |
| **Visual** | Facial Action Units, head pose, gaze, CNN features | OpenFace / face-api.js |

The system uses **attention-based late fusion** that automatically learns modality weights from validation AUC — unreliable modalities (e.g., audio with AUC ~0.55) get near-zero weight while strong modalities dominate.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│           Data Acquisition Layer            │
│   Text │ Audio │ Visual                     │
└────┬────────┬────────┬──────────────────────┘
     │        │        │
┌────▼───┐ ┌──▼────┐ ┌─▼──────┐
│  Text  │ │ Audio │ │ Visual │   Preprocessing
│Preproc.│ │Preproc│ │Preproc.│
└────┬───┘ └──┬────┘ └─┬──────┘
     │        │        │
┌────▼───┐ ┌──▼────┐ ┌─▼──────┐
│  Text  │ │ Audio │ │ Visual │   Feature Extraction
│Features│ │Feature│ │Features│
└────┬───┘ └──┬────┘ └─┬──────┘
     │        │        │
     └────────┼────────┘
              │
     ┌────────▼────────┐
     │ AttentionFusion  │        Learned Weights
     │ (AUC-weighted)   │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │   Depression    │        Classification
     │   Detection     │
     │  Output + Score │
     └─────────────────┘
```

---

## 📂 Project Structure

```
depression_project/
├── app.py                      # Flask web application
├── main.py                     # ML pipeline (train + evaluate)
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── src/
│   ├── load_labels.py          # Load PHQ-8 labels from E-DAIC
│   ├── text_features.py        # Text features (VADER + TF-IDF + SBERT)
│   ├── audio_features.py       # Basic audio features (MFCC + eGeMAPS)
│   ├── audio_features_enhanced.py  # Enhanced audio (prosodic biomarkers)
│   ├── visual_features.py      # Basic visual features (AUs + pose)
│   ├── visual_features_enhanced.py # Enhanced visual (CNN + OpenFace)
│   ├── visual_browser_model.py # Browser-compatible visual model
│   ├── fusion.py               # Model training + AttentionFusion
│   └── evaluate.py             # Metrics, plots, bootstrap CIs
├── models/                     # Trained .pkl model files
├── data/features/              # Extracted feature CSVs
├── results/                    # Evaluation outputs (plots, CSV)
├── templates/index.html        # Web app frontend (SPA)
├── static/
│   ├── css/style.css           # Dark theme + glassmorphism
│   └── js/app.js               # Frontend logic + face-api.js
├── tests/test_app.py           # Unit tests
├── validate_models.py          # Model validation script
└── _archive/                   # Archived experimental scripts
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- E-DAIC dataset (for training pipeline)

### Installation

```bash
git clone https://github.com/TheSpectre542005/Depression-Detection-Multimodal-.git
cd Depression-Detection-Multimodal-
pip install -r requirements.txt
```

### Run the ML Pipeline

```bash
python main.py
```

This runs:
1. **5-Fold Stratified CV** with PCA fitted inside each fold (no leakage)
2. **Model selection** across LR, SVM, RF, GB, XGBoost per modality
3. **AttentionFusion** with AUC-learned weights
4. **Holdout evaluation** with bootstrap 95% CIs
5. **Cost-sensitive thresholding** (FN weighted 5x more than FP)

### Run the Web Application

```bash
python app.py
# Open http://localhost:5000
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🌐 Web Application

The SENTIRA web app provides a complete screening experience:

1. **Landing Page** — Project overview and disclaimer
2. **Device Setup** — Camera and microphone verification
3. **Virtual Interview** — AI chatbot (Mira) asks clinical questions while analyzing text, voice, and facial expressions
4. **PHQ-8 Survey** — Standard 8-question clinical questionnaire
5. **Results Dashboard** — Risk gauge, modality breakdown, expression chart

### Combined Risk Score

The final risk score uses adaptive multimodal fusion:

| Modality | Source | Weight |
|----------|--------|--------|
| PHQ-8 Score | Clinical questionnaire | Adaptive |
| Text Analysis | Trained ML model | Adaptive |
| Voice Analysis | Client-side audio features | Adaptive |
| Facial Analysis | face-api.js expressions | Adaptive |

---

## 📊 Key Techniques

- **AttentionFusion** — Learns modality weights from validation AUC (replaces static weights)
- **SMOTE + Mixup + Noise Augmentation** — Handles class imbalance
- **PCA inside CV folds** — Prevents data leakage
- **Cost-sensitive thresholding** — FN weighted 5x (missing depression is critical)
- **Bootstrap CIs** — 95% confidence intervals on all metrics
- **SBERT embeddings** — Optional sentence-transformer features for improved text AUC

---

## 📁 Dataset

This project uses the **E-DAIC (Extended DAIC-WOZ)** dataset:

- 275 clinical interview recordings
- Audio, video, and text transcripts per participant
- PHQ-8 depression severity labels
- **Binary classification**: PHQ-8 ≥ 10 → Depressed

> The dataset is not included in this repository due to licensing restrictions.

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| ML | scikit-learn, imbalanced-learn, XGBoost |
| NLP | NLTK, VADER Sentiment, sentence-transformers |
| Audio | OpenSMILE (pre-extracted) |
| Visual | OpenFace (training), face-api.js (web) |
| Web | Flask, HTML/CSS/JavaScript |
| Design | Glassmorphism, CSS animations |

---

## ⚠️ Disclaimer

This is a **screening tool for research and educational purposes only**. It does **NOT** provide medical diagnosis. If you or someone you know is struggling with depression, please contact a licensed mental health professional or call a crisis helpline.

---

## 📄 License

This project is for academic use. See [LICENSE](LICENSE) for details.
