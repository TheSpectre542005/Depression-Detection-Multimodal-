# 🧠 Depression Detection by Multimodal Analysis

A multimodal machine learning system that detects depression by analyzing **text**, **speech**, and **facial expressions** from clinical interviews. Built on the E-DAIC (Extended DAIC-WOZ) dataset with a Flask web application for real-time screening.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

Depression is a major mental health disorder often underdiagnosed due to reliance on subjective self-reporting. This project builds an automated screening tool that combines three communication channels:

| Modality | Features Extracted | Method |
|----------|-------------------|--------|
| **Text** | Sentiment (VADER), TF-IDF, linguistic markers | NLP |
| **Audio** | MFCCs, eGeMAPS (pitch, energy, speaking rate) | OpenSMILE |
| **Visual** | Facial Action Units, head pose, gaze | OpenFace / face-api.js |

The system uses **late fusion** to combine predictions from unimodal L1-regularized Logistic Regression models into a final depression risk score.

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
     │  Late Fusion    │        Multimodal Fusion
     │  (Weighted Avg) │
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
├── requirements.txt            # Python dependencies
├── src/
│   ├── load_labels.py          # Load PHQ-8 labels from E-DAIC
│   ├── text_features.py        # Text feature extraction (VADER + TF-IDF)
│   ├── audio_features.py       # Audio feature extraction (MFCC + eGeMAPS)
│   ├── visual_features.py      # Visual feature extraction (AUs + pose)
│   ├── fusion.py               # Model training + late fusion
│   └── evaluate.py             # Metrics, plots, confusion matrices
├── models/                     # Trained .pkl model files
├── data/features/              # Extracted feature CSVs
├── results/                    # Evaluation outputs (plots, CSV)
├── templates/index.html        # Web app frontend (SPA)
├── static/
│   ├── css/style.css           # Dark theme + glassmorphism
│   └── js/app.js               # Frontend logic + face-api.js
└── notebooks/                  # Exploratory analysis (optional)
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

This trains unimodal models, performs late fusion, and saves results to `results/`.

### Run the Web Application

```bash
python app.py
# Open http://localhost:5000
```

---

## 🌐 Web Application

The MindScan web app provides a complete screening experience:

### User Flow

1. **Landing Page** — Project overview and disclaimer
2. **PHQ-8 Survey** — Standard 8-question clinical questionnaire (keyboard shortcuts: 0-3)
3. **Virtual Assistant Interview** — AI chatbot asks 8 clinical-style questions while webcam captures facial expressions via face-api.js
4. **Results Dashboard** — Risk gauge, PHQ-8 breakdown, text analysis with sentiment, and facial expression chart

### Combined Risk Score

The final risk score uses 3-way late fusion:

| Modality | Weight | Source |
|----------|--------|--------|
| PHQ-8 Score | 35% | Clinical questionnaire |
| Text Analysis | 35% | Trained ML model (L1 LogReg) |
| Facial Analysis | 30% | face-api.js expression detection |

---

## 📊 ML Pipeline Results

| Model | Accuracy | F1 | AUC-ROC |
|-------|----------|-----|---------|
| Text Only | 0.758 | 0.429 | 0.591 |
| Audio Only | 0.394 | 0.474 | 0.548 |
| Visual Only | 0.455 | 0.526 | 0.657 |
| Late Fusion | 0.758 | 0.429 | 0.591 |
| Early Fusion | 0.576 | 0.462 | 0.635 |

### Key Techniques

- **SMOTE** applied inside cross-validation folds (prevents data leakage)
- **L1-regularized Logistic Regression** for built-in feature selection
- **PCA** dimensionality reduction (248 audio → 20, 214 visual → 20)
- **Smart fusion weights** excluding modalities with AUC ≤ 0.52
- **Constrained thresholds** (0.25–0.65) to prevent degenerate predictions

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
| ML | scikit-learn, imbalanced-learn |
| NLP | NLTK, VADER Sentiment |
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
