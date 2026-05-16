# SENTIRA Project Functional Report

## 1. Overview

SENTIRA is a multimodal depression screening application that combines a Flask web interface with machine learning models trained on the E-DAIC dataset. The system uses:

- **Text analysis** from interview transcripts
- **Audio/voice analysis** from browser microphone recordings
- **Visual/facial expression analysis** from camera frames
- **PHQ-8 questionnaire** responses as a clinical baseline

The application is designed as a screening tool, not a clinical diagnostic device.

---

## 2. Website Architecture and Workflow

### 2.1 Frontend Flow

The website is built as a single-page app in `templates/index.html` plus `static/js/app.js`.

The user journey is divided into five main screens:

1. **Landing page**
   - Introduces the application and scientific goals
   - Offers a "Start Free Scan" button to begin

2. **Setup screen**
   - Checks camera and microphone access
   - Displays live camera preview and audio waveform
   - Allows users to test the voice assistant with TTS

3. **Interview screen**
   - Asks a sequence of scripted questions in casual conversational format
   - Uses browser speech recognition (`SpeechRecognition`/`webkitSpeechRecognition`) to transcribe user replies
   - Records the audio locally for duration and volume analysis
   - Captures facial expressions via `face-api.js`

4. **PHQ-8 questionnaire**
   - Presents 8 standard clinical questions
   - Collects numeric answers 0–3 for each item
   - Computes a PHQ-8 score and severity level in-browser

5. **Results screen**
   - Displays a combined risk score
   - Shows modality-specific probabilities and weights
   - Presents `Low / Moderate / High` risk categories

### 2.2 Frontend Data Capture

The browser collects three types of real-time data:

- **Text**: transcribed interview responses and user-typed text
- **Audio**: microphone waveform, speech segments, pause ratio, speech rate, energy levels
- **Video**: facial expression probabilities from `face-api.js` (happy, sad, angry, surprised, fearful, disgusted, neutral)

The frontend also supports:

- Text-to-speech voice assistant (`speechSynthesis`)
- Audio recording and playback via `MediaRecorder`
- Camera and microphone checks before the interview starts
- Smooth UI transitions and animated status indicators

### 2.3 Backend API and Inference

The backend is served by `app.py` using Flask.

Key routes:

- `GET /` → serve `index.html`
- `POST /api/phq` → validate and score PHQ-8 responses
- `POST /api/analyze-text` → extract text features, run text model inference, return probability
- `POST /api/predict` → run full combined inference using available modalities

The backend logic includes:

- Loading trained models from `models/`
- Loading `tfidf_vectorizer` for text inference
- Optional SBERT embedding support if dependencies are installed
- Fallback handling when models are absent
- Computing combined risk scores using adaptive weights

---

## 3. How the Models Are Trained

### 3.1 Data Sources

Training uses extracted feature CSV files in `data/features/`:

- `master_labels.csv` — PHQ-8 labels and participant IDs
- `text_features.csv` — precomputed text features
- `audio_features_enhanced.csv` — precomputed audio features
- `visual_features.csv` — precomputed visual features

The data is merged by participant ID and labels are used as binary targets:
- `label = 1` if PHQ-8 score ≥ 10
- `label = 0` otherwise

### 3.2 Training Pipeline in `main.py`

The pipeline is a 5-fold stratified cross-validation process.

For each fold:

1. Split data into training and test sets preserving class ratios
2. Build three modality-specific pipelines:
   - **Text pipeline**: VarianceThreshold → StandardScaler → SMOTE → VotingClassifier
   - **Audio pipeline**: VarianceThreshold → SelectKBest(50) → StandardScaler → SMOTE → VotingClassifier
   - **Visual pipeline**: VarianceThreshold → PCA(30) → StandardScaler → SMOTE → VotingClassifier

3. Train each pipeline on the fold's training set
4. Predict probabilities on the fold's held-out test set
5. Compute internal validation AUC for each modality using nested 3-fold CV on training data
6. Generate fusion weights using modality validation AUC values
7. Compute fused probability as a weighted sum of modality probabilities
8. Record fold metrics for text, audio, visual, and fusion predictions

### 3.3 Modality Models

Each modality uses a `VotingClassifier` ensemble with four submodels:

- Logistic Regression
- Support Vector Classifier
- Random Forest
- Gradient Boosting

Each ensemble is configured with class weighting and regularization tuned for the small, imbalanced dataset.

### 3.4 Feature Engineering

#### Text Features

Text features include:

- VADER sentiment scores (positive, neutral, negative, compound)
- Linguistic statistics (word count, lexical diversity, average word length)
- Clinical NLP markers (depression lexicon frequency, pronoun ratios, absolutist language, negation, hedging)
- TF-IDF vectors of the transcript
- Optional SBERT sentence embeddings reduced via PCA

#### Audio Features

Audio features focus on clinically meaningful acoustic markers:

- Pitch/F0
- Energy and loudness
- Jitter and shimmer
- Harmonics-to-noise ratio
- Pause statistics
- MFCC statistics
- Compact eGeMAPS features

The audio pipeline selects the top 50 audio features from over 1,200 candidates using mutual information.

#### Visual Features

Visual features are based on OpenFace output and include:

- Facial Action Unit intensities and confidences
- Head pose and gaze features
- Frame-level aggregations: mean, standard deviation, min, max
- Percentage of active AU frames

The visual pipeline reduces this high-dimensional feature set to 30 PCA components.

### 3.5 Fusion Strategy

Fusion is performed using a learned late-fusion weight scheme:

- Each modality predicts a depression probability on the test fold
- Inner CV on training data computes AUC for text/audio/visual models
- Weight for each modality = max(0.01, AUC − 0.5)
- Weights are normalized to sum to 1
- Final fused score = weighted sum of modality probabilities

This strategy is designed so that unreliable modalities contribute less to the final output.

### 3.6 Evaluation and Metrics

The system evaluates model quality using:

- ROC-AUC
- Accuracy
- F1-score
- Precision and recall
- Sensitivity and specificity
- Average precision (AP)
- Balanced accuracy

Visual outputs are saved to `results/` with ROC curves, calibration curves, confusion matrices, and probability distributions.

---

## 4. Inference and Production Behavior

### 4.1 Server-side Inference

The server loads trained models and uses them as follows:

- The text model is loaded from `models/final_text_model.pkl` if available
- TF-IDF vectorizer is loaded from `models/text_tfidf.pkl`
- SBERT + PCA components are loaded if the environment has `sentence-transformers`
- A browser-compatible visual model may also be loaded for server-side visual inference

### 4.2 Text Inference

Text inference occurs in `app.py` via `extract_text_features()` and `predict_text_prob()`.

Steps:

1. Clean and preprocess the input text
2. Extract sentiment, lexical, and clinical NLP features
3. Compute TF-IDF vectors or use zeros if vectorizer is missing
4. Compute SBERT features if available
5. Truncate or pad the final feature array to the model's expected size
6. Predict depression probability using the loaded ensemble

### 4.3 Audio and Visual Inference

For the web app, audio and visual inference is primarily handled in the browser.

- Browser collects expression scores from `face-api.js`
- Browser computes audio metrics from Web Audio API
- These values are sent to the server as `audioData` and `visualData`

If a browser-compatible visual model is loaded, the server also runs a secondary visual prediction using aggregated expression features.

### 4.4 Combined Risk Calculation

The final combined risk score is computed using adaptive weights:

- PHQ-8 score is normalized and always included
- Text probability weight scales with word count
- Visual probability weight scales with number of facial samples
- Audio probability weight scales with number of audio samples and a reliability flag

Weights are normalized so they sum to 1.0.

The final risk categories are:

- `High` if fused probability ≥ 0.55
- `Moderate` if fused probability ≥ 0.38
- `Low` otherwise

This combined approach enables the app to integrate subjective clinical responses with objective behavioral signals.

---

## 5. Practical Limitations and Justification of Inaccuracies

### 5.1 Dataset Size and Generalization

- The E-DAIC dataset contains only ~219 usable participants for the final model.
- Small sample size makes the model vulnerable to overfitting.
- This is why the project emphasizes leakage-free cross-validation and nested validation.

### 5.2 Class Imbalance

- Depressed cases are the minority class (~30%).
- Accuracy alone is misleading because a naive majority classifier could appear strong.
- The project uses SMOTE and evaluation metrics such as recall, F1, and ROC-AUC to measure real effectiveness.

### 5.3 Modality Noise and Mismatch

- **Audio quality** varies widely in browser recordings. Background noise, microphone differences, and room acoustics reduce reliability.
- **Visual data** from `face-api.js` is not the same as the training data from OpenFace/AU features. There is a domain mismatch.
- **Text transcripts** may be imperfect if speech recognition fails. The frontend uses browser STT, which can mis-transcribe casual speech.

### 5.4 PHQ-8 as a Proxy Label

- The model uses PHQ-8 thresholding as a binary proxy for clinical depression.
- PHQ-8 is a screening instrument, not a definitive diagnosis.
- This limits the reliability of the label and must be stated explicitly.

### 5.5 Fusion and Weighting Trade-offs

- The final fusion formula is heuristic and uses weights that depend on modality availability and sample confidence.
- While adaptive, this weight scheme is not guaranteed to be optimal for all users.
- It is justified as a safety-oriented design that prioritizes the PHQ-8 instrument and downweights poor-quality signals.

### 5.6 Performance Expectations

- The model is intended for screening, not diagnosis.
- It should be described as an auxiliary tool that provides additional behavioral insights.
- False positives and false negatives are expected due to data noise, subjectivity, and the limited training dataset.

---

## 6. How to Explain This Project

### 6.1 What You Built

- A **Flask-based web interface** for multimodal mental health screening
- A **text model** trained on linguistic and sentiment features
- An **adaptive fusion layer** that combines PHQ-8, text, audio, and facial signals
- A **frontend experience** with voice interaction, camera/mic checks, and real-time feedback

### 6.2 Why It Works

- Because depression affects multiple communication channels: what people say, how they speak, and how they express themselves visually
- Because combining modalities is more robust than using any single source alone
- Because PHQ-8 provides a clinically validated baseline

### 6.3 Why It Is Not Perfect

- The dataset is small and imbalanced
- Browser audio/video is noisier than training data
- PHQ-8 is a screening instrument, not a clinical gold standard
- Modality weights are adaptive, but not universally reliable

### 6.4 How to Justify Inaccuracies

Use these points:

- **Data limitation**: "We had only 219 labeled participants, so the model is a proof-of-concept rather than a production-ready diagnostic tool."
- **Class imbalance**: "We prioritized recall and used SMOTE because missing depressed users is more harmful than raising false alarms."
- **Modal mismatch**: "Browser-captured audio and face data are weaker than the lab-grade features used for training, so we downweight those modalities when quality is low."
- **Clinical realism**: "We do not claim clinical diagnosis. This tool augments screening by highlighting probable risk."

### 6.5 What Makes It Good

- **Multimodal design**: avoids single-signal bias
- **Transparent engineering**: explicit weight calculations, confidence-based fusion, and fallback logic
- **Safety-aware metrics**: recall/F2 and cost-sensitive thresholding
- **User-friendly UI**: step-by-step guided screening with camera and microphone checks

---

## 7. Notes for Presentation

When presenting the project, emphasize:

- The **scientific motivation** for multimodal depression detection
- The **data pipeline** from E-DAIC labels to feature extraction to model training
- The **inference path** from browser input to server fusion output
- The **engineering decisions** made to reduce overfitting and handle imbalance
- The **ethical boundary**: screening aid, not medical diagnosis

Also mention the specific file roles:

- `main.py` → training and cross-validation pipeline
- `app.py` → Flask inference API and fusion logic
- `src/text_features.py` → NLP feature extraction
- `src/audio_features_enhanced.py` → audio biomarker aggregation
- `src/visual_features.py` → visual facial AU extraction
- `static/js/app.js` → browser UX, data collection, audio/video capture

---

## 8. Recommended Report Usage

This document can be used to:

- Create a dedicated methodology section in your final report
- Explain the architecture during project demonstrations
- Document limitations honestly for viva or defense
- Reference concrete file names and system behavior

If you want, I can also convert this into a formatted `Project_Report.md` section or create a shorter executive summary for your presentation.
