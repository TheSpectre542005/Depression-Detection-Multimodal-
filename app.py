# app.py — Depression Detection Web Application
import io
import os
import re
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from scipy.io import wavfile
from scipy.signal import resample, find_peaks, spectrogram
from scipy.stats import kurtosis, skew
from python_speech_features import delta, mfcc

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import FLASK_DEBUG, FLASK_PORT, MODELS_DIR, N_TFIDF, AUDIO_RELIABLE
from src.text_features import (DEPRESSION_WORDS, FIRST_PERSON_SINGULAR, FIRST_PERSON_PLURAL,
                                THIRD_PERSON, ABSOLUTIST_WORDS, NEGATION_WORDS, HEDGING_WORDS,
                                extract_clinical_nlp_features)

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('sentira')

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(MODELS_DIR):
    MODELS_DIR = os.path.join(BASE_DIR, MODELS_DIR)

# ---------------------------------------------------------------------------
# Load trained models & scalers (with error handling)
# ---------------------------------------------------------------------------

# Text model (required)
# Prefer final_text_model.pkl (best VotingClassifier ensemble from main.py)
# which is an ImbPipeline with built-in VT → Scaler → SMOTE → Ensemble.
# Falls back to old text_model.pkl if the final model doesn't exist.
TEXT_MODEL_IS_PIPELINE = False
EXPECTED_TEXT_FEATURES = 96  # default: 4 sentiment + 5 linguistic + 17 clinical + 50 TF-IDF + 20 SBERT
text_scaler = None

try:
    text_model = joblib.load(os.path.join(MODELS_DIR, 'final_text_model.pkl'))
    TEXT_MODEL_IS_PIPELINE = True
    # Ensure compatibility with current sklearn version
    if hasattr(text_model, 'steps'):
        for step_name, step in text_model.steps:
            if hasattr(step, 'n_features_in_'):
                EXPECTED_TEXT_FEATURES = step.n_features_in_
                break
    logger.info(f"✅ Text model loaded: final ImbPipeline ensemble (expects {EXPECTED_TEXT_FEATURES} features)")
except FileNotFoundError:
    logger.error("❌ final_text_model.pkl not found! Ensure the model is trained and available.")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ Failed to load text model: {e}")
    sys.exit(1)

# TF-IDF vectorizer
try:
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'text_tfidf.pkl'))
    logger.info("✅ TF-IDF vectorizer loaded")
except FileNotFoundError:
    tfidf_vectorizer = None
    logger.warning("⚠️  TF-IDF vectorizer not found — text features will use zeros for TF-IDF slots")

# NOTE: Audio/visual models are trained on PCA-reduced OpenSMILE/CNN embeddings
# and CANNOT be used for inference on client-provided browser features (different
# feature space — 8 vs 40+ dimensions). Audio/visual analysis relies on
# client-side probabilities from face-api.js and Web Audio API instead.

# Browser-compatible visual model (trained on AU→expression mapped features)
browser_visual_model = None
browser_visual_scaler = None
HAS_BROWSER_VISUAL = False
try:
    browser_visual_model = joblib.load(os.path.join(MODELS_DIR, 'visual_browser_model.pkl'))
    browser_visual_scaler = joblib.load(os.path.join(MODELS_DIR, 'visual_browser_scaler.pkl'))
    HAS_BROWSER_VISUAL = True
    logger.info("✅ Browser visual model loaded (face-api.js compatible)")
except FileNotFoundError:
    logger.info("ℹ️  Browser visual model not found — visual server-side prediction disabled")

# Sentence-transformers (optional, for enhanced text features)
sbert_model_app = None
sbert_pca_app = None
HAS_SBERT_APP = False
try:
    text_has_sbert = joblib.load(os.path.join(MODELS_DIR, 'text_has_sbert.pkl'))
    if text_has_sbert:
        from sentence_transformers import SentenceTransformer
        from config import SBERT_MODEL_NAME
        sbert_model_app = SentenceTransformer(SBERT_MODEL_NAME)
        sbert_pca_app = joblib.load(os.path.join(MODELS_DIR, 'text_sbert_pca.pkl'))
        HAS_SBERT_APP = True
        logger.info("✅ Sentence-transformer model + PCA loaded")
except Exception:
    logger.info("ℹ️  SBERT not available — using base text features only")

# Text processing tools
sid        = SentimentIntensityAnalyzer()
FALLBACK_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by',
    'for', 'from', 'had', 'has', 'have', 'he', 'her', 'his', 'i', 'in',
    'is', 'it', 'its', 'me', 'my', 'of', 'on', 'or', 'our', 'she', 'so',
    'that', 'the', 'their', 'them', 'they', 'this', 'to', 'was', 'we',
    'were', 'with', 'you', 'your', 'very'
}
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    logger.warning("NLTK stopwords unavailable; using built-in fallback list")
    stop_words = FALLBACK_STOP_WORDS
lemmatizer = WordNetLemmatizer()


def predict_text_prob(features):
    """
    Get depression probability from text features.

    If using the final ImbPipeline (from main.py), the pipeline handles
    VarianceThreshold → StandardScaler → (skip SMOTE) → VotingClassifier.
    If using the old direct model, we apply the external scaler manually.
    """
    arr = features.copy()
    if len(arr) > EXPECTED_TEXT_FEATURES:
        arr = arr[:EXPECTED_TEXT_FEATURES]
    elif len(arr) < EXPECTED_TEXT_FEATURES:
        arr = np.pad(arr, (0, EXPECTED_TEXT_FEATURES - len(arr)))

    try:
        if TEXT_MODEL_IS_PIPELINE:
            return float(text_model.predict_proba(arr.reshape(1, -1))[0][1])
        else:
            X = arr.reshape(1, -1)
            scaled = text_scaler.transform(X)
            return float(text_model.predict_proba(scaled)[0][1])
    except Exception as exc:
        logger.warning(f"Text model inference failed; using heuristic fallback: {exc}")
        neg = float(arr[0]) if len(arr) > 0 else 0.0
        compound = float(arr[3]) if len(arr) > 3 else 0.0
        dep_ratio = float(arr[10]) if len(arr) > 10 else 0.0
        fps_ratio = float(arr[12]) if len(arr) > 12 else 0.0
        score = 0.25 + (0.35 * neg) + (0.20 * max(-compound, 0)) + (1.2 * dep_ratio) + (0.6 * fps_ratio)
        return float(np.clip(score, 0.05, 0.95))

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = []
    for t in text.split():
        if t in stop_words or len(t) <= 1:
            continue
        try:
            tokens.append(lemmatizer.lemmatize(t))
        except LookupError:
            tokens.append(t)
    return ' '.join(tokens)


def extract_text_features(raw_text):
    """
    Build the feature vector for inference.
    
    Dynamically adapts to match the trained model's expected feature count.
    Features: sentiment + linguistic + clinical NLP + TF-IDF + optional SBERT.
    Final vector is truncated/padded to match text_scaler.n_features_in_.
    """
    expected_features = EXPECTED_TEXT_FEATURES
    
    sentiment = sid.polarity_scores(raw_text)
    words = raw_text.split()
    n_words = len(words)

    # ── 4 sentiment features ──
    features = [
        sentiment['neg'],
        sentiment['neu'],
        sentiment['pos'],
        sentiment['compound'],
    ]

    # ── 5 linguistic features ──
    features.extend([
        n_words,
        len(set(w.lower() for w in words)) if words else 0,
        len(set(words)) / n_words if n_words else 0,
        float(np.mean([len(w) for w in words])) if words else 0,
        0.0,  # avg_conf placeholder — training data had ASR confidence scores
    ])

    # ── 17 clinical NLP features ──
    clinical = extract_clinical_nlp_features(raw_text)
    features.extend([
        clinical['dep_lexicon_count'],
        clinical['dep_lexicon_ratio'],
        clinical['dep_lexicon_unique'],
        clinical['fps_ratio'],
        clinical['fpp_ratio'],
        clinical['tp_ratio'],
        clinical['absolutist_count'],
        clinical['absolutist_ratio'],
        clinical['negation_count'],
        clinical['negation_ratio'],
        clinical['sent_variance'],
        clinical['sent_range'],
        clinical['mean_sent_len'],
        clinical['response_brevity'],
        clinical['question_ratio'],
        clinical['hedging_count'],
        clinical['hedging_ratio'],
    ])

    # ── TF-IDF features ──
    if tfidf_vectorizer is not None:
        clean_text = preprocess(raw_text)
        tfidf_vector = tfidf_vectorizer.transform([clean_text]).toarray()[0]
        features.extend(tfidf_vector)
    else:
        logger.warning("TF-IDF vectorizer not found; using zeros for TF-IDF features")
        features.extend([0.0] * N_TFIDF)

    # ── SBERT features (optional) ──
    if HAS_SBERT_APP:
        try:
            emb = sbert_model_app.encode([raw_text])
            reduced = sbert_pca_app.transform(emb)[0]
            features.extend(reduced)
        except Exception as e:
            logger.warning(f"SBERT feature extraction failed: {e}")
            features.extend([0.0] * sbert_pca_app.n_components_)
    else:
        logger.info("SBERT not available; skipping SBERT features")

    # Log feature vector length for debugging
    logger.debug(f"Extracted feature vector length: {len(features)}")

    # Adapt to trained model's expected dimensions
    feature_array = np.array(features, dtype=np.float64)
    if len(feature_array) > expected_features:
        feature_array = feature_array[:expected_features]
    elif len(feature_array) < expected_features:
        feature_array = np.pad(feature_array, (0, expected_features - len(feature_array)))

    logger.debug(f"Final feature vector length: {len(feature_array)}")
    return feature_array


# ---------------------------------------------------------------------------
# Audio model support
# ---------------------------------------------------------------------------
AUDIO_MODEL = None
AUDIO_FEATURE_COLUMNS = []
AUDIO_FEATURE_EXPECTED = 0


def _load_audio_model_and_columns():
    global AUDIO_MODEL, AUDIO_FEATURE_COLUMNS, AUDIO_FEATURE_EXPECTED
    try:
        AUDIO_MODEL = joblib.load(os.path.join(MODELS_DIR, 'final_audio_model.pkl'))
        feature_csv = os.path.join(BASE_DIR, 'data', 'features', 'audio_features_enhanced.csv')
        if os.path.exists(feature_csv):
            cols = pd.read_csv(feature_csv, nrows=0).columns.tolist()
            if cols and cols[0] == 'pid':
                cols = cols[1:]
            AUDIO_FEATURE_COLUMNS = cols
            AUDIO_FEATURE_EXPECTED = len(cols)
            logger.info(f"✅ Audio model loaded: final_audio_model.pkl uses {AUDIO_FEATURE_EXPECTED} features")
        else:
            logger.warning(f"⚠️ Audio feature CSV not found: {feature_csv}")
    except FileNotFoundError:
        logger.warning('⚠️ final_audio_model.pkl not found — server-side audio inference disabled')
    except Exception as exc:
        logger.warning(f'⚠️ Failed loading audio model or feature metadata: {exc}')


def _normalize_audio(signal):
    signal = np.asarray(signal)
    if signal.dtype.kind in ('i', 'u'):
        signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max
    elif signal.dtype.kind == 'f':
        signal = signal.astype(np.float32)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    return signal


def _load_wav_stream(file_storage):
    file_storage.stream.seek(0)
    raw_bytes = file_storage.read()
    wav_io = io.BytesIO(raw_bytes)
    samplerate, audio = wavfile.read(wav_io)
    audio = _normalize_audio(audio)
    if samplerate != 16000:
        target_len = int(len(audio) * 16000 / samplerate)
        if target_len > 0:
            audio = resample(audio, target_len)
        samplerate = 16000
    return samplerate, audio


def _estimate_pitch(signal, samplerate):
    if len(signal) < samplerate // 10:
        return 0.0
    frame = signal[:min(len(signal), samplerate)] * np.hamming(min(len(signal), samplerate))
    corr = np.correlate(frame, frame, mode='full')[len(frame)-1:]
    min_lag = max(1, samplerate // 500)
    corr[:min_lag] = 0
    peak = np.argmax(corr)
    if peak < 1:
        return 0.0
    return float(samplerate / peak)


def _estimate_formants(signal, samplerate):
    n = min(len(signal), 4096)
    if n < 512:
        return 0.0, 0.0, 0.0
    windowed = signal[:n] * np.hamming(n)
    spectrum = np.abs(np.fft.rfft(windowed, n=4096))
    freqs = np.fft.rfftfreq(4096, d=1.0 / samplerate)
    valid = np.where((freqs >= 200) & (freqs <= 4000))[0]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0
    magnitudes = spectrum[valid]
    peaks, _ = find_peaks(magnitudes, distance=20)
    if len(peaks) == 0:
        return 0.0, 0.0, 0.0
    ordered = peaks[np.argsort(magnitudes[peaks])[::-1]]
    peak_freqs = freqs[valid][ordered][:3]
    if len(peak_freqs) < 3:
        peak_freqs = np.pad(peak_freqs, (0, 3 - len(peak_freqs)), constant_values=0.0)
    return float(peak_freqs[0]), float(peak_freqs[1]), float(peak_freqs[2])


def _spectral_flux(signal, samplerate):
    if len(signal) < 512:
        return 0.0, 0.0, np.array([0.0])
    f, t, S = spectrogram(signal, fs=samplerate, window='hann', nperseg=512, noverlap=256, nfft=512, scaling='spectrum', mode='magnitude')
    if S.shape[1] < 2:
        return 0.0, 0.0, np.array([0.0])
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    return float(np.mean(flux)), float(np.min(flux)), flux


def _extract_audio_feature_vector(samplerate, signal):
    named_features = {}
    pitch = _estimate_pitch(signal, samplerate)
    f1, f2, f3 = _estimate_formants(signal, samplerate)
    flux_mean, flux_min, flux_series = _spectral_flux(signal, samplerate)
    flux_delta = np.diff(flux_series) if len(flux_series) > 1 else np.array([0.0])
    flux_delta_std = float(np.std(flux_delta)) if flux_delta.size else 0.0
    logf0 = np.log(np.maximum(1.0, pitch))
    f0_log_std = float(np.std([logf0])) if not np.isnan(logf0) else 0.0
    f0_log_delta_std = 0.0
    if flux_delta.size > 1:
        f0_log_delta_std = float(np.std(np.diff(np.array([logf0]))))

    try:
        mfcc_features = mfcc(signal, samplerate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, appendEnergy=True)
        if mfcc_features.size == 0:
            mfcc_features = np.zeros((1, 13), dtype=np.float32)
    except Exception:
        mfcc_features = np.zeros((1, 13), dtype=np.float32)

    mfcc_delta = delta(mfcc_features, 2)
    mfcc_ddelta = delta(mfcc_delta, 2)
    mfcc_means = np.mean(mfcc_features, axis=0)
    mfcc_stds = np.std(mfcc_features, axis=0)

    egemaps_values = [pitch, flux_mean, flux_min, flux_delta_std, f1, f2, f3, f0_log_std, f0_log_delta_std]

    def _approx_boaw_mfcc(bin_index, stat):
        idx = int(bin_index) % mfcc_features.shape[1]
        values = mfcc_features[:, idx]
        if stat == 'mean':
            return float(np.mean(values))
        if stat == 'std':
            return float(np.std(values))
        return float(np.max(values))

    def _approx_boaw_egemaps(bin_index, stat):
        idx = int(bin_index) % len(egemaps_values)
        val = float(np.abs(egemaps_values[idx]))
        if stat == 'mean':
            return val
        if stat == 'std':
            return val * 0.1
        return val

    slope_500_1500 = 0.0
    try:
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / samplerate)
        spectrum = np.abs(np.fft.rfft(signal * np.hamming(len(signal))))
        mask = (freqs >= 500) & (freqs <= 1500)
        if mask.any():
            xp = freqs[mask]
            yp = np.log(np.maximum(spectrum[mask], 1e-8))
            if len(xp) > 1:
                slope_500_1500 = float(np.polyfit(xp, yp, 1)[0])
    except Exception:
        slope_500_1500 = 0.0

    named_features['prosody_voiced_pitch_mean'] = pitch
    named_features['egemaps_spectralFlux_sma3_mean'] = flux_mean
    named_features['egemaps_spectralFlux_sma3_min'] = flux_min
    named_features['egemaps_spectralFlux_sma3_delta_std'] = flux_delta_std
    named_features['egemaps_slope500-1500_sma3_max'] = slope_500_1500
    named_features['egemaps_slope500-1500_sma3_delta_std'] = 0.0
    named_features['egemaps_logRelF0-H1-H2_sma3nz_std'] = f0_log_std
    named_features['egemaps_logRelF0-H1-A3_sma3nz_delta_std'] = f0_log_delta_std
    named_features['egemaps_F1frequency_sma3nz_mean'] = f1
    named_features['egemaps_F2frequency_sma3nz_skew'] = float(skew([f1, f2, f3])) if not np.isnan(f1 + f2 + f3) else 0.0
    named_features['egemaps_F3frequency_sma3nz_mean'] = f3

    # MFCC-derived features used by the saved audio model
    named_features['mfcc_pcm_fftMag_mfcc[3]_min'] = float(np.min(mfcc_features[:, 3]))
    named_features['mfcc_pcm_fftMag_mfcc[4]_mean'] = float(np.mean(mfcc_features[:, 4]))
    named_features['mfcc_pcm_fftMag_mfcc[8]_max'] = float(np.max(mfcc_features[:, 8]))
    named_features['mfcc_pcm_fftMag_mfcc[10]_p75'] = float(np.percentile(mfcc_features[:, 10], 75))
    named_features['mfcc_pcm_fftMag_mfcc[10]_kurt'] = float(kurtosis(mfcc_features[:, 10], fisher=False, nan_policy='omit'))
    named_features['mfcc_pcm_fftMag_mfcc[10]_ddelta_mean'] = float(np.mean(mfcc_ddelta[:, 10]))
    named_features['mfcc_pcm_fftMag_mfcc_de[4]_mean'] = float(np.mean(mfcc_delta[:, 4]))
    named_features['mfcc_pcm_fftMag_mfcc_de[5]_skew'] = float(skew(mfcc_delta[:, 5], nan_policy='omit'))
    named_features['mfcc_pcm_fftMag_mfcc_de_de[8]_range'] = float(np.max(mfcc_ddelta[:, 8]) - np.min(mfcc_ddelta[:, 8]))
    named_features['mfcc_pcm_fftMag_mfcc_de_de[11]_range'] = float(np.max(mfcc_ddelta[:, 11]) - np.min(mfcc_ddelta[:, 11]))

    for col in AUDIO_FEATURE_COLUMNS:
        if col.startswith('boaw_mfcc_bin'):
            match = re.match(r'boaw_mfcc_bin(\d+)_(mean|std|max)', col)
            if match:
                named_features[col] = _approx_boaw_mfcc(match.group(1), match.group(2))
        elif col.startswith('boaw_egemaps_bin'):
            match = re.match(r'boaw_egemaps_bin(\d+)_(mean|std|max)', col)
            if match:
                named_features[col] = _approx_boaw_egemaps(match.group(1), match.group(2))

    feature_vector = np.zeros(AUDIO_FEATURE_EXPECTED, dtype=np.float64)
    for idx, col in enumerate(AUDIO_FEATURE_COLUMNS):
        feature_vector[idx] = float(named_features.get(col, 0.0))
    return feature_vector


def predict_audio_probability(feature_vector):
    if AUDIO_MODEL is None:
        raise RuntimeError('Audio model is not available')
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)
    return float(AUDIO_MODEL.predict_proba(feature_vector)[0][1])


@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    if AUDIO_MODEL is None or AUDIO_FEATURE_EXPECTED == 0:
        return jsonify({'error': 'Server-side audio model unavailable'}), 503

    audio_file = request.files.get('audioFile')
    if not audio_file:
        return jsonify({'error': 'Missing audioFile in request'}), 400

    try:
        samplerate, signal = _load_wav_stream(audio_file)
        feature_vector = _extract_audio_feature_vector(samplerate, signal)
        audio_prob = predict_audio_probability(feature_vector)
        return jsonify({
            'audioProb': round(audio_prob, 4),
            'source': 'server',
            'message': 'Server-side audio model inference completed.'
        })
    except Exception as exc:
        logger.exception('Server audio upload failed')
        return jsonify({'error': 'Audio processing failed', 'details': str(exc)}), 500


_load_audio_model_and_columns()


def phq8_severity(score):
    if score <= 4:
        return 'Minimal'
    elif score <= 9:
        return 'Mild'
    elif score <= 14:
        return 'Moderate'
    elif score <= 19:
        return 'Moderately Severe'
    else:
        return 'Severe'


def validate_phq_answers(answers):
    """Validate PHQ-8 answers: exactly 8 items, each 0-3."""
    if not isinstance(answers, list) or len(answers) != 8:
        return False
    return all(isinstance(a, (int, float)) and 0 <= int(a) <= 3 for a in answers)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/phq', methods=['POST'])
def phq_score():
    data = request.json
    if not data:
        return jsonify({'error': 'Missing request body'}), 400

    answers = data.get('answers', [])
    if not validate_phq_answers(answers):
        return jsonify({'error': 'Invalid PHQ-8 answers. Expected 8 integers (0-3).'}), 400

    total = sum(int(a) for a in answers)
    return jsonify({
        'score':    total,
        'maxScore': 24,
        'severity': phq8_severity(total),
        'depressed': total >= 10,
    })


@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    if not data:
        return jsonify({'error': 'Missing request body'}), 400

    text = data.get('text', '')
    if not isinstance(text, str) or len(text.strip()) < 10:
        return jsonify({'error': 'Need at least 10 characters of text for analysis'}), 400

    # Truncate excessively long input
    text = text[:10000]

    features = extract_text_features(text)
    prob     = predict_text_prob(features)

    sentiment = sid.polarity_scores(text)
    words     = text.split()

    return jsonify({
        'probability': round(prob, 4),
        'prediction':  int(prob >= 0.38),
        'sentiment':   sentiment,
        'wordCount':   len(words),
        'uniqueWords': len(set(w.lower() for w in words)),
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'Missing request body'}), 400

    results = {}

    # ── PHQ-8 ──────────────────────────────────────────────────
    phq_answers = data.get('phqAnswers', [])
    if phq_answers and not validate_phq_answers(phq_answers):
        return jsonify({'error': 'Invalid PHQ-8 answers'}), 400

    phq_total   = sum(int(a) for a in phq_answers) if phq_answers else 0
    results['phq'] = {
        'score':     phq_total,
        'severity':  phq8_severity(phq_total),
        'depressed': phq_total >= 10,
    }

    # ── Text analysis ──────────────────────────────────────────
    text      = data.get('interviewText', '')
    if isinstance(text, str):
        text = text[:10000]  # Truncate excessively long input
    text_prob = 0.38
    if text and isinstance(text, str) and len(text.strip()) >= 10:
        feats  = extract_text_features(text)
        text_prob = predict_text_prob(feats)

    results['text'] = {
        'probability': round(text_prob, 4),
        'prediction':  int(text_prob >= 0.38),
    }

    # ── Combined assessment ────────────────────────────────────
    phq_norm = min(phq_total / 24.0, 1.0)

    # Include facial analysis if available
    visual_data = data.get('visualData', None)
    visual_prob = None
    if (visual_data and isinstance(visual_data, dict)
            and visual_data.get('samplesCollected', 0) > 0):
        visual_prob = float(visual_data.get('visualProb', 0.5))
        visual_prob = max(0.0, min(1.0, visual_prob))  # Clamp
        results['visual'] = {
            'probability': round(visual_prob, 4),
            'flatAffect':  round(float(visual_data.get('flatAffect', 0)), 4),
            'samples':     visual_data['samplesCollected'],
        }

        # Server-side prediction using browser-compatible visual model
        if HAS_BROWSER_VISUAL:
            try:
                expressions = visual_data.get('expressions', {})
                expr_features = []
                for expr in ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']:
                    val = float(expressions.get(expr, 0))
                    expr_features.extend([val, 0.0, val, 0.0])  # mean, std, max, trend placeholders
                # Add derived features
                happy_mean = float(expressions.get('happy', 0))
                sad_mean = float(expressions.get('sad', 0))
                expr_features.append(float(visual_data.get('flatAffect', 0)))  # flat_affect
                expr_features.append(sad_mean / max(happy_mean, 0.01))  # sad_happy_ratio
                neg_mean = np.mean([float(expressions.get(e, 0)) for e in ['sad', 'angry', 'fearful', 'disgusted']])
                pos_mean = np.mean([float(expressions.get(e, 0)) for e in ['happy', 'surprised']])
                expr_features.append(neg_mean / max(pos_mean, 0.01))  # neg_pos_expr_ratio
                expr_features.append(1.0 if happy_mean > 0.3 else 0.0)  # smile_frequency
                expr_features.append(0.5)  # expression_transition_rate placeholder

                feat_arr = np.array(expr_features, dtype=np.float64).reshape(1, -1)
                # Pad/trim to match model's expected features
                expected = browser_visual_scaler.n_features_in_
                if feat_arr.shape[1] < expected:
                    feat_arr = np.pad(feat_arr, ((0, 0), (0, expected - feat_arr.shape[1])))
                elif feat_arr.shape[1] > expected:
                    feat_arr = feat_arr[:, :expected]

                feat_scaled = browser_visual_scaler.transform(feat_arr)
                server_visual_prob = float(browser_visual_model.predict_proba(feat_scaled)[0][1])
                # Blend client and server predictions (60% server, 40% client)
                visual_prob = 0.6 * server_visual_prob + 0.4 * visual_prob
                results['visual']['server_probability'] = round(server_visual_prob, 4)
                results['visual']['probability'] = round(visual_prob, 4)
            except Exception as e:
                logger.warning(f"Browser visual prediction failed: {e}")

    # Include audio analysis if available
    audio_data = data.get('audioData', None)
    audio_prob = None
    audio_source = 'client'
    if isinstance(audio_data, dict):
        server_audio_prob = audio_data.get('serverAudioProb')
        if server_audio_prob is not None:
            try:
                audio_prob = float(server_audio_prob)
                audio_source = 'server'
            except Exception:
                audio_prob = None

    if audio_prob is None and (audio_data and isinstance(audio_data, dict)
            and audio_data.get('samplesCollected', 0) >= 3):
        audio_prob = float(audio_data.get('audioProb', 0.5))
        audio_source = 'client'

    if audio_prob is not None:
        audio_prob = max(0.0, min(1.0, audio_prob))  # Clamp
        results['audio'] = {
            'probability':       round(audio_prob, 4),
            'prediction':        int(audio_prob >= 0.38),
            'avgEnergy':         round(float(audio_data.get('avgEnergy', 0)) if audio_data else 0, 4),
            'pauseRatio':        round(float(audio_data.get('pauseRatio', 0)) if audio_data else 0, 4),
            'speechRate':        round(float(audio_data.get('speechRate', 0)) if audio_data else 0, 4),
            'samples':           int(audio_data.get('samplesCollected', 0)) if audio_data else 0,
            'source':            audio_source,
            'client_probability': round(float(audio_data.get('audioProb', 0)), 4) if audio_data and 'audioProb' in audio_data else None,
        }



    # ── Intelligent adaptive fusion ───────────────────────────
    # Weights are proportional to modality reliability:
    #   PHQ-8: Gold standard clinical instrument → highest base weight
    #   Text:  ML model trained on E-DAIC → high weight, scales with text length
    #   Visual: face-api.js expressions → moderate weight, scales with sample count
    #   Audio:  Web Audio API features → lower weight (browser audio is noisy)
    has_visual = visual_prob is not None
    has_audio  = audio_prob is not None

    # Confidence-scaled weights
    weights = {}

    # PHQ-8 always included (validated clinical instrument)
    weights['phq'] = 0.35

    # Text: scale weight by input richness (more words = more signal)
    text_word_count = len(text.split()) if text else 0
    text_confidence = min(1.0, text_word_count / 50.0)  # Full confidence at 50+ words
    weights['text'] = 0.30 * (0.5 + 0.5 * text_confidence)  # Range: 0.15 - 0.30

    # Visual: scale by sample count and detection quality
    if has_visual:
        vis_samples = visual_data.get('samplesCollected', 0)
        vis_confidence = min(1.0, vis_samples / 15.0)  # Full confidence at 15+ samples
        weights['visual'] = 0.25 * (0.3 + 0.7 * vis_confidence)  # Range: 0.075 - 0.25

    # Audio: scale by sample count, downweight if unreliable
    if has_audio:
        aud_samples = audio_data.get('samplesCollected', 0)
        aud_confidence = min(1.0, aud_samples / 20.0)  # Full confidence at 20+ samples
        base_audio_weight = 0.15 if AUDIO_RELIABLE else 0.08
        weights['audio'] = base_audio_weight * (0.3 + 0.7 * aud_confidence)

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Compute weighted fusion
    combined_prob = weights.get('phq', 0) * phq_norm + weights.get('text', 0) * text_prob
    if has_audio:
        combined_prob += weights['audio'] * audio_prob
    if has_visual:
        combined_prob += weights['visual'] * visual_prob

    if combined_prob >= 0.55:
        risk = 'High'
    elif combined_prob >= 0.38:
        risk = 'Moderate'
    else:
        risk = 'Low'

    results['combined'] = {
        'probability': round(combined_prob, 4),
        'riskLevel':   risk,
        'prediction':  int(combined_prob >= 0.38),
        'weights':     {k: round(v, 3) for k, v in weights.items()},
        'modalities_used': list(weights.keys()),
    }

    logger.info(f"Prediction: PHQ={phq_total}, text_prob={text_prob:.3f}, "
                f"visual={'Y' if has_visual else 'N'}, audio={'Y' if has_audio else 'N'}, "
                f"weights={weights}, combined={combined_prob:.3f} ({risk})")

    return jsonify(results)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  \U0001f9e0 SENTIRA — Depression Detection Web Application")
    print(f"  \U0001f517 Open http://localhost:{FLASK_PORT}")
    print("=" * 55 + "\n")
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT)
