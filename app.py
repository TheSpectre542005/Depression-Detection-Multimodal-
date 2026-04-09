# app.py — Depression Detection Web Application
import os
import re
import sys
import logging
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import FLASK_DEBUG, FLASK_PORT, MODELS_DIR, N_TFIDF

# Data quality tracking (imported from audio_features)
data_quality_log = {'audio_unreliable': True}  # Marked unreliable until proven otherwise

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('sentira')

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained models & scalers (with error handling)
# ---------------------------------------------------------------------------

# Text model (required)
try:
    text_model  = joblib.load(os.path.join(MODELS_DIR, 'text_model.pkl'))
    text_scaler = joblib.load(os.path.join(MODELS_DIR, 'text_scaler.pkl'))
    logger.info("✅ Text model + scaler loaded")
except FileNotFoundError:
    logger.error("❌ Text model files not found! Run main.py to train models first.")
    sys.exit(1)

# TF-IDF vectorizer
try:
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'text_tfidf.pkl'))
    logger.info("✅ TF-IDF vectorizer loaded")
except FileNotFoundError:
    tfidf_vectorizer = None
    logger.warning("⚠️  TF-IDF vectorizer not found — text features will use zeros for TF-IDF slots")

# Audio model (optional - may not be available or reliable)
audio_model = None
audio_scaler = None
HAS_AUDIO_MODEL = False
try:
    audio_model = joblib.load(os.path.join(MODELS_DIR, 'audio_model.pkl'))
    audio_artifacts = joblib.load(os.path.join(MODELS_DIR, 'audio_scaler.pkl'))
    audio_scaler = audio_artifacts.get('scaler') if isinstance(audio_artifacts, dict) else audio_artifacts
    HAS_AUDIO_MODEL = True
    logger.info("✅ Audio model loaded (use with caution - may have data quality issues)")
except FileNotFoundError:
    logger.info("ℹ️  Audio model not found - audio analysis disabled")

# Visual model (optional)
visual_model = None
visual_scaler = None
HAS_VISUAL_MODEL = False
try:
    visual_model = joblib.load(os.path.join(MODELS_DIR, 'visual_model.pkl'))
    visual_artifacts = joblib.load(os.path.join(MODELS_DIR, 'visual_scaler.pkl'))
    visual_scaler = visual_artifacts.get('scaler') if isinstance(visual_artifacts, dict) else visual_artifacts
    HAS_VISUAL_MODEL = True
    logger.info("✅ Visual model loaded")
except FileNotFoundError:
    logger.info("ℹ️  Visual model not found - visual analysis disabled")

# Text processing tools
sid        = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)


def extract_text_features(raw_text):
    """
    Build the same 59-feature vector the model was trained on.
    Order: sent_neg, sent_neu, sent_pos, sent_compound,
           word_count, unique_words, lexical_div, avg_word_len, avg_conf,
           tfidf_0 … tfidf_49
    """
    sentiment = sid.polarity_scores(raw_text)
    words = raw_text.split()
    n_words = len(words)

    features = [
        sentiment['neg'],
        sentiment['neu'],
        sentiment['pos'],
        sentiment['compound'],
        n_words,
        len(set(w.lower() for w in words)) if words else 0,
        len(set(words)) / n_words if n_words else 0,
        float(np.mean([len(w) for w in words])) if words else 0,
        0.0,  # avg_conf — not available for new text
    ]

    # TF-IDF — use the saved vectorizer if available
    if tfidf_vectorizer is not None:
        clean_text = preprocess(raw_text)
        tfidf_vector = tfidf_vectorizer.transform([clean_text]).toarray()[0]
        features.extend(tfidf_vector)
    else:
        features.extend([0.0] * N_TFIDF)

    return np.array(features, dtype=np.float64)


def extract_audio_features_from_data(audio_data):
    """
    Extract audio features from client-provided audio metrics.
    This is a simplified version - full implementation would process raw audio.
    """
    if not HAS_AUDIO_MODEL:
        return None

    try:
        # Extract key audio features from the data
        features = [
            float(audio_data.get('avgEnergy', 0)),
            float(audio_data.get('pauseRatio', 0)),
            float(audio_data.get('speechRate', 0)),
            float(audio_data.get('pitchMean', 0)),
            float(audio_data.get('pitchStd', 0)),
            float(audio_data.get('jitter', 0)),
            float(audio_data.get('shimmer', 0)),
            float(audio_data.get('hnr', 0)),  # Harmonics-to-noise ratio
        ]

        # Pad to expected feature size if needed
        # The actual model expects specific features from the training pipeline
        # This is a placeholder that would need to match the training features
        return np.array(features, dtype=np.float64)

    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        return None


def extract_visual_features_from_data(visual_data):
    """
    Extract visual features from client-provided facial metrics.
    This processes face-api.js output or similar.
    """
    if not HAS_VISUAL_MODEL:
        return None

    try:
        # Extract key visual features
        features = []

        # Expression probabilities (from face-api.js)
        expressions = visual_data.get('expressions', {})
        for expr in ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']:
            features.append(float(expressions.get(expr, 0)))

        # Facial action units (if available)
        aus = visual_data.get('actionUnits', {})
        for au in ['AU01', 'AU04', 'AU06', 'AU12', 'AU15', 'AU17', 'AU20', 'AU26']:
            features.append(float(aus.get(au, 0)))

        # Head pose (if available)
        pose = visual_data.get('headPose', {})
        features.extend([
            float(pose.get('pitch', 0)),
            float(pose.get('yaw', 0)),
            float(pose.get('roll', 0)),
        ])

        # Flat affect indicator
        features.append(float(visual_data.get('flatAffect', 0)))

        return np.array(features, dtype=np.float64)

    except Exception as e:
        logger.error(f"Error extracting visual features: {e}")
        return None


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
    scaled   = text_scaler.transform(features.reshape(1, -1))
    prob     = float(text_model.predict_proba(scaled)[0][1])

    sentiment = sid.polarity_scores(text)
    words     = text.split()

    return jsonify({
        'probability': round(prob, 4),
        'prediction':  int(prob >= 0.5),
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
    text_prob = 0.5
    if text and isinstance(text, str) and len(text.strip()) >= 10:
        feats  = extract_text_features(text)
        scaled = text_scaler.transform(feats.reshape(1, -1))
        text_prob = float(text_model.predict_proba(scaled)[0][1])

    results['text'] = {
        'probability': round(text_prob, 4),
        'prediction':  int(text_prob >= 0.5),
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

    # Include audio analysis if available
    audio_data = data.get('audioData', None)
    audio_prob = None
    if (audio_data and isinstance(audio_data, dict)
            and audio_data.get('samplesCollected', 0) >= 3):
        audio_prob = float(audio_data.get('audioProb', 0.5))
        audio_prob = max(0.0, min(1.0, audio_prob))  # Clamp
        results['audio'] = {
            'probability':      round(audio_prob, 4),
            'avgEnergy':        round(float(audio_data.get('avgEnergy', 0)), 4),
            'pauseRatio':       round(float(audio_data.get('pauseRatio', 0)), 4),
            'speechRate':       round(float(audio_data.get('speechRate', 0)), 4),
            'samples':          audio_data['samplesCollected'],
        }

    # ── Server-side audio/visual analysis (if models available) ───
    # Override client-provided values with server-side predictions
    if HAS_AUDIO_MODEL and audio_data and isinstance(audio_data, dict):
        try:
            # Extract features from audio data and predict
            audio_feats = extract_audio_features_from_data(audio_data)
            if audio_feats is not None:
                audio_scaled = audio_scaler.transform(audio_feats.reshape(1, -1))
                audio_prob = float(audio_model.predict_proba(audio_scaled)[0][1])
                results['audio']['server_probability'] = round(audio_prob, 4)
        except Exception as e:
            logger.warning(f"Audio prediction failed: {e}")

    if HAS_VISUAL_MODEL and visual_data and isinstance(visual_data, dict):
        try:
            # Extract features from visual data and predict
            visual_feats = extract_visual_features_from_data(visual_data)
            if visual_feats is not None:
                visual_scaled = visual_scaler.transform(visual_feats.reshape(1, -1))
                visual_prob = float(visual_model.predict_proba(visual_scaled)[0][1])
                results['visual']['server_probability'] = round(visual_prob, 4)
        except Exception as e:
            logger.warning(f"Visual prediction failed: {e}")

    # Adaptive fusion with performance-based weights
    # Higher weight for more reliable modalities
    has_visual = visual_prob is not None
    has_audio  = audio_prob is not None and HAS_AUDIO_MODEL

    # Base weights (can be tuned based on validation performance)
    weights = {'phq': 0.35, 'text': 0.35, 'audio': 0.15, 'visual': 0.15}

    if has_visual and has_audio:
        # All modalities available - downweight audio if unreliable
        weights['audio'] = 0.10 if data_quality_log.get('audio_unreliable') else 0.15
        weights['visual'] = 0.20
        weights['phq'] = 0.30
        weights['text'] = 0.30
    elif has_visual:
        weights['visual'] = 0.30
        weights['phq'] = 0.35
        weights['text'] = 0.35
    elif has_audio:
        weights['audio'] = 0.20
        weights['phq'] = 0.40
        weights['text'] = 0.40

    # Normalize weights based on available modalities
    available = {'phq': True, 'text': True, 'visual': has_visual, 'audio': has_audio}
    total_weight = sum(w for k, w in weights.items() if available[k])
    weights = {k: w/total_weight for k, w in weights.items() if available[k]}

    # Compute weighted fusion
    combined_prob = 0
    if 'phq' in weights:
        combined_prob += weights['phq'] * phq_norm
    if 'text' in weights:
        combined_prob += weights['text'] * text_prob
    if has_audio and 'audio' in weights:
        combined_prob += weights['audio'] * audio_prob
    if has_visual and 'visual' in weights:
        combined_prob += weights['visual'] * visual_prob

    if combined_prob >= 0.6:
        risk = 'High'
    elif combined_prob >= 0.4:
        risk = 'Moderate'
    else:
        risk = 'Low'

    results['combined'] = {
        'probability': round(combined_prob, 4),
        'riskLevel':   risk,
        'prediction':  int(combined_prob >= 0.5),
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
