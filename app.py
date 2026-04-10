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
    Build the feature vector the model was trained on.
    76 features (base): 4 sentiment + 5 linguistic + 17 clinical NLP + 50 TF-IDF
    96 features (with SBERT): + 20 PCA-reduced sentence embeddings
    """
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
        0.0,  # avg_conf placeholder — training data had ASR confidence scores, unavailable at inference
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

    # ── 50 TF-IDF features ──
    if tfidf_vectorizer is not None:
        clean_text = preprocess(raw_text)
        tfidf_vector = tfidf_vectorizer.transform([clean_text]).toarray()[0]
        features.extend(tfidf_vector)
    else:
        features.extend([0.0] * N_TFIDF)

    # ── 20 SBERT features (optional) ──
    if HAS_SBERT_APP:
        try:
            emb = sbert_model_app.encode([raw_text])
            reduced = sbert_pca_app.transform(emb)[0]
            features.extend(reduced)
        except Exception:
            features.extend([0.0] * sbert_pca_app.n_components_)

    return np.array(features, dtype=np.float64)



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


    # Adaptive fusion with performance-based weights
    # Higher weight for more reliable modalities
    has_visual = visual_prob is not None
    has_audio  = audio_prob is not None  # Client-side analysis, no server model needed

    # Base weights (can be tuned based on validation performance)
    weights = {'phq': 0.35, 'text': 0.35, 'audio': 0.15, 'visual': 0.15}

    if has_visual and has_audio:
        # Downweight audio if configured as unreliable (AUDIO_RELIABLE in config.py)
        weights['audio'] = 0.15 if AUDIO_RELIABLE else 0.10
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
