# app.py — Depression Detection Web Application
import os
import re
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained models & scalers
# ---------------------------------------------------------------------------
text_model  = joblib.load('models/text_model.pkl')
text_scaler = joblib.load('models/text_scaler.pkl')

# Text processing tools
sid        = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

N_TFIDF = 50  # must match training pipeline

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
    # TF-IDF slots (zeros — vectorizer unavailable for new text)
    features.extend([0.0] * N_TFIDF)
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/phq', methods=['POST'])
def phq_score():
    answers = request.json.get('answers', [])
    total   = sum(int(a) for a in answers)
    return jsonify({
        'score':    total,
        'maxScore': 24,
        'severity': phq8_severity(total),
        'depressed': total >= 10,
    })


@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    text = request.json.get('text', '')
    if not text or len(text.strip()) < 10:
        return jsonify({'error': 'Need more text for analysis'}), 400

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
    data    = request.json
    results = {}

    # ── PHQ-8 ──────────────────────────────────────────────────
    phq_answers = data.get('phqAnswers', [])
    phq_total   = sum(int(a) for a in phq_answers) if phq_answers else 0
    results['phq'] = {
        'score':     phq_total,
        'severity':  phq8_severity(phq_total),
        'depressed': phq_total >= 10,
    }

    # ── Text analysis ──────────────────────────────────────────
    text      = data.get('interviewText', '')
    text_prob = 0.5
    if text and len(text.strip()) >= 10:
        feats  = extract_text_features(text)
        scaled = text_scaler.transform(feats.reshape(1, -1))
        text_prob = float(text_model.predict_proba(scaled)[0][1])

    results['text'] = {
        'probability': round(text_prob, 4),
        'prediction':  int(text_prob >= 0.5),
    }

    # ── Combined assessment ────────────────────────────────────
    phq_norm      = min(phq_total / 24.0, 1.0)

    # Include facial analysis if available
    visual_data = data.get('visualData', None)
    visual_prob = None
    if visual_data and visual_data.get('samplesCollected', 0) > 0:
        visual_prob = visual_data.get('visualProb', 0.5)
        results['visual'] = {
            'probability': round(visual_prob, 4),
            'flatAffect':  round(visual_data.get('flatAffect', 0), 4),
            'samples':     visual_data['samplesCollected'],
        }
        # 3-way fusion: PHQ 35%, Text 35%, Visual 30%
        combined_prob = 0.35 * phq_norm + 0.35 * text_prob + 0.30 * visual_prob
    else:
        # No visual → 50/50 PHQ + Text
        combined_prob = 0.5 * phq_norm + 0.5 * text_prob

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
    }
    return jsonify(results)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  \U0001f9e0 Depression Detection Web Application")
    print("  \U0001f517 Open http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5000)
