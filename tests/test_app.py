# tests/test_app.py
"""Unit tests for the SENTIRA Depression Detection application."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np


# ── Test text preprocessing ─────────────────────────────────────
class TestPreprocess:
    def test_basic_cleaning(self):
        from app import preprocess
        result = preprocess("I'm feeling VERY sad today!!!")
        assert 'very' not in result  # stopword removed
        assert '!' not in result  # punctuation removed
        assert result == result.lower()  # lowercase

    def test_empty_input(self):
        from app import preprocess
        assert preprocess("") == ""
        # str(None) = 'none', which passes length filter
        result = preprocess(None)
        assert isinstance(result, str)

    def test_short_words_filtered(self):
        from app import preprocess
        result = preprocess("I a am ok doing fine today")
        # Single-char words should be filtered (len > 1 check)
        words = result.split()
        assert all(len(w) > 1 for w in words) if words else True

    def test_lemmatization(self):
        from app import preprocess
        result = preprocess("running dogs happily")
        assert 'running' in result or 'run' in result


# ── Test PHQ-8 severity mapping ─────────────────────────────────
class TestPhq8Severity:
    def test_minimal(self):
        from app import phq8_severity
        assert phq8_severity(0) == 'Minimal'
        assert phq8_severity(4) == 'Minimal'

    def test_mild(self):
        from app import phq8_severity
        assert phq8_severity(5) == 'Mild'
        assert phq8_severity(9) == 'Mild'

    def test_moderate(self):
        from app import phq8_severity
        assert phq8_severity(10) == 'Moderate'
        assert phq8_severity(14) == 'Moderate'

    def test_moderately_severe(self):
        from app import phq8_severity
        assert phq8_severity(15) == 'Moderately Severe'
        assert phq8_severity(19) == 'Moderately Severe'

    def test_severe(self):
        from app import phq8_severity
        assert phq8_severity(20) == 'Severe'
        assert phq8_severity(24) == 'Severe'


# ── Test PHQ answer validation ──────────────────────────────────
class TestValidation:
    def test_valid_answers(self):
        from app import validate_phq_answers
        assert validate_phq_answers([0, 1, 2, 3, 0, 1, 2, 3]) is True

    def test_wrong_length(self):
        from app import validate_phq_answers
        assert validate_phq_answers([0, 1, 2]) is False
        assert validate_phq_answers([]) is False

    def test_out_of_range(self):
        from app import validate_phq_answers
        assert validate_phq_answers([0, 1, 2, 3, 4, 1, 2, 3]) is False
        assert validate_phq_answers([0, 1, 2, 3, -1, 1, 2, 3]) is False

    def test_non_numeric(self):
        from app import validate_phq_answers
        assert validate_phq_answers(['a', 1, 2, 3, 0, 1, 2, 3]) is False


# ── Test feature extraction shape ───────────────────────────────
class TestFeatureExtraction:
    def test_output_shape(self):
        from app import extract_text_features, HAS_SBERT_APP
        features = extract_text_features("I feel very sad and hopeless about everything")
        # 76 base features (4 sentiment + 5 linguistic + 17 clinical NLP + 50 TF-IDF)
        # + 20 SBERT features if sentence-transformers is available
        expected = 96 if HAS_SBERT_APP else 76
        assert features.shape == (expected,)
        assert features.dtype == np.float64

    def test_sentiment_range(self):
        from app import extract_text_features
        features = extract_text_features("I am extremely happy and joyful today")
        # Sentiment values should be between 0 and 1
        assert 0 <= features[0] <= 1  # neg
        assert 0 <= features[1] <= 1  # neu
        assert 0 <= features[2] <= 1  # pos

    def test_minimum_text(self):
        from app import extract_text_features, HAS_SBERT_APP
        features = extract_text_features("hello world test input")
        expected = 96 if HAS_SBERT_APP else 76
        assert len(features) == expected


# ── Test Flask API endpoints ────────────────────────────────────
class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        from app import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_index(self, client):
        rv = client.get('/')
        assert rv.status_code == 200

    def test_phq_valid(self, client):
        rv = client.post('/api/phq',
                         json={'answers': [0, 1, 2, 3, 0, 1, 2, 3]},
                         content_type='application/json')
        assert rv.status_code == 200
        data = rv.get_json()
        assert data['score'] == 12
        assert data['severity'] == 'Moderate'

    def test_phq_invalid(self, client):
        rv = client.post('/api/phq',
                         json={'answers': [0, 1, 2]},
                         content_type='application/json')
        assert rv.status_code == 400

    def test_analyze_text_valid(self, client):
        rv = client.post('/api/analyze-text',
                         json={'text': 'I feel very sad and hopeless about everything today'},
                         content_type='application/json')
        assert rv.status_code == 200
        data = rv.get_json()
        assert 'probability' in data
        assert 0 <= data['probability'] <= 1

    def test_analyze_text_too_short(self, client):
        rv = client.post('/api/analyze-text',
                         json={'text': 'hi'},
                         content_type='application/json')
        assert rv.status_code == 400

    def test_predict_valid(self, client):
        rv = client.post('/api/predict',
                         json={
                             'phqAnswers': [0, 0, 0, 0, 0, 0, 0, 0],
                             'interviewText': 'I feel great and happy today, life is wonderful'
                         },
                         content_type='application/json')
        assert rv.status_code == 200
        data = rv.get_json()
        assert 'combined' in data
        assert 'phq' in data
        assert 'text' in data
        assert 'weights' in data['combined']  # New: fusion weights included

    def test_predict_with_visual(self, client):
        rv = client.post('/api/predict',
                         json={
                             'phqAnswers': [2, 2, 2, 2, 2, 2, 2, 2],
                             'interviewText': 'I have been feeling down and hopeless lately',
                             'visualData': {
                                 'samplesCollected': 10,
                                 'visualProb': 0.7,
                                 'flatAffect': 0.5,
                                 'expressions': {'sad': 0.6, 'neutral': 0.3}
                             }
                         },
                         content_type='application/json')
        assert rv.status_code == 200
        data = rv.get_json()
        assert 'visual' in data
        assert 'combined' in data
        # Check that visual modality is included in weights
        assert 'visual' in data['combined'].get('modalities_used', [])

    def test_predict_with_audio(self, client):
        rv = client.post('/api/predict',
                         json={
                             'phqAnswers': [1, 1, 1, 1, 1, 1, 1, 1],
                             'interviewText': 'I am doing okay I guess',
                             'audioData': {
                                 'samplesCollected': 5,
                                 'audioProb': 0.4,
                                 'avgEnergy': 0.5,
                                 'pauseRatio': 0.3,
                                 'speechRate': 120
                             }
                         },
                         content_type='application/json')
        assert rv.status_code == 200
        data = rv.get_json()
        assert 'audio' in data
        assert 'combined' in data

    def test_combined_weights_sum_to_one(self, client):
        rv = client.post('/api/predict',
                         json={
                             'phqAnswers': [1, 2, 1, 2, 1, 2, 1, 2],
                             'interviewText': 'Some days are better than others',
                             'visualData': {'samplesCollected': 5, 'visualProb': 0.5},
                             'audioData': {'samplesCollected': 5, 'audioProb': 0.5}
                         },
                         content_type='application/json')
        assert rv.status_code == 200
        data = rv.get_json()
        weights = data['combined']['weights']
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"
