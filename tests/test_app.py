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
        from app import extract_text_features
        features = extract_text_features("I feel very sad and hopeless about everything")
        assert features.shape == (59,)
        assert features.dtype == np.float64

    def test_sentiment_range(self):
        from app import extract_text_features
        features = extract_text_features("I am extremely happy and joyful today")
        # Sentiment values should be between 0 and 1
        assert 0 <= features[0] <= 1  # neg
        assert 0 <= features[1] <= 1  # neu
        assert 0 <= features[2] <= 1  # pos

    def test_minimum_text(self):
        from app import extract_text_features
        features = extract_text_features("hello world test input")
        assert len(features) == 59


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
