# src/text_features.py
"""
Text feature extraction for depression detection.

Features (76 base + optional 20 SBERT = 76 or 96 total):
  - 4  VADER sentiment scores
  - 5  linguistic statistics (word count, unique words, lexical div, avg word len, avg conf)
  - 17 clinical NLP markers (depression lexicon, pronouns, absolutist, negation, hedging, etc.)
  - 50 TF-IDF features
  - 20 sentence-transformer embeddings (optional, PCA-reduced)
"""
import pandas as pd
import numpy as np
import os, re, logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import joblib

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (DATA_ROOT, FEATURES_DIR, MODELS_DIR, N_TFIDF,
                     SBERT_MODEL_NAME, N_SBERT_COMPONENTS)

logger = logging.getLogger(__name__)

# Optional sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
    logger.info("sentence-transformers available — will extract embeddings")
except ImportError:
    HAS_SBERT = False
    logger.info("sentence-transformers not installed — skipping embeddings")

SAVE_PATH = os.path.join(FEATURES_DIR, "text_features.csv")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sid        = SentimentIntensityAnalyzer()

# ── Depression-specific lexicons ───────────────────────────────────

DEPRESSION_WORDS = {
    'hopeless', 'worthless', 'useless', 'helpless', 'pointless',
    'empty', 'numb', 'alone', 'lonely', 'isolated',
    'miserable', 'depressed', 'anxious', 'suicidal', 'dead',
    'die', 'death', 'kill', 'hate', 'crying',
    'cry', 'pain', 'hurt', 'suffer', 'suffering',
    'tired', 'exhausted', 'fatigue', 'insomnia', 'sleep',
    'guilt', 'guilty', 'shame', 'ashamed', 'failure',
    'failed', 'burden', 'weak', 'broken', 'lost',
    'trapped', 'stuck', 'overwhelmed', 'drained', 'nothing',
    'sad', 'sadness', 'grief', 'mourning', 'despair',
}

FIRST_PERSON_SINGULAR = {'i', 'me', 'my', 'mine', 'myself'}
FIRST_PERSON_PLURAL = {'we', 'us', 'our', 'ours', 'ourselves'}
THIRD_PERSON = {'he', 'she', 'they', 'him', 'her', 'them', 'his', 'their'}

ABSOLUTIST_WORDS = {
    'always', 'never', 'nothing', 'everything', 'completely',
    'totally', 'absolutely', 'entirely', 'definitely', 'certainly',
    'impossible', 'perfect', 'forever', 'constantly', 'every',
}

NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'none',
    'nor', 'cannot', "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't",
    "shouldn't", "wouldn't", "couldn't", "haven't", "hasn't",
}

HEDGING_WORDS = {
    'maybe', 'perhaps', 'probably', 'possibly', 'might',
    'could', 'guess', 'suppose', 'seems', 'apparently',
    'somewhat', 'fairly', 'rather', 'quite', 'sort',
    'kind', 'like',  # "kind of", "sort of", "like"
}


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)


def extract_clinical_nlp_features(text):
    """
    Extract 17 depression-specific clinical NLP features.

    Based on:
    - Rude et al. 2004 (depression lexicons)
    - Stirman & Pennebaker 2001 (first-person pronouns)
    - Al-Mosaiwi & Johnstone 2018 (absolutist language)
    - Tackman et al. 2019 (negation markers)
    """
    words_lower = text.lower().split()
    n_words = max(len(words_lower), 1)  # avoid division by zero

    # ── Depression lexicon ──
    dep_words_found = [w for w in words_lower if w in DEPRESSION_WORDS]
    dep_count = len(dep_words_found)
    dep_ratio = dep_count / n_words
    dep_unique = len(set(dep_words_found))

    # ── Pronoun ratios (elevated first-person singular = depression marker) ──
    fps_count = sum(1 for w in words_lower if w in FIRST_PERSON_SINGULAR)
    fpp_count = sum(1 for w in words_lower if w in FIRST_PERSON_PLURAL)
    tp_count = sum(1 for w in words_lower if w in THIRD_PERSON)

    fps_ratio = fps_count / n_words
    fpp_ratio = fpp_count / n_words
    tp_ratio = tp_count / n_words

    # ── Absolutist language (elevated in depression/anxiety) ──
    abs_count = sum(1 for w in words_lower if w in ABSOLUTIST_WORDS)
    abs_ratio = abs_count / n_words

    # ── Negation markers ──
    neg_count = sum(1 for w in words_lower if w in NEGATION_WORDS)
    neg_ratio = neg_count / n_words

    # ── Sentence-level sentiment variance (emotional instability) ──
    try:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        if len(sentences) >= 2:
            sent_scores = [sid.polarity_scores(s)['compound'] for s in sentences]
            sent_variance = float(np.var(sent_scores))
            sent_range = max(sent_scores) - min(sent_scores)
        else:
            sent_variance = 0.0
            sent_range = 0.0
    except Exception:
        sent_variance = 0.0
        sent_range = 0.0

    # ── Sentence length + brevity ──
    try:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        mean_sent_len = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0
    except Exception:
        mean_sent_len = 0.0

    response_brevity = 1.0 / max(n_words, 1)  # shorter responses → higher value

    # ── Question ratio ──
    question_marks = text.count('?')
    total_sentences = max(len(re.split(r'[.!?]+', text)), 1)
    question_ratio = question_marks / total_sentences

    # ── Hedging language (uncertainty/low confidence) ──
    hedge_count = sum(1 for w in words_lower if w in HEDGING_WORDS)
    hedge_ratio = hedge_count / n_words

    return {
        'dep_lexicon_count': dep_count,
        'dep_lexicon_ratio': dep_ratio,
        'dep_lexicon_unique': dep_unique,
        'fps_ratio': fps_ratio,          # first-person singular
        'fpp_ratio': fpp_ratio,          # first-person plural
        'tp_ratio': tp_ratio,            # third-person
        'absolutist_count': abs_count,
        'absolutist_ratio': abs_ratio,
        'negation_count': neg_count,
        'negation_ratio': neg_ratio,
        'sent_variance': sent_variance,  # emotional instability
        'sent_range': sent_range,        # sentiment swing
        'mean_sent_len': mean_sent_len,
        'response_brevity': response_brevity,
        'question_ratio': question_ratio,
        'hedging_count': hedge_count,
        'hedging_ratio': hedge_ratio,
    }


def extract_text_features(participant_ids):
    records = []
    missing = []

    for pid in participant_ids:
        path = os.path.join(DATA_ROOT, f"{pid}_P", f"{pid}_Transcript.csv")
        if not os.path.exists(path):
            missing.append(pid)
            continue
        try:
            df = pd.read_csv(path)

            # No speaker column — use all Text rows
            # Filter out very short utterances (likely interviewer filler)
            df = df.dropna(subset=['Text'])
            df = df[df['Text'].str.split().str.len() > 2]

            full_text = ' '.join(df['Text'].astype(str).tolist())

            if len(full_text.strip()) < 10:
                full_text = "no text available"

            clean     = preprocess(full_text)
            sentiment = sid.polarity_scores(full_text)
            words     = full_text.split()

            record = {
                'pid'           : pid,
                'clean_text'    : clean,
                'full_text'     : full_text,  # Keep raw text for SBERT
                'sent_neg'      : sentiment['neg'],
                'sent_neu'      : sentiment['neu'],
                'sent_pos'      : sentiment['pos'],
                'sent_compound' : sentiment['compound'],
                'word_count'    : len(words),
                'unique_words'  : len(set(w.lower() for w in words)),
                'lexical_div'   : len(set(words)) / len(words) if words else 0,
                'avg_word_len'  : np.mean([len(w) for w in words]) if words else 0,
                'avg_conf'      : df['Confidence'].mean() if 'Confidence' in df.columns else 0,
            }

            # Add 17 clinical NLP features
            clinical = extract_clinical_nlp_features(full_text)
            record.update(clinical)

            records.append(record)
        except Exception as e:
            logger.warning(f"Error on {pid}: {e}")

    if missing:
        logger.info(f"Missing transcripts: {missing}")

    return pd.DataFrame(records)


def build_text_features(participant_ids, train_pids=None):
    """
    Extract text features. If train_pids is provided, TF-IDF is fit only on
    those participants to prevent data leakage. Otherwise fits on all.
    """
    logger.info("Extracting text features...")
    df = extract_text_features(participant_ids)

    # TF-IDF — fit ONLY on training data to prevent data leakage
    tfidf = TfidfVectorizer(max_features=N_TFIDF, min_df=2, max_df=0.95)

    if train_pids is not None:
        train_mask = df['pid'].isin(train_pids)
        tfidf.fit(df.loc[train_mask, 'clean_text'])
        logger.info(f"  TF-IDF fit on {train_mask.sum()} training samples only (no leakage)")
    else:
        tfidf.fit(df['clean_text'])
        logger.info("  TF-IDF fit on all data (no train_pids provided)")

    matrix = tfidf.transform(df['clean_text']).toarray()
    tfidf_df = pd.DataFrame(matrix,
                            columns=[f'tfidf_{i}' for i in range(matrix.shape[1])])

    # ── Optional: Sentence-transformer embeddings ──
    sbert_df = None
    if HAS_SBERT:
        try:
            logger.info(f"  Extracting sentence-transformer embeddings ({SBERT_MODEL_NAME})...")
            sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
            texts = df['full_text'].tolist()
            embeddings = sbert_model.encode(texts, show_progress_bar=True, batch_size=32)

            # PCA reduce — fit on training data only
            pca = PCA(n_components=min(N_SBERT_COMPONENTS, embeddings.shape[1], len(df) - 1),
                       random_state=42)

            if train_pids is not None:
                train_idx = df.index[df['pid'].isin(train_pids)].tolist()
                pca.fit(embeddings[train_idx])
                logger.info(f"  SBERT PCA fit on {len(train_idx)} training samples only")
            else:
                pca.fit(embeddings)

            reduced = pca.transform(embeddings)
            sbert_df = pd.DataFrame(reduced,
                                     columns=[f'sbert_{i}' for i in range(reduced.shape[1])])

            # Save SBERT PCA for inference
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(pca, os.path.join(MODELS_DIR, 'text_sbert_pca.pkl'))
            joblib.dump(True, os.path.join(MODELS_DIR, 'text_has_sbert.pkl'))
            logger.info(f"  ✅ SBERT embeddings: {embeddings.shape[1]}d → {reduced.shape[1]}d PCA "
                        f"(explained var: {pca.explained_variance_ratio_.sum():.1%})")

        except Exception as e:
            logger.warning(f"  ⚠️ SBERT failed: {e} — proceeding without embeddings")
            sbert_df = None
            joblib.dump(False, os.path.join(MODELS_DIR, 'text_has_sbert.pkl'))
    else:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(False, os.path.join(MODELS_DIR, 'text_has_sbert.pkl'))

    # ── Combine all features ──
    drop_cols = ['clean_text', 'full_text']
    parts = [df.drop(drop_cols, axis=1).reset_index(drop=True), tfidf_df]
    if sbert_df is not None:
        parts.append(sbert_df)

    result = pd.concat(parts, axis=1)

    # Save the TF-IDF vectorizer for use in the web app
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'text_tfidf.pkl'))
    logger.info(f"  ✅ TF-IDF vectorizer saved → {MODELS_DIR}/text_tfidf.pkl")

    result.to_csv(SAVE_PATH, index=False)
    logger.info(f"  ✅ Text features saved → {SAVE_PATH}")
    logger.info(f"  Shape: {result.shape}")
    base_count = 9 + 17  # base + clinical NLP
    sbert_count = sbert_df.shape[1] if sbert_df is not None else 0
    logger.info(f"  Breakdown: {base_count} base+clinical + {N_TFIDF} TF-IDF + {sbert_count} SBERT")
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    labels = pd.read_csv(os.path.join(FEATURES_DIR, 'master_labels.csv'))
    build_text_features(labels['pid'].tolist())