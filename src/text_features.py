# src/text_features.py
import pandas as pd
import numpy as np
import os, re, logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_ROOT, FEATURES_DIR, MODELS_DIR, N_TFIDF

logger = logging.getLogger(__name__)

SAVE_PATH = os.path.join(FEATURES_DIR, "text_features.csv")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sid        = SentimentIntensityAnalyzer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

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

            records.append({
                'pid'           : pid,
                'clean_text'    : clean,
                'sent_neg'      : sentiment['neg'],
                'sent_neu'      : sentiment['neu'],
                'sent_pos'      : sentiment['pos'],
                'sent_compound' : sentiment['compound'],
                'word_count'    : len(words),
                'unique_words'  : len(set(w.lower() for w in words)),
                'lexical_div'   : len(set(words)) / len(words) if words else 0,
                'avg_word_len'  : np.mean([len(w) for w in words]) if words else 0,
                'avg_conf'      : df['Confidence'].mean() if 'Confidence' in df.columns else 0,
            })
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

    result = pd.concat(
        [df.drop('clean_text', axis=1).reset_index(drop=True), tfidf_df],
        axis=1
    )

    # Save the TF-IDF vectorizer for use in the web app
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'text_tfidf.pkl'))
    logger.info(f"  ✅ TF-IDF vectorizer saved → {MODELS_DIR}/text_tfidf.pkl")

    result.to_csv(SAVE_PATH, index=False)
    logger.info(f"  ✅ Text features saved → {SAVE_PATH}")
    logger.info(f"  Shape: {result.shape}")
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    labels = pd.read_csv(os.path.join(FEATURES_DIR, 'master_labels.csv'))
    build_text_features(labels['pid'].tolist())