# src/text_features.py
import pandas as pd
import numpy as np
import os, re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

DATA_ROOT  = r"C:\Users\Rishil\Downloads\E-DAIC\data"
SAVE_PATH  = "data/features/text_features.csv"

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
            print(f"  ⚠️  Error on {pid}: {e}")

    if missing:
        print(f"  Missing transcripts: {missing}")

    return pd.DataFrame(records)

def build_text_features(participant_ids):
    print("\n📝 Extracting text features...")
    df = extract_text_features(participant_ids)

    # TF-IDF (fit on all data — in production fit only on train)
    tfidf  = TfidfVectorizer(max_features=50, min_df=2, max_df=0.95)
    matrix = tfidf.fit_transform(df['clean_text']).toarray()
    tfidf_df = pd.DataFrame(matrix,
                            columns=[f'tfidf_{i}' for i in range(matrix.shape[1])])

    result = pd.concat(
        [df.drop('clean_text', axis=1).reset_index(drop=True), tfidf_df],
        axis=1
    )

    os.makedirs('data/features', exist_ok=True)
    result.to_csv(SAVE_PATH, index=False)
    print(f"  ✅ Text features saved → {SAVE_PATH}")
    print(f"  Shape: {result.shape}")
    return result

if __name__ == "__main__":
    labels = pd.read_csv('data/features/master_labels.csv')
    build_text_features(labels['pid'].tolist())