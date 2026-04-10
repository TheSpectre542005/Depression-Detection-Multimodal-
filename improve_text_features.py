"""
improve_text_features.py
========================
Extracts rich psycholinguistic + linguistic + prosodic-text features
from raw DAIC-WOZ transcripts to push text AUC above 0.72 in honest CV.

Features added beyond the current 59-dim set:
  - LIWC-proxy word lists: neg_affect, pos_affect, anx, sad, anger,
    social, cogmech, tentativeness, certainty, insight
  - 1st/2nd/3rd person pronoun rates (depression -> increased I-focus)
  - Absolutist language (always/never/nothing etc.)
  - Question rate, filler words, hedge words
  - Temporal language (past/future tense ratio)
  - Response latency & duration statistics (from timestamps)
  - Utterance-level features: length distribution, entropy
  - Confidence-weighted features (low-confidence -> disfluency proxy)

Usage:
    python improve_text_features.py            # regenerates text_features_enhanced.csv
    python improve_text_features.py --cv       # also runs honest 5-fold CV
"""

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_ROOT    = os.path.join(os.path.expanduser("~"), "Downloads", "E-DAIC", "data")
FEATURES_DIR = "data/features"
LABELS_CSV   = "data/features/master_labels.csv"
OUT_PATH     = "data/features/text_features_enhanced.csv"
PHQ_THRESH   = 10
RANDOM_STATE = 42


# ── LIWC-proxy word lists ──────────────────────────────────────────────────
# Based on published LIWC 2015 category samples and depression NLP literature

WORD_LISTS = {
    # Negative emotions
    "neg_affect": {
        "bad","worse","worst","terrible","awful","horrible","dreadful","painful",
        "miserable","unfortunate","hate","hated","disgust","disgusting","fear",
        "afraid","scared","terrified","worried","anxious","stress","stressed",
        "frustrated","angry","furious","upset","depressed","hopeless","hopelessness",
        "worthless","useless","failure","failed","sad","sadness","grief","sorrow",
        "regret","regretful","ashamed","shame","guilt","guilty","lonely","alone",
        "empty","numb","broken","helpless","trapped","lost",
    },
    # Positive emotions
    "pos_affect": {
        "good","great","excellent","wonderful","happy","happiness","joy","joyful",
        "glad","pleased","excited","love","loved","loving","proud","pride",
        "grateful","thankful","optimistic","hopeful","cheerful","content",
        "satisfied","enjoy","enjoyed","amazing","fantastic","delighted",
        "confident","motivated","inspired","energetic","positive","better","best",
    },
    # Anxiety / worry
    "anxiety": {
        "nervous","anxious","anxiety","worry","worried","worries","panic","panicking",
        "tense","uneasy","uncertain","unsure","dread","apprehensive","afraid","fear",
        "scared","restless","overwhelmed","overthink","overthinking",
    },
    # Sadness / depression
    "sadness": {
        "sad","sadness","unhappy","depressed","depression","cry","crying","tears",
        "weep","weeping","miserable","gloomy","grief","grieving","mourn","mourning",
        "heartbroken","devastated","hopeless","hopelessness","despair","despairing",
        "empty","hollow","numb","meaningless","worthless","bleak","dark",
    },
    # First person singular pronouns (excessive self-focus in depression)
    "first_person_sg": {"i","me","my","mine","myself"},
    # First person plural
    "first_person_pl": {"we","us","our","ours","ourselves"},
    # Second person
    "second_person": {"you","your","yours","yourself","yourselves"},
    # Third person
    "third_person": {"he","she","they","him","her","them","his","hers","their","theirs"},
    # Absolutist / all-or-nothing thinking (linked to depression severity)
    "absolutist": {
        "always","never","nothing","everything","everyone","no_one","nobody","none",
        "all","every","any","completely","absolutely","totally","entirely","forever",
        "impossible","perfect","perfectly","worst","best","must","cannot","can't",
    },
    # Hedging / tentative language
    "tentative": {
        "maybe","perhaps","possibly","probably","might","could","would","should",
        "seems","seem","think","thought","guess","suppose","suppose","uncertain",
        "unsure","unclear","kind","sort","somewhat","little","like",
    },
    # Certainty language
    "certainty": {
        "definitely","certainly","absolutely","clearly","obviously","undoubtedly",
        "sure","surely","know","knew","always","never","must","will","always",
    },
    # Cognitive processing / insight
    "cognitive": {
        "think","thought","know","knew","realize","realized","understand","understood",
        "consider","considered","believe","believed","imagine","imagined","wonder",
        "wondered","remember","remembered","notice","noticed","feel","felt","sense",
    },
    # Social words
    "social": {
        "family","friend","friends","people","person","someone","anybody","everyone",
        "together","group","community","relationship","partner","spouse","husband",
        "wife","parent","mother","father","child","children","colleague","team",
    },
    # Work / achievement
    "work": {
        "work","job","career","office","business","company","project","task",
        "achieve","success","goal","effort","study","school","university","college",
    },
    # Body / health
    "body": {
        "sleep","tired","fatigue","energy","pain","hurt","sick","illness","health",
        "appetite","eat","food","weight","exercise","doctor","hospital","medicine",
        "medication","headache","muscle","body","physical",
    },
    # Time orientation — past
    "past_focus": {
        "was","were","had","did","used","went","came","saw","knew","thought",
        "felt","said","told","heard","found","left","lost","took","gave","made",
    },
    # Time orientation — future
    "future_focus": {
        "will","would","shall","going","plan","plans","planning","hope","hopes",
        "expect","expects","hope","want","wants","intend","intends","future",
        "tomorrow","next","soon","eventually","someday",
    },
    # Filler / disfluency words
    "filler": {
        "um","uh","like","you_know","kind_of","sort_of","i_mean","basically",
        "actually","literally","honestly","you_know","right","okay","so",
    },
}


def tokenize(text):
    """Lowercase, strip punctuation, return word list."""
    return re.findall(r"\b[a-z']+\b", str(text).lower())


def extract_features_from_transcript(pid, data_root):
    """
    Extract all psycholinguistic + temporal features for one participant.
    Returns a flat dict of feature_name -> value.
    """
    trans_path = os.path.join(data_root, f"{pid}_P", f"{pid}_Transcript.csv")
    if not os.path.exists(trans_path):
        return None

    try:
        df = pd.read_csv(trans_path)
    except Exception:
        return None

    # Filter to participant speech only (exclude interviewer if 'Speaker' col exists)
    if "Speaker" in df.columns:
        df = df[df["Speaker"].str.upper().isin(["PARTICIPANT", "P", "SUBJECT"])]

    if df.empty or "Text" not in df.columns:
        return None

    # ── Concatenated text ──────────────────────────────────────────────────
    all_text  = " ".join(df["Text"].fillna("").astype(str))
    tokens    = tokenize(all_text)
    n_tokens  = max(len(tokens), 1)

    record = {"pid": pid}

    # ── WORD LIST features (rate per token) ────────────────────────────────
    for cat, word_set in WORD_LISTS.items():
        count = sum(1 for t in tokens if t in word_set)
        record[f"wl_{cat}"] = count / n_tokens

    # ── Derived depression-relevant ratios ────────────────────────────────
    record["i_vs_we_ratio"] = (
        (record["wl_first_person_sg"] + 1e-6) /
        (record["wl_first_person_pl"] + 1e-6)
    )
    record["neg_pos_ratio"] = (
        (record["wl_neg_affect"] + 1e-6) /
        (record["wl_pos_affect"] + 1e-6)
    )
    record["past_future_ratio"] = (
        (record["wl_past_focus"] + 1e-6) /
        (record["wl_future_focus"] + 1e-6)
    )
    record["absolutist_x_neg"] = record["wl_absolutist"] * record["wl_neg_affect"]
    record["sadness_x_first"]  = record["wl_sadness"] * record["wl_first_person_sg"]

    # ── Basic lexical features ─────────────────────────────────────────────
    sentences       = [s.strip() for s in re.split(r"[.!?]+", all_text) if s.strip()]
    n_sentences     = max(len(sentences), 1)
    words_per_sent  = n_tokens / n_sentences
    unique_words    = len(set(tokens))
    lexical_div     = unique_words / n_tokens
    avg_word_len    = np.mean([len(t) for t in tokens]) if tokens else 0

    # Type-token ratio in windows (more robust)
    window_size = 50
    ttr_windows = []
    for i in range(0, len(tokens) - window_size + 1, window_size // 2):
        w = tokens[i:i + window_size]
        ttr_windows.append(len(set(w)) / window_size)
    record["word_count"]      = n_tokens
    record["unique_words"]    = unique_words
    record["lexical_div"]     = lexical_div
    record["avg_word_len"]    = avg_word_len
    record["words_per_sent"]  = words_per_sent
    record["n_sentences"]     = n_sentences
    record["ttr_windowed"]    = np.mean(ttr_windows) if ttr_windows else lexical_div

    # ── Token entropy (vocabulary breadth) ────────────────────────────────
    from collections import Counter
    counts = Counter(tokens)
    probs  = np.array(list(counts.values())) / n_tokens
    record["token_entropy"] = -np.sum(probs * np.log(probs + 1e-12))

    # ── Utterance-level features (from df rows) ────────────────────────────
    utt_lengths   = df["Text"].fillna("").apply(lambda x: len(tokenize(str(x))))
    record["utt_len_mean"]  = utt_lengths.mean()
    record["utt_len_std"]   = utt_lengths.std() if len(utt_lengths) > 1 else 0
    record["utt_len_max"]   = utt_lengths.max()
    record["utt_len_min"]   = utt_lengths.min()
    record["n_utterances"]  = len(df)
    record["short_utt_pct"] = (utt_lengths <= 2).mean()   # minimal responses

    # ── Confidence features ────────────────────────────────────────────────
    if "Confidence" in df.columns:
        conf = df["Confidence"].replace(0, np.nan).dropna()
        record["avg_conf"]       = conf.mean() if not conf.empty else 0.5
        record["conf_std"]       = conf.std()  if len(conf) > 1 else 0
        record["low_conf_pct"]   = (conf < 0.7).mean() if not conf.empty else 0
    else:
        record["avg_conf"] = record["conf_std"] = record["low_conf_pct"] = 0

    # ── Temporal / timing features ─────────────────────────────────────────
    if "Start_Time" in df.columns and "End_Time" in df.columns:
        df = df.copy()
        df["duration"]  = (df["End_Time"] - df["Start_Time"]).clip(lower=0)
        df["gap"]       = df["Start_Time"].shift(-1) - df["End_Time"]

        record["utt_dur_mean"]    = df["duration"].mean()
        record["utt_dur_std"]     = df["duration"].std() if len(df) > 1 else 0
        record["speech_rate"]     = n_tokens / max(df["duration"].sum(), 1e-3)
        gaps = df["gap"].dropna()
        record["pause_mean"]      = gaps.mean()   if not gaps.empty else 0
        record["pause_std"]       = gaps.std()    if len(gaps) > 1 else 0
        record["long_pause_pct"]  = (gaps > 3.0).mean() if not gaps.empty else 0
    else:
        for k in ["utt_dur_mean","utt_dur_std","speech_rate","pause_mean","pause_std","long_pause_pct"]:
            record[k] = 0

    # ── VADER sentiment ────────────────────────────────────────────────────
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(all_text)
        record["sent_neg"]      = scores["neg"]
        record["sent_pos"]      = scores["pos"]
        record["sent_neu"]      = scores["neu"]
        record["sent_compound"] = scores["compound"]
        # utterance-level sentiment variance
        sent_per_utt = df["Text"].fillna("").apply(
            lambda x: sia.polarity_scores(str(x))["compound"])
        record["sent_var"]     = sent_per_utt.var()
        record["sent_min"]     = sent_per_utt.min()
        record["sent_max"]     = sent_per_utt.max()
        record["neg_utt_pct"]  = (sent_per_utt < -0.1).mean()
    except ImportError:
        for k in ["sent_neg","sent_pos","sent_neu","sent_compound",
                   "sent_var","sent_min","sent_max","neg_utt_pct"]:
            record[k] = 0

    return record


def build_enhanced_features(labels_path=LABELS_CSV, data_root=DATA_ROOT, out_path=OUT_PATH):
    print(f"\n{'='*60}")
    print("  Text Feature Enhancement")
    print(f"{'='*60}")

    labels = pd.read_csv(labels_path)
    for id_col in ["pid","Participant_ID","participant_id","id","ID"]:
        if id_col in labels.columns:
            pid_col = id_col; break

    pids    = labels[pid_col].tolist()
    print(f"  Participants to process: {len(pids)}")

    records = []
    missing = []
    for pid in pids:
        rec = extract_features_from_transcript(str(pid), data_root)
        if rec is None:
            missing.append(pid)
            print(f"  [X] {pid}")
        else:
            records.append(rec)

    print(f"\n  Extracted: {len(records)} / {len(pids)}")
    if missing:
        print(f"  Missing  : {len(missing)} — {missing[:10]}")

    df = pd.DataFrame(records).fillna(0)
    # Ensure pid col is first
    df = df[["pid"] + [c for c in df.columns if c != "pid"]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    feat_cols = [c for c in df.columns if c != "pid"]
    print(f"  Features : {len(feat_cols)}")
    print(f"  Saved to : {out_path}")
    return df


def run_cv(features_path, labels_path=LABELS_CSV, n_folds=5, n_bootstrap=1000):
    """Leakage-free 5-fold CV on the enhanced features."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.metrics import balanced_accuracy_score

    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        HAS_IMBALANCED = True
    except ImportError:
        HAS_IMBALANCED = False

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    # Load data
    df = pd.read_csv(features_path)
    for id_col in ["pid","Participant_ID","participant_id","id","ID"]:
        if id_col in df.columns: df = df.set_index(id_col); break
    for col in ["PHQ_Score","phq_score","label","Label","depressed","Depressed"]:
        if col in df.columns: df = df.drop(columns=[col])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    labels = pd.read_csv(labels_path)
    for id_col in ["pid","Participant_ID","participant_id","id","ID"]:
        if id_col in labels.columns: labels = labels.set_index(id_col); break
    y_series = (labels["PHQ_Score"] >= PHQ_THRESH).astype(int)

    common = df.index.intersection(y_series.index)
    X = df.loc[common].values
    y = y_series.loc[common].values
    n_feat = X.shape[1]
    k = min(40, n_feat)

    print(f"\n{'='*60}")
    print(f"  Honest {n_folds}-Fold CV on enhanced text features")
    print(f"  Samples: {len(y)}  Depressed: {y.sum()}  Features: {n_feat}")
    print(f"{'='*60}\n")

    # Candidate pipelines
    pipes = {}
    pipes["LR_L2"] = Pipeline([
        ("vt", VarianceThreshold(0.001)),
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=3000, solver="lbfgs")),
    ])
    pipes["LR_L1"] = Pipeline([
        ("vt", VarianceThreshold(0.001)),
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", LogisticRegression(C=0.5, class_weight="balanced",
                                   max_iter=3000, solver="saga", penalty="l1")),
    ])
    pipes["SVM"] = Pipeline([
        ("vt", VarianceThreshold(0.001)),
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"), cv=3)),
    ])
    pipes["RF"] = Pipeline([
        ("vt", VarianceThreshold(0.001)),
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                       class_weight="balanced",
                                       min_samples_leaf=2, random_state=RANDOM_STATE)),
    ])
    pipes["GB"] = Pipeline([
        ("vt", VarianceThreshold(0.001)),
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=3, subsample=0.8,
                                           random_state=RANDOM_STATE)),
    ])
    if HAS_XGB:
        pipes["XGB"] = Pipeline([
            ("vt", VarianceThreshold(0.001)),
            ("sc", StandardScaler()),
            ("sel", SelectKBest(f_classif, k=k)),
            ("clf", xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, scale_pos_weight=2,
                                       use_label_encoder=False, eval_metric="logloss",
                                       random_state=RANDOM_STATE, verbosity=0)),
        ])
    if HAS_IMBALANCED:
        pipes["SMOTE+LR"] = ImbPipeline([
            ("vt", VarianceThreshold(0.001)),
            ("sc", StandardScaler()),
            ("sel", SelectKBest(f_classif, k=k)),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")),
        ])
        pipes["SMOTE+RF"] = ImbPipeline([
            ("vt", VarianceThreshold(0.001)),
            ("sc", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                           min_samples_leaf=2, random_state=RANDOM_STATE)),
        ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    print(f"  {'Pipeline':<20} CV AUC (mean +/- std)")
    print(f"  {'-'*48}")
    best_name, best_auc, best_pipe = None, 0, None
    for name, pipe in pipes.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  {name:<20} {scores.mean():.4f} +/- {scores.std():.4f}")
        if scores.mean() > best_auc:
            best_auc, best_name, best_pipe = scores.mean(), name, pipe

    print(f"\n  [OK] Best: {best_name}  CV AUC = {best_auc:.4f}")

    # OOF predictions with best pipeline
    oof_y, oof_p = [], []
    for tr_i, te_i in cv.split(X, y):
        p = best_pipe.__class__(**best_pipe.get_params()) if False else \
            type(best_pipe)(best_pipe.steps) if hasattr(best_pipe, 'steps') else best_pipe
        # Re-clone properly
        from sklearn.base import clone
        p = clone(best_pipe)
        p.fit(X[tr_i], y[tr_i])
        prob = p.predict_proba(X[te_i])[:, 1]
        oof_y.append(y[te_i]); oof_p.append(prob)

    oof_y = np.concatenate(oof_y)
    oof_p = np.concatenate(oof_p)
    oof_auc = roc_auc_score(oof_y, oof_p)
    oof_acc = accuracy_score(oof_y, (oof_p >= 0.5).astype(int))
    oof_bac = balanced_accuracy_score(oof_y, (oof_p >= 0.5).astype(int))
    oof_f1  = f1_score(oof_y, (oof_p >= 0.5).astype(int), zero_division=0)

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(oof_y), len(oof_y))
        if len(np.unique(oof_y[idx])) < 2: continue
        boot_aucs.append(roc_auc_score(oof_y[idx], oof_p[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

    print(f"\n  OOF Results ({n_folds}-fold, {best_name}):")
    print(f"  AUC-ROC    : {oof_auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Accuracy   : {oof_acc:.4f}")
    print(f"  Balanced   : {oof_bac:.4f}")
    print(f"  F1         : {oof_f1:.4f}")
    if oof_auc >= 0.72:
        print(f"\n  [TARGET MET] AUC >= 0.72")
    else:
        print(f"\n  [NOTE] AUC = {oof_auc:.4f}, gap to 0.72 = {0.72 - oof_auc:.4f}")

    # Save final model (retrain on full data)
    from sklearn.base import clone
    final_model = clone(best_pipe)
    final_model.fit(X, y)
    import joblib
    os.makedirs("models", exist_ok=True)
    bundle = {"pipeline": final_model, "threshold": 0.5, "feature_file": features_path}
    joblib.dump(bundle, "models/text_model.pkl")
    joblib.dump(final_model, "models/text_model_plain.pkl")
    print(f"\n  Model saved -> models/text_model.pkl  (trained on all {len(y)} samples)")
    return oof_auc, ci_lo, ci_hi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default=DATA_ROOT)
    parser.add_argument("--out_path",   default=OUT_PATH)
    parser.add_argument("--cv",         action="store_true",
                        help="Run 5-fold CV after extraction")
    parser.add_argument("--cv_only",    action="store_true",
                        help="Skip extraction, run CV on existing enhanced file")
    parser.add_argument("--n_folds",    type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    if not args.cv_only:
        build_enhanced_features(
            labels_path=LABELS_CSV,
            data_root=args.data_root,
            out_path=args.out_path,
        )

    if args.cv or args.cv_only:
        feat_path = args.out_path if os.path.exists(args.out_path) else OUT_PATH
        run_cv(feat_path, n_folds=args.n_folds, n_bootstrap=args.n_bootstrap)
