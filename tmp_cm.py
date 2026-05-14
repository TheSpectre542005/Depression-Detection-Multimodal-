import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

def get_text_ensemble():
    return VotingClassifier([
        ('lr', LogisticRegression(C=0.5, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE)),
        ('svc', SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3,
                                      class_weight='balanced', random_state=RANDOM_STATE)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                                          subsample=0.8, random_state=RANDOM_STATE))
    ], voting='soft')

def get_audio_ensemble():
    return VotingClassifier([
        ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3,
                                          subsample=0.8, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2,
                                      class_weight='balanced', random_state=RANDOM_STATE)),
        ('lr', LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE)),
        ('svc', SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE))
    ], voting='soft')

def get_visual_ensemble():
    return VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=3,
                                      class_weight='balanced', random_state=RANDOM_STATE)),
        ('svc', SVC(C=1.0, kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_STATE)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                                          subsample=0.8, random_state=RANDOM_STATE)),
        ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE))
    ], voting='soft')

def find_optimal_threshold(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]

labels = pd.read_csv('data/features/master_labels.csv')
text_df = pd.read_csv('data/features/text_features.csv')
audio_df = pd.read_csv('data/features/audio_features_enhanced.csv')
visual_df = pd.read_csv('data/features/visual_features.csv')

def fix_phq_leak(df):
    leak_cols = [c for c in df.columns if any(x in c.lower() for x in ['phq', 'score', 'label', 'binary', 'dep_'])]
    to_drop = [c for c in leak_cols if c != 'pid' and c != 'label']
    return df.drop(columns=to_drop)

audio_df = fix_phq_leak(audio_df)
visual_df = fix_phq_leak(visual_df)
merged = labels.merge(text_df, on='pid').merge(audio_df, on='pid').merge(visual_df, on='pid')
y = merged['label'].values
text_cols = [c for c in text_df.columns if c not in ('pid', 'label')]
audio_cols = [c for c in audio_df.columns if c not in ('pid', 'label')]
visual_cols = [c for c in visual_df.columns if c not in ('pid', 'label')]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
all_y = []
all_prob_fusion = []
for train_idx, test_idx in skf.split(merged, y):
    X_train_text = merged.iloc[train_idx][text_cols]
    X_test_text = merged.iloc[test_idx][text_cols]
    X_train_audio = merged.iloc[train_idx][audio_cols]
    X_test_audio = merged.iloc[test_idx][audio_cols]
    X_train_visual = merged.iloc[train_idx][visual_cols]
    X_test_visual = merged.iloc[test_idx][visual_cols]
    y_train, y_test = y[train_idx], y[test_idx]
    smote_k = min(3, sum(y_train == 1) - 1)
    text_pipe = ImbPipeline([('vt', VarianceThreshold()), ('sc', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k)), ('clf', get_text_ensemble())])
    text_pipe.fit(X_train_text, y_train)
    p_text = text_pipe.predict_proba(X_test_text)[:,1]
    audio_pipe = ImbPipeline([('vt', VarianceThreshold()), ('sk', SelectKBest(mutual_info_classif, k=min(50, len(audio_cols)))), ('sc', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k)), ('clf', get_audio_ensemble())])
    audio_pipe.fit(X_train_audio, y_train)
    p_audio = audio_pipe.predict_proba(X_test_audio)[:,1]
    visual_pipe = ImbPipeline([('vt', VarianceThreshold()), ('pca', PCA(n_components=min(30, len(visual_cols)))), ('sc', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k)), ('clf', get_visual_ensemble())])
    visual_pipe.fit(X_train_visual, y_train)
    p_visual = visual_pipe.predict_proba(X_test_visual)[:,1]
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    inner_aucs = {'text': [], 'audio': [], 'visual': []}
    for i_train, i_val in inner_cv.split(X_train_text, y_train):
        yt, yv = y_train[i_train], y_train[i_val]
        sk = min(3, sum(yt == 1) - 1)
        tp = ImbPipeline([('vt', VarianceThreshold()), ('sc', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=sk)), ('clf', get_text_ensemble())])
        tp.fit(X_train_text.values[i_train], yt)
        try: inner_aucs['text'].append(roc_auc_score(yv, tp.predict_proba(X_train_text.values[i_val])[:,1]))
        except: inner_aucs['text'].append(0.5)
        ap = ImbPipeline([('vt', VarianceThreshold()), ('sk', SelectKBest(mutual_info_classif, k=min(50, len(audio_cols)))), ('sc', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=sk)), ('clf', get_audio_ensemble())])
        ap.fit(X_train_audio.values[i_train], yt)
        try: inner_aucs['audio'].append(roc_auc_score(yv, ap.predict_proba(X_train_audio.values[i_val])[:,1]))
        except: inner_aucs['audio'].append(0.5)
        vp = ImbPipeline([('vt', VarianceThreshold()), ('pca', PCA(n_components=min(30, len(visual_cols)))), ('sc', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=sk)), ('clf', get_visual_ensemble())])
        vp.fit(X_train_visual.values[i_train], yt)
        try: inner_aucs['visual'].append(roc_auc_score(yv, vp.predict_proba(X_train_visual.values[i_val])[:,1]))
        except: inner_aucs['visual'].append(0.5)
    w = {mod: max(0.01, np.mean(inner_aucs[mod]) - 0.5) for mod in inner_aucs}
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    p_fusion = w['text'] * p_text + w['audio'] * p_audio + w['visual'] * p_visual
    all_y.extend(y_test.tolist())
    all_prob_fusion.extend(p_fusion.tolist())
all_y = np.array(all_y)
all_prob_fusion = np.array(all_prob_fusion)
threshold = find_optimal_threshold(all_y, all_prob_fusion)
preds = (all_prob_fusion >= threshold).astype(int)
cm = confusion_matrix(all_y, preds)
print(threshold)
print(cm)
