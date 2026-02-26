# src/load_labels.py
import pandas as pd
import os

LABELS_DIR = r"C:\Users\Rishil\Downloads\E-DAIC\labels"

def load_labels():
    train = pd.read_csv(os.path.join(LABELS_DIR, "train_split.csv"))
    dev   = pd.read_csv(os.path.join(LABELS_DIR, "dev_split.csv"))
    
    print("=== TRAIN SPLIT ===")
    print("Columns:", train.columns.tolist())
    print(train.head(3))
    
    print("\n=== DEV SPLIT ===")
    print("Columns:", dev.columns.tolist())
    print(dev.head(3))
    
    return train, dev

def build_master_labels(save=True):
    train, dev = load_labels()
    df = pd.concat([train, dev], ignore_index=True)
    
    # ── Auto-detect column names ──────────────────────────
    # Find participant ID column
    pid_col = None
    for candidate in ['Participant_ID','participant_id','ID','id','SubjectID']:
        if candidate in df.columns:
            pid_col = candidate
            break
    
    # Find PHQ score column
    score_col = None
    for candidate in ['PHQ_Score','PHQ8_Score','phq_score','PHQ8','phq8','Score']:
        if candidate in df.columns:
            score_col = candidate
            break
    
    if pid_col is None or score_col is None:
        print("\n⚠️  Could not auto-detect columns!")
        print("All columns:", df.columns.tolist())
        print("Please manually set pid_col and score_col")
        return None
    
    print(f"\n✅ Using: pid_col='{pid_col}', score_col='{score_col}'")
    
    # Binary label: PHQ >= 10 → Depressed
    df['label'] = (df[score_col] >= 10).astype(int)
    df = df[[pid_col, score_col, 'label']].rename(columns={pid_col: 'pid'})
    df['pid'] = df['pid'].astype(int)
    
    print(f"\n📊 Class Distribution:")
    print(df['label'].value_counts())
    print(f"\nTotal participants: {len(df)}")
    print(f"Depressed (1):     {df['label'].sum()}")
    print(f"Not Depressed (0): {(df['label']==0).sum()}")
    print(f"Imbalance ratio:   {df['label'].mean():.2%} positive")
    
    if save:
        os.makedirs('data/features', exist_ok=True)
        df.to_csv('data/features/master_labels.csv', index=False)
        print(f"\n✅ Saved → data/features/master_labels.csv")
    
    return df

if __name__ == "__main__":
    df = build_master_labels()