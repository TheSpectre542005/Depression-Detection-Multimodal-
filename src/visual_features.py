# src/visual_features.py
import pandas as pd
import numpy as np
import os

DATA_ROOT = r"C:\Users\Rishil\Downloads\E-DAIC\data"
SAVE_PATH = "data/features/visual_features.csv"

def build_visual_features(participant_ids):
    print("\n👁️  Extracting visual features...")
    records = []
    missing = []

    for pid in participant_ids:
        path = os.path.join(DATA_ROOT, f"{pid}_P", "features",
                            f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        if not os.path.exists(path):
            missing.append(pid)
            continue
        try:
            df = pd.read_csv(path)

            # Only keep high-confidence frames
            if 'confidence' in df.columns:
                df = df[df['confidence'] >= 0.80]
            if 'success' in df.columns:
                df = df[df['success'] == 1]

            # Select feature groups
            au_r_cols   = [c for c in df.columns if 'AU' in c and '_r' in c]
            au_c_cols   = [c for c in df.columns if 'AU' in c and '_c' in c]
            pose_cols   = [c for c in df.columns if 'pose_' in c]
            gaze_cols   = [c for c in df.columns if 'gaze_' in c]

            use_cols = au_r_cols + au_c_cols + pose_cols + gaze_cols

            if not use_cols:
                print(f"  ⚠️  No feature columns found for {pid}")
                continue

            sub    = df[use_cols].select_dtypes(include=[np.number])
            record = {'pid': pid}

            # Aggregate: mean, std, min, max per feature
            for col in sub.columns:
                record[f'{col}_mean'] = sub[col].mean()
                record[f'{col}_std']  = sub[col].std()
                record[f'{col}_min']  = sub[col].min()
                record[f'{col}_max']  = sub[col].max()

            # Extra: % of frames where each AU is active (from _c cols)
            for col in au_c_cols:
                if col in df.columns:
                    record[f'{col}_pct_active'] = df[col].mean()

            records.append(record)

        except Exception as e:
            print(f"  ⚠️  Error on {pid}: {e}")

    if missing:
        print(f"  Missing visual features: {len(missing)} participants")

    df = pd.DataFrame(records).fillna(0)
    os.makedirs('data/features', exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f"  ✅ Visual features saved → {SAVE_PATH}")
    print(f"  Shape: {df.shape}")
    return df

if __name__ == "__main__":
    labels = pd.read_csv('data/features/master_labels.csv')
    build_visual_features(labels['pid'].tolist())