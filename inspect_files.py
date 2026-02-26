# inspect_files.py
import pandas as pd
import os

DATA_ROOT = r"C:\Users\Rishil\Downloads\E-DAIC\data"
TEST_PID = 300  # we'll inspect this participant

base = os.path.join(DATA_ROOT, f"{TEST_PID}_P")

files_to_check = {
    "Transcript"    : f"{TEST_PID}_Transcript.csv",
    "MFCC"          : f"features/{TEST_PID}_OpenSMILE2.3.0_mfcc.csv",
    "eGeMAPS"       : f"features/{TEST_PID}_OpenSMILE2.3.0_egemaps.csv",
    "OpenFace AUs"  : f"features/{TEST_PID}_OpenFace2.1.0_Pose_gaze_AUs.csv",
}

for name, rel_path in files_to_check.items():
    full_path = os.path.join(base, rel_path)
    print(f"\n{'='*55}")
    print(f"📄 {name}")
    print(f"   Path: {full_path}")
    if not os.path.exists(full_path):
        print("   ❌ FILE NOT FOUND")
        continue
    try:
        df = pd.read_csv(full_path, nrows=3)
        print(f"   ✅ Shape (first 3 rows): {df.shape}")
        print(f"   Columns ({len(df.columns)} total): {df.columns.tolist()[:10]}")
        print(f"   First row sample:\n{df.iloc[0][:5]}")
    except Exception as e:
        # Try with different separator
        try:
            df = pd.read_csv(full_path, nrows=3, sep=';')
            print(f"   ✅ Shape (sep=';'): {df.shape}")
            print(f"   Columns: {df.columns.tolist()[:10]}")
        except Exception as e2:
            print(f"   ❌ Error: {e2}")