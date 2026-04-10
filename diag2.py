import pandas as pd, joblib, numpy as np

print("MODEL DIAGNOSTICS")
print("-"*60)
for name, path in [("audio","models/audio_model.pkl"),("text","models/text_model.pkl"),("visual","models/visual_model.pkl")]:
    bundle = joblib.load(path)
    model = bundle["pipeline"] if isinstance(bundle, dict) else bundle
    thresh = bundle.get("threshold", 0.5) if isinstance(bundle, dict) else 0.5
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.keys())
        scaler = model.named_steps.get("scaler", None)
        exp = scaler.n_features_in_ if scaler is not None else "?"
        clf = type(list(model.named_steps.values())[-1]).__name__
    else:
        steps = []
        exp = getattr(model, "n_features_in_", "?")
        clf = type(model).__name__
    print(f"  {name}")
    print(f"    threshold : {thresh:.4f}")
    print(f"    exp_feats : {exp}")
    print(f"    steps     : {steps}")
    print(f"    classifier: {clf}")

print()
print("FEATURE FILE DIMENSIONS")
print("-"*60)
fmap = {
    "audio_features.csv":          "data/features/audio_features.csv",
    "audio_features_enhanced.csv": "data/features/audio_features_enhanced.csv",
    "text_features.csv":           "data/features/text_features.csv",
    "visual_features.csv":         "data/features/visual_features.csv",
}
for n, p in fmap.items():
    df = pd.read_csv(p)
    print(f"  {n}: {df.shape}")
