# config.py — Centralized project configuration
import os

# ── Data Paths ─────────────────────────────────────────────────
# Supports environment variables for portability across systems
DATA_ROOT = os.environ.get(
    'EDAIC_DATA_ROOT',
    os.path.join(os.path.expanduser('~'), 'Downloads', 'E-DAIC', 'data')
)
LABELS_DIR = os.environ.get(
    'EDAIC_LABELS_DIR',
    os.path.join(os.path.expanduser('~'), 'Downloads', 'E-DAIC', 'labels')
)

# ── Output Paths ───────────────────────────────────────────────
FEATURES_DIR = 'data/features'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# ── Model Hyperparameters ──────────────────────────────────────
N_TFIDF = 50
PCA_COMPONENTS = 20
SMOTE_K_NEIGHBORS = 3
CV_SPLITS = 5
CV_REPEATS = 3           # Repeated Stratified K-Fold (5×3 = 15 evaluations)
RANDOM_STATE = 42

# ── Sentence-Transformers (optional) ───────────────────────────
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'  # 384-dim, ~80MB, very fast
N_SBERT_COMPONENTS = 20                 # PCA-reduced embedding dimensions

# ── Data Augmentation ─────────────────────────────────────────
AUGMENT_MIXUP = True          # Blend features from different samples
AUGMENT_NOISE = True          # Add Gaussian noise to features
MIXUP_ALPHA = 0.2             # Beta distribution parameter for Mixup
NOISE_STD = 0.05              # Gaussian noise standard deviation
AUGMENT_FACTOR = 2            # Multiply minority class by this factor

# ── Regularization search grid ─────────────────────────────────
C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# ── Threshold tuning ───────────────────────────────────────────
THRESHOLD_MIN = 0.25
THRESHOLD_MAX = 0.65
THRESHOLD_STEP = 0.01

# ── Fusion settings ────────────────────────────────────────────
MIN_AUC_FOR_FUSION = 0.52
# Set to True when audio model validation AUC exceeds 0.60
AUDIO_RELIABLE = os.environ.get('AUDIO_RELIABLE', 'false').lower() == 'true'

# ── Clinical thresholds ────────────────────────────────────────
MIN_CLINICAL_AUC = 0.70       # Minimum acceptable AUC for clinical use
MIN_CLINICAL_SENSITIVITY = 0.70
MIN_CLINICAL_F1 = 0.60

# ── Bootstrap CI ───────────────────────────────────────────────
N_BOOTSTRAP = 1000            # Number of bootstrap iterations for CIs
CI_LEVEL = 0.95               # 95% confidence intervals

# ── Flask settings ─────────────────────────────────────────────
FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
FLASK_PORT = int(os.environ.get('FLASK_PORT', 5000))
