from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "world_cup_training_matches.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

DATA_CUTOFF = "2024-06-14"
TARGET_MATCH_COUNT = None
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42

for directory in [PROCESSED_DIR, FEATURES_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
