from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

for directory in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
