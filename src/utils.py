from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "quant_project") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
