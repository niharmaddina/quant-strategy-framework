import yaml
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
