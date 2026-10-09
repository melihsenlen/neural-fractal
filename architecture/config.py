import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(path) as y:
        return yaml.safe_load(y)

def path(config: dict, entry: str) -> Path:
    return Path(config["paths"][entry])