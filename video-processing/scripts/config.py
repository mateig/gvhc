"""Config loaded once from config.yaml at import time."""

import yaml

with open("config.yaml") as f:
    cfg: dict = yaml.safe_load(f) or {}
