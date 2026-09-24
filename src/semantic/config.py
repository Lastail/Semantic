from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import yaml


def _merge(a: dict, b: dict) -> dict:
    """Recursively merge b on top of a without mutating either input."""
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(*paths: str | Path) -> dict:
    """Load YAML files from left to right; later files override earlier ones."""
    cfg: dict = {}
    for path in paths:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg
