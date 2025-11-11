import json
from pathlib import Path

def norm_code(x):
    if x is None: return None
    return str(x).strip().upper()

def load_all_geo(paths: dict[int, Path]) -> dict[int, dict]:
    out = {}
    for lvl, p in paths.items():
        with open(p, "r", encoding="utf-8") as f:
            out[lvl] = json.load(f)
    return out
