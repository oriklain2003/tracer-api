from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

RULE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "rules" / "rule_config.json"

# Training Bounding Box (Levant Region)
TRAIN_NORTH = 34.597042
TRAIN_SOUTH = 28.536275
TRAIN_WEST  = 32.299805
TRAIN_EAST  = 37.397461

@lru_cache(maxsize=None)
def load_rule_config(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else RULE_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
