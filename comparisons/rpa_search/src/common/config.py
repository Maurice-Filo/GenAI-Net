from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value
