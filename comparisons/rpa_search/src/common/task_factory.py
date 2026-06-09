from __future__ import annotations

from typing import Any, Dict, Tuple

from comparisons.rpa_search.src.common.logic_task import build_logic_components
from comparisons.rpa_search.src.common.rpa_task import build_rpa_components


def build_components(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    task_name = config.get("task", config.get("benchmark", {}).get("task", "rpa"))
    if task_name == "rpa":
        return build_rpa_components(config)
    if task_name == "logic":
        return build_logic_components(config)
    raise ValueError(f"Unknown benchmark task: {task_name}")
