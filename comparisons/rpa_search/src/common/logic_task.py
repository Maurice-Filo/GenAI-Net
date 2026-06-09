from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Tuple

import numpy as np

from RL4CRN.utils.crn_builders import build_logic_IOCRN
from RL4CRN.utils.default_tasks.LogicTaskKind import LogicTaskKind  # noqa: F401
from RL4CRN.utils.input_interface import Configurator, make_task
from RL4CRN.utils.library_builders import build_MAK_library

from comparisons.rpa_search.src.common.rpa_task import QuietRewardFunction, _tuple_or_value


def build_logic_components(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    logic = config["logic"]

    cfg = Configurator.preset(logic.get("preset", "paper"))
    cfg.solver.algorithm = logic.get("solver", "LSODA")
    cfg.solver.rtol = float(logic.get("rtol", 1e-8))
    cfg.solver.atol = float(logic.get("atol", 1e-8))

    n_inputs = int(logic.get("n_inputs", 4))
    crn, species_labels = build_logic_IOCRN(
        n_inputs=n_inputs,
        include_dilution=bool(logic.get("include_dilution", False)),
        solver=cfg.solver,
        n_support_species=int(logic.get("n_support_species", 0)),
        dilution_rate=float(logic.get("dilution_rate", 0.05)),
    )

    library_components = build_MAK_library(
        crn,
        species_labels,
        order=int(logic.get("library_order", 2)),
    )

    task = make_task(
        template_crn=crn,
        library_components=library_components,
        kind="logic",
        species_labels=species_labels,
        params={
            "n_inputs": n_inputs,
            "t_f": float(logic.get("t_f", 100.0)),
            "n_t": int(logic.get("n_t", 1000)),
            "ic": _tuple_or_value(logic.get("ic", ("constant", 0.01))),
            "weights": _tuple_or_value(logic.get("weights", "transient")),
            "logic_fn": build_logic_fn(logic.get("formula", "chain_or_pairs_4")),
        },
    )
    task.compute_reward = QuietRewardFunction(task.compute_reward)

    return crn, library_components, task, cfg


def build_logic_fn(formula: str):
    if formula != "chain_or_pairs_4":
        raise ValueError(f"Unsupported logic formula: {formula}")

    def logic_fn(u):
        x = np.asarray(u, dtype=float) > 0.5
        return bool((x[0] and x[1]) or (x[1] and x[2]) or (x[2] and x[3]))

    return logic_fn
