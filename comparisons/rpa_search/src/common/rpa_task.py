from __future__ import annotations

from typing import Any, Dict, Tuple
from contextlib import redirect_stdout
from io import StringIO

from RL4CRN.utils.input_interface import Configurator, make_task
from RL4CRN.utils.crn_builders import build_simple_IOCRN
from RL4CRN.utils.library_builders import build_MAK_library
from RL4CRN.utils.default_tasks.TrackingTaskKind import TrackingTaskKind  # noqa: F401


def build_rpa_components(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Build the canonical RPA IOCRN, library, task, and RL4CRN config."""
    rpa = config["rpa"]

    cfg = Configurator.preset("balanced")
    cfg.solver.algorithm = rpa.get("solver", "LSODA")
    cfg.solver.rtol = float(rpa.get("rtol", 1e-8))
    cfg.solver.atol = float(rpa.get("atol", 1e-8))

    crn, species_labels = build_simple_IOCRN(
        species=list(rpa["species"]),
        production_input_map=dict(rpa["production_input_map"]),
        degradation_input_map=dict(rpa.get("degradation_input_map", {})),
        dilution_map=dict(rpa.get("dilution_map", {})),
        output_species=rpa["output_species"],
        solver=cfg.solver,
    )

    library_components = build_MAK_library(
        crn,
        species_labels,
        order=int(rpa.get("library_order", 2)),
    )

    task = make_task(
        template_crn=crn,
        library_components=library_components,
        kind="tracking",
        species_labels=species_labels,
        params={
            "t_f": float(rpa.get("t_f", 100.0)),
            "n_t": int(rpa.get("n_t", 1000)),
            "ic": _tuple_or_value(rpa.get("ic", ("constant", 0.01))),
            "weights": _tuple_or_value(rpa.get("weights", "transient")),
            "u_values": list(rpa.get("u_values", [0.5, 1.0, 1.5])),
            "target": lambda u_1: u_1,
        },
    )
    task.compute_reward = QuietRewardFunction(task.compute_reward)

    return crn, library_components, task, cfg


def _tuple_or_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


class QuietRewardFunction:
    """Suppress verbose prints from notebook-oriented reward functions."""

    def __init__(self, reward_fn):
        self.reward_fn = reward_fn

    def __call__(self, crn):
        with redirect_stdout(StringIO()):
            return self.reward_fn(crn)
