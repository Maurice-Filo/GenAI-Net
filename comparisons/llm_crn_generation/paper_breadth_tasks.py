"""Reproducible builders for the paper-extension breadth experiments."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np

from RL4CRN.utils.crn_builders import build_simple_IOCRN
from RL4CRN.utils.default_tasks.ClassificationTaskKind import ClassificationTaskKind
from RL4CRN.utils.default_tasks.DoseResponseTaskKind import DoseResponseTaskKind
from RL4CRN.utils.default_tasks.OscillatorFreqTaskKind import OscillatorFreqTaskKind
from RL4CRN.utils.default_tasks.OscillatorMeanTaskKind import OscillatorMeanTaskKind
from RL4CRN.utils.default_tasks.SSARobustTaskKind import SSARobustTaskKind
from RL4CRN.utils.input_interface import Configurator, make_task
from RL4CRN.utils.library_builders import build_MAK_library


def hill_target(u_1: float) -> float:
    return float(2.0 * u_1**2 / (0.25**2 + u_1**2))


def ultrasensitive_target(u_1: float) -> float:
    return float(2.0 * u_1**8 / (0.5**8 + u_1**8))


def biphasic_target(u_1: float) -> float:
    return float(8.0 * u_1 / (1.0 + u_1) / (1.0 + (u_1 / 0.55) ** 4))


DOSE_TARGETS: dict[str, Callable[[float], float]] = {
    "dose_hill": hill_target,
    "dose_ultrasensitive": ultrasensitive_target,
    "dose_biphasic": biphasic_target,
}


def build_paper_breadth_components(config: dict[str, Any], task_name: str):
    cfg = Configurator.preset("paper")
    task_cfg = config[task_name]
    cfg.solver.algorithm = str(task_cfg.get("solver", "CVODE"))
    cfg.solver.rtol = float(task_cfg.get("rtol", 1e-10))
    cfg.solver.atol = float(task_cfg.get("atol", 1e-10))

    if task_name in DOSE_TARGETS:
        crn, species = build_simple_IOCRN(
            species=["X_1", "X_2", "X_3"],
            production_input_map={"X_1": "u_1"},
            degradation_input_map={},
            dilution_map={},
            output_species="X_3",
            solver=cfg.solver,
        )
        library_components = build_MAK_library(crn, species, order=2)
        task = make_task(
            template_crn=crn,
            library_components=library_components,
            kind="dose_response",
            species_labels=species,
            params={
                "t_f": 100,
                "n_t": 1000,
                "ic": ("constant", 0.01),
                "weights": "transient",
                "u_spec": ("linspace", 0.0, 1.0, 10),
                "target": DOSE_TARGETS[task_name],
            },
        )
    elif task_name == "classifier":
        crn, species = build_simple_IOCRN(
            species=["X_1", "X_2"],
            production_input_map={},
            degradation_input_map={},
            dilution_map={},
            production_map={},
            output_species=["X_1", "X_2"],
            solver=cfg.solver,
        )
        library_components = build_MAK_library(crn, species, order=2)
        eps = 1e-3
        diagonal_1 = [[x / 10, x / 10 - 0.3] for x in range(3, 13)]
        diagonal_2 = [[x / 10, x / 10 + 0.3] for x in range(10)]
        cluster_1 = [[0.9, eps], [1, eps], [1.1, eps], [0.9, 0.1], [1, 0.1], [1.1, 0.1]]
        cluster_2 = [[eps, 0.9], [eps, 1], [eps, 1.1], [0.1, 0.9], [0.1, 1], [0.1, 1.1]]
        task = make_task(
            template_crn=crn,
            library_components=library_components,
            kind="classification",
            species_labels=species,
            params={
                "ic_r_maps": [
                    (diagonal_1, [1, 0]),
                    (diagonal_2, [0, 1]),
                    (cluster_1, [1, 0]),
                    (cluster_2, [0, 1]),
                ],
                # IOCRN retains one inert input slot even without an input-modulated reaction.
                "u_list": [1.0],
                "t_f": 100,
                "n_t": 1000,
                "weights": "uniform",
                "norm": 1,
                "relative": False,
                "LARGE_NUMBER": 1e4,
            },
        )
    elif task_name in {"oscillator_mean", "oscillator_frequency"}:
        frequency = task_name == "oscillator_frequency"
        crn, species = build_simple_IOCRN(
            species=["X_1", "X_2", "X_3"],
            production_input_map={"X_1": "u_1"} if frequency else {},
            dilution_map={},
            output_species="X_3",
            solver=cfg.solver,
        )
        library_components = build_MAK_library(crn, species, order=2)
        params: dict[str, Any] = {
            "t_f": 100,
            "n_t": 1000,
            "ic": ("constant", 0.01),
            "weights": "transient",
            "osc_w": [0.0, 0.6, 0.1, 0.3],
        }
        if frequency:
            params.update(
                {"u_values": [0.1, 1 / 15, 0.05], "freq_target": lambda u_1: u_1}
            )
            kind = "oscillator_freq"
        else:
            params.update({"u_values": [0.0], "mean_target": 1.0})
            kind = "oscillator_mean"
        task = make_task(
            template_crn=crn,
            library_components=library_components,
            kind=kind,
            species_labels=species,
            params=params,
        )
    elif task_name == "stochastic_rpa":
        crn, species = build_simple_IOCRN(
            species=["Z_1", "Z_2", "Z_3", "X_1"],
            production_input_map={"Z_1": "u_1"},
            degradation_input_map={"X_1": "u_2"},
            dilution_map={},
            output_species="X_1",
            solver=cfg.solver,
        )
        library_components = build_MAK_library(crn, species, order=2)
        task = make_task(
            template_crn=crn,
            library_components=library_components,
            kind="ssa_robust",
            species_labels=species,
            params={
                "t_f": 100,
                "n_t": 100,
                "ic": ("constant", 0.01),
                "weights": "transient",
                "u_values": [0.5, 1.0, 1.5],
                "target": lambda u_1: 3.0 * u_1,
                "n_trajectories": 1000,
                "LARGE_NUMBER": 1e3,
            },
        )
    else:
        raise ValueError(f"Unknown breadth task: {task_name}")

    return crn, library_components, task, cfg


def make_builder(task_name: str):
    return lambda config: build_paper_breadth_components(config, task_name)


BREADTH_TASKS = tuple(DOSE_TARGETS) + (
    "classifier",
    "oscillator_mean",
    "oscillator_frequency",
    "stochastic_rpa",
)
DETERMINISTIC_TASKS = tuple(task for task in BREADTH_TASKS if task != "stochastic_rpa")

BREADTH_BUILDERS = {task: make_builder(task) for task in BREADTH_TASKS}
