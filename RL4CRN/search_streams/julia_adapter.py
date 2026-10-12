"""Explicit deterministic tracking adapter; the canonical evaluator stays authoritative."""

from types import SimpleNamespace

import numpy as np

from RL4CRN.parameter_optimization import Experiment, ReactionLibrary


def task_experiment(evaluator, task):
    from RL4CRN.utils.input_interface import get_task_kind, overrides_get
    from RL4CRN.utils.target_fn import build_r_list_from_target

    if task.kind == "classification":
        return _classification_experiment(evaluator, task)
    if task.kind not in ("tracking", "logic", "dose_response") or task.norm != 1:
        raise ValueError("Julia adapter supports deterministic tracking, Logic and dose-response L1 only")
    crn = evaluator.crn_template
    if str(crn.solver).upper() != "CVODE" or len(crn.output_labels) != 1:
        raise ValueError("Julia pilot requires CVODE and one named species output")
    library = ReactionLibrary.from_iocrn(crn, evaluator.library)
    kind = get_task_kind(task.kind)
    times = kind.build_time_horizon(task)
    u = kind.build_u_list(task, {})
    view = SimpleNamespace(species_labels=list(library.species), num_species=len(library.species),
        species_idx_dict={s: i for i, s in enumerate(library.species)})
    ic = kind.build_ic(task, {})
    if isinstance(ic, str):
        raise ValueError("Steady-state-derived initial conditions are not supported")
    x0 = ic.get_ic(view)
    if task.kind == "logic":
        if len(x0) != 1:
            raise ValueError("Logic pilot requires one initial condition, matching canonical row targets")
        fn = overrides_get(task, {}, "logic_fn", fallback_attr="logic_fn")
        targets = [[float(bool(fn(v)))] for v in u]
    else:
        view.input_idx_dict = dict(crn.input_idx_dict)
        targets = build_r_list_from_target(
            target=overrides_get(task, {}, "target", fallback_attr="target"), template_crn=view,
            u_list=u, x0_list=x0, expand_over_ic=True, q=1)
        if len(targets) == len(u) and len(x0) > 1:
            targets = targets * len(x0)
    pairs = [(x, v) for x in x0 for v in u]
    output = np.zeros((1, len(library.species)))
    output[0, library.species.index(crn.output_labels[0])] = 1.
    targets = np.broadcast_to(np.asarray(targets)[:, :, None], (len(pairs), 1, len(times)))
    experiment = Experiment(times, np.asarray([p[0] for p in pairs]),
        np.asarray([p[1] for p in pairs]), output, targets,
        weights=kind.build_weights(task, {}), rtol=crn.rtol, atol=crn.atol,
        max_species=float(task.LARGE_NUMBER))
    return library, experiment


def _classification_experiment(evaluator, task):
    """Preserve the canonical IC/label pairing, two-output mean and time grid."""
    from RL4CRN.utils.input_interface import build_weights, overrides_get
    from RL4CRN.utils.default_tasks.ClassificationTaskKind import (
        _build_ic_and_r_list_from_ic_r_maps, _normalize_u_list,
    )

    def value(name, default=None):
        return overrides_get(task, {}, name, fallback_attr=name, default=default)

    crn = evaluator.crn_template
    if str(crn.solver).upper() != "CVODE" or int(value("norm", 1)) != 1 or value("relative", False):
        raise ValueError("Classification adapter requires CVODE and absolute L1 loss")
    library = ReactionLibrary.from_iocrn(crn, evaluator.library)
    if list(library.species) != list(task.species_labels):
        raise ValueError("Classification IC order must match the canonical species order")
    ic, labels = _build_ic_and_r_list_from_ic_r_maps(task.species_labels, value("ic_r_maps"))
    inputs = _normalize_u_list(value("u_list"), int(crn.num_inputs or task.n_inputs or 1))
    if len(inputs) != 1:
        raise ValueError("Classification adapter requires one input vector per IC/label pair")
    times = np.linspace(0., float(value("t_f")), int(value("n_t")), dtype=np.float32)
    states = np.asarray(ic.get_ic(crn))
    q = len(crn.output_labels)
    labels = np.asarray(labels)
    if labels.shape != (len(states), q):
        raise ValueError("Classification targets must align with scenarios and output labels")
    output = np.zeros((q, len(library.species)))
    for i, name in enumerate(crn.output_labels):
        output[i, library.species.index(name)] = 1.
    targets = np.broadcast_to(labels[:, :, None], (len(states), q, len(times)))
    # IOCRN keeps an inert numerical input slot for autonomous systems, while
    # the Julia model lists only named channels (none in this benchmark).
    scenario_inputs = (np.repeat(inputs, len(states), axis=0) if library.inputs
                       else np.empty((len(states), 0)))
    experiment = Experiment(times, states, scenario_inputs,
        output, targets, weights=build_weights(q=q, n_t=len(times), w_spec=value("weights", "uniform")),
        rtol=crn.rtol, atol=crn.atol, max_species=float(task.LARGE_NUMBER))
    return library, experiment
