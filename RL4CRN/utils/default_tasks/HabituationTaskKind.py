from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from itertools import product

import numpy as np
from scipy.optimize import fsolve

from RL4CRN.utils.input_interface import (
    register_task_kind,
    overrides_get,
    TaskKindBase,
    TaskSpec,
)

from RL4CRN.rewards.deterministic import (
    habituation_metric_with_gap,
    habituation_error_piecewise,
)


def build_on_off_gap_time_horizon(
    *,
    t_on: float,
    t_off: float,
    t_gap: float,
    n_repeats_pre: int,
    n_repeats_post: int,
    n_t: int,
    dtype=np.float32,
) -> List[np.ndarray]:
    """Build a piecewise time horizon with an OFF gap between two pulse trains.

    Each segment time grid is a *local* linspace from 0..Tseg (legacy format).
    Segment layout:
        [ON, OFF] * n_repeats_pre  +  [GAP_OFF]  +  [ON, OFF] * n_repeats_post

    Args:
        t_on: Duration of each ON segment.
        t_off: Duration of each OFF segment.
        t_gap: Duration of the long OFF gap between trains.
        n_repeats_pre: Number of ON/OFF repetitions before the gap.
        n_repeats_post: Number of ON/OFF repetitions after the gap.
        n_t: Total number of time samples distributed across segments.
        dtype: Numpy dtype for time arrays.

    Returns:
        List of 1D numpy arrays, one per segment, each spanning [0, Tseg].

    Raises:
        ValueError: If durations or repeats are invalid, or n_t too small.
    """
    if n_repeats_pre <= 0 or n_repeats_post <= 0:
        raise ValueError("n_repeats_pre and n_repeats_post must be >= 1.")
    if t_on <= 0 or t_off <= 0 or t_gap <= 0:
        raise ValueError("t_on, t_off, and t_gap must be > 0.")

    n_segments = 2 * n_repeats_pre + 1 + 2 * n_repeats_post
    if n_t < 2 * n_segments:
        raise ValueError(f"n_t={n_t} too small for {n_segments} segments.")

    base_pts = max(2, int(np.floor(n_t / n_segments)))

    def lin(T: float, pts: int) -> np.ndarray:
        return np.linspace(0.0, float(T), int(pts), dtype=dtype)

    # Keep ON/OFF stable; allocate extra resolution to long gaps.
    pts_on = max(2, int(base_pts * (t_on / (t_on + t_off))))
    pts_off = max(2, base_pts - pts_on + 1)
    pts_gap = max(2, int(base_pts * (t_gap / (t_on + t_off))) + 1)

    nested: List[np.ndarray] = []
    for _ in range(n_repeats_pre):
        nested.append(lin(t_on, pts_on))
        nested.append(lin(t_off, pts_off))

    nested.append(lin(t_gap, pts_gap))

    for _ in range(n_repeats_post):
        nested.append(lin(t_on, pts_on))
        nested.append(lin(t_off, pts_off))

    return nested


def build_u_nested_list_with_gap(
    *,
    u_list: List[np.ndarray],
    n_repeats_pre: int,
    n_repeats_post: int,
    off_value: float = 0.0,
) -> List[List[np.ndarray]]:
    """Build ON/OFF input protocols with an OFF gap between trains.

    For each `u` in `u_list`, produces:
        [u, u_off] * n_repeats_pre  +  [u_off]  +  [u, u_off] * n_repeats_post

    Args:
        u_list: List of constant input vectors (shape (p,)).
        n_repeats_pre: Number of ON/OFF repetitions before the gap.
        n_repeats_post: Number of ON/OFF repetitions after the gap.
        off_value: Scalar OFF value broadcast to all inputs.

    Returns:
        List of protocols, one per u. Each protocol is a list of (p,) arrays,
        length (2*n_repeats_pre + 1 + 2*n_repeats_post).
    """
    u_nested_list: List[List[np.ndarray]] = []
    for u in u_list:
        u = np.asarray(u, dtype=np.float32).reshape(-1)
        u_off = np.full_like(u, float(off_value), dtype=np.float32)

        protocol: List[np.ndarray] = []
        for _ in range(n_repeats_pre):
            protocol.append(u)
            protocol.append(u_off)

        protocol.append(u_off)

        for _ in range(n_repeats_post):
            protocol.append(u)
            protocol.append(u_off)

        u_nested_list.append(protocol)

    return u_nested_list


def extract_peaks_pre_post_from_piecewise(
    intervals: Sequence[Tuple[float, float]],
    t: np.ndarray,
    y: np.ndarray,
    n_repeats_pre: int,
    n_repeats_post: int,
    LARGE_NUMBER: float = 1e4,
) -> Tuple[List[float], List[float]]:
    """Extract stimulus peaks before and after a gap from a piecewise protocol.

    Assumes segment layout:
        [ON, OFF]*n_repeats_pre + [GAP_OFF] + [ON, OFF]*n_repeats_post

    Intervals are legacy local grids (0..Tseg). This function converts them to
    absolute [start, end] bounds by cumulative durations.

    Args:
        intervals: Segment intervals in legacy format (0..Tseg).
        t: Stitched absolute time vector (T,).
        y: Single-scenario output trajectory, shape (q, T) or (T,).
        n_repeats_pre: Pulses before gap.
        n_repeats_post: Pulses after gap.
        LARGE_NUMBER: Value to return if a segment has no samples.

    Returns:
        peaks_pre: List of peaks in ON segments before the gap.
        peaks_post: List of peaks in ON segments after the gap.
    """
    durations = np.array([float(e - s) for (s, e) in intervals], dtype=float)
    starts = np.cumsum(np.concatenate([[0.0], durations[:-1]]))
    ends = starts + durations
    abs_intervals = list(zip(starts, ends))

    gap_idx = 2 * n_repeats_pre
    stim_pre_idx = list(range(0, 2 * n_repeats_pre, 2))
    stim_post_idx = list(range(gap_idx + 1, gap_idx + 1 + 2 * n_repeats_post, 2))

    y0 = y[0] if y.ndim == 2 else y

    def seg_peak(seg_idx: int) -> float:
        start, end = abs_intervals[seg_idx]
        mask = (t >= start) & (t <= end)
        if not np.any(mask):
            return float(LARGE_NUMBER)
        return float(np.max(y0[mask]))

    peaks_pre = [seg_peak(i) for i in stim_pre_idx]
    peaks_post = [seg_peak(i) for i in stim_post_idx]
    return peaks_pre, peaks_post


def steady_state_ic_list(crn, u_list: List[np.ndarray], x0_guess=None) -> List[np.ndarray]:
    """Compute steady-state initial conditions for a list of constant inputs.

    Uses fsolve on: rate_function(t=0, x, u) = 0, warm-started from the previous
    solution.

    Args:
        crn: CRN object with `rate_function(t, x, u)`.
        u_list: List of input vectors to solve steady state for.
        x0_guess: Optional initial guess for the first solve (shape (n,)).

    Returns:
        List of steady-state state vectors, one per u in u_list.

    Raises:
        ValueError: If x0_guess has wrong length.
        RuntimeError: If solver returns unexpected size.
    """
    n = int(getattr(crn, "num_species", None) or len(crn.species_labels))
    if x0_guess is None:
        x_prev = np.zeros(n, dtype=np.float64)
    else:
        x_prev = np.asarray(x0_guess, dtype=np.float64).reshape(-1)
        if x_prev.size != n:
            raise ValueError(f"x0_guess has length {x_prev.size} but num_species is {n}")

    out: List[np.ndarray] = []
    for u in u_list:
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        x_ss = fsolve(lambda x: crn.rate_function(0.0, x, u), x_prev)
        x_ss = np.asarray(x_ss, dtype=np.float32).reshape(-1)
        if x_ss.size != n:
            raise RuntimeError(f"fsolve returned length {x_ss.size}, expected {n}")
        out.append(x_ss)
        x_prev = x_ss.astype(np.float64)

    return out


def habituation_metric_multifreq_with_gap(
    *,
    pulse_shapes: Sequence[Tuple[float, float]],
    t_gap: float,
    n_repeats_pre: int,
    n_repeats_post: int,
    n_t: int,
    crn,
    u_nested_builder,
    u_list_local: List[np.ndarray],
    x0_list: List[np.ndarray],
    ratio_weights,
    gap_weight: float,
    recovery_tol: float,
    dishabituate_rho: float,
    min_peak: float,
    max_peak: float,
    freq_weight: float = 1.0,
    LARGE_NUMBER: float = 1e4,
    single_frequency_mode: bool = False,
    sensitization: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate habituation (or sensitization) across multiple pulse frequencies with a gap.

    For each pulse shape (t_on, t_off), this function:
      1) builds an ON/OFF protocol with a long OFF gap,
      2) simulates the CRN,
      3) computes a per-frequency loss via `habituation_metric_with_gap`,
      4) optionally adds a cross-frequency monotonicity penalty based on early-peak slope.

    The returned debug info includes a `freq_runs` payload suitable for plotting.

    Args:
        pulse_shapes: List of (t_on, t_off) pairs.
        t_gap: Duration of the long OFF gap.
        n_repeats_pre: Pulses before the gap.
        n_repeats_post: Pulses after the gap.
        n_t: Total time samples for each simulation.
        crn: CRN object providing `transient_response_piecewise(...)`.
        u_nested_builder: Builder for the piecewise input list (kept for API compatibility).
        u_list_local: List of constant input vectors defining scenarios.
        x0_list: List of initial conditions (usually length 1 in your setup).
        ratio_weights: Weights for ratio-based terms (passed through).
        gap_weight: Weight for the gap-consistency penalty.
        recovery_tol: Relative tolerance for recovery across the gap.
        dishabituate_rho: Optional constraint on post-gap response.
        min_peak: Minimum allowed peak amplitude.
        max_peak: Maximum allowed peak amplitude.
        freq_weight: Weight for the cross-frequency monotonicity penalty.
        LARGE_NUMBER: Penalty value for invalid simulations.
        single_frequency_mode: If True, skip cross-frequency penalty.
        sensitization: If True, flip slope sign to encourage increasing response.

    Returns:
        total_loss: Scalar loss.
        info: Debug dictionary including per-frequency losses and `freq_runs`.
    """
    per_freq_losses: List[float] = []
    slopes: List[float] = []
    periods: List[float] = []
    freq_runs: List[Dict[str, Any]] = []

    eps = 1e-12

    for (t_on, t_off) in pulse_shapes:
        nested_time_horizon = build_on_off_gap_time_horizon(
            t_on=float(t_on),
            t_off=float(t_off),
            t_gap=float(t_gap),
            n_repeats_pre=int(n_repeats_pre),
            n_repeats_post=int(n_repeats_post),
            n_t=int(n_t),
            dtype=np.float32,
        )
        u_nested_list = build_u_nested_list_with_gap(
            u_list=u_list_local,
            n_repeats_pre=int(n_repeats_pre),
            n_repeats_post=int(n_repeats_post),
            off_value=0.0,
        )

        t, x_list, y_list, _ = crn.transient_response_piecewise(
            u_nested_list,
            x0_list,
            nested_time_horizon,
            LARGE_NUMBER=LARGE_NUMBER,
            force=True,
        )
        intervals = [(float(tk[0]), float(tk[-1])) for tk in nested_time_horizon]

        Lf = habituation_metric_with_gap(
            intervals=intervals,
            t=t,
            y_list=y_list,
            w=ratio_weights,
            n_repeats_pre=n_repeats_pre,
            n_repeats_post=n_repeats_post,
            gap_weight=gap_weight,
            recovery_tol=recovery_tol,
            dishabituate_rho=dishabituate_rho,
            min_peak=min_peak,
            max_peak=max_peak,
            LARGE_NUMBER=LARGE_NUMBER,
            sensitization=sensitization,
        )
        per_freq_losses.append(float(Lf))

        if not single_frequency_mode:
            p1_list: List[float] = []
            p2_list: List[float] = []
            for y in y_list:
                peaks_pre, _ = extract_peaks_pre_post_from_piecewise(
                    intervals,
                    t,
                    y,
                    n_repeats_pre,
                    n_repeats_post,
                    LARGE_NUMBER=LARGE_NUMBER,
                )
                if len(peaks_pre) < 2:
                    return float(LARGE_NUMBER), {"reason": "need >=2 pre peaks"}

                p1_list.append(max(float(peaks_pre[0]), float(min_peak)))
                p2_list.append(max(float(peaks_pre[1]), float(min_peak)))

            p1 = float(np.mean(p1_list))
            p2 = float(np.mean(p2_list))

            period = float(t_on + t_off)
            if sensitization:
                slope = (np.log(p1 + eps) - np.log(p2 + eps)) / max(period, eps)
            else:
                slope = (np.log(p2 + eps) - np.log(p1 + eps)) / max(period, eps)

            slopes.append(float(slope))
            periods.append(period)

        # Snapshot for plotting/debug.
        freq_runs.append(
            {
                "pulse_shape": (float(t_on), float(t_off)),
                "time_horizon": np.asarray(t, dtype=float),
                "outputs": y_list,
                "input_intervals": intervals,
                "input_pulse": u_nested_list[0][0],
            }
        )

    # Single-frequency: no cross-frequency penalty.
    if single_frequency_mode or len(pulse_shapes) <= 1:
        total = float(np.mean(per_freq_losses)) if per_freq_losses else float(LARGE_NUMBER)
        return total, {
            "per_freq_losses": per_freq_losses,
            "freq_runs": freq_runs,
            "single_frequency_mode": True,
        }

    # Cross-frequency penalty: higher frequency = smaller period.
    order = np.argsort(np.array(periods, dtype=float))
    slopes_sorted = [slopes[i] for i in order]
    periods_sorted = [periods[i] for i in order]

    # Enforce: slope_highfreq <= slope_lowfreq.
    freq_pen = 0.0
    for i in range(len(slopes_sorted) - 1):
        hi = slopes_sorted[i]
        lo = slopes_sorted[i + 1]
        freq_pen += max(0.0, hi - lo)

    freq_pen /= max(1, (len(slopes_sorted) - 1))
    total = float(np.mean(per_freq_losses) + freq_weight * freq_pen)

    return total, {
        "per_freq_losses": per_freq_losses,
        "periods": periods,
        "slopes": slopes,
        "periods_sorted": periods_sorted,
        "slopes_sorted": slopes_sorted,
        "freq_pen": float(freq_pen),
        "single_frequency_mode": False,
        "freq_runs": freq_runs,
    }


@register_task_kind
class HabituationGapTaskKind(TaskKindBase):
    """Habituation/sensitization task with two pulse trains separated by a gap."""
    kind = "habituation_gap"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "pulse_shapes": "List[(t_on, t_off)] OR a single (t_on, t_off)",
                "gap_time": "float OFF gap duration",
                "n_repeats_pre": "int pulses before gap",
                "n_repeats_post": "int pulses after gap",
                "u_values": "List[float] grid for u",
            },
            "optional": {
                "freq_weight": "float frequency penalty weight (default 1.0)",
                "gap_weight": "float gap penalty weight (default 5.0)",
                "recovery_tol": "float recovery tolerance (default 0.05)",
                "dishabituate_rho": "float dishabituation constraint (default 1.0)",
                "ratio_weights": "float or list (default 1.0)",
                "min_peak": "float (default 0.1)",
                "max_peak": "float (default 2.0)",
                "n_t": "int samples per simulation (default task.n_t)",
                "sensitization": "bool (default False)",
            },
            "notes": (
                "If pulse_shapes has one entry, cross-frequency slope penalties are disabled. "
                "If multiple shapes are provided, a monotonicity penalty encourages faster "
                "habituation at higher frequency."
            ),
        }

    def validate(self, task: TaskSpec) -> None:
        pulse_shapes = overrides_get(task, {}, "pulse_shapes", fallback_attr="pulse_shapes")
        if pulse_shapes is None:
            raise ValueError("habituation_gap requires pulse_shapes.")

        if (
            isinstance(pulse_shapes, (tuple, list))
            and len(pulse_shapes) == 2
            and not isinstance(pulse_shapes[0], (tuple, list))
        ):
            pulse_shapes = [pulse_shapes]

        if not isinstance(pulse_shapes, list) or len(pulse_shapes) < 1:
            raise ValueError("pulse_shapes must be a non-empty list or a single (t_on,t_off).")

        for ps in pulse_shapes:
            if not (isinstance(ps, (tuple, list)) and len(ps) == 2):
                raise ValueError("each pulse_shape must be (t_on, t_off).")
            t_on, t_off = float(ps[0]), float(ps[1])
            if t_on <= 0 or t_off <= 0:
                raise ValueError("all t_on and t_off must be > 0.")

        gap_time = overrides_get(task, {}, "gap_time", fallback_attr="gap_time")
        if gap_time is None or float(gap_time) <= 0:
            raise ValueError("habituation_gap requires gap_time > 0.")

        if int(overrides_get(task, {}, "n_repeats_pre", fallback_attr="n_repeats_pre")) <= 0:
            raise ValueError("n_repeats_pre must be >= 1.")
        if int(overrides_get(task, {}, "n_repeats_post", fallback_attr="n_repeats_post")) <= 0:
            raise ValueError("n_repeats_post must be >= 1.")

        if overrides_get(task, {}, "u_values", fallback_attr="u_values") is None:
            raise ValueError("habituation_gap requires u_values.")

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        u_values = overrides_get(task, {}, "u_values", fallback_attr="u_values")
        if u_values is None:
            raise ValueError("need u_values")
        if task.n_inputs is None:
            raise ValueError("need n_inputs")

        return [
            np.asarray(u, dtype=np.float32)
            for u in product(list(u_values), repeat=int(task.n_inputs))
        ]

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        pulse_shapes = overrides_get(task, overrides, "pulse_shapes", fallback_attr="pulse_shapes")
        if (
            isinstance(pulse_shapes, (tuple, list))
            and len(pulse_shapes) == 2
            and not isinstance(pulse_shapes[0], (tuple, list))
        ):
            pulse_shapes = [pulse_shapes]
        if not pulse_shapes:
            raise ValueError("habituation_gap requires pulse_shapes.")

        single_frequency_mode = (len(pulse_shapes) == 1)

        freq_weight = float(overrides_get(task, overrides, "freq_weight", fallback_attr="freq_weight", default=1.0))
        t_gap = float(overrides_get(task, overrides, "gap_time", fallback_attr="gap_time"))

        n_pre = int(overrides_get(task, overrides, "n_repeats_pre", fallback_attr="n_repeats_pre"))
        n_post = int(overrides_get(task, overrides, "n_repeats_post", fallback_attr="n_repeats_post"))

        n_t = int(overrides_get(task, overrides, "n_t", fallback_attr="n_t", default=task.n_t))

        ratio_weights = overrides_get(task, overrides, "ratio_weights", fallback_attr="ratio_weights", default=1.0)
        min_peak = float(overrides_get(task, overrides, "min_peak", fallback_attr="min_peak", default=0.1))
        max_peak = float(overrides_get(task, overrides, "max_peak", fallback_attr="max_peak", default=2.0))

        gap_weight = float(overrides_get(task, overrides, "gap_weight", fallback_attr="gap_weight", default=5.0))
        recovery_tol = float(overrides_get(task, overrides, "recovery_tol", fallback_attr="recovery_tol", default=0.05))
        dishabituate_rho = float(overrides_get(task, overrides, "dishabituate_rho", fallback_attr="dishabituate_rho", default=1.0))

        sensitization = bool(overrides_get(task, overrides, "sensitization", fallback_attr="sensitization", default=False))

        u_list_local = self.build_u_list(task, overrides)
        ic_obj = "from_ss" if task.ic == "from_ss" else self.build_ic(task, overrides)

        pulse_shapes_f = [(float(ps[0]), float(ps[1])) for ps in pulse_shapes]

        def reward_fn(state: Any):
            if ic_obj == "from_ss":
                u_off = np.zeros_like(u_list_local[0], dtype=np.float32)
                x0_list = steady_state_ic_list(state, [u_off])
            else:
                x0_list = ic_obj.get_ic(state)

            ss_offset = x0_list[0][state.species_idx_dict[state.output_labels[0]]]
            mn = min_peak + ss_offset
            mx = max_peak + ss_offset

            loss, info = habituation_metric_multifreq_with_gap(
                pulse_shapes=pulse_shapes_f,
                t_gap=t_gap,
                n_repeats_pre=n_pre,
                n_repeats_post=n_post,
                n_t=n_t,
                crn=state,
                u_nested_builder=build_u_nested_list_with_gap,
                u_list_local=u_list_local,
                x0_list=x0_list,
                ratio_weights=ratio_weights,
                gap_weight=gap_weight,
                recovery_tol=recovery_tol,
                dishabituate_rho=dishabituate_rho,
                min_peak=mn,
                max_peak=mx,
                freq_weight=freq_weight,
                LARGE_NUMBER=task.LARGE_NUMBER,
                single_frequency_mode=single_frequency_mode,
                sensitization=sensitization,
            )

            state.last_task_info["reward"] = float(loss)
            state.last_task_info["reward type"] = "habituation_gap"
            state.last_task_info["multifreq_info"] = info
            state.last_task_info["single_frequency_mode"] = bool(single_frequency_mode)
            state.last_task_info["pulse_shapes"] = pulse_shapes_f
            state.last_task_info["freq_runs"] = info.get("freq_runs", [])

            # Backward-compatible single-run payload.
            if single_frequency_mode and state.last_task_info["freq_runs"]:
                run0 = state.last_task_info["freq_runs"][0]
                state.last_task_info["input_intervals"] = run0.get("input_intervals")
                state.last_task_info["input_pulse"] = run0.get("input_pulse")
                state.last_task_info["time_horizon"] = run0.get("time_horizon")
                state.last_task_info["outputs"] = run0.get("outputs")

            return float(loss), state.last_task_info

        return reward_fn
