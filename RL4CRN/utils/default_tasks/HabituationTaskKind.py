from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
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


def _as_pulse_shapes(pulse_shapes) -> List[Tuple[float, float]]:
    if (
        isinstance(pulse_shapes, (tuple, list))
        and len(pulse_shapes) == 2
        and not isinstance(pulse_shapes[0], (tuple, list))
    ):
        pulse_shapes = [pulse_shapes]
    if not pulse_shapes:
        raise ValueError("pulse_shapes must contain at least one (t_on, t_off).")
    return [(float(ps[0]), float(ps[1])) for ps in pulse_shapes]


def _absolute_intervals(nested_time_horizon) -> List[Tuple[float, float]]:
    durations = np.asarray([float(t[-1]) for t in nested_time_horizon], dtype=float)
    starts = np.cumsum(np.concatenate([[0.0], durations[:-1]]))
    return [(float(s), float(s + d)) for s, d in zip(starts, durations)]


def _build_typed_protocol(
    *,
    u: np.ndarray,
    t_on: float,
    t_off: float,
    block_repeats: Sequence[int],
    gap_times: Sequence[float],
    n_t: int,
    off_value: float = 0.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    if len(gap_times) != max(0, len(block_repeats) - 1):
        raise ValueError("gap_times must have len(block_repeats) - 1 entries.")

    u = np.asarray(u, dtype=np.float32).reshape(-1)
    u_off = np.full_like(u, float(off_value), dtype=np.float32)

    durations: List[float] = []
    u_steps: List[np.ndarray] = []
    segment_types: List[str] = []
    for b, n_rep in enumerate(block_repeats):
        for _ in range(int(n_rep)):
            durations.extend([float(t_on), float(t_off)])
            u_steps.extend([u, u_off])
            segment_types.extend(["on", "off"])
        if b < len(gap_times):
            durations.append(float(gap_times[b]))
            u_steps.append(u_off)
            segment_types.append("gap")

    if any(d <= 0 for d in durations):
        raise ValueError("All segment durations must be positive.")

    n_segments = len(durations)
    base_pts = max(2, int(np.floor(float(n_t) / max(1, n_segments))))
    total = max(float(sum(durations)), 1e-12)
    nested = [
        np.linspace(0.0, d, max(2, int(base_pts * d / total * n_segments) + 1), dtype=np.float32)
        for d in durations
    ]
    return nested, u_steps, segment_types


def _extract_on_peaks(
    *,
    intervals: Sequence[Tuple[float, float]],
    t: np.ndarray,
    y: np.ndarray,
    segment_types: Sequence[str],
    min_peak: float,
    max_peak: float,
    LARGE_NUMBER: float,
    floor_min_peak: bool = True,
    include_following_off: bool = False,
) -> List[float]:
    y0 = y[0] if y.ndim == 2 else y
    peaks: List[float] = []
    for idx, typ in enumerate(segment_types):
        if typ != "on":
            continue
        start, end = intervals[idx]
        if include_following_off:
            off_idx = idx + 1
            if off_idx < len(segment_types) and segment_types[off_idx] == "off":
                end = intervals[off_idx][1]
        mask = (t >= start) & (t <= end)
        if not np.any(mask):
            return [float(LARGE_NUMBER)]
        peak = float(np.max(y0[mask]))
        if peak > max_peak:
            return [float(LARGE_NUMBER)]
        peaks.append(max(peak, min_peak) if floor_min_peak else peak)
    return peaks


def _extract_pulse_troughs(
    *,
    intervals: Sequence[Tuple[float, float]],
    t: np.ndarray,
    y: np.ndarray,
    segment_types: Sequence[str],
    min_value: float,
    LARGE_NUMBER: float,
) -> List[float]:
    y0 = y[0] if y.ndim == 2 else y
    troughs: List[float] = []
    for idx, typ in enumerate(segment_types):
        if typ != "on":
            continue
        off_idx = idx + 1
        if off_idx >= len(segment_types) or segment_types[off_idx] != "off":
            off_idx = idx
        start, end = intervals[off_idx]
        mask = (t >= start) & (t <= end)
        if not np.any(mask):
            return [float(LARGE_NUMBER)]
        troughs.append(max(float(np.min(y0[mask])), float(min_value)))
    return troughs


def _hinge(x: float, scale: float = 1.0) -> float:
    return max(0.0, float(x)) / max(abs(float(scale)), 1e-12)


def _first_habituation_time(
    peaks: Sequence[float],
    *,
    tolerance: float,
    start_index: int = 0,
    eps: float = 1e-12,
) -> float:
    if len(peaks) < 2:
        return float("inf")
    start = max(0, int(start_index))
    for i in range(start, len(peaks) - 1):
        p_i = max(float(peaks[i]), eps)
        rel_drop = (float(peaks[i]) - float(peaks[i + 1])) / p_i
        if rel_drop >= 0.0 and rel_drop <= float(tolerance):
            return float(i + 1)
    return float("inf")


def _mmc2_habituation_time(
    peaks: Sequence[float],
    *,
    tolerance: float,
    start_index: int = 0,
    eps: float = 1e-12,
) -> float:
    """Return the first pulse index whose next peak changes by at most tolerance."""
    if len(peaks) < 2:
        return float("inf")
    p = np.asarray(peaks, dtype=float)
    if (not np.all(np.isfinite(p))) or np.max(p) <= 0:
        return float("inf")
    start = max(0, int(start_index))
    for i in range(start, len(p) - 1):
        p_i = max(float(p[i]), eps)
        rel_drop = (float(p[i]) - float(p[i + 1])) / p_i
        if rel_drop >= 0.0 and rel_drop <= float(tolerance):
            return float(i + 1)
    return float("inf")


def _relative_error(a: float, b: float, eps: float = 1e-12) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), eps)


def _habituation_log_loss(peaks: Sequence[float], *, s: int, eps: float = 1e-12) -> float:
    if len(peaks) < 2:
        return 1e4
    ratios = []
    for i in range(len(peaks) - 1):
        r = float(peaks[i + 1]) / max(float(peaks[i]), eps)
        ratios.append(max(r, eps) ** int(s))
    return float(np.log(float(np.max(ratios)) + eps))


def _early_peak_separation_loss(
    peaks: Sequence[float],
    *,
    s: int,
    min_change: float,
    count: int,
    eps: float = 1e-12,
) -> float:
    n_pairs = min(max(0, int(count) - 1), max(0, len(peaks) - 1))
    if n_pairs <= 0:
        return 1e4
    terms = []
    for i in range(n_pairs):
        r = float(peaks[i + 1]) / max(float(peaks[i]), eps)
        signed_change = int(s) * (1.0 - r)
        terms.append(max(0.0, float(min_change) - signed_change))
    return float(np.mean(terms))


def _valid_peaks(peaks: Sequence[float], n_required: int, LARGE_NUMBER: float) -> bool:
    if len(peaks) < int(n_required):
        return False
    return not any((not np.isfinite(float(p))) or float(p) >= float(LARGE_NUMBER) for p in peaks)


def _relative_close(a: float, b: float, tol: float, eps: float = 1e-12) -> float:
    rel = abs(float(a) - float(b)) / max(abs(float(b)), eps)
    return max(0.0, rel - float(tol)) / max(float(tol), eps)


def _state_distance(x: np.ndarray, x_ref: np.ndarray, *, floor: float = 1.0) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    denom = max(float(np.linalg.norm(x_ref, ord=2)), float(floor), 1e-12)
    return float(np.linalg.norm(x - x_ref, ord=2) / denom)


def _state_close_penalty(
    x: np.ndarray,
    x_ref: np.ndarray,
    *,
    tol: float,
    floor: float = 1.0,
) -> float:
    rel = _state_distance(x, x_ref, floor=floor)
    return max(0.0, rel - float(tol)) / max(float(tol), 1e-12)


def _solve_off_steady_state(crn, x_guess: np.ndarray, u_off: np.ndarray) -> np.ndarray:
    x_guess = np.asarray(x_guess, dtype=np.float64).reshape(-1)
    u_off = np.asarray(u_off, dtype=np.float64).reshape(-1)
    x_ss = fsolve(lambda x: crn.rate_function(0.0, x, u_off), x_guess)
    return np.asarray(x_ss, dtype=float).reshape(-1)


def _segment_end_state(
    *,
    t: np.ndarray,
    x: np.ndarray,
    intervals: Sequence[Tuple[float, float]],
    segment_idx: int,
    LARGE_NUMBER: float,
) -> np.ndarray:
    end = float(intervals[int(segment_idx)][1])
    candidates = np.flatnonzero(np.asarray(t, dtype=float) <= end + 1e-9)
    if candidates.size == 0:
        return np.full(x.shape[0], LARGE_NUMBER, dtype=float)
    return np.asarray(x[:, int(candidates[-1])], dtype=float)


def _gap_state_terms(
    run: Dict[str, Any],
    x0_list: Sequence[np.ndarray],
    *,
    tol: float,
    floor: float,
    LARGE_NUMBER: float,
) -> Tuple[List[float], List[float]]:
    penalties: List[float] = []
    distances: List[float] = []
    gap_indices = [i for i, typ in enumerate(run.get("segment_types", [])) if typ == "gap"]
    if not gap_indices:
        return penalties, distances

    t = np.asarray(run["time_horizon"], dtype=float)
    intervals = run.get("absolute_intervals", run.get("input_intervals", []))
    for scenario_idx, x in enumerate(run.get("trajectories", [])):
        x_ref = np.asarray(x0_list[scenario_idx % len(x0_list)], dtype=float)
        for gap_idx in gap_indices:
            x_gap = _segment_end_state(
                t=t,
                x=np.asarray(x, dtype=float),
                intervals=intervals,
                segment_idx=gap_idx,
                LARGE_NUMBER=LARGE_NUMBER,
            )
            distances.append(_state_distance(x_gap, x_ref, floor=floor))
            penalties.append(_state_close_penalty(x_gap, x_ref, tol=tol, floor=floor))
    return penalties, distances


def _longest_gap_steady_state_reset_terms(
    *,
    crn,
    run: Dict[str, Any],
    x0_list: Sequence[np.ndarray],
    u_off: np.ndarray,
    tol: float,
    floor: float,
    LARGE_NUMBER: float,
) -> Tuple[List[float], List[float]]:
    penalties: List[float] = []
    distances: List[float] = []
    gap_indices = [i for i, typ in enumerate(run.get("segment_types", [])) if typ == "gap"]
    if not gap_indices:
        return penalties, distances

    gap_idx = gap_indices[-1]
    t = np.asarray(run["time_horizon"], dtype=float)
    intervals = run.get("absolute_intervals", run.get("input_intervals", []))
    for scenario_idx, x in enumerate(run.get("trajectories", [])):
        x_initial_ss = np.asarray(x0_list[scenario_idx % len(x0_list)], dtype=float)
        x_gap = _segment_end_state(
            t=t,
            x=np.asarray(x, dtype=float),
            intervals=intervals,
            segment_idx=gap_idx,
            LARGE_NUMBER=LARGE_NUMBER,
        )
        if (not np.all(np.isfinite(x_gap))) or np.any(np.abs(x_gap) >= float(LARGE_NUMBER)):
            distances.append(float(LARGE_NUMBER))
            penalties.append(float(LARGE_NUMBER))
            continue
        try:
            ss_from_start = _solve_off_steady_state(crn, x_initial_ss, u_off)
            ss_from_gap = _solve_off_steady_state(crn, x_gap, u_off)
        except Exception:
            distances.append(float(LARGE_NUMBER))
            penalties.append(float(LARGE_NUMBER))
            continue
        distances.append(_state_distance(ss_from_gap, ss_from_start, floor=floor))
        penalties.append(_state_close_penalty(ss_from_gap, ss_from_start, tol=tol, floor=floor))
    return penalties, distances


def _simulate_hallmark_protocol(
    *,
    crn,
    u: np.ndarray,
    x0_list: List[np.ndarray],
    t_on: float,
    t_off: float,
    block_repeats: Sequence[int],
    gap_times: Sequence[float],
    n_t: int,
    min_peak: float,
    max_peak: float,
    LARGE_NUMBER: float,
    label: str,
    group: str,
    floor_min_peak: bool = True,
    include_following_off_for_peaks: bool = False,
) -> Tuple[Dict[str, Any], List[List[float]]]:
    nested_time_horizon, u_steps, segment_types = _build_typed_protocol(
        u=u,
        t_on=t_on,
        t_off=t_off,
        block_repeats=block_repeats,
        gap_times=gap_times,
        n_t=n_t,
    )
    t, x_list, y_list, _ = crn.transient_response_piecewise(
        [u_steps],
        x0_list,
        nested_time_horizon,
        LARGE_NUMBER=LARGE_NUMBER,
        force=True,
    )
    intervals = _absolute_intervals(nested_time_horizon)
    peak_lists = [
        _extract_on_peaks(
            intervals=intervals,
            t=t,
            y=y,
            segment_types=segment_types,
            min_peak=min_peak,
            max_peak=max_peak,
            LARGE_NUMBER=LARGE_NUMBER,
            floor_min_peak=floor_min_peak,
            include_following_off=include_following_off_for_peaks,
        )
        for y in y_list
    ]
    trough_lists = [
        _extract_pulse_troughs(
            intervals=intervals,
            t=t,
            y=y,
            segment_types=segment_types,
            min_value=0.0,
            LARGE_NUMBER=LARGE_NUMBER,
        )
        for y in y_list
    ]
    run = {
        "group": group,
        "label": label,
        "pulse_shape": (float(t_on), float(t_off)),
        "block_repeats": [int(n) for n in block_repeats],
        "gap_times": [float(g) for g in gap_times],
        "segment_types": list(segment_types),
        "time_horizon": np.asarray(t, dtype=float),
        "trajectories": x_list,
        "outputs": y_list,
        "absolute_intervals": intervals,
        "input_intervals": [(0.0, float(tk[-1])) for tk in nested_time_horizon],
        "input_pulse": np.asarray(u, dtype=float),
        "peaks": peak_lists,
        "troughs": trough_lists,
    }
    return run, peak_lists


def _mmc2_run_stability_loss(
    run: Dict[str, Any],
    *,
    max_abs_output: float,
    max_abs_state: float,
    LARGE_NUMBER: float,
) -> float:
    arrays: List[Tuple[str, np.ndarray, float]] = []
    arrays.extend(
        ("output", np.asarray(y, dtype=float), float(max_abs_output))
        for y in run.get("outputs", [])
    )
    arrays.extend(
        ("state", np.asarray(x, dtype=float), float(max_abs_state))
        for x in run.get("trajectories", [])
    )
    for _name, arr, bound in arrays:
        if arr.size == 0:
            return float(LARGE_NUMBER)
        if not np.all(np.isfinite(arr)):
            return float(LARGE_NUMBER)
        if np.nanmax(np.abs(arr)) >= float(LARGE_NUMBER):
            return float(LARGE_NUMBER)
        if np.nanmax(np.abs(arr)) > max(float(bound), 1e-12):
            return float(LARGE_NUMBER)
    return 0.0


def _mmc2_validity_loss(
    peaks: Sequence[float],
    troughs: Sequence[float],
    *,
    delta_p_max: float,
    monotone_drop_tol: float,
    i_max: int,
    n_post_min: int,
    delta_min: float,
    delta_t: float,
    trough_thr: float,
    trough_tail: float,
    trough_count_thr: float,
    trough_count_max: int,
    LARGE_NUMBER: float,
    min_peak: float,
    max_peak: float,
    eps: float = 1e-12,
) -> Tuple[float, Dict[str, Any]]:
    p = np.asarray(peaks, dtype=float)
    tr = np.asarray(troughs, dtype=float)
    if p.size < 2 or tr.size < 2:
        return float(LARGE_NUMBER), {"reason": "need at least two peaks and troughs"}
    if (not np.all(np.isfinite(p))) or (not np.all(np.isfinite(tr))):
        return float(LARGE_NUMBER), {"reason": "non-finite peaks or troughs"}
    if np.any(p >= float(LARGE_NUMBER)) or np.any(tr >= float(LARGE_NUMBER)):
        return float(LARGE_NUMBER), {"reason": "large sentinel in peaks or troughs"}

    i_star = int(np.argmax(p))
    p_star = max(float(p[i_star]), eps)
    tail = p[i_star:]
    tail_trough = tr[i_star:min(tr.size, p.size)]

    terms: Dict[str, float] = {}
    max_observed_peak = float(np.max(p))
    terms["valid_response_amplitude"] = (
        0.0 if max_observed_peak >= float(min_peak) else float(LARGE_NUMBER)
    )
    terms["valid_response_upper_bound"] = (
        0.0 if max_observed_peak <= float(max_peak) else float(LARGE_NUMBER)
    )
    if i_star + 1 < p.size:
        first_drop = (float(p[i_star]) - float(p[i_star + 1])) / p_star
        terms["valid_first_drop"] = _hinge(first_drop - float(delta_p_max), delta_p_max)
    else:
        terms["valid_first_drop"] = 1.0

    if tail.size > 1:
        min_relative_drop = max(float(monotone_drop_tol), 0.0)
        monotone_terms = [
            _hinge(
                min_relative_drop - (float(tail[i]) - float(tail[i + 1])) / p_star,
                max(min_relative_drop, eps),
            )
            for i in range(tail.size - 1)
        ]
        terms["valid_monotone_tail"] = float(np.mean(monotone_terms))
    else:
        terms["valid_monotone_tail"] = 1.0

    terms["valid_early_maximum"] = _hinge(i_star - int(i_max), max(1, int(i_max)))
    post_count = int(p.size - i_star - 1)
    terms["valid_post_count"] = _hinge(int(n_post_min) - post_count, max(1, int(n_post_min)))

    if post_count > 0:
        min_peak_ratio = float(np.min(p[i_star + 1:])) / p_star
        max_late_peak_ratio = max(float(delta_min), eps)
        terms["valid_min_peak"] = _hinge(
            min_peak_ratio - max_late_peak_ratio,
            max_late_peak_ratio,
        )
    else:
        terms["valid_min_peak"] = 1.0

    k_sep = min(max(1, int(n_post_min)), tail.size, tail_trough.size)
    if k_sep > 0:
        separations = (tail[:k_sep] - tail_trough[:k_sep]) / p_star
        terms["valid_peak_trough_separation"] = _hinge(float(delta_t) - float(np.min(separations)), delta_t)
        trough_ratios = tail_trough[:k_sep] / p_star
        terms["valid_trough_bound"] = _hinge(float(np.max(trough_ratios)) - float(trough_thr), trough_thr)
    else:
        terms["valid_peak_trough_separation"] = 1.0
        terms["valid_trough_bound"] = 1.0

    terms["valid_tail_trough"] = _hinge(float(tr[-1]) / p_star - float(trough_tail), trough_tail)
    high_trough_count = int(np.sum((tr / p_star) > float(trough_count_thr)))
    terms["valid_high_trough_count"] = _hinge(high_trough_count - int(trough_count_max), max(1, int(trough_count_max)))

    info = {
        "i_star": i_star,
        "p_star": p_star,
        "high_trough_count": high_trough_count,
        "terms": terms,
    }
    return float(np.mean(list(terms.values()))), info


def _simulate_mmc2_train(
    *,
    crn,
    u: np.ndarray,
    x0_list: List[np.ndarray],
    t_on: float,
    t_off: float,
    n_repeats: int,
    n_t: int,
    min_peak: float,
    max_peak: float,
    LARGE_NUMBER: float,
    label: str,
    group: str,
) -> Tuple[Dict[str, Any], List[List[float]], List[List[float]]]:
    run, peak_lists = _simulate_hallmark_protocol(
        crn=crn,
        u=u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        block_repeats=[int(n_repeats)],
        gap_times=[],
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        floor_min_peak=False,
        include_following_off_for_peaks=True,
        label=label,
        group=group,
    )
    return run, peak_lists, run.get("troughs", [])


def _mmc2_recovery_time(
    *,
    crn,
    u: np.ndarray,
    x0_list: List[np.ndarray],
    t_on: float,
    t_off: float,
    train_repeats: int,
    n_t: int,
    min_peak: float,
    max_peak: float,
    LARGE_NUMBER: float,
    recovery_tol: float,
    max_gap: float,
    search_depth: int,
    label: str,
    group: str,
) -> Tuple[float, Dict[str, Any]]:
    hi = max(float(max_gap), 1e-6)
    probe_n_t = max(64, int(np.ceil(float(n_t) / max(1, 2 * int(train_repeats) + 2))) * 2)

    long_run, train_peak_lists = _simulate_hallmark_protocol(
        crn=crn,
        u=u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        block_repeats=[int(train_repeats), 0],
        gap_times=[hi],
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        floor_min_peak=False,
        include_following_off_for_peaks=True,
        label=f"{label} cached train and max gap G={hi:.4g}",
        group=group,
    )
    gap_indices = [i for i, typ in enumerate(long_run.get("segment_types", [])) if typ == "gap"]
    if not gap_indices:
        return hi, {
            "recovered": False,
            "relative_error": float(LARGE_NUMBER),
            "hallmark_run": long_run,
            "optimized_recovery_search": True,
            "reason": "missing cached recovery gap",
        }
    gap_idx = gap_indices[-1]
    gap_start = float(long_run["absolute_intervals"][gap_idx][0])
    t_long = np.asarray(long_run["time_horizon"], dtype=float)
    x_long_list = [np.asarray(x, dtype=float) for x in long_run.get("trajectories", [])]
    reference_peaks = []
    for peaks in train_peak_lists:
        if not _valid_peaks(peaks, int(train_repeats), LARGE_NUMBER):
            return hi, {
                "recovered": False,
                "relative_error": float(LARGE_NUMBER),
                "hallmark_run": long_run,
                "optimized_recovery_search": True,
                "reason": "invalid cached train peaks",
            }
        reference_peaks.append(float(peaks[0]))

    def state_after_gap(gap: float) -> List[np.ndarray]:
        target = gap_start + min(max(float(gap), 0.0), hi)
        states = []
        for x in x_long_list:
            state = np.asarray(
                [np.interp(target, t_long, x_i) for x_i in x],
                dtype=float,
            )
            states.append(state)
        return states

    def simulate_probe(gap: float) -> Tuple[bool, float]:
        probe_states = state_after_gap(gap)
        _probe_run, probe_peak_lists = _simulate_hallmark_protocol(
            crn=crn,
            u=u,
            x0_list=probe_states,
            t_on=t_on,
            t_off=t_off,
            block_repeats=[1],
            gap_times=[],
            n_t=probe_n_t,
            min_peak=min_peak,
            max_peak=max_peak,
            LARGE_NUMBER=LARGE_NUMBER,
            floor_min_peak=False,
            include_following_off_for_peaks=True,
            label=f"{label} probe G={float(gap):.4g}",
            group=group,
        )
        rels = []
        for peaks, first_peak in zip(probe_peak_lists, reference_peaks):
            if not _valid_peaks(peaks, 1, LARGE_NUMBER):
                return False, float(LARGE_NUMBER)
            rels.append(_relative_error(peaks[0], first_peak))
        rel = float(np.mean(rels)) if rels else float(LARGE_NUMBER)
        return rel <= float(recovery_tol), rel

    def simulate_full_gap(gap: float) -> Dict[str, Any]:
        run, peak_lists = _simulate_hallmark_protocol(
            crn=crn,
            u=u,
            x0_list=x0_list,
            t_on=t_on,
            t_off=t_off,
            block_repeats=[int(train_repeats), 1],
            gap_times=[max(float(gap), 1e-6)],
            n_t=n_t,
            min_peak=min_peak,
            max_peak=max_peak,
            LARGE_NUMBER=LARGE_NUMBER,
            floor_min_peak=False,
            include_following_off_for_peaks=True,
            label=f"{label} G={float(gap):.4g}",
            group=group,
        )
        return run

    recovered, hi_rel = simulate_probe(hi)
    if not recovered:
        return hi, {
            "recovered": False,
            "relative_error": hi_rel,
            "hallmark_run": simulate_full_gap(hi),
            "optimized_recovery_search": True,
            "probe_evaluations": 1,
        }

    lo = 0.0
    best_rel = hi_rel
    evaluations = 1
    for _ in range(max(0, int(search_depth))):
        mid = 0.5 * (lo + hi)
        recovered_mid, rel_mid = simulate_probe(mid)
        evaluations += 1
        if recovered_mid:
            hi = mid
            best_rel = rel_mid
        else:
            lo = mid
    return hi, {
        "recovered": True,
        "relative_error": best_rel,
        "hallmark_run": simulate_full_gap(hi),
        "optimized_recovery_search": True,
        "probe_evaluations": evaluations,
    }

def habituation_hallmarks_metric(
    *,
    crn,
    u_list_local: List[np.ndarray],
    x0_list: List[np.ndarray],
    pulse_shapes: Sequence[Tuple[float, float]],
    n_repeats_pre: int,
    n_repeats_post: int,
    gap_time: float,
    n_t: int,
    s: int,
    min_peak: float,
    max_peak: float,
    LARGE_NUMBER: float,
    freq_weight: float = 1.0,
    habituation_weight: float = 1.0,
    early_peak_sep_weight: float = 1.0,
    early_peak_min_change: float = 0.1,
    early_peak_count: int = 3,
    gap_weight: float = 5.0,
    recovery_tol: float = 0.05,
    potentiation_blocks: int = 3,
    potentiation_gap_time: float = 100.0,
    potentiation_weight: float = 1.0,
    intensity_values: Sequence[float] = (1.0, 0.66, 0.33),
    intensity_weight: float = 1.0,
    long_train_repeats: int = 30,
    short_gap_time: float = 20.0,
    subliminal_weight: float = 1.0,
    asymptote_window: int = 3,
    long_gap_times: Sequence[float] = (100.0, 300.0, 1000.0),
    long_term_weight: float = 1.0,
    long_term_min_memory: float = 0.05,
    memory_difference_weight: float = 1.0,
    state_reset_weight: float = 1.0,
    state_reset_tol: float = 0.2,
    state_reset_floor: float = 1.0,
    peak_tol: float = 0.1,
    margin: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    eps = 1e-12
    runs: List[Dict[str, Any]] = []
    component_losses: Dict[str, float] = {}
    early_sep_terms: List[float] = []
    base_u = np.asarray(u_list_local[0], dtype=np.float32).reshape(-1)
    pulse_shapes_f = _as_pulse_shapes(pulse_shapes)
    default_shape = pulse_shapes_f[len(pulse_shapes_f) // 2]
    state_reset_terms: List[float] = []

    # Hallmarks 1/2: habituation followed by spontaneous recovery at the default period.
    t_on, t_off = default_shape
    run, peak_lists = _simulate_hallmark_protocol(
        crn=crn,
        u=base_u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        block_repeats=[n_repeats_pre, n_repeats_post],
        gap_times=[gap_time],
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        label=f"recovery T={t_on + t_off:g}",
        group="1/2 habituation and recovery",
    )
    runs.append(run)
    base_terms = []
    for peaks in peak_lists:
        if not _valid_peaks(peaks, n_repeats_pre + n_repeats_post, LARGE_NUMBER):
            return float(LARGE_NUMBER), {"reason": "invalid peak", "hallmark_runs": runs}
        pre = peaks[:n_repeats_pre]
        post = peaks[n_repeats_pre:]
        h = _habituation_log_loss(pre, s=s)
        early_sep_terms.append(_early_peak_separation_loss(
            pre,
            s=s,
            min_change=early_peak_min_change,
            count=early_peak_count,
        ))
        rec = _relative_close(post[0], pre[0], recovery_tol) if post else LARGE_NUMBER
        base_terms.append(habituation_weight * h + gap_weight * rec)
    component_losses["habituation_recovery"] = float(np.mean(base_terms))

    # Hallmark 4: frequency sensitivity from the same IC and with no gap/memory test.
    slopes = []
    periods = []
    first_peaks = []
    for t_on, t_off in pulse_shapes_f:
        run, peak_lists = _simulate_hallmark_protocol(
            crn=crn,
            u=base_u,
            x0_list=x0_list,
            t_on=t_on,
            t_off=t_off,
            block_repeats=[n_repeats_pre],
            gap_times=[],
            n_t=n_t,
            min_peak=min_peak,
            max_peak=max_peak,
            LARGE_NUMBER=LARGE_NUMBER,
            label=f"frequency T={t_on + t_off:g}",
            group="4 frequency",
        )
        runs.append(run)
        period_scores = []
        period_first = []
        for peaks in peak_lists:
            if not _valid_peaks(peaks, n_repeats_pre, LARGE_NUMBER):
                return float(LARGE_NUMBER), {"reason": "invalid frequency peak", "hallmark_runs": runs}
            h = _habituation_log_loss(peaks, s=s)
            early_sep_terms.append(_early_peak_separation_loss(
                peaks,
                s=s,
                min_change=early_peak_min_change,
                count=early_peak_count,
            ))
            period_scores.append(h)
            period_first.append(peaks[0])
        first_peaks.append(float(np.mean(period_first)))
        slopes.append(float(np.mean(period_scores)) / max(float(t_on + t_off), eps))
        periods.append(float(t_on + t_off))

    freq_pen = 0.0
    if len(slopes) > 1:
        order = np.argsort(np.asarray(periods, dtype=float))
        sorted_slopes = [slopes[i] for i in order]
        sorted_first = [first_peaks[i] for i in order]
        for i in range(len(sorted_slopes) - 1):
            freq_pen += max(0.0, sorted_slopes[i] - sorted_slopes[i + 1] + margin)
            freq_pen += peak_tol * _relative_close(sorted_first[i], sorted_first[i + 1], peak_tol)
        freq_pen /= max(1, len(sorted_slopes) - 1)
    component_losses["frequency"] = float(freq_weight * freq_pen)

    # Hallmark 3: potentiation across repeated trains, while first peaks remain comparable.
    run, peak_lists = _simulate_hallmark_protocol(
        crn=crn,
        u=base_u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        block_repeats=[n_repeats_pre] * int(potentiation_blocks),
        gap_times=[potentiation_gap_time] * (int(potentiation_blocks) - 1),
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        label="potentiation repeated trains",
        group="3 potentiation",
    )
    runs.append(run)
    pot_terms = []
    for peaks in peak_lists:
        if not _valid_peaks(peaks, n_repeats_pre * int(potentiation_blocks), LARGE_NUMBER):
            return float(LARGE_NUMBER), {"reason": "invalid potentiation peak", "hallmark_runs": runs}
        block_scores = []
        block_first = []
        for b in range(int(potentiation_blocks)):
            block = peaks[b * n_repeats_pre:(b + 1) * n_repeats_pre]
            block_scores.append(habituation_weight * _habituation_log_loss(block, s=s))
            early_sep_terms.append(_early_peak_separation_loss(
                block,
                s=s,
                min_change=early_peak_min_change,
                count=early_peak_count,
            ))
            block_first.append(block[0])
        pen = 0.0
        for b in range(len(block_scores) - 1):
            pen += max(0.0, block_scores[b + 1] - block_scores[b] + margin)
            pen += peak_tol * _relative_close(block_first[b + 1], block_first[0], peak_tol)
        pot_terms.append(pen / max(1, len(block_scores) - 1))
    component_losses["potentiation"] = float(potentiation_weight * np.mean(pot_terms))

    # Hallmark 5: lower intensities should habituate more, with peak amplitude scaling preserved.
    intensity_scores = []
    intensity_first = []
    for a in intensity_values:
        run, peak_lists = _simulate_hallmark_protocol(
            crn=crn,
            u=base_u * float(a),
            x0_list=x0_list,
            t_on=t_on,
            t_off=t_off,
            block_repeats=[n_repeats_pre],
            gap_times=[],
            n_t=n_t,
            min_peak=min_peak,
            max_peak=max_peak,
            LARGE_NUMBER=LARGE_NUMBER,
            label=f"intensity {float(a):.2g}",
            group="5 intensity",
        )
        runs.append(run)
        for peaks in peak_lists:
            if not _valid_peaks(peaks, n_repeats_pre, LARGE_NUMBER):
                return float(LARGE_NUMBER), {"reason": "invalid intensity peak", "hallmark_runs": runs}
        scores = [habituation_weight * _habituation_log_loss(peaks, s=s) for peaks in peak_lists]
        early_sep_terms.extend([
            _early_peak_separation_loss(
                peaks,
                s=s,
                min_change=early_peak_min_change,
                count=early_peak_count,
            )
            for peaks in peak_lists
        ])
        intensity_scores.append(float(np.mean(scores)))
        intensity_first.append(float(np.mean([peaks[0] for peaks in peak_lists])))
    int_pen = 0.0
    order = np.argsort(np.asarray(intensity_values, dtype=float))[::-1]
    for left, right in zip(order[:-1], order[1:]):
        int_pen += max(0.0, intensity_scores[right] - intensity_scores[left] + margin)
    high = max(intensity_first[order[0]], eps)
    for idx in order[1:]:
        expected = float(intensity_values[idx]) / max(float(intensity_values[order[0]]), eps)
        observed = intensity_first[idx] / high
        int_pen += peak_tol * abs(observed - expected)
    component_losses["intensity"] = float(intensity_weight * int_pen / max(1, len(order) - 1))

    # Hallmark 6: extra stimulation after asymptote delays recovery after a short gap.
    normal_run, normal_peaks = _simulate_hallmark_protocol(
        crn=crn,
        u=base_u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        block_repeats=[n_repeats_pre, n_repeats_post],
        gap_times=[short_gap_time],
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        label="normal train, short gap",
        group="6 subliminal accumulation",
    )
    long_run, long_peaks = _simulate_hallmark_protocol(
        crn=crn,
        u=base_u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        block_repeats=[long_train_repeats, n_repeats_post],
        gap_times=[short_gap_time],
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        label="extended train, short gap",
        group="6 subliminal accumulation",
    )
    runs.extend([normal_run, long_run])
    sub_terms = []
    for pn, pl in zip(normal_peaks, long_peaks):
        if not _valid_peaks(pn, n_repeats_pre + n_repeats_post, LARGE_NUMBER):
            return float(LARGE_NUMBER), {"reason": "invalid normal subliminal peak", "hallmark_runs": runs}
        if not _valid_peaks(pl, long_train_repeats + n_repeats_post, LARGE_NUMBER):
            return float(LARGE_NUMBER), {"reason": "invalid long subliminal peak", "hallmark_runs": runs}
        n_post_first = pn[n_repeats_pre]
        l_post_first = pl[long_train_repeats]
        asym = abs(pl[long_train_repeats - 1] - pl[max(0, long_train_repeats - asymptote_window)]) / max(pl[0], eps)
        delay = max(0.0, (l_post_first - n_post_first) * s / max(pl[0], eps) + margin)
        sub_terms.append(asym + delay)
    component_losses["subliminal"] = float(subliminal_weight * np.mean(sub_terms))

    # Hallmark 10: response memory remains after long gaps, while the state reset
    # check is anchored on the longest gap to avoid rewarding premature reset.
    memories = []
    state_memories = []
    longest_gap_run = None
    longest_gap_state_distances: List[float] = []
    largest_gap = max(float(g) for g in long_gap_times) if long_gap_times else 0.0
    for g in long_gap_times:
        run, peak_lists = _simulate_hallmark_protocol(
            crn=crn,
            u=base_u,
            x0_list=x0_list,
            t_on=t_on,
            t_off=t_off,
            block_repeats=[n_repeats_pre, n_repeats_post],
            gap_times=[float(g)],
            n_t=n_t,
            min_peak=min_peak,
            max_peak=max_peak,
            LARGE_NUMBER=LARGE_NUMBER,
            label=f"long gap {float(g):g}",
            group="10 long-term",
        )
        runs.append(run)
        if float(g) == largest_gap:
            longest_gap_run = run
        _, gap_distances = _gap_state_terms(
            run,
            x0_list,
            tol=state_reset_tol,
            floor=state_reset_floor,
            LARGE_NUMBER=LARGE_NUMBER,
        )
        if float(g) == largest_gap:
            longest_gap_state_distances = gap_distances
        state_memories.append(float(np.mean(gap_distances)) if gap_distances else float(LARGE_NUMBER))
        vals = []
        for peaks in peak_lists:
            if not _valid_peaks(peaks, n_repeats_pre + n_repeats_post, LARGE_NUMBER):
                return float(LARGE_NUMBER), {"reason": "invalid long-term peak", "hallmark_runs": runs}
            pre0 = peaks[0]
            post0 = peaks[n_repeats_pre]
            vals.append(s * (1.0 - post0 / max(pre0, eps)))
        memories.append(float(np.mean(vals)))
    if longest_gap_run is not None:
        u_off = np.zeros_like(base_u, dtype=np.float32)
        longest_reset_terms, ss_reset_distances = _longest_gap_steady_state_reset_terms(
            crn=crn,
            run=longest_gap_run,
            x0_list=x0_list,
            u_off=u_off,
            tol=state_reset_tol,
            floor=state_reset_floor,
            LARGE_NUMBER=LARGE_NUMBER,
        )
        state_reset_terms.extend(longest_reset_terms)
    else:
        ss_reset_distances = []

    memory_differences = [
        abs(float(m) - float(sm)) for m, sm in zip(memories, state_memories)
    ]
    lt_pen = 0.0
    for i in range(len(memories) - 1):
        lt_pen += max(0.0, memories[i + 1] - memories[i] + margin)
    diff_pen = 0.0
    for i in range(len(memory_differences) - 1):
        diff_pen += max(0.0, memory_differences[i + 1] - memory_differences[i] + margin)
    if memory_differences:
        diff_pen += max(0.0, float(long_term_min_memory) - memory_differences[-1])
    lt_pen += max(0.0, float(long_term_min_memory) - memories[-1])
    component_losses["long_term"] = float(long_term_weight * (lt_pen + memory_difference_weight * diff_pen))
    component_losses["state_reset"] = (
        float(state_reset_weight * np.mean(state_reset_terms))
        if state_reset_terms
        else 0.0
    )
    component_losses["early_peak_separation"] = (
        float(early_peak_sep_weight * np.mean(early_sep_terms))
        if early_sep_terms
        else 0.0
    )

    total = (
        component_losses["habituation_recovery"]
        + component_losses["frequency"]
        + component_losses["potentiation"]
        + component_losses["intensity"]
        + component_losses["subliminal"]
        + component_losses["long_term"]
        + component_losses["state_reset"]
        + component_losses["early_peak_separation"]
    )
    return float(total), {
        "component_losses": component_losses,
        "hallmark_runs": runs,
        "s": int(s),
        "pulse_shapes": pulse_shapes_f,
        "long_gap_times": [float(g) for g in long_gap_times],
        "intensity_values": [float(a) for a in intensity_values],
        "memory": memories,
        "state_memory": state_memories,
        "memory_difference": memory_differences,
        "longest_gap_state_distance": longest_gap_state_distances,
        "longest_gap_ss_distance": ss_reset_distances,
        "frequency_periods": periods,
        "frequency_slopes": slopes,
        "frequency_first_peaks": first_peaks,
        "intensity_scores": intensity_scores,
        "intensity_first_peaks": intensity_first,
    }


def habituation_hallmarks_mmc2_metric(
    *,
    crn,
    u_list_local: List[np.ndarray],
    x0_list: List[np.ndarray],
    pulse_shapes: Sequence[Tuple[float, float]],
    n_repeats_pre: int,
    n_repeats_post: int,
    gap_time: float,
    n_t: int,
    min_peak: float,
    max_peak: float,
    LARGE_NUMBER: float,
    eps_h: float = 0.01,
    eps_subliminal: float = 0.005,
    recovery_tol: float = 0.05,
    rt_search_depth: int = 8,
    rt_max_gap: float = 200.0,
    potentiation_gap_fraction: float = 0.5,
    intensity_values: Sequence[float] = (1.0, 0.5, 0.25),
    delta_p_max: float = 0.5,
    monotone_drop_tol: float = 1e-4,
    i_max: int = 2,
    n_post_min: int = 2,
    delta_min: float = 0.2,
    delta_t: float = 0.05,
    trough_thr: float = 0.6,
    trough_tail: float = 0.02,
    trough_count_thr: float = 0.10,
    trough_count_max: int = 5,
    validity_weight: float = 1.0,
    decrement_weight: float = 25.0,
    monotone_weight: float = 10.0,
    hard_shape_validity: bool = True,
    habituation_weight: float = 1.0,
    recovery_weight: float = 1.0,
    potentiation_weight: float = 1.0,
    frequency_weight: float = 1.0,
    intensity_weight: float = 1.0,
    subliminal_weight: float = 1.0,
    ordering_margin: float = 0.0,
    reference_pulse_shape_index: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    eps = 1e-12
    base_u = np.asarray(u_list_local[0], dtype=np.float32).reshape(-1)
    pulse_shapes_f = _as_pulse_shapes(pulse_shapes)
    if reference_pulse_shape_index is None:
        reference_idx = len(pulse_shapes_f) // 2
    else:
        reference_idx = int(np.clip(int(reference_pulse_shape_index), 0, len(pulse_shapes_f) - 1))
    default_shape = pulse_shapes_f[reference_idx]
    runs: List[Dict[str, Any]] = []
    component_losses: Dict[str, float] = {}
    validity_terms: List[float] = []
    validity_term_values: Dict[str, List[float]] = {}
    stability_terms: List[float] = []
    strict_failures: List[str] = []

    def add_stability(run: Dict[str, Any]) -> None:
        stability_terms.append(
            _mmc2_run_stability_loss(
                run,
                max_abs_output=max_peak,
                max_abs_state=max(10.0 * max_peak, max_peak),
                LARGE_NUMBER=LARGE_NUMBER,
            )
        )

    def add_validity(peaks: Sequence[float], troughs: Sequence[float]) -> Dict[str, Any]:
        loss, detail = _mmc2_validity_loss(
            peaks,
            troughs,
            delta_p_max=delta_p_max,
            monotone_drop_tol=monotone_drop_tol,
            i_max=i_max,
            n_post_min=n_post_min,
            delta_min=delta_min,
            delta_t=delta_t,
            trough_thr=trough_thr,
            trough_tail=trough_tail,
            trough_count_thr=trough_count_thr,
            trough_count_max=trough_count_max,
            LARGE_NUMBER=LARGE_NUMBER,
            min_peak=min_peak,
            max_peak=max_peak,
        )
        validity_terms.append(float(loss))
        for name, value in detail.get("terms", {}).items():
            name_s = str(name)
            value_f = float(value)
            validity_term_values.setdefault(name_s, []).append(value_f)
            if (
                bool(hard_shape_validity)
                and name_s in {"valid_min_peak", "valid_monotone_tail"}
                and value_f > 1e-9
            ):
                failure_name = f"{name_s}_failed"
                if failure_name not in strict_failures:
                    strict_failures.append(failure_name)
        return detail

    t_on, t_off = default_shape
    base_run, base_peaks, base_troughs = _simulate_mmc2_train(
        crn=crn,
        u=base_u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        n_repeats=n_repeats_pre,
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        label=f"reference train T={t_on + t_off:g}",
        group="MMC2 1 habituation",
    )
    runs.append(base_run)
    add_stability(base_run)
    ht_values: List[float] = []
    validity_details = []
    for peaks, troughs in zip(base_peaks, base_troughs):
        if not _valid_peaks(peaks, n_repeats_pre, LARGE_NUMBER):
            component_losses["habituation_time"] = float(LARGE_NUMBER)
            component_losses["strict_feasibility"] = float(LARGE_NUMBER)
            return float(LARGE_NUMBER), {
                "reason": "invalid reference peaks",
                "component_losses": component_losses,
                "hallmark_runs": runs,
                "strict_failures": ["invalid_reference_peaks"],
            }
        detail = add_validity(peaks, troughs)
        validity_details.append(detail)
        ht = _mmc2_habituation_time(
            peaks,
            tolerance=eps_h,
            start_index=detail.get("i_star", 0),
        )
        ht_values.append(ht)
    finite_ht = [h for h in ht_values if np.isfinite(h)]
    if not finite_ht:
        component_losses["habituation_time"] = float(LARGE_NUMBER)
        strict_failures.append("reference_habituation_time_missing")
        train_for_rt = int(n_repeats_pre)
    else:
        component_losses["habituation_time"] = float(habituation_weight * np.mean([
            _hinge(h - n_repeats_pre, max(1, n_repeats_pre)) for h in ht_values
        ]))
        train_for_rt = max(2, int(np.ceil(float(np.mean(finite_ht)))) + 1)

    rt, rt_info = _mmc2_recovery_time(
        crn=crn,
        u=base_u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        train_repeats=min(train_for_rt, int(n_repeats_pre)),
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        recovery_tol=recovery_tol,
        max_gap=rt_max_gap,
        search_depth=rt_search_depth,
        label="recovery search",
        group="MMC2 2 spontaneous recovery",
    )
    runs.append(rt_info["hallmark_run"])
    add_stability(rt_info["hallmark_run"])
    component_losses["recovery_time"] = (
        0.0 if rt_info.get("recovered", False) else float(recovery_weight * _hinge(rt_info["relative_error"] - recovery_tol, recovery_tol))
    )

    pot_gap = max(float(rt) * float(potentiation_gap_fraction), 1e-6)
    pot_run, pot_peaks = _simulate_hallmark_protocol(
        crn=crn,
        u=base_u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        block_repeats=[n_repeats_pre, n_repeats_pre],
        gap_times=[pot_gap],
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        floor_min_peak=False,
        include_following_off_for_peaks=True,
        label=f"two trains G={pot_gap:.4g}",
        group="MMC2 3 potentiation",
    )
    runs.append(pot_run)
    add_stability(pot_run)
    pot_terms = []
    for peaks in pot_peaks:
        if not _valid_peaks(peaks, 2 * n_repeats_pre, LARGE_NUMBER):
            pot_terms.append(float(LARGE_NUMBER))
            continue
        h1 = _mmc2_habituation_time(peaks[:n_repeats_pre], tolerance=eps_h)
        h2 = _mmc2_habituation_time(peaks[n_repeats_pre:], tolerance=eps_h)
        if not np.isfinite(h1) or not np.isfinite(h2):
            strict_failures.append("potentiation_habituation_time_missing")
            pot_terms.append(float(LARGE_NUMBER))
        else:
            pot_terms.append(_hinge(h2 / max(h1, eps) - 1.0, 1.0))
    component_losses["potentiation"] = float(potentiation_weight * np.mean(pot_terms))

    freq_ht: List[float] = []
    freq_periods: List[float] = []
    freq_all_finite = True
    for t_on_f, t_off_f in pulse_shapes_f:
        run, peak_lists, trough_lists = _simulate_mmc2_train(
            crn=crn,
            u=base_u,
            x0_list=x0_list,
            t_on=t_on_f,
            t_off=t_off_f,
            n_repeats=n_repeats_pre,
            n_t=n_t,
            min_peak=min_peak,
            max_peak=max_peak,
            LARGE_NUMBER=LARGE_NUMBER,
            label=f"frequency T={t_on_f + t_off_f:g}",
            group="MMC2 4 frequency sensitivity",
        )
        runs.append(run)
        add_stability(run)
        period_h = []
        for peaks, troughs in zip(peak_lists, trough_lists):
            detail = add_validity(peaks, troughs)
            period_h.append(
                _mmc2_habituation_time(
                    peaks,
                    tolerance=eps_h,
                    start_index=detail.get("i_star", 0),
                )
            )
        if not all(np.isfinite(h) for h in period_h):
            freq_all_finite = False
            strict_failures.append(f"frequency_habituation_time_missing_T_{t_on_f + t_off_f:g}")
        freq_ht.append(float(np.mean([h if np.isfinite(h) else n_repeats_pre + 1 for h in period_h])))
        freq_periods.append(float(t_on_f + t_off_f))
    freq_pen = float(LARGE_NUMBER) if not freq_all_finite else 0.0
    if freq_all_finite and len(freq_ht) > 1:
        order = np.argsort(np.asarray(freq_periods, dtype=float))
        ht_sorted = [freq_ht[i] for i in order]
        for a, b in zip(ht_sorted[:-1], ht_sorted[1:]):
            freq_pen += _hinge(a / max(b, eps) - 1.0 + float(ordering_margin), 1.0)
        freq_pen /= max(1, len(ht_sorted) - 1)
    component_losses["frequency_sensitivity"] = float(frequency_weight * freq_pen)

    intensity_ht: List[float] = []
    intensity_first_peaks: List[float] = []
    intensity_all_finite = True
    intensity_values_f = [float(a) for a in intensity_values]
    for amp in intensity_values_f:
        run, peak_lists, trough_lists = _simulate_mmc2_train(
            crn=crn,
            u=base_u * amp,
            x0_list=x0_list,
            t_on=t_on,
            t_off=t_off,
            n_repeats=n_repeats_pre,
            n_t=n_t,
            min_peak=min_peak,
            max_peak=max_peak,
            LARGE_NUMBER=LARGE_NUMBER,
            label=f"intensity A={amp:g}",
            group="MMC2 5 intensity sensitivity",
        )
        runs.append(run)
        add_stability(run)
        amp_h = []
        amp_first = []
        for peaks, troughs in zip(peak_lists, trough_lists):
            detail = add_validity(peaks, troughs)
            amp_h.append(
                _mmc2_habituation_time(
                    peaks,
                    tolerance=eps_h,
                    start_index=detail.get("i_star", 0),
                )
            )
            amp_first.append(float(peaks[0]) if peaks else 0.0)
        if not all(np.isfinite(h) for h in amp_h):
            intensity_all_finite = False
            strict_failures.append(f"intensity_habituation_time_missing_A_{amp:g}")
        intensity_ht.append(float(np.mean([h if np.isfinite(h) else n_repeats_pre + 1 for h in amp_h])))
        intensity_first_peaks.append(float(np.mean(amp_first)))
    int_pen = float(LARGE_NUMBER) if not intensity_all_finite else 0.0
    if intensity_all_finite and len(intensity_ht) > 1:
        order = np.argsort(np.asarray(intensity_values_f, dtype=float))
        ht_sorted = [intensity_ht[i] for i in order]
        for a, b in zip(ht_sorted[:-1], ht_sorted[1:]):
            int_pen += _hinge(a / max(b, eps) - 1.0 + float(ordering_margin), 1.0)
        int_pen /= max(1, len(ht_sorted) - 1)
    component_losses["intensity_sensitivity"] = float(intensity_weight * int_pen)

    strict_ht = []
    for peaks in base_peaks:
        strict_ht.append(_mmc2_habituation_time(peaks, tolerance=eps_subliminal))
    strict_ht_all_finite = all(np.isfinite(h) for h in strict_ht)
    if not strict_ht_all_finite:
        strict_failures.append("strict_habituation_time_missing")
    strict_train = max(2, int(np.ceil(np.nanmean([
        h if np.isfinite(h) else n_repeats_pre for h in strict_ht
    ]))) + 1)
    rt_strict, rt_strict_info = _mmc2_recovery_time(
        crn=crn,
        u=base_u,
        x0_list=x0_list,
        t_on=t_on,
        t_off=t_off,
        train_repeats=min(strict_train, int(n_repeats_pre)),
        n_t=n_t,
        min_peak=min_peak,
        max_peak=max_peak,
        LARGE_NUMBER=LARGE_NUMBER,
        recovery_tol=recovery_tol,
        max_gap=rt_max_gap,
        search_depth=rt_search_depth,
        label="strict recovery search",
        group="MMC2 6 subliminal accumulation",
    )
    runs.append(rt_strict_info["hallmark_run"])
    add_stability(rt_strict_info["hallmark_run"])
    if not strict_ht_all_finite:
        sub_loss = float(LARGE_NUMBER)
    elif not rt_strict_info.get("recovered", False):
        strict_failures.append("strict_recovery_not_found")
        sub_loss = float(LARGE_NUMBER)
    else:
        sub_loss = _hinge(rt / max(rt_strict, eps) - 1.0, 1.0)
    component_losses["subliminal_accumulation"] = float(subliminal_weight * sub_loss)

    if validity_term_values:
        for name, values in sorted(validity_term_values.items()):
            term_weight = 1.0
            if name == "valid_min_peak":
                term_weight = float(decrement_weight)
            elif name == "valid_monotone_tail":
                term_weight = float(monotone_weight)
            component_losses[name] = float(validity_weight * term_weight * np.mean(values))
    else:
        component_losses["validity"] = (
            float(validity_weight * np.mean(validity_terms)) if validity_terms else float(LARGE_NUMBER)
        )
    component_losses["trajectory_stability"] = (
        float(np.max(stability_terms)) if stability_terms else 0.0
    )
    if component_losses["trajectory_stability"] > 0:
        strict_failures.append("trajectory_unstable")
    component_losses["strict_feasibility"] = float(LARGE_NUMBER) if strict_failures else 0.0
    total = float(sum(component_losses.values()))
    return total, {
        "component_losses": component_losses,
        "hallmark_runs": runs,
        "pulse_shapes": pulse_shapes_f,
        "intensity_values": intensity_values_f,
        "frequency_periods": freq_periods,
        "frequency_slopes": freq_ht,
        "intensity_scores": intensity_ht,
        "intensity_first_peaks": intensity_first_peaks,
        "reference_pulse_shape_index": int(reference_idx),
        "reference_pulse_shape": default_shape,
        "rt": float(rt),
        "rt_strict": float(rt_strict),
        "rt_recovered": bool(rt_info.get("recovered", False)),
        "rt_strict_recovered": bool(rt_strict_info.get("recovered", False)),
        "validity_details": validity_details,
        "strict_failures": strict_failures,
        "constants": {
            "eps_h": float(eps_h),
            "eps_subliminal": float(eps_subliminal),
            "recovery_tol": float(recovery_tol),
            "delta_p_max": float(delta_p_max),
            "monotone_drop_tol": float(monotone_drop_tol),
            "i_max": int(i_max),
            "n_post_min": int(n_post_min),
            "delta_min": float(delta_min),
            "delta_t": float(delta_t),
            "trough_thr": float(trough_thr),
            "trough_tail": float(trough_tail),
            "trough_count_thr": float(trough_count_thr),
            "trough_count_max": int(trough_count_max),
            "decrement_weight": float(decrement_weight),
            "monotone_weight": float(monotone_weight),
            "hard_shape_validity": bool(hard_shape_validity),
        },
    }


@register_task_kind
class HabituationHallmarksMMC2TaskKind(TaskKindBase):
    """MMC2-style habituation task with hinge penalties and binary-search recovery time."""
    kind = "habituation_hallmarks_mmc2"

    @staticmethod
    def help() -> Dict[str, Any]:
        mmc2_thresholds = {
            "eps_h": "habituation tolerance, default 0.01",
            "eps_subliminal": "stricter habituation tolerance, default 0.005",
            "recovery_tol": "recovery response tolerance, default 0.05",
            "delta_p_max": "Delta P_max first post-maximum drop bound, default 0.5",
            "monotone_drop_tol": "minimum relative drop required between consecutive post-maximum peaks, default 1e-4",
            "i_max": "largest peak must occur within first i_max+1 pulses, default 2",
            "n_post_min": "minimum post-maximum pulses, default 2",
            "delta_min": "maximum allowed late-peak ratio after the maximum peak, default 0.2 means min late peak <= 0.2 * max peak",
            "delta_t": "minimum peak-trough separation, default 0.05",
            "trough_thr": "T_thr trough ratio bound, default 0.6",
            "trough_tail": "T_tail final trough ratio bound, default 0.02",
            "trough_count_thr": "T_thr^c high-trough threshold, default 0.10",
            "trough_count_max": "N_thr^c max high-trough count, default 5",
            "decrement_weight": "extra weight for valid_min_peak, default 25",
            "monotone_weight": "extra weight for valid_monotone_tail, default 10",
            "hard_shape_validity": "if true, nonzero decrement/monotone penalties trigger strict_feasibility, default true",
        }
        search = {
            "rt_search_depth": "bounded binary search iterations, default 8",
            "rt_max_gap": "largest recovery gap to test, default gap_time or 200",
            "potentiation_gap_fraction": "fraction of rt used between potentiation trains, default 0.5",
            "reference_pulse_shape_index": "index of pulse_shapes used for reference/recovery/potentiation/intensity, default middle",
        }
        weights = {
            "validity_weight": "weight for MMC2 validity constraints",
            "habituation_weight": "weight for habituation time feasibility",
            "recovery_weight": "weight for recovery feasibility",
            "potentiation_weight": "weight for second-train faster habituation",
            "frequency_weight": "weight for frequency ordering",
            "intensity_weight": "weight for intensity ordering",
            "subliminal_weight": "weight for delayed recovery after stricter habituation",
        }
        return {
            "required": {
                "pulse_shapes": "List[(t_on, t_off)] OR a single (t_on, t_off)",
                "n_repeats_pre": "int pulses in each train",
                "n_repeats_post": "int kept for compatibility/presentation",
                "gap_time": "float default gap scale",
                "u_values": "List[float] grid for u",
            },
            "optional": {**search, **mmc2_thresholds, **weights},
            "mmc2_thresholds": mmc2_thresholds,
            "search": search,
            "weights": weights,
        }

    def validate(self, task: TaskSpec) -> None:
        _as_pulse_shapes(overrides_get(task, {}, "pulse_shapes", fallback_attr="pulse_shapes"))
        if int(overrides_get(task, {}, "n_repeats_pre", fallback_attr="n_repeats_pre")) <= 2:
            raise ValueError("n_repeats_pre must be >= 3.")
        if overrides_get(task, {}, "u_values", fallback_attr="u_values") is None:
            raise ValueError("habituation_hallmarks_mmc2 requires u_values.")

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        u_values = overrides_get(task, {}, "u_values", fallback_attr="u_values")
        if task.n_inputs is None:
            raise ValueError("need n_inputs")
        return [
            np.asarray(u, dtype=np.float32)
            for u in product(list(u_values), repeat=int(task.n_inputs))
        ]

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        pulse_shapes = _as_pulse_shapes(overrides_get(task, overrides, "pulse_shapes", fallback_attr="pulse_shapes"))
        n_pre = int(overrides_get(task, overrides, "n_repeats_pre", fallback_attr="n_repeats_pre"))
        n_post = int(overrides_get(task, overrides, "n_repeats_post", fallback_attr="n_repeats_post", default=n_pre))
        gap_time = float(overrides_get(task, overrides, "gap_time", fallback_attr="gap_time", default=50.0))
        n_t = int(overrides_get(task, overrides, "n_t", fallback_attr="n_t", default=task.n_t))
        min_peak = float(overrides_get(task, overrides, "min_peak", fallback_attr="min_peak", default=0.1))
        max_peak = float(overrides_get(task, overrides, "max_peak", fallback_attr="max_peak", default=10.0))

        u_list_local = self.build_u_list(task, overrides)
        ic_obj = "from_ss" if task.ic == "from_ss" else self.build_ic(task, overrides)

        metric_kwargs = dict(
            eps_h=float(overrides_get(task, overrides, "eps_h", fallback_attr="eps_h", default=0.01)),
            eps_subliminal=float(overrides_get(task, overrides, "eps_subliminal", fallback_attr="eps_subliminal", default=0.005)),
            recovery_tol=float(overrides_get(task, overrides, "recovery_tol", fallback_attr="recovery_tol", default=0.05)),
            rt_search_depth=int(overrides_get(task, overrides, "rt_search_depth", fallback_attr="rt_search_depth", default=8)),
            rt_max_gap=float(overrides_get(task, overrides, "rt_max_gap", fallback_attr="rt_max_gap", default=max(gap_time, 200.0))),
            potentiation_gap_fraction=float(overrides_get(task, overrides, "potentiation_gap_fraction", fallback_attr="potentiation_gap_fraction", default=0.5)),
            intensity_values=overrides_get(task, overrides, "intensity_values", fallback_attr="intensity_values", default=[1.0, 0.5, 0.25]),
            delta_p_max=float(overrides_get(task, overrides, "delta_p_max", fallback_attr="delta_p_max", default=0.5)),
            monotone_drop_tol=float(overrides_get(task, overrides, "monotone_drop_tol", fallback_attr="monotone_drop_tol", default=1e-4)),
            i_max=int(overrides_get(task, overrides, "i_max", fallback_attr="i_max", default=2)),
            n_post_min=int(overrides_get(task, overrides, "n_post_min", fallback_attr="n_post_min", default=2)),
            delta_min=float(overrides_get(task, overrides, "delta_min", fallback_attr="delta_min", default=0.2)),
            delta_t=float(overrides_get(task, overrides, "delta_t", fallback_attr="delta_t", default=0.05)),
            trough_thr=float(overrides_get(task, overrides, "trough_thr", fallback_attr="trough_thr", default=0.6)),
            trough_tail=float(overrides_get(task, overrides, "trough_tail", fallback_attr="trough_tail", default=0.02)),
            trough_count_thr=float(overrides_get(task, overrides, "trough_count_thr", fallback_attr="trough_count_thr", default=0.10)),
            trough_count_max=int(overrides_get(task, overrides, "trough_count_max", fallback_attr="trough_count_max", default=5)),
            validity_weight=float(overrides_get(task, overrides, "validity_weight", fallback_attr="validity_weight", default=1.0)),
            decrement_weight=float(overrides_get(task, overrides, "decrement_weight", fallback_attr="decrement_weight", default=25.0)),
            monotone_weight=float(overrides_get(task, overrides, "monotone_weight", fallback_attr="monotone_weight", default=10.0)),
            hard_shape_validity=bool(overrides_get(task, overrides, "hard_shape_validity", fallback_attr="hard_shape_validity", default=True)),
            habituation_weight=float(overrides_get(task, overrides, "habituation_weight", fallback_attr="habituation_weight", default=1.0)),
            recovery_weight=float(overrides_get(task, overrides, "recovery_weight", fallback_attr="recovery_weight", default=1.0)),
            potentiation_weight=float(overrides_get(task, overrides, "potentiation_weight", fallback_attr="potentiation_weight", default=1.0)),
            frequency_weight=float(overrides_get(task, overrides, "frequency_weight", fallback_attr="frequency_weight", default=1.0)),
            intensity_weight=float(overrides_get(task, overrides, "intensity_weight", fallback_attr="intensity_weight", default=1.0)),
            subliminal_weight=float(overrides_get(task, overrides, "subliminal_weight", fallback_attr="subliminal_weight", default=1.0)),
            ordering_margin=float(overrides_get(task, overrides, "ordering_margin", fallback_attr="ordering_margin", default=0.0)),
            reference_pulse_shape_index=overrides_get(
                task,
                overrides,
                "reference_pulse_shape_index",
                fallback_attr="reference_pulse_shape_index",
                default=None,
            ),
        )

        def reward_fn(state: Any):
            if ic_obj == "from_ss":
                u_off = np.zeros_like(u_list_local[0], dtype=np.float32)
                x0_list = steady_state_ic_list(state, [u_off])
            else:
                x0_list = ic_obj.get_ic(state)

            ss_offset = x0_list[0][state.species_idx_dict[state.output_labels[0]]]
            mn = min_peak + ss_offset
            mx = max_peak + ss_offset

            loss, info = habituation_hallmarks_mmc2_metric(
                crn=state,
                u_list_local=u_list_local,
                x0_list=x0_list,
                pulse_shapes=pulse_shapes,
                n_repeats_pre=n_pre,
                n_repeats_post=n_post,
                gap_time=gap_time,
                n_t=n_t,
                min_peak=mn,
                max_peak=mx,
                LARGE_NUMBER=task.LARGE_NUMBER,
                **metric_kwargs,
            )

            state.last_task_info["reward"] = float(loss)
            state.last_task_info["reward type"] = "habituation_hallmarks_mmc2"
            state.last_task_info["hallmark_info"] = info
            state.last_task_info["hallmark_runs"] = info.get("hallmark_runs", [])
            state.last_task_info["component_losses"] = info.get("component_losses", {})
            state.last_task_info["type"] = "transient response"
            return float(loss), state.last_task_info

        return reward_fn


@register_task_kind
class HabituationHallmarksTaskKind(TaskKindBase):
    """Habituation/sensitization task covering hallmarks 1, 2, 3, 4, 5, 6, and 10."""
    kind = "habituation_hallmarks"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "pulse_shapes": "List[(t_on, t_off)] OR a single (t_on, t_off)",
                "n_repeats_pre": "int pulses in the reference train",
                "n_repeats_post": "int pulses after recovery gaps",
                "gap_time": "float OFF gap for spontaneous recovery",
                "u_values": "List[float] grid for u",
            },
            "optional": {
                "s": "+1 habituation, -1 sensitization (default inferred from sensitization)",
                "sensitization": "bool alias setting s=-1",
                "intensity_values": "List[float] e.g. [1.0, 0.66, 0.33]",
                "long_train_repeats": "int extended train length for hallmark 6",
                "long_gap_times": "List[float] gaps for hallmark 10",
                "potentiation_blocks": "int repeated trains for hallmark 3",
                "n_t": "int samples per protocol simulation",
                "habituation_weight": "float weight on peak-ratio separation terms",
                "early_peak_sep_weight": "float weight for minimum early peak separation",
                "early_peak_min_change": "minimum signed fractional change among early peaks",
                "early_peak_count": "number of early peaks included in separation check",
                "state_reset_weight": "float weight for all-species gap reset penalty",
                "state_reset_tol": "relative L2 tolerance for gap reset to initial state",
                "memory_difference_weight": "float weight for transient-vs-state memory gap trend",
            },
        }

    def validate(self, task: TaskSpec) -> None:
        _as_pulse_shapes(overrides_get(task, {}, "pulse_shapes", fallback_attr="pulse_shapes"))
        if int(overrides_get(task, {}, "n_repeats_pre", fallback_attr="n_repeats_pre")) <= 1:
            raise ValueError("n_repeats_pre must be >= 2.")
        if int(overrides_get(task, {}, "n_repeats_post", fallback_attr="n_repeats_post")) <= 0:
            raise ValueError("n_repeats_post must be >= 1.")
        if overrides_get(task, {}, "u_values", fallback_attr="u_values") is None:
            raise ValueError("habituation_hallmarks requires u_values.")

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        u_values = overrides_get(task, {}, "u_values", fallback_attr="u_values")
        if task.n_inputs is None:
            raise ValueError("need n_inputs")
        return [
            np.asarray(u, dtype=np.float32)
            for u in product(list(u_values), repeat=int(task.n_inputs))
        ]

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        pulse_shapes = _as_pulse_shapes(overrides_get(task, overrides, "pulse_shapes", fallback_attr="pulse_shapes"))
        n_pre = int(overrides_get(task, overrides, "n_repeats_pre", fallback_attr="n_repeats_pre"))
        n_post = int(overrides_get(task, overrides, "n_repeats_post", fallback_attr="n_repeats_post"))
        gap_time = float(overrides_get(task, overrides, "gap_time", fallback_attr="gap_time"))
        n_t = int(overrides_get(task, overrides, "n_t", fallback_attr="n_t", default=task.n_t))
        min_peak = float(overrides_get(task, overrides, "min_peak", fallback_attr="min_peak", default=0.1))
        max_peak = float(overrides_get(task, overrides, "max_peak", fallback_attr="max_peak", default=2.0))
        sensitization = bool(overrides_get(task, overrides, "sensitization", fallback_attr="sensitization", default=False))
        s = int(overrides_get(task, overrides, "s", fallback_attr="s", default=(-1 if sensitization else 1)))
        if s not in (-1, 1):
            raise ValueError("s must be +1 for habituation or -1 for sensitization.")

        u_list_local = self.build_u_list(task, overrides)
        ic_obj = "from_ss" if task.ic == "from_ss" else self.build_ic(task, overrides)

        metric_kwargs = dict(
            freq_weight=float(overrides_get(task, overrides, "freq_weight", fallback_attr="freq_weight", default=1.0)),
            habituation_weight=float(overrides_get(task, overrides, "habituation_weight", fallback_attr="habituation_weight", default=1.0)),
            early_peak_sep_weight=float(overrides_get(task, overrides, "early_peak_sep_weight", fallback_attr="early_peak_sep_weight", default=1.0)),
            early_peak_min_change=float(overrides_get(task, overrides, "early_peak_min_change", fallback_attr="early_peak_min_change", default=0.1)),
            early_peak_count=int(overrides_get(task, overrides, "early_peak_count", fallback_attr="early_peak_count", default=3)),
            gap_weight=float(overrides_get(task, overrides, "gap_weight", fallback_attr="gap_weight", default=5.0)),
            recovery_tol=float(overrides_get(task, overrides, "recovery_tol", fallback_attr="recovery_tol", default=0.05)),
            potentiation_blocks=int(overrides_get(task, overrides, "potentiation_blocks", fallback_attr="potentiation_blocks", default=3)),
            potentiation_gap_time=float(overrides_get(task, overrides, "potentiation_gap_time", fallback_attr="potentiation_gap_time", default=gap_time)),
            potentiation_weight=float(overrides_get(task, overrides, "potentiation_weight", fallback_attr="potentiation_weight", default=1.0)),
            intensity_values=overrides_get(task, overrides, "intensity_values", fallback_attr="intensity_values", default=[1.0, 0.66, 0.33]),
            intensity_weight=float(overrides_get(task, overrides, "intensity_weight", fallback_attr="intensity_weight", default=1.0)),
            long_train_repeats=int(overrides_get(task, overrides, "long_train_repeats", fallback_attr="long_train_repeats", default=max(20, 2 * n_pre))),
            short_gap_time=float(overrides_get(task, overrides, "short_gap_time", fallback_attr="short_gap_time", default=max(1.0, 0.2 * gap_time))),
            subliminal_weight=float(overrides_get(task, overrides, "subliminal_weight", fallback_attr="subliminal_weight", default=1.0)),
            asymptote_window=int(overrides_get(task, overrides, "asymptote_window", fallback_attr="asymptote_window", default=3)),
            long_gap_times=overrides_get(task, overrides, "long_gap_times", fallback_attr="long_gap_times", default=[gap_time, 3.0 * gap_time, 10.0 * gap_time]),
            long_term_weight=float(overrides_get(task, overrides, "long_term_weight", fallback_attr="long_term_weight", default=1.0)),
            long_term_min_memory=float(overrides_get(task, overrides, "long_term_min_memory", fallback_attr="long_term_min_memory", default=0.05)),
            memory_difference_weight=float(overrides_get(task, overrides, "memory_difference_weight", fallback_attr="memory_difference_weight", default=1.0)),
            state_reset_weight=float(overrides_get(task, overrides, "state_reset_weight", fallback_attr="state_reset_weight", default=1.0)),
            state_reset_tol=float(overrides_get(task, overrides, "state_reset_tol", fallback_attr="state_reset_tol", default=0.2)),
            state_reset_floor=float(overrides_get(task, overrides, "state_reset_floor", fallback_attr="state_reset_floor", default=1.0)),
            peak_tol=float(overrides_get(task, overrides, "peak_tol", fallback_attr="peak_tol", default=0.1)),
            margin=float(overrides_get(task, overrides, "margin", fallback_attr="margin", default=0.0)),
        )

        def reward_fn(state: Any):
            if ic_obj == "from_ss":
                u_off = np.zeros_like(u_list_local[0], dtype=np.float32)
                x0_list = steady_state_ic_list(state, [u_off])
            else:
                x0_list = ic_obj.get_ic(state)

            ss_offset = x0_list[0][state.species_idx_dict[state.output_labels[0]]]
            mn = min_peak + ss_offset
            mx = max_peak + ss_offset

            loss, info = habituation_hallmarks_metric(
                crn=state,
                u_list_local=u_list_local,
                x0_list=x0_list,
                pulse_shapes=pulse_shapes,
                n_repeats_pre=n_pre,
                n_repeats_post=n_post,
                gap_time=gap_time,
                n_t=n_t,
                s=s,
                min_peak=mn,
                max_peak=mx,
                LARGE_NUMBER=task.LARGE_NUMBER,
                **metric_kwargs,
            )

            state.last_task_info["reward"] = float(loss)
            state.last_task_info["reward type"] = "habituation_hallmarks"
            state.last_task_info["hallmark_info"] = info
            state.last_task_info["hallmark_runs"] = info.get("hallmark_runs", [])
            state.last_task_info["component_losses"] = info.get("component_losses", {})
            state.last_task_info["s"] = int(s)
            state.last_task_info["type"] = "transient response"
            return float(loss), state.last_task_info

        return reward_fn


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
        if gap_time is None or float(gap_time) < 0:
            raise ValueError("habituation_gap requires gap_time >= 0.")

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



# --- habituation

def build_on_off_time_horizon(
    *,
    t_on: float,
    t_off: float,
    n_repeats: int,
    n_t: int,
    dtype=np.float32,
) -> List[np.ndarray]:
    if n_repeats <= 0:
        raise ValueError("n_repeats must be >= 1.")
    if t_on <= 0 or t_off <= 0:
        raise ValueError("t_on and t_off must be > 0.")

    n_segments = 2 * int(n_repeats)
    if n_t < 2 * n_segments:
        raise ValueError(f"n_t={n_t} too small for {n_segments} segments.")

    pts_per_segment = max(2, int(np.floor(n_t / n_segments)))

    nested_time_horizon = []
    for _ in range(int(n_repeats)):
        nested_time_horizon.append(np.linspace(0.0, float(t_on), pts_per_segment, dtype=dtype))
        nested_time_horizon.append(np.linspace(0.0, float(t_off), pts_per_segment, dtype=dtype))

    return nested_time_horizon


def build_u_nested_list_on_off(
    *,
    u_list: List[np.ndarray],
    n_repeats: int,
    off_value: float = 0.0,
) -> List[List[np.ndarray]]:
    u_nested_list = []

    for u in u_list:
        u = np.asarray(u, dtype=np.float32).reshape(-1)
        u_off = np.full_like(u, float(off_value), dtype=np.float32)

        protocol = []
        for _ in range(int(n_repeats)):
            protocol.append(u)
            protocol.append(u_off)

        u_nested_list.append(protocol)

    return u_nested_list


@register_task_kind
class HabituationTaskKind(TaskKindBase):
    """Habituation task with a repeated ON/OFF pulse train."""

    kind = "habituation"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "pulse_shape": "Single (t_on, t_off)",
                "n_repeats": "int number of ON/OFF pulse repeats",
                "u_values": "List[float] grid for u",
            },
            "optional": {
                "weights": "float or list of peak-ratio weights (default 1.0)",
                "ratio_weights": "alias for weights",
                "min_peak": "float (default 0.1)",
                "max_peak": "float (default 2.0)",
                "n_t": "int samples per simulation (default task.n_t)",
            },
        }

    def validate(self, task: TaskSpec) -> None:
        pulse_shape = overrides_get(task, {}, "pulse_shape", fallback_attr="pulse_shape")
        if not (isinstance(pulse_shape, (tuple, list)) and len(pulse_shape) == 2):
            raise ValueError("habituation requires pulse_shape=(t_on, t_off).")

        t_on, t_off = float(pulse_shape[0]), float(pulse_shape[1])
        if t_on <= 0 or t_off <= 0:
            raise ValueError("t_on and t_off must be > 0.")

        n_repeats = overrides_get(task, {}, "n_repeats", fallback_attr="n_repeats")
        if n_repeats is None or int(n_repeats) <= 1:
            raise ValueError("n_repeats must be >= 2.")

        if overrides_get(task, {}, "u_values", fallback_attr="u_values") is None:
            raise ValueError("habituation requires u_values.")

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
        pulse_shape = overrides_get(task, overrides, "pulse_shape", fallback_attr="pulse_shape")
        if not (isinstance(pulse_shape, (tuple, list)) and len(pulse_shape) == 2):
            raise ValueError("habituation requires pulse_shape=(t_on, t_off).")

        t_on, t_off = float(pulse_shape[0]), float(pulse_shape[1])
        n_repeats = int(overrides_get(task, overrides, "n_repeats", fallback_attr="n_repeats"))
        n_t = int(overrides_get(task, overrides, "n_t", fallback_attr="n_t", default=task.n_t))

        weights = overrides_get(task, overrides, "ratio_weights", fallback_attr="ratio_weights", default=None)
        if weights is None:
            weights = overrides_get(task, overrides, "weights", fallback_attr="weights", default=1.0)
        if isinstance(weights, str):
            weights = 1.0

        min_peak = float(overrides_get(task, overrides, "min_peak", fallback_attr="min_peak", default=0.1))
        max_peak = float(overrides_get(task, overrides, "max_peak", fallback_attr="max_peak", default=2.0))

        u_list_local = self.build_u_list(task, overrides)
        ic_obj = "from_ss" if task.ic == "from_ss" else self.build_ic(task, overrides)

        nested_time_horizon = build_on_off_time_horizon(
            t_on=t_on,
            t_off=t_off,
            n_repeats=n_repeats,
            n_t=n_t,
            dtype=np.float32,
        )

        u_nested_list = build_u_nested_list_on_off(
            u_list=u_list_local,
            n_repeats=n_repeats,
            off_value=0.0,
        )

        def reward_fn(state: Any):
            if ic_obj == "from_ss":
                u_off = np.zeros_like(u_list_local[0], dtype=np.float32)
                x0_list = steady_state_ic_list(state, [u_off])
            else:
                x0_list = ic_obj.get_ic(state)

            loss, info = habituation_error_piecewise(
                crn=state,
                u_nested_list=u_nested_list,
                x0_list=x0_list,
                nested_time_horizon=nested_time_horizon,
                w=weights,
                LARGE_NUMBER=task.LARGE_NUMBER,
                min_peak=min_peak,
                max_peak=max_peak,
            )

            state.last_task_info.update(info)
            state.last_task_info["reward"] = float(loss)
            state.last_task_info["reward type"] = "habituation"
            state.last_task_info["pulse_shape"] = (t_on, t_off)
            state.last_task_info["n_repeats"] = n_repeats

            return float(loss), state.last_task_info

        return reward_fn
