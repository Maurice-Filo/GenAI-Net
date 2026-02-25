"""
RL4CRN.rewards

Reward / cost functions for evaluating IOCRN behaviors under different design tasks.

This module provides task-specific wrappers around an IOCRN's simulation interface
(e.g., `IOCRN.transient_response` and `IOCRN.transient_response_piecewise`)
and converts the resulting trajectories into scalar performance measures that can be
used as RL rewards (or costs).

Included objectives:

- **Dynamic tracking** (continuous-valued): weighted L1/L2 tracking error to a reference
  trajectory or setpoint across multiple scenarios.
- **Piecewise tracking**: same as above, but with piecewise-constant inputs and segmented
  time horizons (useful for protocols / sequences).
- **Oscillation shaping**: penalizes deviations from desired oscillatory features such as
  frequency, mean level, damping, and peak ratios using `oscillation_metrics`.
- **Logic circuit scoring**: evaluates steady-state binary behavior (via BCE or thresholded
  mismatch) for combinational circuits and piecewise logic protocols (e.g., latches).
- **Custom relationship tracking**: evaluates arbitrary algebraic constraints between species
  trajectories defined by a user-supplied function (targeting zero error).

All functions return a scalar `performance` (interpretable as cost unless you negate it)
and update `crn.last_task_info` with metadata such as the reward value, task type, and
the simulation settings that produced it.
"""

from RL4CRN.utils.utils import performance_metric
from RL4CRN.utils.utils import oscillation_metrics
import numpy as np

def dynamic_tracking_error(crn, u_list, x0_list, time_horizon, r_list, w, norm=1, relative=False, LARGE_NUMBER=1e4):
    """
    Compute a dynamic tracking cost for an IOCRN over a batch of scenarios.

    The function simulates the CRN for each scenario in the Cartesian product
    of `u_list` and `x0_list` (as implemented by `crn.transient_response`) and
    evaluates how well the output trajectories track the provided references.

    Args:
        crn : IOCRN
            An IOCRN-like object implementing
            `transient_response(u_list, x0_list, time_horizon, LARGE_NUMBER=...)`.
            The simulation is expected to return `(t, x_list, y_list, last_task_info)`,
            where `y_list` is a list of output trajectories.
        u_list : list[np.ndarray]
            List of constant input vectors. Each element has shape `(p,)`, where `p`
            is the number of CRN inputs.
        x0_list : list[np.ndarray]
            List of initial state vectors. Each element has shape `(n,)`, where `n`
            is the number of CRN species.
        time_horizon : np.ndarray
            1D array of evaluation times with shape `(T,)`.
        r_list : list[np.ndarray]
            List of reference signals/targets for each scenario. The expected shape
            and interpretation depend on `performance_metric`. Common usage is a list
            of arrays with shape `(q, T)` or `(q,)` (setpoints), where `q` is the number
            of outputs.
        w : np.ndarray
            Weights for the tracking error. Typically shape `(q, T)` so each output
            and time point can be weighted differently. (Exact expectations follow
            `performance_metric`.)
        norm : int, default=1
            Norm used in the tracking error. Passed to `performance_metric`.
            Common options are 1 (L1) or 2 (L2 / squared error).
        relative : bool, default=False
            If True, compute a relative error (as supported by `performance_metric`).
        LARGE_NUMBER : float, default=1e4
            Divergence penalty passed to `transient_response`. If the integrator fails
            or becomes unstable, trajectories may be filled with `LARGE_NUMBER`, which
            typically yields a large cost.

    Returns:
        performance : float
            Scalar tracking cost aggregated across scenarios, outputs, and time.
        last_task_info : dict
            Updated `crn.last_task_info` dictionary, augmented with:

            - 'reward': performance
            - 'setpoint': r_list
            - 'initial_conditions': x0_list
            - 'reward type': 'dynamic_tracking_error'
    """

    t, x_list, y_list, last_task_info = crn.transient_response(u_list, x0_list, time_horizon, LARGE_NUMBER=LARGE_NUMBER)
    performance = performance_metric(r_list, y_list, w, norm=norm, relative=relative)
    crn.last_task_info['reward'] = performance
    crn.last_task_info['setpoint'] = r_list
    crn.last_task_info['initial_conditions'] = x0_list
    crn.last_task_info['reward type'] = 'dynamic_tracking_error'
    return performance, crn.last_task_info


import numpy as np
from typing import List, Sequence, Tuple, Union, Optional

def habituation_error_piecewise(
    crn,
    u_nested_list,
    x0_list,
    nested_time_horizon,
    w,
    LARGE_NUMBER: float = 1e4,
    min_peak: float = 0.1,
    max_peak: float = 2.0,
):
    """Compute a habituation cost for a piecewise protocol using peak ratios.

    This function simulates the CRN under a piecewise-constant input protocol and
    evaluates "habituation" as a change in peak response across repeated stimulus
    windows.

    Convention used here:
      - `nested_time_horizon` defines K segments with time grids t_0, t_1, ..., t_{K-1}.
      - We treat *even-indexed* segments (0, 2, 4, ...) as "stimulus" windows.
        Peaks are measured in those windows for each scenario output trajectory.
      - A habituation score is computed from ratios of consecutive stimulus peaks:
            ratio_k = peak_{k+1} / peak_k
        (Lower ratios indicate stronger habituation.)

    The function also enforces peak bounds. If any measured peak is outside
    [min_peak, max_peak], it returns `LARGE_NUMBER`.

    Args:
        crn: IOCRN
            IOCRN-like object implementing:
            `transient_response_piecewise(u_nested_list, x0_list, nested_time_horizon, LARGE_NUMBER=...)`.
        u_nested_list: list[list[np.ndarray]]
            List of input protocols. Each protocol is a list of input vectors (p,)
            applied segment-wise. The inner list length must match
            `len(nested_time_horizon)`.
        x0_list: list[np.ndarray]
            List of initial conditions (n,).
        nested_time_horizon: list[np.ndarray]
            List of time grids, one per segment. These will be stitched by the simulator.
        w: Union[float, Sequence[float], np.ndarray]
            Weights applied to each peak ratio term. If a scalar, the same weight is
            applied to all ratios. If a sequence, it should have length equal to the
            number of ratios (n_peaks - 1).
        LARGE_NUMBER: float, default=1e4
            Penalty returned if simulation diverges or peak constraints fail.
        min_peak: float, default=0.1
            Minimum acceptable peak value. Peaks below this are considered invalid.
        max_peak: float, default=2.0
            Maximum acceptable peak value. Peaks above this are considered invalid.

    Returns:
        Tuple[float, dict]:
            - performance: Scalar habituation cost (lower is better).
            - last_task_info: `crn.last_task_info` updated with metadata:
                - 'reward': performance
                - 'min_peak', 'max_peak'
                - 'initial_conditions'
                - 'reward type': 'habituation_error_piecewise'
    """
    t, x_list, y_list, last_task_info = crn.transient_response_piecewise(
        u_nested_list, x0_list, nested_time_horizon, LARGE_NUMBER=LARGE_NUMBER
    )

    # Convert each segment's time grid to (start, end) interval
    intervals = [(float(t_k[0]), float(t_k[-1])) for t_k in nested_time_horizon]

    if len(crn.output_labels) == 0 or len(crn.output_labels) > 1:
        raise ValueError("habituation_error_piecewise currently supports exactly one output species.")

    # we assume the IC dictates a base for the peaks
    ss_offset = x0_list[0][crn.species_idx_dict[crn.output_labels[0]]]

    min_peak += ss_offset
    max_peak += ss_offset

    # comupute steady state values for each off segment and check they match
    durations = np.array([float(tk[-1] - tk[0]) for tk in nested_time_horizon], dtype=float)
    starts = np.cumsum(np.concatenate([[0.0], durations[:-1]]))
    ends = starts + durations
    abs_intervals = list(zip(starts, ends))
    off_intervals = [(0.,0.)] + abs_intervals[1::2]

    # For each scenario, check OFF steady state is invariant
    # (assuming u_nested_list length == number of scenarios)
    _ok = True
    for s in range(len(x_list)):
        # Determine OFF input for this scenario (single input)
        # Here: assume the protocol alternates and off is the second segment value if provided,
        # otherwise 0.
        if len(u_nested_list[s]) > 1:
            u_off = np.asarray(u_nested_list[s][1], dtype=np.float64).reshape(-1)
        else:
            u_off = np.array([0.0], dtype=np.float64)

        ok, _ = check_off_ss_invariant(crn, t, x_list[s], off_intervals, u_off,
                                    ss_tol_abs=1e-4, ss_tol_rel=1e-3)
        
        if not ok:
            _ok = False
            break
    
    if _ok:
        performance = habituation_metric(
            intervals=intervals,
            t=t,
            y_list=y_list,
            w=w,
            min_peak=min_peak,
            max_peak=max_peak,
            base_valley=ss_offset,
            LARGE_NUMBER=LARGE_NUMBER
        )
    else:
        performance = float(LARGE_NUMBER)

    crn.last_task_info["reward"] = performance
    crn.last_task_info["min_peak"] = float(min_peak)
    crn.last_task_info["max_peak"] = float(max_peak)
    crn.last_task_info["initial_conditions"] = x0_list
    crn.last_task_info["reward type"] = "habituation_error_piecewise"
    # save input info for pulse plotting
    if len(u_nested_list) == 1: # plotting is supported only for a single input 
        crn.last_task_info["input_intervals"] = intervals
        crn.last_task_info["input_pulse"] = u_nested_list[0][0]

    return performance, crn.last_task_info

import numpy as np
from scipy.optimize import fsolve

def state_at_time(t, x, t_query):
    # x shape (n, T). Take nearest time index.
    idx = int(np.argmin(np.abs(t - t_query)))
    return x[:, idx]

def check_off_ss_invariant(crn, t, x, off_intervals, u_off,
                           ss_tol_abs=1e-4, ss_tol_rel=1e-3,
                           xtol=1e-9, maxfev=2000):
    """
    x: full state trajectory, shape (n, T)
    off_intervals: list of (start,end) absolute
    u_off: input vector for OFF
    """

    ss_solutions = []
    for (_, end) in off_intervals:
        x_guess = state_at_time(t, x, end)
        x_ss, ok, _ = fsolve_ss(crn, u_off, x_guess, xtol=xtol, maxfev=maxfev)
        if not ok:
            return False, None
        ss_solutions.append(x_ss)

    ref = ss_solutions[0]
    for sol in ss_solutions[1:]:
        diff = np.linalg.norm(sol - ref, ord=np.inf)
        scale = max(1.0, np.linalg.norm(ref, ord=np.inf))
        if diff > ss_tol_abs and diff > ss_tol_rel * scale:
            return False, ss_solutions
    return True, ss_solutions


def fsolve_ss(crn, u, x0, xtol=1e-9, maxfev=2000):
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)

    sol, infodict, ier, msg = fsolve(
        lambda x: crn.rate_function(0.0, x, u),
        x0,
        full_output=True,
        xtol=xtol,
        maxfev=maxfev,
    )
    sol = np.asarray(sol, dtype=np.float64)
    success = (ier == 1) and np.all(np.isfinite(sol))
    return sol, success, msg

def habituation_metric(
    intervals: Sequence[Tuple[float, float]],
    t: np.ndarray,
    y_list: Sequence[np.ndarray],
    w: Union[float, Sequence[float], np.ndarray],
    min_peak: float = 0.1,
    max_peak: float = 2.0,
    base_valley: float = 0.0,
    LARGE_NUMBER: float = 1e4,
) -> float:
    """Compute a habituation cost from output peaks across repeated stimulus windows.

    This metric extracts peak amplitudes from specified time intervals and computes
    ratios of consecutive stimulus peaks. By default, it assumes even-indexed
    intervals (0, 2, 4, ...) correspond to repeated stimulus windows.

    Steps:
      1) For each scenario output trajectory in `y_list`, compute peaks in stimulus
         windows (even intervals).
      2) Enforce peak bounds: if any peak is outside [min_peak, max_peak], return
         `LARGE_NUMBER`.
      3) Compute peak ratios: ratio_k = peak_{k+1} / peak_k.
      4) Return weighted mean of ratios (lower implies stronger habituation).

    Notes:
      - If there are fewer than 2 stimulus windows, this metric cannot form a ratio
        and returns `LARGE_NUMBER`.
      - If any peak is zero or extremely small, division can blow up; we guard with
        a small epsilon.

    Args:
        intervals: Sequence[Tuple[float, float]]
            Time intervals (start, end) defining protocol segments.
        t: np.ndarray
            Stitched time vector from the simulator, shape (T,).
        y_list: Sequence[np.ndarray]
            List of output trajectories, one per scenario.
            Each element is typically shape (q, T) (q outputs).
        w: Union[float, Sequence[float], np.ndarray]
            Weights for each ratio term. If scalar, repeated. If sequence, must match
            number of ratios (n_peaks - 1) or will be broadcast/clipped.
        min_peak: float, default=0.1
            Minimum acceptable peak amplitude.
        max_peak: float, default=2.0
            Maximum acceptable peak amplitude.
        LARGE_NUMBER: float, default=1e4
            Penalty returned if constraints fail or insufficient windows exist.

    Returns:
        float: Scalar habituation cost. Lower is better.
    """
    # Stimulus windows: even-indexed intervals
    rep_len = intervals[1][1] + intervals[0][1]
    stim_intervals = [intervals[i] for i in range(0, len(intervals), 2)]
    stim_intervals = [(float(start) + i*rep_len, float(end) + i*rep_len) for i, (start, end) in enumerate(stim_intervals)]

    off_intervals = [intervals[i] for i in range(1, len(intervals), 2)]
    off_intervals = [(float(start) + i*rep_len + intervals[0][1], float(end) + i*rep_len + intervals[0][1]) for i, (start, end) in enumerate(off_intervals)]

    if len(stim_intervals) < 2:
        raise ValueError("At least two stimulus intervals are required to compute habituation.")

    eps = 1e-12
    all_ratios: List[float] = []

    # Prepare weights for ratios
    n_ratios = len(stim_intervals) - 1
    if np.isscalar(w):
        w_arr = np.full(n_ratios, float(w), dtype=np.float32)
    else:
        w_arr = np.asarray(list(w), dtype=np.float32).reshape(-1)
        if w_arr.size < n_ratios:
            # pad with last value
            w_arr = np.pad(w_arr, (0, n_ratios - w_arr.size), mode="edge")
        elif w_arr.size > n_ratios:
            w_arr = w_arr[:n_ratios]

    # For each scenario (each y), compute stimulus peaks and ratios
    for y in y_list:
        # y assumed shape (q, T). Use max over outputs q as a single "response amplitude".
        # If you prefer a specific output channel, replace this with y[channel_idx, :].
        stim_peaks: List[float] = []
        for (start, end) in stim_intervals:
            mask = (t >= start) & (t <= end)
            if not np.any(mask):
                stim_peaks.append(float(LARGE_NUMBER))
                continue
            y_seg = y[:, mask]  # (q, T_seg)
            peak = float(np.max(y_seg))
            stim_peaks.append(peak)

        off_valleys = []
        for (start, end) in off_intervals:
            mask = (t >= start) & (t <= end)
            if not np.any(mask):
                off_valleys.append(float(LARGE_NUMBER))
                continue
            y_seg = y[:, mask]  # (q, T_seg)
            valley = float(np.min(y_seg))
            off_valleys.append(valley)

        # Peak bounds check
        peaks_ok = [(p <= max_peak) for p in stim_peaks]
        if not all(peaks_ok):
            return float(LARGE_NUMBER) # penalize if any peak is out of bounds (upper)
        # otherwise, set size to min_peak if any peak is below min_peak
        stim_peaks = [max(p, min_peak) for p in stim_peaks]

        # check valleys 
        # print(off_valleys)
        # valleys_ok = [(base_valley <= p) for p in off_valleys]
        # if not all(valleys_ok):
        #     return float(LARGE_NUMBER)

        # Ratios peak_{k+1}/peak_k
        ratios = [
            (stim_peaks[i + 1] / max(stim_peaks[i], eps))
            for i in range(len(stim_peaks) - 1)
        ]

        # Weighted mean for this scenario
        ratios = np.asarray(ratios, dtype=np.float32)
        # scenario_score = float(np.mean(w_arr * ratios))
        scenario_score = np.log(float(np.max(ratios)) + eps) # use log of max ratio to focus on worst habituation step
        all_ratios.append(scenario_score)

    return float(np.mean(all_ratios)) if all_ratios else float(LARGE_NUMBER)


def dynamic_tracking_error_piecewise(crn, u_nested_list, x0_list, nested_time_horizon, r_list, w, norm=1, relative=False, LARGE_NUMBER=1e4):
    """
    Compute a dynamic tracking cost for piecewise-constant input protocols.

    This is the piecewise analogue of `dynamic_tracking_error`. Instead of
    constant inputs over a single horizon, each scenario specifies a *sequence*
    of inputs applied over segmented time horizons.

    Args:
        crn : IOCRN
            An IOCRN-like object implementing
            `transient_response_piecewise(u_nested_list, x0_list, nested_time_horizon, LARGE_NUMBER=...)`.
        u_nested_list : list[list[np.ndarray]]
            List of input protocols. Each element is a sequence `[u_0, u_1, ..., u_K]`,
            where each `u_k` has shape `(p,)`. The inner list length must match
            `len(nested_time_horizon)`.
        x0_list : list[np.ndarray]
            List of initial state vectors, each of shape `(n,)`.
        nested_time_horizon : list[np.ndarray]
            List of time grids `[t_0, t_1, ..., t_K]`, one per protocol segment.
            Each `t_k` is a 1D array of times for that segment. The CRN simulator is
            responsible for stitching the full trajectory.
        r_list : list[np.ndarray]
            Reference signals/targets for each scenario (see `performance_metric`).
        w : np.ndarray
            Weights for the tracking error (typically shape `(q, T_full)`), where
            `T_full` matches the concatenated time grid used in the simulation.
        norm : int, default=1
            Norm used in the tracking error (passed to `performance_metric`).
        relative : bool, default=False
            If True, compute a relative error (as supported by `performance_metric`).
        LARGE_NUMBER : float, default=1e4
            Divergence penalty passed to the simulator.

    Returns:
        performance : float
            Scalar tracking cost.
        last_task_info : dict
            Updated `crn.last_task_info` with reward metadata (same keys as
            `dynamic_tracking_error`).
    """
    t, x_list, y_list, last_task_info = crn.transient_response_piecewise(u_nested_list, x0_list, nested_time_horizon, LARGE_NUMBER=LARGE_NUMBER)
    performance = performance_metric(r_list, y_list, w, norm=norm, relative=relative)
    crn.last_task_info['reward'] = performance
    crn.last_task_info['setpoint'] = r_list
    crn.last_task_info['initial_conditions'] = x0_list
    crn.last_task_info['reward type'] = 'dynamic_tracking_error'
    return performance, crn.last_task_info


def oscillation_error(crn, u_list, x0_list, time_horizon, f_list=None, mean_list=None, w=[1/4, 1/4, 1/4, 1/4], t0=0, LARGE_NUMBER=1e4):
    """
    Compute an oscillation-shaping cost based on output time-series metrics.

    The CRN is simulated (as in `transient_response`), then oscillatory features
    are extracted via `oscillation_metrics`. The returned scalar cost is a weighted
    sum of several error components:

      - mean error (if `mean_list` is provided)
      - frequency error (if `f_list` is provided)
      - damping deviation from 1
      - peak ratio `r1` deviation from 1

    Args:
        crn : IOCRN
            IOCRN-like object implementing `transient_response`.
        u_list : list[np.ndarray]
            List of constant input vectors, each of shape `(p,)`.
        x0_list : list[np.ndarray]
            List of initial states, each of shape `(n,)`.
        time_horizon : np.ndarray
            1D array of evaluation times with shape `(T,)`.
        f_list : list[np.ndarray] or None, default=None
            Desired oscillation frequencies per scenario and output. Expected format
            follows `oscillation_metrics`. If None, frequency error is not included.
        mean_list : list[np.ndarray] or None, default=None
            Desired mean values per scenario and output (format follows `oscillation_metrics`).
            If None, mean error is not included.
        w : list[float], default=[1/4, 1/4, 1/4, 1/4]
            Weights `[mean_error, frequency_error, damping_error, r1_error]`.
        t0 : float, default=0
            Time threshold after which oscillation metrics are evaluated (to ignore transients).
        LARGE_NUMBER : float, default=1e4
            Divergence penalty passed to the simulator.

    Returns:
        performance : float
            Scalar oscillation cost.
        last_task_info : dict
            Updated `crn.last_task_info`, augmented with:

            - 'reward': performance
            - 'frequency': f_list
            - 'reward type': 'oscillation_error'
    """
    t, x_list, y_list, last_task_info = crn.transient_response(u_list, x0_list, time_horizon, LARGE_NUMBER=LARGE_NUMBER)
    frequency_error, mean_error, damping, r1, peaks_flag = oscillation_metrics(y_list, t, t0, f_list, mean_list)

    if mean_error is None:
        mean_error = 0.0
    if frequency_error is None:
        frequency_error = 0.0
        
    performance = w[0]*mean_error + w[1]*frequency_error + w[2]*np.abs(1 - damping) + w[3]*np.abs(1 - r1)

    crn.last_task_info['reward'] = performance
    crn.last_task_info['frequency'] = f_list
    crn.last_task_info['reward type'] = 'oscillation_error'
    return performance, crn.last_task_info

def logic_circuit_reward(crn, u_list, x0_list, time_horizon, r_list, w, norm=1, relative=False, LARGE_NUMBER=1e4):
    r"""
    Compute a steady-state logic circuit cost using binary cross-entropy (BCE).

    The CRN is simulated for each scenario. For each output trace, the final time
    point is treated as the steady-state output `y_ss` and compared against the
    target logic value `r` using BCE:
    
    $$\text{BCE}(r, y_ss) = - [ r \log(y_ss) + (1-r) \log(1-y_ss) ].$$

    Notes:
        - This function currently ignores `w`, `norm`, and `relative` (kept for API
        compatibility with tracking rewards).
        - Outputs are clipped to `[1e-6, 1-1e-6]` to avoid `log(0)`.

    Parameters:
        crn : IOCRN
            IOCRN-like object implementing `transient_response`.
        u_list : list[np.ndarray]
            List of constant inputs, each shape `(p,)`.
        x0_list : list[np.ndarray]
            List of initial states, each shape `(n,)`.
        time_horizon : np.ndarray
            1D array of evaluation times with shape `(T,)`.
        r_list : list[np.ndarray]
            List of desired binary targets per scenario. Each `r` is expected to have
            shape `(q,)` (one target per output).
        w : np.ndarray
            Unused (present for signature compatibility).
        norm : int
            Unused.
        relative : bool
            Unused.
        LARGE_NUMBER : float, default=1e4
            Divergence penalty passed to the simulator.

    Returns:
        performance : float
            Mean BCE across scenarios and outputs (lower is better).
        last_task_info : dict
            Updated `crn.last_task_info`, augmented with:

            - 'reward': performance
            - 'setpoint': r_list
            - 'initial_conditions': x0_list
            - 'reward type': 'dynamic_tracking_error'  (kept as-is; consider renaming to 'logic_circuit_reward')
    """

    t, x_list, y_list, last_task_info = crn.transient_response(u_list, x0_list, time_horizon, LARGE_NUMBER=LARGE_NUMBER)
    # use binary cross-entropy as performance metric
    scores = []
    for i in range(len(r_list)):
        r = r_list[i]
        y = y_list[i]
        # take the last time point as steady-state output
        y_ss = y[-1,:]
        # clip values to avoid log(0)
        y_ss = np.clip(y_ss, 1e-6, 1-1e-6)
        # compute binary cross-entropy
        bce = - (r * np.log(y_ss) + (1 - r) * np.log(1 - y_ss))
        scores.append(bce)
    performance = np.array(scores)
    performance = np.mean(performance)
    crn.last_task_info['reward'] = performance
    crn.last_task_info['setpoint'] = r_list
    crn.last_task_info['initial_conditions'] = x0_list
    crn.last_task_info['reward type'] = 'dynamic_tracking_error'
    return performance, crn.last_task_info



# For latch:

def dynamic_tracking_error_piecewise_logic(crn, u_nested_list, x0_list, nested_time_horizon, r_list, w, norm=1, relative=False, LARGE_NUMBER=1e4):
    r"""
    Compute a piecewise logic tracking cost using thresholded mismatch.

    This is intended for sequential / protocol-driven logic tasks (e.g. latches),
    where targets are specified as binary values and outputs are evaluated using a
    0.5 threshold across the entire time horizon (not only at steady state).

    Internally, the CRN is simulated with `transient_response_piecewise`, then
    the score is computed by `performance_metric_logic`:
    $\text{mean}(|1[r>0.5] - 1[y>0.5]|)$ over scenarios, outputs, and time.

    Parameters:
        crn : IOCRN
            IOCRN-like object implementing `transient_response_piecewise`.
        u_nested_list : list[list[np.ndarray]]
            List of input protocols (see `dynamic_tracking_error_piecewise`).
        x0_list : list[np.ndarray]
            Initial states, each shape `(n,)`.
        nested_time_horizon : list[np.ndarray]
            List of time grids per segment.
        r_list : list[np.ndarray]
            Target logic values per scenario, each expected shape `(q,)` or broadcastable
            to outputs (used as constant targets across time by `performance_metric_logic`).
        w : np.ndarray
            Unused (present for signature compatibility).
        norm : int
            Unused.
        relative : bool
            Unused.
        LARGE_NUMBER : float, default=1e4
            Divergence penalty passed to the simulator.

    Returns:
        performance : float
            Mean thresholded mismatch across scenarios, outputs, and time.
        last_task_info : dict
            Updated `crn.last_task_info` with reward metadata.
    """

    t, x_list, y_list, last_task_info = crn.transient_response_piecewise(u_nested_list, x0_list, nested_time_horizon, LARGE_NUMBER=LARGE_NUMBER)
    performance = performance_metric_logic(r_list, y_list)
    crn.last_task_info['reward'] = performance
    crn.last_task_info['setpoint'] = r_list
    crn.last_task_info['initial_conditions'] = x0_list
    crn.last_task_info['reward type'] = 'dynamic_tracking_error'
    return performance, crn.last_task_info


def performance_metric_logic(r_list, y_list):
    r"""
    Compute a binary (thresholded) mismatch score between targets and outputs.

    Targets `r_list` are treated as desired binary outputs (thresholded at 0.5).
    Outputs `y_list` are thresholded at 0.5 across all time points. The returned
    score is the mean absolute mismatch:
    $\text{mean}(|1[r>0.5] - 1[y>0.5]|)$
    averaged over scenarios, outputs, and time.

    Parameters:
        r_list : list[np.ndarray]
            List of reference logic targets, typically each of shape `(q,)` where `q`
            is the number of outputs.
        y_list : list[np.ndarray]
            List of output trajectories, each of shape `(q, T)`.

    Returns:
        float
            Mean mismatch rate in [0, 1], where 0 indicates perfect logic behavior.
    """
    
    # Check if dimensions match
    if len(r_list) != len(y_list):
        raise ValueError(f"Length of reference and output lists must match. Got {len(r_list)} and {len(y_list)}.")
    if r_list[0].shape[0] != y_list[0].shape[0]:
        raise ValueError("Reference signal and output must have the same number of dimensions (q).")
    
    # Convert lists to numpy arrays
    r_array = np.stack(r_list)   # shape (list_length, q)
    y_array = np.stack(y_list)   # shape (list_length, q, time_steps)

    # Compute logic error using 0.5 threshold
    error = np.abs((r_array[:, :, None] > 0.5).astype(float) - (y_array > 0.5).astype(float))
    
    return error.mean()



import numpy as np

def track_relationship(crn, u_list, x0_list, time_horizon, w, species_names, relationship_func, norm=1, LARGE_NUMBER=1e4):
    """
    Compute a cost for enforcing an algebraic relationship between species trajectories.

    This utility is for tasks where the objective is not tracking a pre-specified
    reference trajectory, but rather satisfying a constraint among species, e.g.

    $$A(t) - B(t) = 0,$$

    $$A(t) + B(t) - C(t) = 0,$$

    $$A(t)B(t) - C(t) = 0,$$
    
    etc.

    The user supplies `relationship_func`, which is called on the requested species
    trajectories and should return an *error signal* that is zero when the desired
    relationship holds. The function then aggregates the error into a scalar cost
    using a weighted L1 or L2 norm across time (and across scenarios).

    Parameters:
        crn : IOCRN
            IOCRN-like object implementing `transient_response`.
        u_list : list[np.ndarray]
            List of constant input vectors, each shape `(p,)`.
        x0_list : list[np.ndarray]
            List of initial conditions, each shape `(n,)`.
        time_horizon : np.ndarray
            1D time grid of shape `(T,)`.
        w : np.ndarray
            Weight array for the relationship error over time. Typically shape `(q_rel, T)`
            where `q_rel` is the output dimension of `relationship_func` (often 1).
            If `w` is 1D `(T,)`, it is treated as `(1, T)`.
        species_names : list[str]
            Names of species to feed into `relationship_func`, in the same order as the
            function arguments.
        relationship_func : callable
            Function mapping species trajectories to an error signal. It will be called as:
                relationship_func(traj_1, traj_2, ..., traj_N)
            where each `traj_i` has shape `(T,)`. The function should return either:

            - a 1D array `(T,)` (interpreted as a single error channel), or
            - a 2D array `(q_rel, T)` for multiple error channels.
        norm : int, default=1
            Norm for aggregation:

            1. mean weighted absolute error
            2. mean weighted squared error
        LARGE_NUMBER : float, default=1e4
            Divergence penalty passed to `transient_response`. If trajectories contain
            values near `LARGE_NUMBER`, the relationship error is set to `LARGE_NUMBER`
            to strongly penalize divergence.

    Returns:
        performance : float
            Scalar relationship-tracking cost.
        last_task_info : dict
            Updated `crn.last_task_info`, augmented with:

            - 'reward': performance
            - 'initial_conditions': x0_list
            - 'reward type': 'relationship_tracking'
            - 'tracked_species': species_names
    """

    # 1. Map Names to Indices
    try:
        species_indices = [crn.species_labels.index(name) for name in species_names]
    except ValueError as e:
        raise ValueError(f"Species name error: {e}")

    # 2. Run Simulation (Get full state x)
    # t: time vector
    # x_list: list of arrays, each shape (n_species, time_steps)
    t, x_list, _, last_task_info = crn.transient_response(u_list, x0_list, time_horizon, LARGE_NUMBER=LARGE_NUMBER)

    # 3. Compute the Relationship Error Signal
    error_list = []
    
    for x in x_list:
        # Extract the trajectories for the requested species
        # inputs will be a list of arrays [traj_A, traj_B, traj_C...]
        inputs = [x[idx, :] for idx in species_indices]
        
        # Apply the user's function. 
        # We assume the function returns the "error" (difference from target)
        # e.g., if we want A = B, func returns (A - B).
        computed_error = relationship_func(*inputs)
        
        # Ensure it has the right shape (q, time_steps). 
        # If the lambda returns a 1D array (time_steps,), we reshape to (1, time_steps)
        if computed_error.ndim == 1:
            computed_error = computed_error[None, :]

        # check if LARGE_NUMBER was returned (indicating divergence)
        if np.any(np.abs(x) >= LARGE_NUMBER - 1):
            computed_error = np.full_like(computed_error, LARGE_NUMBER)
            
        error_list.append(computed_error)

    # 4. Compute Performance (Weighted Norm of the Error)
    # Since r_list (target) is implicitly zero for this formulation, we just penalize 'error_list'
    
    # Convert to array for batch processing: (Batch, q, Time)
    error_array = np.stack(error_list)
    
    # Expand weights to match batch: (1, q, Time) -> (Batch, q, Time)
    # Ensure w is at least 2D (q, time)
    if w.ndim == 1:
        w = w[None, :]
    w_expanded = np.repeat(w[None, :, :], len(error_list), axis=0)

    match norm:
        case 1:
            performance = (w_expanded * np.abs(error_array)).mean()
        case 2:
            performance = (w_expanded * error_array**2).mean()
        case _:
            raise ValueError(f"Unsupported norm: {norm}")

    # 5. Log Info
    crn.last_task_info['reward'] = performance
    crn.last_task_info['initial_conditions'] = x0_list
    crn.last_task_info['reward type'] = 'relationship_tracking'
    crn.last_task_info['tracked_species'] = species_names
    
    return performance, crn.last_task_info


# --- habituation with gap ---

from typing import Sequence, Tuple, List, Dict, Any, Union



def habituation_metric_with_gap(
    *,
    intervals: Sequence[Tuple[float, float]],  # legacy (0..Tk)
    t: np.ndarray,
    y_list: Sequence[np.ndarray],              # each (q,T)
    w: Union[float, Sequence[float], np.ndarray],
    n_repeats_pre: int,
    n_repeats_post: int,
    gap_weight: float = 5.0,
    recovery_tol: float = 0.05,   # |pB1 - pA1| / pA1 <= 5%
    dishabituate_rho: float = 1.0, # require pB1 >= rho * pA_last (rho=1 is minimal)
    min_peak: float = 0.1,
    max_peak: float = 2.0,
    LARGE_NUMBER: float = 1e4,
    sensitization: bool = False, # if True, reward increasing response instead of decreasing
) -> float:
    """
    Returns: habituation loss + gap consistency penalty.
    Keeps your "log(max ratio)" style for habituation.
    """

    # Build absolute intervals from legacy durations
    durations = np.array([float(end - start) for (start, end) in intervals], dtype=float)  # start=0 legacy OK
    starts = np.cumsum(np.concatenate([[0.0], durations[:-1]]))
    ends = starts + durations
    abs_intervals = list(zip(starts, ends))

    # Segment layout:
    # pre: [ON0, OFF0, ON1, OFF1, ...] => stim segments at indices 0,2,...,2*(n_pre-1)
    # gap: one OFF segment at index 2*n_pre
    # post: resumes with ON at index 2*n_pre+1, then OFF, etc.
    gap_idx = 2 * n_repeats_pre
    stim_pre_idx  = list(range(0, 2 * n_repeats_pre, 2))
    stim_post_idx = list(range(gap_idx + 1, gap_idx + 1 + 2 * n_repeats_post, 2))

    eps = 1e-12
    all_scores: List[float] = []

    # weights for ratios (pre-train only; you can extend to post too)
    n_ratios = max(1, len(stim_pre_idx) - 1)
    if np.isscalar(w):
        w_arr = np.full(n_ratios, float(w), dtype=np.float32)
    else:
        w_arr = np.asarray(list(w), dtype=np.float32).reshape(-1)
        if w_arr.size < n_ratios:
            w_arr = np.pad(w_arr, (0, n_ratios - w_arr.size), mode="edge")
        elif w_arr.size > n_ratios:
            w_arr = w_arr[:n_ratios]

    for y in y_list:
        y0 = y[0] if y.ndim == 2 else y  # single output expected

        # --- extract peaks in each stim segment ---
        def seg_peak(seg_idx: int) -> float:
            start, end = abs_intervals[seg_idx]
            mask = (t >= start) & (t <= end)
            if not np.any(mask):
                return float(LARGE_NUMBER)
            return float(np.max(y0[mask]))

        peaks_pre  = [seg_peak(i) for i in stim_pre_idx]
        peaks_post = [seg_peak(i) for i in stim_post_idx]

        if any(p > max_peak for p in peaks_pre + peaks_post):
            return float(LARGE_NUMBER)

        peaks_pre  = [max(p, min_peak) for p in peaks_pre]
        peaks_post = [max(p, min_peak) for p in peaks_post]

        # --- habituation on pre train (your original max-ratio log) ---
        if not sensitization:
            ratios_pre = np.array(
                [peaks_pre[i+1] / max(peaks_pre[i], eps) for i in range(len(peaks_pre) - 1)],
                dtype=np.float32,
            )
            hab_score = float(np.log(float(np.median(ratios_pre)) + eps))
        else:
            ratios_pre = np.array(
                [peaks_pre[i] / max(peaks_pre[i+1], eps) for i in range(len(peaks_pre) - 1)],
                dtype=np.float32,
            )
            hab_score = float(np.log(float(np.median(ratios_pre)) + eps))

        # --- gap dynamics check ---
        # (1) recovery: first post peak close to first pre peak
        pA1 = peaks_pre[0]
        # pAlast = peaks_pre[-1]
        pB1 = peaks_post[0]

        rel_diff = abs(pB1 - pA1) / max(pA1, eps)
        rec_pen = max(0.0, rel_diff - recovery_tol) / max(recovery_tol, eps)

        # (2) dishabituation: first post peak should be == to the first pre peak
        dis_pen = np.abs(pB1 - pA1) / max(pA1, eps)

        gap_pen = rec_pen + dis_pen

        scenario_score = hab_score + gap_weight * gap_pen
        all_scores.append(float(scenario_score))

    return float(np.mean(all_scores)) if all_scores else float(LARGE_NUMBER)


