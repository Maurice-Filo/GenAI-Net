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