"""
Stochastic rewards (SSA).

This module defines reward / loss functions that evaluate IOCRN controllers under
intrinsic stochasticity by running Stochastic Simulation Algorithm (SSA) rollouts
via `IOCRN.transient_response_SSA`. The rewards are computed primarily from the
mean output trajectories, with optional robustness terms that penalize variability
(e.g., steady-state coefficient of variation), and they populate `crn.last_task_info`
with metadata for downstream logging and analysis.
"""

from RL4CRN.utils.utils import performance_metric
import numpy as np

def dynamic_tracking_error_SSA(crn, u_list, x0_list, time_horizon, r_list, w, 
                               n_trajectories=100, max_threads=10000, 
                               norm=1, relative=False, LARGE_NUMBER=1e4, LARGE_PENALTY=1e4):
    """
    Compute a dynamic tracking cost using stochastic simulation (SSA).

    This function evaluates tracking performance under intrinsic noise by running
    multiple SSA trajectories per (input, initial-condition) scenario and comparing
    the *mean* output trajectory against the reference `r_list` using
    `performance_metric`.

    The cost is computed on the mean trajectories only (variance is ignored here).
    If the SSA simulator reports divergence, a large constant penalty is returned.

    Parameters:
        crn : IOCRN
            IOCRN-like object implementing
            `transient_response_SSA(u_list, x0_list, time_horizon, n_trajectories=..., max_threads=..., max_value=...)`.
            Expected to populate `crn.last_task_info['has_diverged']`.
        u_list : list[np.ndarray]
            List of constant input vectors, each of shape `(p,)`.
        x0_list : list[np.ndarray]
            List of initial state vectors, each of shape `(n,)`.
        time_horizon : np.ndarray
            1D time grid (or a simulator-specific time specification) over which the
            SSA trajectories are sampled.
        r_list : list[np.ndarray]
            List of reference targets per scenario, in the same format expected by
            `performance_metric` (commonly `(q, T)` trajectories or `(q,)` setpoints).
        w : np.ndarray
            Weights for the tracking metric. Typically shape `(q, T)` (or compatible
            with `performance_metric`).
        n_trajectories : int, default=100
            Number of SSA trajectories simulated per scenario.
        max_threads : int, default=10000
            Upper bound on GPU threads / parallelism for SSA (passed through to the CRN).
        norm : int, default=1
            Norm used by `performance_metric` (commonly 1 for L1, 2 for L2 / squared error).
        relative : bool, default=False
            If True, compute a relative error (as supported by `performance_metric`).
        LARGE_NUMBER : float, default=1e4
            Maximum value / divergence threshold passed to the SSA simulator.
        LARGE_PENALTY : float, default=1e4
            Returned cost when the simulator indicates divergence.

    Returns:
        performance : float
            Scalar tracking cost (lower is better).
        last_task_info : dict
            Updated `crn.last_task_info`, augmented with:
            - 'reward': performance
            - 'setpoint': r_list
            - 'reward type': 'dynamic_tracking_error_SSA'
            - 'weights': w
    """

    # 1. Run Stochastic Simulation
    # Unpack the extended return values from the SSA version
    # We ignore x_std_list and y_std_list for the error calculation as requested
    (time_horizon, x_mean_list, y_mean_list, 
     x_std_list, y_std_list, last_task_info) = crn.transient_response_SSA(
        u_list, x0_list, time_horizon, 
        n_trajectories=n_trajectories, 
        max_threads=max_threads,
        max_value=LARGE_NUMBER
    )

    # 2. Compute Performance Metric
    # We compare the Mean Trajectory (y_mean_list) against the Reference (r_list)
    if not crn.last_task_info['has_diverged']:
        performance = performance_metric(r_list, y_mean_list, w, norm=norm, relative=relative)
    else:
        performance = LARGE_PENALTY

    # 3. Update Metadata
    # The SSA method already populates 'inputs', 'trajectories', etc.
    # We add the tracking-specific fields.
    crn.last_task_info['reward'] = performance
    crn.last_task_info['setpoint'] = r_list
    crn.last_task_info['reward type'] = 'dynamic_tracking_error_SSA'
    
    # Optional: You might want to store the weights used
    crn.last_task_info['weights'] = w

    return performance, crn.last_task_info


import numpy as np

def robust_tracking_loss_SSA(crn, u_list, x0_list, time_horizon, r_list, w, 
                             n_trajectories=100, max_threads=10000, 
                             norm=2, relative=False, 
                             LARGE_NUMBER=1e4, LARGE_PENALTY=1e4,
                             lambda_std=0.5,
                             rpa_weight=1.0,
                             cv_weight=1.0):
    """
    Compute a robustness-aware tracking loss under SSA.

    This loss combines:
    
    1. an accuracy term (tracking error on the *mean* trajectory), and
    2. a precision term (steady-state coefficient-of-variation penalty).

    The accuracy term is computed via `performance_metric(r_list, y_mean_list, w, ...)`.
    The precision term uses the coefficient of variation (CV = std / |mean|) computed
    from the SSA output standard deviation and mean in time regions where the weight
    vector `w` is positive (interpreted here as the “steady-state” window).

    The final loss is:
        
        loss = rpa_weight * base_error + cv_weight * (lambda_std * mean_cv)

    Notes:
        - This implementation assumes a single-output layout sometimes produced as
        `(B, 1, T)`; if so, it squeezes the singleton dimension to `(B, T)`.
        - If `crn.last_task_info['has_diverged']` is True, the loss components are
        overridden with a large penalty / safe defaults.

    Parameters:
        crn : IOCRN
            IOCRN-like object implementing `transient_response_SSA`.
        u_list : list[np.ndarray]
            List of constant input vectors, each of shape `(p,)`.
        x0_list : list[np.ndarray]
            List of initial state vectors, each of shape `(n,)`.
        time_horizon : np.ndarray
            Time grid passed to SSA.
        r_list : list[np.ndarray]
            Reference targets per scenario (format expected by `performance_metric`).
        w : np.ndarray
            Weights used for the tracking error and to identify the steady-state window.
            For the CV penalty, `w` is flattened and indices with `w > 0` are used.
        n_trajectories : int, default=100
            Number of SSA trajectories per scenario.
        max_threads : int, default=10000
            Upper bound on GPU threads / parallelism for SSA.
        norm : int, default=2
            Norm used by `performance_metric` for the accuracy term.
        relative : bool, default=False
            If True, compute a relative tracking error (as supported by `performance_metric`).
        LARGE_NUMBER : float, default=1e4
            Maximum value / divergence threshold passed to the SSA simulator.
        LARGE_PENALTY : float, default=1e4
            Penalty used when divergence is detected.
        lambda_std : float, default=0.5
            Scaling factor applied to the mean CV penalty.
        rpa_weight : float, default=1.0
            Weight multiplying the accuracy (mean-trajectory tracking) term.
        cv_weight : float, default=1.0
            Weight multiplying the precision (CV) term.

    Returns:
        performance : float
            Scalar robustness-aware loss (lower is better).
        last_task_info : dict
            Updated `crn.last_task_info`, augmented with:
            
            - 'reward': performance
            - 'setpoint': r_list
            - 'reward type': 'dynamic_tracking_error_SSA'
            - 'loss_components': {'error': ..., 'cv_penalty': ...}
    """

    # 1. Run Stochastic Simulation
    (t_steps_out, x_mean_list, y_mean_list, 
     x_std_list, y_std_list, last_task_info) = crn.transient_response_SSA(
        u_list, x0_list, time_horizon, 
        n_trajectories=n_trajectories, 
        max_threads=max_threads,
        max_value=LARGE_NUMBER
    )

    # for debug, print all the shapes
    # Debug: y_mean_list shape: (9, 1, 100), y_std_list shape: (9, 1, 100)
    # print(f"Debug: y_mean_list shape: {np.array(y_mean_list).shape}, y_std_list shape: {np.array(y_std_list).shape}")

    # --- A. Accuracy (Base Error) ---
    base_error = performance_metric(r_list, y_mean_list, w, norm=norm, relative=relative)

    # --- B. Precision (CV Penalty) ---
    
    # 1. Convert to Arrays
    y_mean_arr = np.array(y_mean_list) # Shape: (9, 1, 100)
    y_std_arr = np.array(y_std_list)   # Shape: (9, 1, 100)
    w_arr = np.array(w).flatten()      # Shape: (100,)

    # 2. FIX THE SHAPE: Remove the middle dimension
    # If shape is (N, 1, T), we want (N, T)
    if y_mean_arr.ndim == 3 and y_mean_arr.shape[1] == 1:
        y_mean_arr = y_mean_arr.squeeze(1)
        y_std_arr = y_std_arr.squeeze(1)
        
    # 3. Identify Steady State Indices
    # valid_indices will contain [40, 41, ... 99]
    valid_indices = np.where(w_arr > 0)[0]

    if len(valid_indices) == 0:
        L_precision = 0.0
    else:
        # 4. Extract Steady State Data
        # Now y_mean_arr is (9, 100), so we can slice along axis 1
        ss_means = y_mean_arr[:, valid_indices]
        ss_stds  = y_std_arr[:, valid_indices]

        # 5. Calculate CV
        epsilon = 1e-6
        cv_matrix = ss_stds / (np.abs(ss_means) + epsilon)
        mean_cv = np.mean(cv_matrix)
        
        L_precision = lambda_std * mean_cv

    performance = rpa_weight * base_error + cv_weight * L_precision

    # 3. Update Metadata
    crn.last_task_info['reward'] = performance
    crn.last_task_info['setpoint'] = r_list
    crn.last_task_info['reward type'] = 'dynamic_tracking_error_SSA'
    
    crn.last_task_info['loss_components'] = {
        'error': float(performance - L_precision) if not crn.last_task_info['has_diverged'] else LARGE_PENALTY,
        'cv_penalty': float(L_precision) if not crn.last_task_info['has_diverged'] else 0.0
    }
    
    return performance, crn.last_task_info