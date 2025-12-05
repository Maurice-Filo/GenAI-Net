from RL4CRN.utils.utils import performance_metric
import numpy as np

def dynamic_tracking_error_SSA(crn, u_list, x0_list, time_horizon, r_list, w, 
                               n_trajectories=100, max_threads=10000, 
                               norm=1, relative=False, LARGE_NUMBER=1e4, LARGE_PENALTY=1e4):
    """ 
    Computes the dynamic tracking error for an IOCRN using Stochastic Simulation (SSA).
    The error is calculated based on the MEAN trajectory of the stochastic simulations.
    
    Args:
        crn: An IOCRN object with a transient_response_SSA method.
        u_list: A list of control inputs.
        x0_list: A list of initial states.
        time_horizon: The time horizon.
        r_list: A list of reference signals (setpoints).
        w: Weights for the error metric.
        n_trajectories: Number of SSA trajectories per configuration.
        max_threads: Max GPU threads.
        norm: 1 or 2 (L1 or L2 norm).
        relative: Boolean for relative error.
        LARGE_NUMBER: Penalty for instability.
        
    Returns:
        performance: Float representing the error metric.
        last_task_info: Updated dictionary with simulation details.
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
                             lambda_std=0.5):

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

    performance = base_error + L_precision

    # 3. Update Metadata
    crn.last_task_info['reward'] = performance
    crn.last_task_info['setpoint'] = r_list
    crn.last_task_info['reward type'] = 'dynamic_tracking_error_SSA'
    
    crn.last_task_info['loss_components'] = {
        'error': float(performance - L_precision) if not crn.last_task_info['has_diverged'] else LARGE_PENALTY,
        'cv_penalty': float(L_precision) if not crn.last_task_info['has_diverged'] else 0.0
    }
    
    return performance, crn.last_task_info