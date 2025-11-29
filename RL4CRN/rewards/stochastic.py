from RL4CRN.utils.utils import performance_metric

def dynamic_tracking_error_SSA(crn, u_list, x0_list, time_horizon, r_list, w, 
                               n_trajectories=100, max_threads=10000, 
                               norm=1, relative=False, LARGE_NUMBER=1e4):
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
    performance = performance_metric(r_list, y_mean_list, w, norm=norm, relative=relative)

    # 3. Update Metadata
    # The SSA method already populates 'inputs', 'trajectories', etc.
    # We add the tracking-specific fields.
    crn.last_task_info['reward'] = performance
    crn.last_task_info['setpoint'] = r_list
    crn.last_task_info['reward type'] = 'dynamic_tracking_error_SSA'
    
    # Optional: You might want to store the weights used
    crn.last_task_info['weights'] = w

    return performance, crn.last_task_info