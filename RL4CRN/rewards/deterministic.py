from RL4CRN.utils.utils import performance_metric
from RL4CRN.utils.utils import oscillation_metrics
import numpy as np

def dynamic_tracking_error(crn, u_list, x0_list, time_horizon, r_list, w, norm=1, relative=False, LARGE_NUMBER=1e4):
    """ Computes the dynamic tracking error for an IOCRN given a list of control inputs, initial states, and reference signals.
    Args:
        crn: An IOCRN object with a transient_response method.
        u_list: A list of control inputs, each of shape (p,).
        x0_list: A list of initial states, each of shape (n,).
        time_horizon: The time horizon for the transient response.
        r_list: A list of reference signals, each of shape (q,).
        w: A numpy array of weights, shape (q, time_steps).
        norm: An integer indicating the norm to use for the metric calculation (1 or 2).
        relative: A boolean indicating whether to compute relative error.
        LARGE_NUMBER: A large number to handle cases where the CRN does not converge.
    Returns:
        performance: A float representing the computed performance metric.
        last_task_info: A dictionary containing the last task information, including the reward and setpoint. """

    t, x_list, y_list, last_task_info = crn.transient_response(u_list, x0_list, time_horizon, LARGE_NUMBER=LARGE_NUMBER)
    performance = performance_metric(r_list, y_list, w, norm=norm, relative=relative)
    crn.last_task_info['reward'] = performance
    crn.last_task_info['setpoint'] = r_list
    crn.last_task_info['reward type'] = 'dynamic_tracking_error'
    return performance, crn.last_task_info

def oscillation_error(crn, u_list, x0_list, time_horizon, f_list, w, t0, LARGE_NUMBER=1e4):
    t, x_list, y_list, last_task_info = crn.transient_response(u_list, x0_list, time_horizon, LARGE_NUMBER=LARGE_NUMBER)
    frequency_error, damping, r1, peaks_flag = oscillation_metrics(f_list, y_list, t, t0)

    performance = w[0] * frequency_error + w[1] * np.abs(1 - damping) + w[2] * np.abs(1 - r1)

    crn.last_task_info['reward'] = performance
    crn.last_task_info['frequency'] = f_list
    crn.last_task_info['reward type'] = 'oscillation_error'
    return performance, crn.last_task_info