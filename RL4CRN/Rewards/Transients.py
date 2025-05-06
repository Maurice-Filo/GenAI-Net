from RL4CRN.Utils.Utils import unzip
import numpy as np

def performance_metric(r, y):
    # return np.linalg.norm(r - y) / np.sqrt(len(y)) # + np.abs(y[-1] - r) 
    y = np.transpose(y, (1, 0, 2))
    weight = np.ones(y.shape[0])
    weight[(len(weight)//5)*4:] = weight[(len(weight)//5)*4:]*2
    weight[:(len(weight)//5)] = weight[:(len(weight)//5)]*0.25

    # reshape weight to match the shape of y
    weight = np.repeat(weight, y.shape[1]*y.shape[2]).reshape(y.shape[0], y.shape[1], y.shape[2])

    return (weight*np.abs(r - y)).mean()

def ss_metric(r, y_ss):
    return np.abs(r - y_ss)

def dynamic_tracking_error(crn, inputs, initial_condition, time_horizon, r, threshold=1000):
    y, x = crn.transient_response(inputs, initial_condition, time_horizon, return_states=True)
    y = np.minimum(y, threshold)
    y = np.maximum(y,0)
    y[np.isnan(y)] = threshold
    y[np.isinf(y)] = threshold
    return performance_metric(r, y), y, time_horizon