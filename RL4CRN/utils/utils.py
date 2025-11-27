import torch
import numpy as np
from scipy.signal import find_peaks

def performance_metric(r_list, y_list, w, norm=1, relative=False):
    """ Computes the performance metric based on the difference between reference signal r and output y.
    Args:
        r_list: A list of reference signals, each of shape (q,).
        y_list: A list of outputs, each of shape (q, time_steps).
        w: A numpy array of weights, shape (q, time_steps).
        norm: An integer indicating the norm to use for the metric calculation.
        relative: A boolean indicating whether to compute relative error.
    Returns:
        float: Computed performance metric. """
    
    # Check if dimensions match
    if len(r_list) != len(y_list):
        raise ValueError(f"Length of reference and output lists must match. Got {len(r_list)} and {len(y_list)}.")
    if r_list[0].shape[0] != y_list[0].shape[0]:
        raise ValueError("Reference signal and output must have the same number of dimensions (q).")
    
    # Convert lists to numpy arrays
    r_array = np.stack(r_list)   # shape (list_length, q)
    y_array = np.stack(y_list)   # shape (list_length, q, time_steps)

    # Compute the error and apply weights
    error = r_array[:,:,None] - y_array if not relative else (r_array[:,:,None] - y_array) / np.maximum(np.abs(r_array[:,:,None]), 1e-6)
    w = np.repeat(w[None, :, :], len(y_list), axis=0)
    match norm:
        case 1:
            return (w * np.abs(error)).mean()
        case 2:
            return (w * error**2).mean()
        case _:
            raise ValueError(f"Unsupported norm: {norm}")

def batch_multi_hot(indices, num_classes, intensities=None, device=None, pad_val=0):
    """ Converts (B, R) numpy arrays of indices and intensities into:
    - (B, num_classes) multi-hot tensor
    - (B, num_classes) intensity tensor if intensities are provided.

    Args:
        indices (np.ndarray): (B, R) integer indices, padded with pad_val.
        num_classes (int): Total number of possible categories.
        intensities (np.ndarray): (B, R) float intensities, aligned with indices.
        device (torch.device or None): Optional torch device.
        pad_val (int): Value used for padding in indices.

    Returns:
        (multi_hot, intensity_tensor): both torch.FloatTensors of shape (B, num_classes). """
    
    batch_size, num_reactions = indices.shape
    valid_mask = indices != pad_val
    row_indices = np.repeat(np.arange(batch_size), num_reactions)[valid_mask.ravel()]
    col_indices = indices[valid_mask]
    multi_hot = torch.zeros((batch_size, num_classes), dtype=torch.float32, device=device)
    multi_hot[row_indices, col_indices] = 1.0
    if intensities is not None:
        values = intensities[valid_mask]
        intensity_tensor = torch.zeros_like(multi_hot)
        intensity_tensor[row_indices, col_indices] = torch.tensor(values, dtype=torch.float32, device=device)
        return multi_hot, intensity_tensor
    else:
        return multi_hot
    
def cartesian_prod(arrays, *, dtype=None):
    """ arrays: list/tuple of 1D numpy arrays.
    returns: (prod(len(a) for a in arrays), len(arrays)) ndarray. """
    
    arrays = [np.asarray(a).ravel() for a in arrays]
    if not arrays:
        raise ValueError("arrays must be a non-empty list of 1D arrays")

    # If any input is empty, the product is empty with the right number of columns
    if any(a.size == 0 for a in arrays):
        n = len(arrays)
        dt = dtype if dtype is not None else np.result_type(*arrays)
        return np.empty((0, n), dtype=dt)

    grids = np.meshgrid(*arrays, indexing='ij')
    out = np.stack(grids, axis=-1).reshape(-1, len(arrays))
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out
    
def print_task_info(last_task_info, mode='sizes'):
    """ Prints the information of the last task performed on the IOCRN.
    Arguments:
    - last_task_info: A dictionary containing the information of the last task performed.
    - mode: A string indicating the mode of printing. It can be 'sizes' to print the sizes and types of the last task information, or 'values' to print the values of the last task information.
    If no task has been performed yet, it prints a message indicating that. """
    if not last_task_info:
        print("No task has been performed yet.")
        return
    
    if mode == 'sizes':
        for key, value in last_task_info.items():
            value_type = type(value).__name__
            if isinstance(value, list):
                value_size = len(value) 
                if all(isinstance(v, np.ndarray) for v in value):
                    shapes = [v.shape for v in value]
                    if all(shape == shapes[0] for shape in shapes):
                        array_shape = shapes[0]
                    else:
                        array_shape = "Variable shapes"
                    print(f"{key} --- Type: {value_type} of numpy arrays, List size: {value_size}, Numpy Arrays shape: {array_shape}")
                else:
                    print(f"{key} --- Type: {value_type}, Size: {value_size}")
            elif isinstance(value, np.ndarray):
                print(f"{key} --- Type: {value_type}, Shape: {value.shape}")
            else:
                print(f"{key} --- Type: {value_type}, Value: {value}")
    else:
        for key, value in last_task_info.items():
            print(f"{key}: {value}")

def oscillation_metrics(y_list, t, t0, f_list=None, mean_list=None):
    """ Computes oscillation metrics: frequency error, damping metric, periodicity index, and means.
    Args:
    - y_list: A list of outputs, each of shape (1, time_steps).
    - t: A 1D numpy array representing the time vector.
    - t0: A float representing the time after which to start considering peaks.
    - f_list: A list of frequencies.
    - mean_list: A list of desired mean values for each output.
    Returns:
    - frequency_error: A float representing the mean absolute error between desired and estimated frequencies.
    - avg_damping_metric: A float representing the average damping metric across all outputs. 
    - periodicity_index: A float representing the average periodicity index across all outputs.
    - peaks_flag: A boolean indicating if peaks were found for all outputs. """
    
    # Check if dimensions match
    if 1 != y_list[0].shape[0]:
            raise ValueError("Reference signal and output must have the same number of dimensions.")
    
    if f_list is not None:
        if len(f_list) != len(y_list):
            raise ValueError(f"Length of frequency and output lists must match. Got {len(f_list)} and {len(y_list)}.")
        f_array = np.stack(f_list)   # shape (list_length,)

    if mean_list is not None:
        if len(mean_list) != len(y_list):
            raise ValueError(f"Length of mean and output lists must match. Got {len(mean_list)} and {len(y_list)}.")
        mean_array = np.stack(mean_list)   # shape (list_length,)
    
    # Focus on the time after t0
    time_mask = t >= t0
    t = t[time_mask]
    y_list = [y[:, time_mask] for y in y_list]

    # Compute the peaks of the output signals and the temporal means
    peaks_indices_list = []
    y_mean_list = []
    for y in y_list:
        yy = np.squeeze(y)
        dyn = float(np.max(yy) - np.min(yy)) if yy.size else 0.0
        prom = max(0.01, 0.05 * dyn)  # 5% of dynamic range (with epsilon floor)
        peaks_indices_list.append(find_peaks(yy, prominence=prom)[0])
        y_mean_list.append(np.mean(yy))

    # Compute the frequencies from the peaks
    estimated_frequencies = []
    damping_metrics = []
    peaks_flag = True
    for peaks_indices, y in zip(peaks_indices_list, y_list):
        if len(peaks_indices) < 2:
            estimated_frequencies.append(0.0)
            damping_metrics.append(0.0)
            peaks_flag = False
        else:
            peak_times = t[peaks_indices]
            periods = np.diff(peak_times)
            avg_period = np.mean(periods)
            estimated_frequencies.append(1.0 / avg_period if avg_period > 0 else 0.0)

            peak_heights = y[0, peaks_indices]
            decrements = peak_heights[:-1] / peak_heights[1:]
            avg_decrement = np.mean(decrements)
            damping_metrics.append(avg_decrement)
    estimated_frequencies = np.array(estimated_frequencies).reshape(-1, 1)  # shape (list_length, 1)
    damping_metrics = np.array(damping_metrics).reshape(-1, 1)  # shape (list_length, 1)

    # Compute the relative frequency error
    if f_list is not None:
        frequency_error = np.mean(np.abs(f_array - estimated_frequencies)/ f_array)
    else:
        frequency_error = None

    # Compute the relative means error if mean_list is provided
    if mean_list is not None:
        y_mean_array = np.stack(y_mean_list)   # shape (list_length,)
        mean_error = np.mean(np.abs(y_mean_array - mean_array) / np.maximum(np.abs(mean_array), 1e-6))
    else:
        mean_error = None

    # Compute the average damping metric
    damping = np.mean(damping_metrics) 

    # Compute the periodicity index
    r1_list = []
    for y in y_list:
        x = np.squeeze(y) - np.mean(y)
        if np.allclose(x, 0.0, atol=1e-2):
            r1_list.append(0.0)
            continue

        R_full = np.correlate(x, x, mode='full')
        mid = len(R_full) // 2

        if R_full[mid] <= 0:
            r1_list.append(0.0)
            continue

        R = R_full[mid:] / R_full[mid]  # normalize so R[0] = 1

        # Find first nonzero-lag local maximum
        if len(R) < 3:
            r1_list.append(0.0)
            continue

        # Ignore lag=0
        R_search = R[1:]
        if len(R_search) < 3:
            r1_list.append(0.0)
            continue
        
        # Find indices where R[i-1] < R[i] > R[i+1]
        candidates = np.where((R_search[1:-1] > R_search[:-2]) &
                            (R_search[1:-1] > R_search[2:]))[0] + 1
        if len(candidates) == 0:
            r1_list.append(0.0)
            continue

        # Choose the first such peak
        tau1 = candidates[0] + 1
        r1 = R[tau1]
        r1_list.append(r1)

    r1 = np.mean(r1_list) if len(r1_list) else 0.0
    return frequency_error, mean_error, damping, r1, peaks_flag