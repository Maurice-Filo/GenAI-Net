"""
General-purpose utilities for RL4CRN.

This module collects small, reusable helpers that are shared across the project:

- scalar performance metrics for tracking tasks (`performance_metric`)
- batch encodings for discrete selections (`batch_multi_hot`)
- combinatorial helpers (`cartesian_prod`)
- convenience printing (`print_task_info`)
- oscillation-specific diagnostics (`oscillation_metrics`)

The functions are intentionally framework-light (NumPy/Torch/Scipy only) and are
written to be called from environments, policies, and evaluation scripts.
"""

import torch
import numpy as np
from scipy.signal import find_peaks

def performance_metric(r_list, y_list, w, norm=1, relative=False):
    """
    Compute a weighted tracking error between reference signals and output trajectories.

    This metric compares a list of reference vectors ``r_list`` (one per rollout/task)
    to a list of output trajectories ``y_list`` over time, and aggregates the weighted
    error across batch, outputs, and time.

    Args:
        r_list (list[np.ndarray]): List of reference/setpoint vectors.
            Each element has shape ``(q,)`` where ``q`` is the number of measured outputs.
        y_list (list[np.ndarray]): List of output trajectories.
            Each element has shape ``(q, T)`` where ``T`` is the number of time steps.
        w (np.ndarray): Weights over outputs and time with shape ``(q, T)``.
            Larger weights penalize errors more strongly at the corresponding output/time.
        norm (int): Error norm to use:

            - ``1``: L1 error (mean absolute error)
            - ``2``: L2 error (mean squared error)
        relative (bool): If True, compute relative error per component:
            ``(r - y) / max(|r|, 1e-6)``.
            Useful when outputs have different scales.

    Returns:
        float: Scalar performance value (lower is better).

    Raises:
        ValueError: If list lengths or signal dimensions are inconsistent.
    """
    
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
    """
    Convert a padded batch of indices into a multi-hot encoding (optionally with intensities).

    Given an integer array of shape ``(B, R)`` containing up to ``R`` indices per batch element,
    this function produces a tensor of shape ``(B, num_classes)`` where each selected index is 1.
    Padded entries (equal to ``pad_val``) are ignored.

    If ``intensities`` is provided (aligned with ``indices``), an additional tensor is returned
    where selected positions store the provided intensity values.

    Args:
        indices (np.ndarray): Integer array of shape ``(B, R)`` containing indices.
            Entries equal to ``pad_val`` are treated as padding.
        num_classes (int): Total number of categories in the multi-hot representation.
        intensities (np.ndarray | None): Optional float array of shape ``(B, R)`` containing
            values associated with each index (e.g., weights). Must align with ``indices``.
        device (torch.device | None): Optional device for the returned tensors.
        pad_val (int): Padding value in ``indices`` to ignore.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

            - If ``intensities is None``: ``multi_hot`` tensor of shape ``(B, num_classes)``.
            - If ``intensities`` provided: ``(multi_hot, intensity_tensor)``, both float tensors
              of shape ``(B, num_classes)``.

    Notes:
        If the same index appears multiple times in a row of ``indices``, the multi-hot
        entry will be set to 1 (no counting). For intensities, later assignments overwrite
        earlier ones due to direct indexing.
    """
    
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
    """
    Compute the cartesian product of 1D arrays.

    Args:
        arrays (Sequence[np.ndarray]): Non-empty list/tuple of 1D arrays. Each input is flattened.
        dtype (np.dtype | None): Optional dtype to cast the output to.

    Returns:
        np.ndarray: Array of shape ``(Π_i len(arrays[i]), len(arrays))`` where each row is one
        combination of values, ordered in 'ij' meshgrid order.

    Raises:
        ValueError: If ``arrays`` is empty.

    Notes:
        If any input array is empty, the result is an empty array with the correct
        number of columns.
    """
    
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
    """
    Pretty-print the contents of an environment/task info dictionary.

    Args:
        last_task_info (dict): Dictionary containing metadata about the last task/simulation
            (e.g., reward, setpoints, trajectories, diagnostics).
        mode (str): Printing mode:

            - ``'sizes'``: print key, type, and shape/size summaries (default).
            - otherwise: print full values (can be verbose).

    Returns:
        None
    """
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
    """
    Compute oscillation diagnostics from output trajectories.

    The function extracts peaks to estimate frequency and a simple damping metric,
    computes mean values, and estimates a periodicity index based on the first
    nonzero-lag local maximum of the normalized autocorrelation.

    Args:
        y_list (list[np.ndarray]): List of output trajectories. Each entry is expected to have
            shape ``(1, T)`` (single-output signal). (If you have multiple outputs, pass
            them as separate list entries.)
        t (np.ndarray): Time vector of shape ``(T,)``.
        t0 (float): Start time for analysis. Only samples with ``t >= t0`` are considered
            for peak detection and statistics.
        f_list (list[float] | None): Desired frequencies for each output (same length as ``y_list``).
            If provided, the function returns a relative frequency error; otherwise returns None.
        mean_list (list[float] | None): Desired means for each output (same length as ``y_list``).
            If provided, the function returns a relative mean error; otherwise returns None.

    Returns:
        tuple:

            - `frequency_error` (float | None): Mean relative frequency error across outputs,
              or None if ``f_list`` is not provided.
            - `mean_error` (float | None): Mean relative mean error across outputs,
              or None if ``mean_list`` is not provided.
            - `damping` (float): Average damping metric across outputs. Computed from ratios of
              successive peak heights; 0 if insufficient peaks.
            - `r1` (float): Average periodicity index across outputs, defined as the value of the
              first nonzero-lag local maximum of the normalized autocorrelation.
            - `peaks_flag` (bool): True if at least two peaks were found for *every* output;
              False otherwise.

    Raises:
        ValueError: If input dimensions are inconsistent.

    Notes:
        - Peak prominence is chosen adaptively as ``max(0.01, 0.05 * dynamic_range)`` per signal.
        - If a signal is nearly constant (autocorrelation ill-defined), its r1 contribution is 0.
    """
    
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

def compute_oscillation_specs(y_list, t, t0):
    """ Computes the fundamental frequencies of oscillations in the output signals.
    Args:
    - y_list: A list of outputs, each of shape (1, time_steps).
    - t: A 1D numpy array representing the time vector.
    - t0: A float representing the time after which to start considering peaks.
    Returns:
    - estimated_frequencies: A numpy array of estimated frequencies, shape (list_length, 1).
    - damping_metrics: A numpy array of damping metrics, shape (list_length, 1).
    - amplitude: A numpy array of amplitudes, shape (list_length, 1).
    - mean: A numpy array of mean values, shape (list_length, 1).
    - periodicity_index: A numpy array of periodicity indices, shape (list_length, 1).
    """
    
    # Check if dimensions match
    if 1 != y_list[0].shape[0]:
            raise ValueError("Reference signal and output must have the same number of dimensions.")
    
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
    amplitude = []
    peaks_flag = True
    for peaks_indices, y in zip(peaks_indices_list, y_list):
        if len(peaks_indices) < 2:
            estimated_frequencies.append(0.0)
            damping_metrics.append(0.0)
            amplitude.append(0.0)
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
            avg_amplitude = np.mean(peak_heights)
            amplitude.append(avg_amplitude)

    # Compute the mean and periodicity index
    periodicity_index = []
    mean = []
    for y in y_list:
        mean.append(np.mean(np.squeeze(y)))
        x = np.squeeze(y) - np.mean(y)
        if np.allclose(x, 0.0, atol=1e-2):
            periodicity_index.append(0.0)
            continue

        R_full = np.correlate(x, x, mode='full')
        mid = len(R_full) // 2

        if R_full[mid] <= 0:
            periodicity_index.append(0.0)
            continue

        R = R_full[mid:] / R_full[mid]  # normalize so R[0] = 1

        # Find first nonzero-lag local maximum
        if len(R) < 3:
            periodicity_index.append(0.0)
            continue

        # Ignore lag=0
        R_search = R[1:]
        if len(R_search) < 3:
            periodicity_index.append(0.0)
            continue
        
        # Find indices where R[i-1] < R[i] > R[i+1]
        candidates = np.where((R_search[1:-1] > R_search[:-2]) &
                            (R_search[1:-1] > R_search[2:]))[0] + 1
        if len(candidates) == 0:
            periodicity_index.append(0.0)
            continue

        # Choose the first such peak
        tau1 = candidates[0] + 1
        r1 = R[tau1]
        periodicity_index.append(r1)


    estimated_frequencies = np.array(estimated_frequencies).reshape(-1, 1)  # shape (list_length, 1)
    damping_metrics = np.array(damping_metrics).reshape(-1, 1)  # shape (list_length, 1)
    amplitude = np.array(amplitude).reshape(-1, 1)  # shape (list_length, 1)
    mean = np.array(mean).reshape(-1, 1)  # shape (list_length, 1)
    periodicity_index = np.array(periodicity_index).reshape(-1, 1)  # shape (list_length, 1)

    return estimated_frequencies, damping_metrics, amplitude, mean, periodicity_index