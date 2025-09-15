import torch
import numpy as np

def performance_metric(r_list, y_list, w, norm=1):
    """
    Computes the performance metric based on the difference between reference signal r and output y.
    Args:
        r: A list of reference signals, each of shape (q,).
        y: A list of outputs, each of shape (q, time_steps).
        w: A numpy array of weights, shape (q, time_steps).
        norm: An integer indicating the norm to use for the metric calculation.
    Returns:
        float: Computed performance metric.
    """
    # Check if dimensions match
    if len(r_list) != len(y_list):
        raise ValueError("Length of reference and output lists must match.")
    if r_list[0].shape[0] != y_list[0].shape[0]:
        raise ValueError("Reference signal and output must have the same number of dimensions (q).")
    
    # Convert lists to numpy arrays
    r_array = np.stack(r_list)   # shape (list_length, q)
    y_array = np.stack(y_list)   # shape (list_length, q, time_steps)

    # Compute the error and apply weights
    error = r_array[:,:,None] - y_array
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
    Converts (B, R) numpy arrays of indices and intensities into:
    - (B, num_classes) multi-hot tensor
    - (B, num_classes) intensity tensor if intensities are provided.

    Args:
        indices (np.ndarray): (B, R) integer indices, padded with pad_val.
        num_classes (int): Total number of possible categories.
        intensities (np.ndarray): (B, R) float intensities, aligned with indices.
        device (torch.device or None): Optional torch device.
        pad_val (int): Value used for padding in indices.

    Returns:
        (multi_hot, intensity_tensor): both torch.FloatTensors of shape (B, num_classes)
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
    arrays: list/tuple of 1D numpy arrays
    returns: (prod(len(a) for a in arrays), len(arrays)) ndarray
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
    Prints the information of the last task performed on the IOCRN.
    Arguments:
    - last_task_info: A dictionary containing the information of the last task performed.
    - mode: A string indicating the mode of printing. It can be 'sizes' to print the sizes and types of the last task information, or 'values' to print the values of the last task information.
    If no task has been performed yet, it prints a message indicating that.
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