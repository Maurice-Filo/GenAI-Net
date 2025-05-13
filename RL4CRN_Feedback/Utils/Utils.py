import torch
import numpy as np

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

