"""Device selection utility."""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU).

    Returns:
        A ``torch.device`` pointing at CUDA if available, then Apple
        Silicon MPS, then CPU as a fallback.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
