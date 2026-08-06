"""Reproducibility seed helper."""

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for PyTorch, NumPy, and CUDA.

    Args:
        seed: Integer seed value for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
