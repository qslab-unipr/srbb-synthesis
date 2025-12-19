# unitary_generation/io_utils.py

# Unused module in version v1.1, it is ready for possible future use.

import numpy as np

# --- 1) NumPy .npy/.npz ---
def save_npy(path: str, U: np.ndarray) -> None:
    np.save(path, U)

def load_npy(path: str, mmap_mode: str | None = None) -> np.ndarray:
    """
    mmap_mode='r' allows memory-mapped reading (does not load everything at once).
    Useful for large matrices.
    """
    return np.load(path, mmap_mode=mmap_mode)

def save_npz(path: str, **arrays) -> None:
    """Save multiple arrays in a compressed file."""
    np.savez_compressed(path, **arrays)