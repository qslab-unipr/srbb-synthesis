# unitary_generation/generators.py

# List of benchmark generators.
# The only working generator is expiH.
# The other methods may be used for future research purposes.

import numpy as np
from scipy.linalg import expm

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])

"""
def random_haar(dim: int, seed: int | None = None, dtype=np.complex128) -> np.ndarray:
    #Random Haar-Unitary via QR with R phase correction
    rng = np.random.default_rng(seed)   # random number generator with seed
    Z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))).astype(dtype) # complex Ginibre, complex matrix with gaussian entries
    Q, R = qr(Z)                # QR decomposition
    d = np.diag(R)              # R diagonal matrix
    ph = d / np.abs(d)          # unitary phases
    U = (Q * ph)                # phase correction
    return U.astype(dtype, copy=False)

def fourier(dim: int, dtype=np.complex128) -> np.ndarray:
    #Discrete Fourier Transformation (DFT) with dimension dim
    k = np.arange(dim)                          # array with indexes from 0 to dim-1
    omega = np.exp(-2j * np.pi / dim)           # dth root of the unit (conventional - symbol)
    F = omega ** np.outer(k, k)                 # outer product (matrix) implements the exponent
    return (F / np.sqrt(dim)).astype(dtype)     # divided to confirm unitariety

def kron_power(U: np.ndarray, k: int) -> np.ndarray:
    #Kronecker Product(k times)
    out = U
    for _ in range(k - 1):
        out = np.kron(out, U)
    return out

def pauli(op: str) -> np.ndarray:
    #Returns a Pauli matrix ('I','X','Y','Z')
    if op == 'I':
        return np.array([[1,0],[0,1]], dtype=np.complex128)
    if op == 'X':
        return np.array([[0,1],[1,0]], dtype=np.complex128)
    if op == 'Y':
        return np.array([[0,-1j],[1j,0]], dtype=np.complex128)
    if op == 'Z':
        return np.array([[1,0],[0,-1]], dtype=np.complex128)
    raise ValueError("op must be one of 'I','X','Y','Z'.")

def pauli_string(ops: str) -> np.ndarray:
    #Builds a pauli string with a given input string, such as XIZ
    M = np.array([[1]], dtype=np.complex128)
    for ch in ops:
        M = np.kron(M, pauli(ch))
    return M

def block_unitary(blocks: list[np.ndarray]) -> np.ndarray:
    #Unitary diagonal blocks -> Diagonal block unitary
    return block_diag(*blocks).astype(np.complex128)

def unitary_from_seed(n_qubits: int, kind: str = "haar", seed: int | None = None) -> np.ndarray:
    #generates a unitary matrix with 2^n qubits
    dim = 2 ** n_qubits
    if kind == "haar":
        return random_haar(dim, seed=seed)
    if kind == "fourier":
        return fourier(dim)
    raise ValueError("unsupported kind.")
"""

def expiH(H: np.ndarray, t: float = 1.0) -> np.ndarray:
    """If H is hermitian, U is unitary"""
    return expm(-1j * t * H)