# unitary_generation/checks.py

# Module used by the main program to perform checks.

import numpy as np

def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """verifies that U is a valid unitary"""
    I = np.eye(U.shape[0], dtype=U.dtype)
    err = np.linalg.norm(U.conj().T @ U - I, ord='fro')
    return err <= tol

"""
def unitary_report(U: np.ndarray) -> dict:
    I = np.eye(U.shape[0], dtype=U.dtype)
    left = np.linalg.norm(U.conj().T @ U - I, ord='fro')
    right = np.linalg.norm(U @ U.conj().T - I, ord='fro')
    return {"fro_err_left": float(left), "fro_err_right": float(right)}
"""
 
def is_hamiltonian(H, tol=1e-10):
    """verifies that H is a valid Hamiltonian"""
    if H.shape[0] != H.shape[1]:
        raise ValueError("[WARN] The matrix is not square\n")
    if not np.allclose(H, H.conj().T, atol=tol):
        raise ValueError("[WARN] The matrix is not hermitian\n")
    eigvals = np.linalg.eigvals(H)
    if not np.allclose(eigvals.imag, 0, atol=tol):
        raise ValueError("[WARN] Eigenvalues have a non-zero immaginary part\n")
    print("[INFO] The given matrix is hamiltonian.\n")
    return H