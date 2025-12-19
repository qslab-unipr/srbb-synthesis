#unitary_generation/hamiltonians.py

#Module in which the hamiltonian matrices are defined.
#The user inserts its hamiltonian matrix, which is added to a list of available matrices.

import numpy as np
import sympy as sp
from . import generators, checks

#Pauli matrices
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])

#returns a list of all available hamiltonians to the main program
def getList():
    return [name for name, obj in globals().items() 
            if callable(obj) and name not in ("generate_unitary", "getList")]

#generates the unitary matrix with the chosen hamiltonian, then passes it to the main program
def generate_unitary(fname):
    func = globals()[fname]
    if fname == "H2":
        print("[WARNING] The H2 hamiltonian requires 6 parameters (a0, a1, a2, a3, a4, a5).\n")
        params = []
        for i in range(6):
            val = float(input(f"Insert value for a{i}: "))
            params.append(val)
        H = func(*params)
    else:
        H = func()
    i = input("Please insert the value of the t/theta rotation parameter: ")
    if fname == "LiH":
        if i not in ["pi/2", "pi/4", "pi/8"]:
            raise ValueError(f"{i}: Parameter invalid! Permitted values are " + "pi/2"+ ", " + "pi/4" + ", " + "pi/8" + "\n")
    t = float(sp.sympify(i))
    U = generators.expiH(H, t)
    return U

# ==== Hamiltonian matrix definition section ====

def pauli_z():
    H = np.array([[1, 0],
                  [0, -1]])
    return checks.is_hamiltonian(H)

def pauli_x():
    H = np.array([[0, 1],
                  [1, 0]])
    return checks.is_hamiltonian(H)

def pauli_y():
    H = np.array([[0, -1j],
                  [1j, 0]])
    return checks.is_hamiltonian(H)

def non_hermitian():  # invalid example
    H = np.array([[0, 1],
                  [0, 0]])
    return checks.is_hamiltonian(H)

def H2(a0, a1, a2, a3, a4, a5):
    """Builds the parametric hamiltonian matrix related to H_2 in the 2-qubit STO-3G set"""
    """in this set, the dynamics is defined by the time parameter -> modify generators.expiH according to the desired dynamics"""
    term0 = a0 * np.kron(I, I)
    term1 = a1 * np.kron(Z, I)
    term2 = a2 * np.kron(I, Z)
    term3 = a3 * np.kron(Z, Z)
    term4 = a4 * np.kron(X, X)
    term5 = a5 * np.kron(Y, Y)
    H = term0 + term1 + term2 + term3 + term4 + term5
    return checks.is_hamiltonian(H)

def LiH():
    """Builds the double excitation Hamiltonian for LiH on 3 qubits in the UCCSD system"""
    """In the UCC system the evolution time becomes the excitation/rotation parameter -> modify generators.expiH according to the desired transition"""
    term0 = np.kron(X, np.kron(X, X))
    term1 = np.kron(Y, np.kron(Y, X))
    H = term0 + term1
    return checks.is_hamiltonian(H)
