#The function called "predefined" is needed to generate the SU matrix associated with a predefined hamiltonian (as defined in hamiltonians.py)
#The other nodes may be useful for research purposes if the unitary is known (via a static definition)

import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
import numpy as np
import cmath
import config
from unitary_generation import io_utils as io

n_qubit = config.n_qubit
dev = qml.device('default.qubit', wires = n_qubit)
#dev = config.dev

@qml.qnode(dev)
def predefined(x, matrix):
    #matrix[:] = io.load_npy('example.npy', 'r')
    AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = range(n_qubit))

    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

"""
@qml.qnode(dev)
def h2(x, matrix):
    n_qubit = 2
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = np.array([
				[0.7669 + 0.6402j,       0,                 0,             -0.0267 + 0.0356j],
				[0,               0.8062 + 0.5242j,   0.2280 - 0.1526j,     0],
				[0,               0.2280 - 0.1526j,   0.1768 + 0.9452j,     0],
				[-0.0267 + 0.0356j,       0,                 0,             0.8288 + 0.5578j]
			], dtype=complex)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def lihpi2(x, matrix):
    n_qubit = 3
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    theta = np.pi / 2
    cos2t = np.cos(2 * theta)
    isin2t = 1j * np.sin(2 * theta)

    matrix[:] = np.array([
            [1,       0,       0,       0,       0,       0,       0,       0],
            [0,       1,       0,       0,       0,       0,       0,       0],
            [0,       0,   cos2t,       0,       0, -isin2t,       0,       0],
            [0,       0,       0,   cos2t, -isin2t,       0,       0,       0],
            [0,       0,       0, -isin2t,   cos2t,       0,       0,       0],
            [0,       0, -isin2t,       0,       0,   cos2t,       0,       0],
            [0,       0,       0,       0,       0,       0,       1,       0],
            [0,       0,       0,       0,       0,       0,       0,       1]
        ], dtype=complex)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))
    
@qml.qnode(dev)
def lihpi4(x, matrix):
    n_qubit = 3
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    theta = np.pi / 4
    cos2t = np.cos(2 * theta)
    isin2t = 1j * np.sin(2 * theta)

    matrix[:] = np.array([
            [1,       0,       0,       0,       0,       0,       0,       0],
            [0,       1,       0,       0,       0,       0,       0,       0],
            [0,       0,   cos2t,       0,       0, -isin2t,       0,       0],
            [0,       0,       0,   cos2t, -isin2t,       0,       0,       0],
            [0,       0,       0, -isin2t,   cos2t,       0,       0,       0],
            [0,       0, -isin2t,       0,       0,   cos2t,       0,       0],
            [0,       0,       0,       0,       0,       0,       1,       0],
            [0,       0,       0,       0,       0,       0,       0,       1]
        ], dtype=complex)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))
    
@qml.qnode(dev)
def lihpi8(x, matrix):
    n_qubit = 3
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    theta = np.pi / 8
    cos2t = np.cos(2 * theta)
    isin2t = 1j * np.sin(2 * theta)

    matrix[:] = np.array([
            [1,       0,       0,       0,       0,       0,       0,       0],
            [0,       1,       0,       0,       0,       0,       0,       0],
            [0,       0,   cos2t,       0,       0, -isin2t,       0,       0],
            [0,       0,       0,   cos2t, -isin2t,       0,       0,       0],
            [0,       0,       0, -isin2t,   cos2t,       0,       0,       0],
            [0,       0, -isin2t,       0,       0,   cos2t,       0,       0],
            [0,       0,       0,       0,       0,       0,       1,       0],
            [0,       0,       0,       0,       0,       0,       0,       1]
        ], dtype=complex)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))
"""