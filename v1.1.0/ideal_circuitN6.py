import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
import numpy as np
import cmath
import config

n_qubit = config.n_qubit
device_type = config.device_type
dev = qml.device('default.qubit', wires = n_qubit)

@qml.qnode(dev)
def random(x, matrix):
    AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    if device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def qft6(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = qml.QFT.compute_matrix(6)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    if device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def grover6(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = qml.GroverOperator.compute_matrix(6, 0)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    if device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))



