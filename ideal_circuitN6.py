import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
import numpy as np
import cmath

def random(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def qft6(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        matrix[:] = qml.QFT.compute_matrix(6)
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def grover6(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        matrix[:] = qml.GroverOperator.compute_matrix(6, 0)
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit


