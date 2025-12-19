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
