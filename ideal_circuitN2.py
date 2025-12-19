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

def cnot(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = qml.CNOT.compute_matrix()
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def xx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.kron(qml.PauliX.compute_matrix(), qml.PauliX.compute_matrix())
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def yy(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.kron(qml.PauliY.compute_matrix(), qml.PauliY.compute_matrix())
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def zz(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.kron(qml.PauliZ.compute_matrix(), qml.PauliZ.compute_matrix())
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def swap(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = qml.SWAP.compute_matrix()
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def xz(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.kron(qml.PauliX.compute_matrix(), qml.PauliZ.compute_matrix())
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def zx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.kron(qml.PauliZ.compute_matrix(), qml.PauliX.compute_matrix())
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def zy(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.kron(qml.PauliZ.compute_matrix(), qml.PauliY.compute_matrix())
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def cnot_reverse(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]) 
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def hi(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.kron(qml.Hadamard.compute_matrix(), np.eye(2, dtype=int))
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def hh(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.kron(qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix())
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def iswap(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = qml.ISWAP.compute_matrix()
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def cs(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, +1j]]) 
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def ct(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, (1/np.sqrt(2) + 1j/np.sqrt(2))]]) 
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def root_not_i(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = 1/2 * np.kron(np.array([[1 + 1j, 1 -1j], [1 - 1j, 1 + 1j]]), np.eye(2, dtype=int)) 
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def xx_yy(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = np.kron(qml.PauliY.compute_matrix(), qml.PauliY.compute_matrix()) @ np.kron(qml.PauliX.compute_matrix(), qml.PauliX.compute_matrix())
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def siswap(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = qml.SISWAP.compute_matrix()
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def entangled2(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = qml.CNOT.compute_matrix() @ np.kron(qml.Hadamard.compute_matrix(), np.eye(2, dtype=int))
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def qft2(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = qml.QFT.compute_matrix(2)
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def grover2(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, [0, 1], normalize=True)
        matrix[:] = qml.GroverOperator.compute_matrix(2, 0)
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = [0, 1])
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit
