import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
import numpy as np
import cmath
import config

n_qubit = config.n_qubit
dev = qml.device('default.qubit', wires = n_qubit)
#dev = config.dev

@qml.qnode(dev)
def random(x, matrix):
    AmplitudeEmbedding(x, wires = range(n_qubit), normalize=True)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = range(n_qubit))

    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def cnot(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = qml.CNOT.compute_matrix()
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def xx(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.kron(qml.PauliX.compute_matrix(), qml.PauliX.compute_matrix())
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def yy(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.kron(qml.PauliY.compute_matrix(), qml.PauliY.compute_matrix())
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def zz(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.kron(qml.PauliZ.compute_matrix(), qml.PauliZ.compute_matrix())
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def swap(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = qml.SWAP.compute_matrix()
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def xz(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.kron(qml.PauliX.compute_matrix(), qml.PauliZ.compute_matrix())
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def zx(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.kron(qml.PauliZ.compute_matrix(), qml.PauliX.compute_matrix())
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def zy(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.kron(qml.PauliZ.compute_matrix(), qml.PauliY.compute_matrix())
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def cnot_reverse(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]) 
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def hi(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.kron(qml.Hadamard.compute_matrix(), np.eye(2, dtype=int))
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def hh(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.kron(qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix())
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def iswap(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = qml.ISWAP.compute_matrix()
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def cs(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, +1j]]) 
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def ct(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, (1/np.sqrt(2) + 1j/np.sqrt(2))]]) 
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def root_not_i(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = 1/2 * np.kron(np.array([[1 + 1j, 1 -1j], [1 - 1j, 1 + 1j]]), np.eye(2, dtype=int)) 
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def xx_yy(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = np.kron(qml.PauliY.compute_matrix(), qml.PauliY.compute_matrix()) @ np.kron(qml.PauliX.compute_matrix(), qml.PauliX.compute_matrix())
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def siswap(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = qml.SISWAP.compute_matrix()
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def entangled2(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = qml.CNOT.compute_matrix() @ np.kron(qml.Hadamard.compute_matrix(), np.eye(2, dtype=int))
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def qft2(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = qml.QFT.compute_matrix(2)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))

@qml.qnode(dev)
def grover2(x, matrix):
    AmplitudeEmbedding(x, [0, 1], normalize=True)
    matrix[:] = qml.GroverOperator.compute_matrix(2, 0)
    SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
    qml.QubitUnitary(SU, wires = [0, 1])
    if config.device_type == "sim":
        return qml.density_matrix(wires = range(n_qubit))
    else:
        return qml.probs(wires=range(n_qubit))
