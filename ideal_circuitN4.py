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
        return qml.density_matrix(wires = range(n_qubit))
    return circuit

def cnotcnot(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()  #layer A, component 1
        matrix[:] = np.kron(A1, A1)
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def cnotcnot_rev(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()
        A2 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        matrix[:] = np.kron(A1, A2)
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def cnot01_cnot02_cnot03(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        block1 = np.eye(8)
        block2 = np.zeros((8,8))
        block3 = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 1, 0]])
        matrix[:] = np.block([[block1, block2], [block2, block3]])
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def cnot01_cnot02_cnot03(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()
        A2 = np.eye(4)
        A = np.kron(A1, A2)
        block1 = np.eye(4)
        block2 = np.zeros((4,4))
        block3 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        B1 = np.block([[block1, block2], [block2, block3]])
        B2 = np.eye(2)
        B = np.kron(B1, B2)
        block4 = np.eye(8)
        block5 = np.zeros((8,8))
        block6 = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 1, 0]])
        C = np.block([[block4, block5], [block5, block6]])
        matrix[:] = C @ B @ A
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit


def cnot10_cnot02_cnot23_cnot31(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = np.eye(2)
        block1 = np.array([[1, 0], [0, 0]])
        block2 = np.array([[0, 0], [0, 1]])
        block3 = np.zeros((2,2))
        A2 = np.block([[block1, block3, block2, block3],
                       [block3, block1, block3, block2],
                       [block2, block3, block1, block3],
                       [block3, block2, block3, block1]])
        A = np.kron(A1, A2)
        B1 = np.eye(4)
        B2 = qml.CNOT.compute_matrix()
        B = np.kron(B1, B2)
        block1 = np.eye(4)
        block2 = np.zeros((4,4))
        block3 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        C1 = np.block([[block1, block2], [block2, block3]])
        C2 = np.eye(2)
        C = np.kron(C1, C2)
        D1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        D2 = np.eye(4)
        D = np.kron(D1, D2)
        matrix[:] = A @ B @ C @ D
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def entangled4(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = np.eye(4)
        A2 = qml.CNOT.compute_matrix()
        A = np.kron(A1, A2)
        B1 = np.eye(4)
        B2 = qml.Hadamard.compute_matrix()
        B3 = np.eye(2)
        B = np.kron(np.kron(B1, B2), B3)
        C1 = np.eye(2)
        C2 = qml.CNOT.compute_matrix()
        C = np.kron(np.kron(C1, C2), C1)
        D1 = np.eye(2)
        D2 = qml.Hadamard.compute_matrix()
        D3 = np.eye(4)
        D = np.kron(np.kron(D1, D2), D3)
        E1 = qml.CNOT.compute_matrix()
        E2 = np.eye(4)
        E = np.kron(E1, E2)
        F1 = qml.Hadamard.compute_matrix()
        F2 = np.eye(8)
        F = np.kron(F1, F2)
        matrix[:] = A @ B @ C @ D @ E @ F
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def hhii_icnoti_iihh(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = np.eye(4)
        A2 = qml.Hadamard.compute_matrix()
        A = np.kron(np.kron(A1, A2), A2)
        B1 = np.eye(2)
        B2 = qml.CNOT.compute_matrix()
        B = np.kron(np.kron(B1, B2), B1)
        C1 = qml.Hadamard.compute_matrix()
        C2 = np.eye(4)
        C = np.kron(np.kron(C1, C1), C2)
        matrix[:] = A @ B @ C
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def hhhh_xyzx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = qml.PauliX.compute_matrix()
        A2 = qml.PauliY.compute_matrix()
        A3 = qml.PauliZ.compute_matrix()
        A = np.kron(np.kron(np.kron(A1, A2), A3), A1)
        B1 = qml.Hadamard.compute_matrix()
        B = np.kron(np.kron(np.kron(B1, B1), B1), B1)
        matrix[:] = A @ B
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def swaprootnoti_iicnot(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = np.eye(4)
        A2 = qml.CNOT.compute_matrix()
        A = np.kron(A1, A2)
        B1 = np.eye(4)
        B2 = 1/2 * np.array([[1 + 1j, 1 -1j], [1 - 1j, 1 + 1j]])
        B3 = np.eye(2)
        B = np.kron(np.kron(B1, B2), B3)
        C1 = qml.SWAP.compute_matrix()
        C2 = np.eye(4)
        C = np.kron(C1, C2)
        matrix[:] = A @ B @ C
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def swapii_irootnoti_iicnot(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = np.eye(4)
        A2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, +1j]])
        A = np.kron(A1, A2)
        B1 = np.eye(2)
        B2 = qml.CNOT.compute_matrix()
        B = np.kron(np.kron(B1, B2), B1)
        C1 = qml.ISWAP.compute_matrix()
        C2 = np.eye(4)
        C = np.kron(C1, C2)
        matrix[:] = A @ B @ C
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def c1c1c1x(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        matrix[:] = qml.MultiControlledX.compute_matrix([0, 1, 2], control_values='111')
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def c0c0c1x(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        matrix[:] = qml.MultiControlledX.compute_matrix([0, 1, 2], control_values='001')
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def c0c1c0roty(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        U = qml.RY(np.pi/4, wires = [3])
        matrix[:] = qml.matrix(qml.ControlledQubitUnitary(U, control_wires = [0, 1, 2], wires = [3], control_values = [0, 1, 0]))
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def toffolii(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = qml.Toffoli.compute_matrix()
        A2 = np.eye(2)
        matrix[:] = np.kron(A1, A2)
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def xcnoty_cnotyx(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        A1 = qml.CNOT.compute_matrix()
        A2 = qml.PauliY.compute_matrix()
        A3 = qml.PauliX.compute_matrix()
        A = np.kron(np.kron(A1, A2), A3)
        B1 = qml.PauliX.compute_matrix()
        B2 = qml.CNOT.compute_matrix()
        B3 = qml.PauliY.compute_matrix()
        B = np.kron(np.kron(B1, B2), B3)
        matrix[:] = A @ B
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def qft4(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        matrix[:] = qml.QFT.compute_matrix(4)
        SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit

def grover4(dev, n_qubit):
    @qml.qnode(dev)
    def circuit(x, matrix, device_type):
        AmplitudeEmbedding(x, range(n_qubit), normalize=True)
        matrix[:] = qml.GroverOperator.compute_matrix(4, 0)
        SU = matrix/cmath.sqrt(cmath.sqrt(np.linalg.det(matrix)))
        qml.QubitUnitary(SU, wires = range(n_qubit))
        if device_type == "sim":
            return qml.density_matrix(wires = range(n_qubit))
        else:
            return qml.probs(wires=range(n_qubit))
    return circuit
