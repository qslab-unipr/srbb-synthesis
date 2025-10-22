import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
import numpy as np
import cmath
import config

n_qubit = config.n_qubit
dev = qml.device('default.qubit', wires = n_qubit)

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
def cnoth(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.CNOT.compute_matrix() #layer A, component 1
    A2 = qml.Hadamard.compute_matrix() #layer A, component 2
    matrix[:] = np.kron(A1, A2)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def cnoti(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.CNOT.compute_matrix()
    A2 = np.eye(2, dtype=int)
    matrix[:] = np.kron(A1, A2)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def icnot_rev(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = np.eye(2)
    A2 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    matrix[:] = np.kron(A1, A2)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def cnot02(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    block1 = np.eye(4)
    block2 = np.zeros((4,4))
    block3 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    matrix[:] = np.block([[block1, block2], [block2, block3]])
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def cnot20(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    block1 = np.array([[1, 0], [0, 0]])
    block2 = np.array([[0, 0], [0, 1]])
    block3 = np.zeros((2,2))
    matrix[:] = np.block([[block1, block3, block2, block3],
                          [block3, block1, block3, block2],
                          [block2, block3, block1, block3],
                          [block3, block2, block3, block1]])
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def cnotx(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.CNOT.compute_matrix()
    A2 = qml.PauliX.compute_matrix()
    matrix[:] = np.kron(A1, A2)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def cnoty(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.CNOT.compute_matrix()
    A2 = qml.PauliY.compute_matrix()
    matrix[:] = np.kron(A1, A2)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def cnotz(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.CNOT.compute_matrix()
    A2 = qml.PauliZ.compute_matrix()
    matrix[:] = np.kron(A1, A2)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def xxx(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.PauliX.compute_matrix()
    matrix[:] = np.kron(np.kron(A1, A1), A1)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def xyx(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.PauliX.compute_matrix()
    A2 = qml.PauliY.compute_matrix()
    matrix[:] = np.kron(np.kron(A1, A2), A1)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def xyz(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.PauliX.compute_matrix()
    A2 = qml.PauliY.compute_matrix()
    A3 = qml.PauliZ.compute_matrix()
    matrix[:] = np.kron(np.kron(A1, A2), A3)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def hhh(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.Hadamard.compute_matrix()
    matrix[:] = np.kron(np.kron(A1, A1), A1)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def icnot_cnoti(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.CNOT.compute_matrix()
    A2 = np.eye(2)
    A = np.kron(A1, A2) #layer A
    B1 = np.eye(2)
    B2 = qml.CNOT.compute_matrix()
    B = np.kron(B1, B2) #layer B
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def icnotrev_cnotrevi(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    A2 = np.eye(2)
    A = np.kron(A1, A2)
    B1 = np.eye(2)
    B2 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    B = np.kron(B1, B2)
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def cnot02_icnot(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = np.eye(2)
    A2 = qml.CNOT.compute_matrix()
    A = np.kron(A1, A2)
    block1 = np.eye(4)
    block2 = np.zeros((4,4))
    block3 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    B = np.block([[block1, block2], [block2, block3]])
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def toffoli(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = qml.Toffoli.compute_matrix()
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def c0c0x(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = qml.MultiControlledX.compute_matrix([0, 1], [0, 0])
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def c1c0roty(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    U = qml.RY(np.pi/4, wires = [2])
    matrix[:] = qml.matrix(qml.ControlledQubitUnitary(U, control_wires = [0, 1], wires = [2], control_values = [1, 0]))
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def xcnot_cnoty(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.CNOT.compute_matrix()
    A2 = qml.PauliY.compute_matrix()
    A = np.kron(A1, A2)
    B1 = qml.PauliX.compute_matrix()
    B2 = qml.CNOT.compute_matrix()
    B = np.kron(B1, B2)
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def hhh_xyx(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.PauliX.compute_matrix()
    A2 = qml.PauliY.compute_matrix()
    A = np.kron(np.kron(A1, A2), A1)
    B1 = qml.Hadamard.compute_matrix()
    B = np.kron(np.kron(B1, B1), B1)
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def hhh_xyz(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.PauliX.compute_matrix()
    A2 = qml.PauliY.compute_matrix()
    A3 = qml.PauliZ.compute_matrix()
    A = np.kron(np.kron(A1, A2), A3)
    B1 = qml.Hadamard.compute_matrix()
    B = np.kron(np.kron(B1, B1), B1)
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def hii_cnoti_ihi_icnot(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.Hadamard.compute_matrix()
    A2 = np.eye(4)
    A = np.kron(A1, A2)
    B1 = qml.CNOT.compute_matrix()
    B2 = np.eye(2)
    B = np.kron(B1, B2)
    C1 = np.eye(2)
    C2 = qml.Hadamard.compute_matrix()
    C3 = np.eye(2)
    C = np.kron(np.kron(C1, C2), C3)
    D1 = np.eye(2)
    D2 = qml.CNOT.compute_matrix()
    D = np.kron(D1, D2)
    matrix[:] = A @ B @ C @ D
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def hhh_xxx(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.Hadamard.compute_matrix()
    A = np.kron(np.kron(A1, A1), A1)
    B1 = qml.PauliX.compute_matrix()
    B = np.kron(np.kron(B1, B1), B1)
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def hxx_iyz(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = np.eye(2)
    A2 = qml.PauliY.compute_matrix()
    A3 = qml.PauliZ.compute_matrix()
    A = np.kron(np.kron(A1, A2), A3)
    B1 = qml.Hadamard.compute_matrix()
    B2 = qml.PauliX.compute_matrix() 
    B = np.kron(np.kron(B1, B2), B2)
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def swapi_iswap(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = np.eye(2)
    A2 = qml.SWAP.compute_matrix()
    A = np.kron(A1, A2)
    B1 = qml.SWAP.compute_matrix()
    B2 = np.eye(2)
    B = np.kron(B1, B2)
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def iswapi_iswap(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = np.eye(2)
    A2 = qml.SWAP.compute_matrix()
    A = np.kron(A1, A2)
    B1 = qml.ISWAP.compute_matrix()
    B2 = np.eye(2)
    B = np.kron(B1, B2)
    matrix[:] = A @ B
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def rootnotii_ihh_ycs(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A1 = qml.PauliY.compute_matrix()
    A2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, +1j]])
    A = np.kron(A1, A2)
    B1 = np.eye(2)
    B2 = qml.Hadamard.compute_matrix()
    B = np.kron(np.kron(B1, B2), B2)
    C1 = 1/2 * np.array([[1 + 1j, 1 -1j], [1 - 1j, 1 + 1j]])
    C2 = np.eye(4)
    C = np.kron(C1, C2)
    matrix[:] = A @ B @ C
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def qft3(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = qml.QFT.compute_matrix(3)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def grover3(x, matrix): 
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = qml.GroverOperator.compute_matrix(3, 0)
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def fredkin(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = qml.CSWAP.compute_matrix()
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def ccx(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = qml.MultiControlledX.compute_matrix(control_wires  = [0, 1], control_values  = [1, 1])
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def orCircuit(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    A = np.kron(np.kron(qml.PauliX.compute_matrix(), qml.PauliX.compute_matrix()), qml.PauliX.compute_matrix())
    B = np.kron(np.kron(qml.PauliX.compute_matrix(), qml.PauliX.compute_matrix()), np.eye(2))
    matrix[:] = B @ qml.MultiControlledX.compute_matrix([0, 1], [1, 1]) @ A
                
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))

@qml.qnode(dev)
def peres(x, matrix):
    AmplitudeEmbedding(x, range(n_qubit), normalize=True)
    matrix[:] = np.kron(np.eye(2), qml.CNOT.compute_matrix()) @ qml.MultiControlledX.compute_matrix([0, 1], [1, 1])
    SU = matrix/cmath.sqrt(cmath.sqrt(cmath.sqrt(np.linalg.det(matrix))))
    qml.QubitUnitary(SU, wires = range(n_qubit))
    return qml.density_matrix(range(n_qubit))