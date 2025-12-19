import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from pennylane.templates.embeddings import AmplitudeEmbedding

n_qubit = 2

dev = qml.device('default.qubit', wires = n_qubit)
@qml.qnode(dev)
def circuit(params, x = [], U_approx = []):
    if len(U_approx) == 0:
        if len(x) != 0:
            AmplitudeEmbedding(x, [0, 1], normalize = True)

        circuitN2(params)  

        if len(x) == 0:
            return qml.density_matrix([0, 1])
        else:
           return qml.density_matrix([0, 1]), qml.state(), qml.probs([0, 1])       

    else:
        AmplitudeEmbedding(x, [0, 1], normalize = True)
        qml.QubitUnitary(U_approx, wires = [0, 1])

        return qml.state(), qml.probs([0, 1]), qml.density_matrix(range(n_qubit))
    
@qml.qnode(dev)
def amplitude_density(x):
    AmplitudeEmbedding(x, [0, 1], normalize = True)
    return qml.density_matrix([0, 1])

def circuitN2(params):    
    #PHI_FACTOR[Transpositions for M1_odd: (2,3) -- Related elements for M1_odd: (5,7,11,14)]
    ProdT1_odd()
    #M1_odd
    #ProdT1_odd
    #some simplifications arise, so we replace with...
    qml.RZ(params[0],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(params[1],wires=[1])
    qml.RY(params[2],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RY(params[3],wires=[1])
    qml.RZ(params[4],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(params[5],wires=[1])
    qml.CNOT(wires=[1,0])
    qml.CNOT(wires=[0,1])


    #PSI FACTOR [Transpositions for M1_even: (2,4) -- Related elements for M1_even: (10,13,4,6)]

    ProdT1_even()
    M1(params[6:12])
    ProdT1_even()

    #Exp(1,2,9,12)
    #since Exp(1,2,9,12) has a ZYZ decomposition, it can be implemented as M1...
    M1(params[12:18], True)

    #ZETA FACTOR [Elements: 3,8,15]
    qml.RZ(params[18],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(params[19],wires=[0])
    qml.RZ(params[20],wires=[1])
    



    
#M1 definition -> 4 CNOT, 6 R (for 2 qubits, even and odd cases coincide)
def M1(parameters, delete = False):
    qml.RZ(parameters[0],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[1],wires=[1])
    qml.RY(parameters[2],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RY(parameters[3],wires=[1])
    qml.RZ(parameters[4],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[5],wires=[1])
    if not delete:
        qml.CNOT(wires=[0,1])

#ProdT1_odd definition -> 3 CNOT
def ProdT1_odd():
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,0])
    qml.CNOT(wires=[0,1])

#ProdT1_even definition -> 1 CNOT
def ProdT1_even():
    qml.CNOT(wires=[1,0])

