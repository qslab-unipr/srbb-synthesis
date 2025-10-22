import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from pennylane.templates.embeddings import AmplitudeEmbedding

n_qubit = 3

dev = qml.device('default.qubit', wires = n_qubit)

@qml.qnode(dev)
def circuit(params, x = [], U_approx = []):
    if len(U_approx) == 0:
        if len(x) != 0:
            AmplitudeEmbedding(x, range(n_qubit), normalize = True)

        circuitN3(params)  

        if len(x) == 0:
            return qml.density_matrix(range(n_qubit))
        else:
           return qml.density_matrix(range(n_qubit)), qml.state(), qml.probs(range(n_qubit))       

    else:
        AmplitudeEmbedding(x, range(n_qubit), normalize = True)
        qml.QubitUnitary(U_approx, wires = range(n_qubit))

        return qml.state(), qml.probs(range(n_qubit)), qml.density_matrix(range(n_qubit))
    
@qml.qnode(dev)
def amplitude_density(x):
    AmplitudeEmbedding(x, range(n_qubit), normalize = True)
    return qml.density_matrix(range(n_qubit))

def circuitN3(parameters):

    #PHI_FACTOR [Transpositions for M1_odd: (2,3),(6,7) -- Related elements for M1_odd: (5,7,11,14),(41,47,55,62)]
    #PHI_FACTOR [Transpositions for M2_odd: (2,5),(4,7) -- Related elements for M2_odd: (17,21,27,32),(39,45,53,60)]
    #PHI_FACTOR [Transpositions for M3_odd: (2,7),(4,5) -- Related elements for M3_odd: (37,43,51,58),(19,23,29,34)]

    ProdT3_odd()
    M123_odd(parameters[0:18])
    
    #ProdT3_odd
    #Prod T2_odd
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[2,1])
    qml.CNOT(wires=[0,2])
    
    M123_odd(parameters[18:36])
    ProdT2_odd()
    ProdT1_odd()
    M123_odd(parameters[36:54])
    ProdT1_odd()

    #PSI FACTOR [Transpositions for M1_even: (2,4),(6,8) -- Related elements for M1_even: (10,13,4,6),(54,61,36,42)]
    #PSI FACTOR [Transpositions for M2_even: (2,6),(4,8) -- Related elements for M2_even: (26,31,18,22),(52,59,40,46)]
    #PSI FACTOR [Transpositions for M3_even: (2,8),(4,6) -- Related elements for M3_even: (50,57,38,44),(28,33,16,20)]

    ProdT3_even()
    M123_even(parameters[54:66])

    #Prod T3_even
    #Prod T2_even
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[2,1])

    M123_even(parameters[66:78])
    ProdT2_even()
    ProdT1_even()
    M123_even(parameters[78:90])
    ProdT1_even()

    #Exp(1,2,9,12,25,30,49,56)
    #since Exp(1,2,9,12,25,30,49,56) has a ZYZ decomposition, it can be implemented as M123_even...
    M123_even(parameters[90:102])

    #ZETA FACTOR [Elements: 3,8,15,24,35,48,63]

    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[102],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.RZ(parameters[103],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[104],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[105],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[106],wires=[0])
    qml.RZ(parameters[107],wires=[1])
    qml.RZ(parameters[108],wires=[2])


#M123_odd definition -> 14 CNOT, 18 R
def M123_odd(parameters): 
    qml.RZ(parameters[0],wires=[0])
    qml.RZ(parameters[1],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[2],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[3],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[4],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.RZ(parameters[5],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[6],wires=[2])
    qml.RY(parameters[7],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RY(parameters[8],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.RY(parameters[9],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RY(parameters[10],wires=[2])
    qml.RZ(parameters[11],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[12],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.RZ(parameters[13],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[14],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[15],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[16],wires=[1])
    qml.RZ(parameters[17],wires=[1])

#ProdT3_odd definition -> 4 CNOT
def ProdT3_odd():
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[2,0])
    qml.CNOT(wires=[2,1])
    qml.CNOT(wires=[0,2])

#ProdT2_odd definition -> 3 CNOT
def ProdT2_odd():
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[2,0])
    qml.CNOT(wires=[0,2])

#ProdT1_odd definition -> 3 CNOT
def ProdT1_odd():
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,1])
    qml.CNOT(wires=[1,2])

#M123_even definition -> 10 CNOT, 12 R
def M123_even(parameters):
    qml.RZ(parameters[0],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[1],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.RZ(parameters[2],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[3],wires=[2])
    qml.RY(parameters[4],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RY(parameters[5],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.RY(parameters[6],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RY(parameters[7],wires=[2])
    qml.RZ(parameters[8],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[9],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.RZ(parameters[10],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[11],wires=[2])
    qml.CNOT(wires=[0,2])

#ProdT3_even definition -> 2 CNOT
def ProdT3_even():
    qml.CNOT(wires=[2,0])
    qml.CNOT(wires=[2,1])

#ProdT2_even definition -> 1 CNOT
def ProdT2_even():
    qml.CNOT(wires=[2,0])

#ProdT1_even definition -> 1 CNOT
def ProdT1_even():
    qml.CNOT(wires=[2,1]) 

