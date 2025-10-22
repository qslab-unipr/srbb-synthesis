import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from pennylane.templates.embeddings import AmplitudeEmbedding

n_qubit = 4
dev=qml.device('default.qubit',wires=n_qubit)

@qml.qnode(dev)
def circuit(params, x = [], U_approx = []):
    if len(U_approx) == 0:
        if len(x) != 0:
            AmplitudeEmbedding(x, range(n_qubit), normalize = True)

        circuitN4(params)  

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

def circuitN4(parameters):

    #PHI_FACTOR [Transpositions for M1_odd: (2,3),(14,15),(12,15),(12,13) -- Related elements for M1_odd: to add]
    #PHI_FACTOR [Transpositions for M2_odd: (2,5),(10,15),(10,13),(10,11) -- Related elements for M2_odd: to add]
    #PHI_FACTOR [Transpositions for M3_odd: (2,7),(8,15),(8,13),(8,11) -- Related elements for M3_odd: to add]
    #PHI_FACTOR [Transpositions for M4_odd: (2,9),(6,15),(6,13),(6,11) -- Related elements for M4_odd: to add]
    #PHI_FACTOR [Transpositions for M5_odd: (2,11),(8,9),(6,9),(6,7) -- Related elements for M5_odd: to add]
    #PHI_FACTOR [Transpositions for M6_odd: (2,13),(4,15),(4,11),(4,9) -- Related elements for M6_odd: to add]
    #PHI_FACTOR [Transpositions for M7_odd: (2,15),(4,13),(4,7),(4,5) -- Related elements for M7_odd: to add]
    
    ProdT7_odd()
    M_odd(parameters[0:38])     #M7_odd

    #ProdT7_odd
    #ProdT6_odd
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[3,2])

    M_odd(parameters[38:76])    #M6_odd

    #ProdT6_odd
    #ProdT5_odd
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[3,1])
    qml.CNOT(wires=[3,2])

    M_odd(parameters[76:114])   #M5_odd

    #ProdT5_odd
    #ProdT4_odd
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[3,2])

    M_odd(parameters[114:152])  #M4_odd
    ProdT4_odd()
    ProdT3_odd()
    M_odd(parameters[152:190])  #M3_odd

    #ProdT3_odd
    #ProdT2_odd
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[3,2])

    M_odd(parameters[190:228])  #M2_odd
    ProdT2_odd()
    ProdT1_odd()
    M_odd(parameters[228:266])  #M1_odd
    ProdT1_odd()

    #PSI FACTOR [Transpositions for M1_even: (2,4),(14,16),(12,16),(12,14) -- Related elements for M1_even: to add]
    #PSI FACTOR [Transpositions for M2_even: (2,6),(10,16),(10,14),(10,12) -- Related elements for M2_even: to add]
    #PSI FACTOR [Transpositions for M3_even: (2,8),(6,16),(6,14),(6,12) -- Related elements for M3_even: to add]
    #PSI FACTOR [Transpositions for M4_even: (2,10),(8,16),(8,14),(8,12) -- Related elements for M4_even: to add]
    #PSI FACTOR [Transpositions for M5_even: (2,12),(8,10),(6,10),(6,8) -- Related elements for M5_even: to add]
    #PSI FACTOR [Transpositions for M6_even: (2,14),(4,16),(4,12),(4,10) -- Related elements for M6_even: to add]
    #PSI FACTOR [Transpositions for M7_even: (2,16),(4,14),(4,8),(4,6) -- Related elements for M7_even: to add]

    ProdT7_even()
    M_even(parameters[266:290])     #M7_even

    #ProdT7_even
    #ProdT6_even
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[3,2])

    M_even(parameters[290:314])     #M6_even

    #ProdT6_even
    #ProdT5_even
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[3,1])
    qml.CNOT(wires=[3,2])

    M_even(parameters[314:338])     #M5_even

    #ProdT5_even
    #ProdT4_even
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[3,2])

    M_even(parameters[338:362])     #M4_even
    ProdT4_even()
    ProdT3_even()
    M_even(parameters[362:386])     #M3_even

    #ProdT3_even
    #ProdT2_even
    #some simplifications arise, so we replace with...
    qml.CNOT(wires=[3,2])

    M_even(parameters[386:410])     #M2_even
    ProdT2_even()
    ProdT1_even()
    M_even(parameters[410:434])     #M1_even
    ProdT1_even()

    #Exp(1,2,9,12,25,30,49,56,81,90,121,132,169,182,225,240)
    #since it has a ZYZ decomposition, it can be implemented as M_even...
    M_even(parameters[434:458])

    #ZETA FACTOR [Elements: 3,8,15,24,35,48,63,80,99,120,143,168,195,224,255]

    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[458],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[0,3])
    qml.RZ(parameters[459],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[460],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RZ(parameters[461],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[462],wires=[3])
    qml.CNOT(wires=[0,3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[463],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[464],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[465],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.RZ(parameters[466],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[467],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[468],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[469],wires=[0])
    qml.RZ(parameters[470],wires=[1])
    qml.RZ(parameters[471],wires=[2])
    qml.RZ(parameters[472],wires=[3])



#M_odd definition -> 34 CNOT, 38 R
def M_odd(parameters):
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
    qml.CNOT(wires=[0,2])
    
    qml.RZ(parameters[7],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[8],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RZ(parameters[9],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[10],wires=[3])
    qml.CNOT(wires=[0,3])
    qml.RZ(parameters[11],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[12],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RZ(parameters[13],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[14],wires=[3])
    
    qml.RY(parameters[15],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RY(parameters[16],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RY(parameters[17],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RY(parameters[18],wires=[3])
    qml.CNOT(wires=[0,3])
    qml.RY(parameters[19],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RY(parameters[20],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RY(parameters[21],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RY(parameters[22],wires=[3])

    qml.RZ(parameters[23],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[24],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RZ(parameters[25],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[26],wires=[3])
    qml.CNOT(wires=[0,3])
    qml.RZ(parameters[27],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[28],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RZ(parameters[29],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[30],wires=[3])
    qml.CNOT(wires=[0,3])

    qml.CNOT(wires=[0,2])
    qml.RZ(parameters[31],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[32],wires=[2])
    qml.CNOT(wires=[0,2])
    qml.RZ(parameters[33],wires=[2])
    qml.CNOT(wires=[1,2])
    qml.RZ(parameters[34],wires=[2])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[35],wires=[1])
    qml.CNOT(wires=[0,1])
    qml.RZ(parameters[36],wires=[1])
    qml.RZ(parameters[37],wires=[0])

#ProdT7_odd definition -> 5 CNOT
def ProdT7_odd():
    qml.CNOT(wires=[0,3])
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[3,1])
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[0,3])

#ProdT6_odd definition -> 4 CNOT
def ProdT6_odd():
    qml.CNOT(wires=[0,3])
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[3,1])
    qml.CNOT(wires=[0,3])

#ProdT5_odd definition -> 4 CNOT
def ProdT5_odd():
    qml.CNOT(wires=[0,3])
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[0,3])

#ProdT4_odd definition -> 3 CNOT
def ProdT4_odd():
    qml.CNOT(wires=[0,3])
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[0,3])

#ProdT3_odd definition -> 4 CNOT
def ProdT3_odd():
    qml.CNOT(wires=[1,3])
    qml.CNOT(wires=[3,1])
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[1,3])

#ProdT2_odd definition -> 3 CNOT
def ProdT2_odd():
    qml.CNOT(wires=[1,3])
    qml.CNOT(wires=[3,1])
    qml.CNOT(wires=[1,3])

#ProdT1_odd definition -> 3 CNOT
def ProdT1_odd():
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[2,3])

#M_even definition -> 22 CNOT, 24 R
def M_even(parameters):
    qml.RZ(parameters[0],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[1],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RZ(parameters[2],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[3],wires=[3])
    qml.CNOT(wires=[0,3])
    qml.RZ(parameters[4],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[5],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RZ(parameters[6],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[7],wires=[3])
    
    qml.RY(parameters[8],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RY(parameters[9],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RY(parameters[10],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RY(parameters[11],wires=[3])
    qml.CNOT(wires=[0,3])
    qml.RY(parameters[12],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RY(parameters[13],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RY(parameters[14],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RY(parameters[15],wires=[3])

    qml.RZ(parameters[16],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[17],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RZ(parameters[18],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[19],wires=[3])
    qml.CNOT(wires=[0,3])
    qml.RZ(parameters[20],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[21],wires=[3])
    qml.CNOT(wires=[1,3])
    qml.RZ(parameters[22],wires=[3])
    qml.CNOT(wires=[2,3])
    qml.RZ(parameters[23],wires=[3])
    qml.CNOT(wires=[0,3])

#ProdT7_even definition -> 3 CNOT
def ProdT7_even():
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[3,1])
    qml.CNOT(wires=[3,2])

#ProdT6_even definition -> 2 CNOT
def ProdT6_even():
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[3,1])

#ProdT5_even definition -> 2 CNOT
def ProdT5_even():
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[3,2])

#ProdT4_even definition -> 1 CNOT
def ProdT4_even():
    qml.CNOT(wires=[3,0])

#ProdT3_even definition -> 2 CNOT
def ProdT3_even():
    qml.CNOT(wires=[3,1])
    qml.CNOT(wires=[3,2])

#ProdT2_even definition -> 1 CNOT
def ProdT2_even():
    qml.CNOT(wires=[3,1])

#ProdT1_even definition -> 1 CNOT
def ProdT1_even():
    qml.CNOT(wires=[3,2])


