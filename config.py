# Configuration module to set the device (sim or hw)

import pennylane as qml
#from qiskit import *
#from qiskit_ibm_runtime import QiskitRuntimeService

def set_device(n_qubit: int, device_type: str ='sim'):
    if device_type == 'sim':
        dev = qml.device('default.qubit', wires = n_qubit)
    
    #elif device_type == 'hw':
        # Or save your credentials on disk.
        # QiskitRuntimeService.save_account(channel='ibm_quantum', token='', overwrite=True, set_as_default=True)

        #service = QiskitRuntimeService(channel="ibm_quantum", instance='ibm-q/open/main')
        #bck = service.least_busy(operational = True, simulator = False, min_num_qubits = n_qubit)
        #print(bck)
        #print(bck.provider)
        #dev = qml.device('qiskit.remote', wires = n_qubit, backend = bck, shots = 1024)
    
    else:
        raise ValueError(f"Device type not recognized: {device_type}")
    
    return dev

"""
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token=''
)
"""