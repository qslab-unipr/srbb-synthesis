# Approximate Gate Synthesis with the Standard Recursive Block Basis (SRBB) decomposition (v1.0.0)

## Acknowledgements

We acknowledge financial support from the European Union - NextGenerationEU, PNRR MUR project PE0000023-NQSTI.

## Description

This repository documents the implementation of the SRBB-based approximate synthesis algorithm first published at QCE24 [1] and then extended in a preprint (arXiv:2412.03083). The algorithm exploits Quantum Neural Networks (QNNs) to approximate a target unitary operator, where the related Variational Quantum Circuits (VQCs) are scalable and recursive structured ansatz derived from the SRBB decomposition [2]. Compared to previous literature, this implementation translates the unitary synthesis problem via SRBB into a quantum context, where the approximation process is handled in a hybrid manner via a single-layer QNN. This is also possible thanks to a CNOT gate optimization algorithm and the identification of a recursive structure within the diagonal subalgebra (Z-block).

[1] G. Belli, M. Mordacci and M. Amoretti, *"A Scalable Quantum Neural Network for Approximate Unitary Synthesis"*, IEEE International Conference on Quantum Computing and Engineering (QCE), Montreal, QC, Canada, 2024, pp. 49-54, doi: 10.1109/QCE60285.2024.10251.

[2] R. S. Sarkar and B. Adhikari, *"Scalable Quantum Circuits for N-Qubit Unitary Matrices"*, IEEE International Conference on Quantum Computing and Engineering (QCE), Bellevue, WA, USA, 2023, pp. 1078-1088, doi: 10.1109/QCE57702.2023.00122.

All modules have been developed in Python with the support of the PennyLane library, which enables quantum computing simulations. The repository is organised into various files:
- **circuit_main.py** is the main program to approximate a target unitary using the QNN designed on the SRBB decomposition. It also allows to run a wide range of performance tests by modifying the learning process parameters (such as loss function, learning rate, batch size, etc.), visualize the variational circuit and its features, test two different types of optimization, and compare it with other algorithms known in the literature (such as Vidal et al. [3]);
- **config.py** contains the methods necessary to initialize the type of device on which to perform the approximate synthesis algorithm (the simulator provided by PennyLane or real IBM hardware);
- **ideal_circuitNx.py** contains the entire set of quantum circuits used to initially test the approximate synthesis algorithm. They are divided by number of qubits (from 2 to 6) and from them the target ideal operators are derived in matrix representation;
- **USynthesis_circuitNgeneral.py** contains the scalable VQC designed on the SRBB decomposition and optimized in terms of CNOT gates. It is used to optimize the parameters according to the metric criterion defined via loss function, in order to obtain the specified target operator;
- **USynthesis_circuitNx.py** are non-scalable versions with a fixed number of qubits (from 2 to 4) of the *USynthesis_circuitNgeneral.py* module. They are used for development purposes (future scalable optimizations) and testing.

[3] G. Vidal and C. M. Dawson, *"Universal quantum circuit for two-qubit transformations with three controlled-NOT gates"*, Physical Review A **69**, 1 (2004).

## Requirements

The project has been implemented using Python 3.9 (Python 3.10 or later should also work) and version 0.38.0 of the *PennyLane* library. The implementation also makes use of the *cmath*, *numpy*, *matplotlib*, *autograd*, *scipy*, *os*, *re*, *csv*, *pickle*, *sympy* and *time* modules. Some of these are already present in a base Python installation, while others are dependent on the installed *PennyLane* module, such as *scipy* and *numpy*.


## Functions & Methods

More details about the module **circuit_main.py**:
- **get_args()**: utilizes the parameters and variables given to the program when running the script from the terminal. Each parameter is a customizable aspect of the algorithm and will yield a different result. There are numerous parameters that can be chosen:
    - --n is the number of qubits of the quantum register. This number also labels the folder of the results. The default value is 2;
    - --run_name decides the execution name and the subdirectory in which the results are saved;
    - --epochs is the number of epochs of the training process. The default value is 2;
    - --batch_size is the number of elements in each batch. The default value is 64;
    - --lr is the learning rate of the training process. The default value is 0,01;
    - --opt is the optimizer used for the network training and can be either Adam or Nelder Mead. The default choice is Adam;
    - --loss decides the loss function (metric criterion). There are 12 different loss functions. The default selection is the Trace Distance;
    - --num_samples sets the number of states of the training set. The default value is 1000;
    - --test_size defines the size of the test set. The default value is 500;
    - --unitary sets the name of the ideal operator to approximate, deriving it from the *ideal_circuit* modules. If not specified, the program will consider all possible unitaries based on the number of qubits chosen; 
    - --num_random is the number of random unitaries to test. It applies only if the chosen unitary is *random*;
    - --num_layer is the number of repetitions of the VQC. As this functionality is currently broken, any reference to this parameter has been commented;
    - --only_test selects the test mode, which measures all possible metrics between the target unitary and the approximated one. Requires to have executed the program at least once;
    - --device chooses the execution device, which can be either the PennyLane simulator or IBM Real Hardware. The default choice is the simulator;
    - --VQC defines the parameterized quantum circuit to be considered as ansatz for the synthesis algorithm. It can be the circuit designed on the SRBB (scalable decomposition) or the minimum one defined by Vidal for 2 qubit;
    - --path determines the directory in which to save the files and from which to reload them for the testing phase;
    - --newtest requires that the *random* case be chosen and that the number of qubits be 2 or 3.
- **main()**: unlike a traditional entry point, this function is primarily used to set up the execution environment for the subsequent call of the main function *main_program()*, which is responsible for the   training and testing phase of the circuit. The function includes:
    - Device initialization: the backend for executing quantum circuits is selected. The device selection function is defined in the config.py module;
    - Loading of ideal circuits: depending on the number of qubits specified by the user (parameter $n$), a set of ideal circuits is imported from an external module. These circuits represent the target unitary matrices to be synthesized. If no specific circuit is provided, the program trains on all circuits in the list of ideal circuits. If the unitary parameter is set to "random", the program generates a set of random unitary matrices and iteratively trains on each of them. The number of matrices to generate is determined by the num_random parameter;
    - Management of the output path: a custom path is configured for saving output files (metrics, logs), using the run_name parameter to distinguish between different executions.
- **main_program()** represents the main function to perform the approximate synthesis algorithm, given the target unitary and some preliminary steps motivated by the SRBB decomposition; 
- **cost()** chooses which cost function to calculate, defined by the --loss parameter specified in the command line. This function is called inside the network training function and executed for each epoch in the case of the Adam optimizer; otherwise, it is used as the optimising function in the *scipy.optimize.fmin* method in the Nelder Mead optimisation;
- **training()** executes the training of the quantum neural network. It returns the optimised parameters, the approximated target matrix, and the total cost.

The loss functions utilised are defined as follows:
- **traceDistanceLoss(Y_ideal, predictions)** calculates the trace distance between the ideal and synthesized density matrices. Y_ideal is the list of ideal density matrices and predictions is the list of calculated predictions: $||\rho - \sigma||_1 = \text{Tr}\sqrt{(\rho - \sigma)^\dagger (\rho - \sigma)}$
- **fidelityLoss(Y_ideal, predictions, dev)** calculates the fidelity between the ideal and synthesized density matrices: $F(\rho,\sigma) = (\text{Tr}\sqrt{\sqrt{\rho}\sigma \sqrt{\rho}})^2 $
- **hellinger(prob1, prob2)** calculates the Hellinger distance between 2 probabilities distribution. It works only if the device is set to "hw";
- **frobeniusLoss(params, x_max, SU_ideal, rot_count, VQC, U_ideal, dev, n_qubit)** calculates the Frobenius Norm between the ideal unitary matrix and the synthesized one: $||{A}_F|| = \sqrt{\text{Tr}(A^\dagger A)}$
- **qfastLoss(params, x_max, SU_ideal, rot_count, VQC, dev, n_qubit)** calculates the loss QFAST proposed in the paper arXiv:2003.04462 between the ideal unitary matrix and the one synthesized by the QNN: $\Delta(U_C,U_T ) = 1 - \frac{|\text{Tr}(U_T^\dagger U_C)|}{d}$, where $U_C$ and $U_T$ are the ideal unitary matrix and the target unitary matrix respectively;
- **hilbert_schmidt_inner_product(U, V)** calculates the Hilbert-Schmidt inner product between two operators U and V: $F_{HS} = \frac{1}{d}\Re(\text{Tr}(A^\dagger B))$
- **operator_norm(U, V)** calculates the Operator Norm between two operators U and V: $||{A}|| = \sup_{||{\psi}||=1} ||{A\psi}||$. **This loss metric currently does not work: it does not calculate the descending gradient properly and needs to be fixed**;
- **choi_matrix(U)** calculates the Choi matrix implementation of a unitary operator U. This function is used in the Choi Trace Distance loss to compute two Choi matrices and calculate their distance;
- **choi_trace_distance(U, V)** calculates the Choi Trace Distance between two Choi matrices U and V: $D_{\text{tr}}(U, V) = \frac{1}{2} ||J(U) - J(V)||_1$, where $J(U)$ and $J(V)$ are the corresponding Choi matrix representation of operators U and V. **This loss metric currently does not work: it does not calculate the descending gradient properly and needs to be fixed**;
- **geodetic_distance(U, V)** calculates the Geodetic Distance between two operators U and V: $D_{\text{geo}}(U, V) = \arccos (\frac{1}{d}|\text{Tr}(U^\dagger V)|)$
- **average_gate_fidelity(U, V)** calculates the Average Gate Fidelity between two operators U and V: $\overline{F}(U, \mathcal{V}) = \frac{|\text{Tr}(U^\dagger V)|^2+d}{d(d+1)}$
- **opfidelity(U, V)** calculates the Operator Fidelity between two operators U and V:
$F(U, V) = \frac{1}{d^2} \left| \text{Tr}(U^\dagger V) \right|^2$
- **bures_distance(U, V)** calculates the Bures Distance between two operators U and V: $D_B(U, V) = \sqrt{1 - \frac{1}{d^2}|\text{Tr}(U^\dagger V)^2|}$
- **state_induced_distance(U, V, psi)** calculates the State Induced Distance between two operators U and V. A statevector $\ket{\psi}$ is created as to calculate the distance between the two modified operators: $d_{\psi}(U, V) = ||U\ket{\psi}-V\ket{\psi}||$. **This loss metric currently does not work: it does not calculate the descending gradient properly and needs to be fixed**.

More details about the module **USynthesis_circuitNgeneral.py**:
- **make_vidal_qnode(dev, n_qubit)** creates a qnode using the Vidal parameterized circuit;
- **vidalCircuit(params)** constructs the minimal circuit proposed by Vidal et al. [3];
- **make_circuit_qnode(dev, n_qubit)** creates a qnode using the VQC designed on the SRBB;
- **make_amplitude_density_qnode(dev, n_qubit)**: creates the amplitude encoded qnode. It is used in the main code as a component to create the final ideal density matrix;
- All other functions implement the construction of the VQC based on the algebraic decomposition of the SRBB. The mathematical details are presented in the paper arXiv:2412.03083, where the CNOT gate optimization algorithms and the composition of the $Z,\Psi,\Phi$ main blocks are transcribed via pseudocode.


## Usage

The code can be tested by running the **circuit_main.py**. An example of a correct command line syntax is the following:
```
python3.9 circuit_main.py --n 2 --unitary random --epochs 20 --loss frob --path ../results/
```
which will approximate a random unitary matrix with 2 qubits in 20 epochs with the Frobenius loss.

The output file directory where the results are saved is the following: path/N(number of qubits)/run_name/unitary/. An example is: ../N2/run_1/random/.

If no parameters are specified, the default ones will be chosen.

An example of running the program with the above mentioned terminal input command is shown:
```
==============================
[DEBUG] Examined Circuit: random


[DEBUG] STEP 1: UNITARY MATRIX GENERATION

[INFO] The target matrix is unitary.
[INFO] The target matrix is:

  -0.334-0.125j   +0.534+0.205j   +0.728-0.040j   +0.062-0.098j
  -0.469+0.482j   -0.480-0.187j   +0.244+0.241j   +0.402-0.059j
  +0.573-0.278j   -0.190-0.488j   +0.485+0.006j   +0.268+0.115j
  -0.014+0.122j   -0.161+0.328j   +0.137-0.312j   +0.028+0.857j

==============================
```
The first phase of the program generates the target matrix to be approximated. There is also a check for unitarity.
```
==============================

[DEBUG] STEP 2: SU GENERATION

[INFO] Determinant of the chosen matrix:
(-0.05166415503535988+0.9986645157831944j)

[INFO] Roots of det(U) and associated phases

Roots of det(U): [(-0.9188575843226502-0.39458933048518113j), (-0.39458933048518113+0.9188575843226502j), (0.39458933048518113-0.9188575843226502j), (0.9188575843226502+0.39458933048518113j)]

Polar coordinates: [(0.9999999999999999, -2.735971780337036), (0.9999999999999999, 1.9764172000476536), (0.9999999999999999, -1.1651754535421397), (0.9999999999999999, 0.40562087325275703)]

Polar coordinates in order: [(0.9999999999999999, 0.40562087325275703), (0.9999999999999999, 1.9764172000476536), (0.9999999999999999, 3.54721352684255), (0.9999999999999999, 5.118009853637447)]

Sorted roots: [(0.9188575843226502+0.39458933048518113j), (-0.39458933048518113+0.9188575843226502j), (-0.9188575843226502-0.3945893304851811j), (0.394589330485181-0.9188575843226503j)]

[INFO] SU matrix 1/4:
  -0.356+0.017j   +0.572-0.022j   +0.653-0.324j   +0.018-0.115j
  -0.241+0.628j   -0.515+0.017j   +0.319+0.125j   +0.347-0.213j
  +0.417-0.482j   -0.367-0.373j   +0.448-0.186j   +0.292+0.000j
  +0.035+0.117j   -0.019+0.365j   +0.002-0.340j   +0.364+0.777j

[INFO] SU matrix 1/4 determinant:
(1.0000000000000007+1.110223024625157e-16j)

[INFO] SU matrix 2/4:
  +0.017+0.356j   -0.022-0.572j   -0.324-0.653j   -0.115-0.018j
  +0.628+0.241j   +0.017+0.515j   +0.125-0.319j   -0.213-0.347j
  -0.482-0.417j   -0.373+0.367j   -0.186-0.448j   +0.000-0.292j
  +0.117-0.035j   +0.365+0.019j   -0.340-0.002j   +0.777-0.364j

[INFO] SU matrix 2/4 determinant:
(1.0000000000000007+1.110223024625157e-16j)

[INFO] SU matrix 3/4:
  +0.356-0.017j   -0.572+0.022j   -0.653+0.324j   -0.018+0.115j
  +0.241-0.628j   +0.515-0.017j   -0.319-0.125j   -0.347+0.213j
  -0.417+0.482j   +0.367+0.373j   -0.448+0.186j   -0.292-0.000j
  -0.035-0.117j   +0.019-0.365j   -0.002+0.340j   -0.364-0.777j

[INFO] SU matrix 3/4 determinant:
(1.0000000000000004+2.7755575615628923e-16j)

[INFO] SU matrix 4/4:
  -0.017-0.356j   +0.022+0.572j   +0.324+0.653j   +0.115+0.018j
  -0.628-0.241j   -0.017-0.515j   -0.125+0.319j   +0.213+0.347j
  +0.482+0.417j   +0.373-0.367j   +0.186+0.448j   -0.000+0.292j
  -0.117+0.035j   -0.365-0.019j   +0.340+0.002j   -0.777+0.364j

[INFO] SU matrix 4/4 determinant:
(0.9999999999999998+6.106226635438361e-16j)

==============================
```
In this phase of the program, the SU matrices are generated, together with their respective determinants and associated roots. The latter are then converted into polar coordinates, organised with the *bubble_sort* function and saved onto a figure file in the form of a .PNG image.
```
==============================

[DEBUG] STEP 3: NETWORK TRAINING

[BEGIN NETWORK TRAINING]

[TRAINING] Optimizing parameters with ADAM

------------------------------
[TRAINING] EPOCH 1
------------------------------

Cost (loss): 2.3379326016561857

[MATRIX] SU_approx after EPOCH 1
------------------------------

  +0.245+0.267j   +0.208+0.021j   +0.249+0.167j   +0.761-0.395j
  -0.629+0.105j   -0.229-0.088j   -0.498+0.228j   +0.098-0.472j
  +0.215+0.002j   -0.923-0.209j   +0.161-0.016j   +0.169+0.059j
  +0.514+0.389j   -0.005+0.024j   -0.704-0.298j   -0.008-0.004j

------------------------------
```
This section follows the SU generation and focuses on parameter optimization (or network training). The chosen optimizer is used a number of times equal to the given number of epochs, with the corresponding loss function chosen by the user. For each epoch, the approximate SU matrix is shown.
```
------------------------------

[TRAINING] Choosing best SU

[INFO] Spectral norm distance between each ideal SU and SU_approx

  Δ(SU_0) spectral norm: (0.005885213297844609-1.989686114566825e-19j)
  Δ(SU_1) spectral norm: (1.416438437224486-5.486269136966177e-18j)
  Δ(SU_2) spectral norm: (1.9999998397210619-1.0433097086404474e-17j)
  Δ(SU_3) spectral norm: (1.418368913800605+1.0043041402388963e-17j)

[INFO] Best matching SU index (min norm): 0

[INFO] Approximate Unitary:

  -0.334-0.126j   +0.534+0.206j   +0.728-0.039j   +0.061-0.099j
  -0.470+0.481j   -0.479-0.188j   +0.243+0.241j   +0.403-0.059j
  +0.574-0.277j   -0.188-0.488j   +0.485+0.007j   +0.268+0.115j
  -0.013+0.121j   -0.160+0.328j   +0.136-0.312j   +0.034+0.857j

[DEBUG] END TRAINING

==============================
```
The final section of the network training process is the selection of the best SU matrix. There are multiple ideal SU matrices derived from the roots of the determinant, and only one approximated SU. Therefore, the SU approximated by the network is compared with the set of ideal SU matrices computed during the preliminary phase.
```
==============================

[DEBUG] STEP 4: LOSS CALCULATION

[INFO] TESTING VECTOR
[-0.4005281 -0.32835403j  0.43523694-0.06285049j -0.12012624-0.0250944j
 -0.20866606+0.69266004j]

[INFO] TRACE DISTANCE between ideal and synth density matrix
0.004284923514114934

[INFO] DENSITY MATRIX EVOLUTION COMPARISON via trace distance
0.004284923514115009

[INFO] Synthesized statevector:
[ 0.30525978+0.26564966j  0.05913164+0.16596557j -0.62697058-0.12915815j
 -0.62870677-0.01262764j]

[INFO] Synthesized probabilities:
[0.16375328 0.03104112 0.40977394 0.39543166]

[INFO] Synthesized density:
[[ 0.16375328+0.j          0.06213921-0.03495431j -0.22569972-0.12712773j
  -0.19527342-0.16316103j]
 [ 0.06213921+0.03495431j  0.03104112+0.j         -0.05850961-0.09641819j
  -0.03927222-0.10359698j]
 [-0.22569972+0.12712773j -0.05850961+0.09641819j  0.40977394+0.j
   0.39581161+0.07328545j]
 [-0.19527342+0.16316103j -0.03927222+0.10359698j  0.39581161-0.07328545j
   0.39543166+0.j        ]]

==============================

[DEBUG] STEP 5: LOSS EVALUATION AND ERROR RATE

[INFO] Test Set Error Rate
0.0029542989597691276

[INFO] Test Complete

[INFO] Results posted in : run_1_random_Results.txt
```
The final phase of the program generates a normalized vector and calculates the trace distance (if using the frobenius norm) or frobenius norm (if using the trace distance) between the ideal density matrix and the synthesized one. This final check verifies the correct unitary evolution. Finally, the test set error is calculated by measuring the trace distance between the ideal and synthesized density matrices. The synthesized density matrix is calculated by executing the VQC for each vector in the test set.

The resulting files are then saved in the previously described directory.

## File Saving and Result Organising

As previously anticipated in the *Usage* section, the results are saved in a tree of subdirectories consisting of the *path* defined in the parameters, the *number of qubits*, the *run_name* and the name of the approximated *unitary*.

Each execution creates 6 initial files:
- 3 files .PKL used for the test mode flag;
- 1 file .PNG, the plot of the roots associated to SU matrices;
- *"Results.txt"*, containing the ideal and approximate SU, the test set error, the evolution rate, the loss and every useful parameter which can be used to do subsequent data analysis;
- *"SU_Results.txt"* containsa a list of all the ideal SU matrices together with their associated determinant and roots. At the end of the list is the best approximated SU generated by the VQC.

The most important data, such as Machine Learning parameters, test set error, evolution rate and the type of circuit examined, is all collected into a .CSV file corresponding to the examined case (e.g. Frobenius loss with the CNOT unitary). These files are created and stored in the same directory where the results files are saved. They are ready to be used in data analysis programs to construct diagrams and tables.
