#SRBB-SYNTHESIS CIRCUIT_MAIN.PY, v1.1.0

#Main script to run the approximate synthesis algorithm designed on SRBB decomposition.
#The algorithm uses a QNN to approximate unitary operators and the corresponding VQC is based on the SRBB decomposition.
#The scalable VQC is optimized in terms of CNOTs.
#This script covers simulation and execution on real hardware with different metrics.

import USynthesis_circuitNgeneral as USC
import config
import importlib
import inspect
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import argparse
import autograd.numpy as anp
import cmath
import sympy as sp
import pickle
import scipy.optimize
import time
import os
import re
import csv
import sys
from unitary_generation import generators, checks, io_utils, hamiltonians
from scipy.stats import unitary_group
from scipy.linalg import svdvals
"""
HW device import
#from qiskit.quantum_info.analysis import hellinger_fidelity
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_args():

	"""
	get_args() utilizes the parameters and variables given to the program when running the script from the terminal.
	Each parameter is a customizable aspect of the circuit and will yield a different result.
	ALL PARAMETERS ARE CASE-SENSITIVE.
	"""

	parser = argparse.ArgumentParser(description="global settings for the qnn")	 

	parser.add_argument('--n', type = int, default = 2, help = 'number of qubits')
	parser.add_argument('--run_name', type = str, default = 'run_1', help = 'name of the current run')
	parser.add_argument('--epochs', type = int, default = 2, help = 'number of epochs')
	parser.add_argument('--batch_size', type = int, default = 64, help = 'number of elements in each batch')
	parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')
	parser.add_argument('--opt', type = str, default = 'Adam', choices = ['Adam', 'Nelder_Mead'], help = 'optimizer used for training')
	parser.add_argument('--loss', type = str, default = 'trace_distance', choices = ['trace_distance', 'frob', 'fidelity', 'qfast', 'hilbert', 'opnorm', 'geodetic', 'avg_fidelity', 'opfidelity', 'bures'], help = 'loss function')
	parser.add_argument('--num_samples', type = int, default = 1000, help = "if trace of fidelity losses are used, you can set the number of states for the training set")
	parser.add_argument('--test_size', type = int, default = 500, help = "size of the test set")
	parser.add_argument('--unitary', type = str, nargs = '+', default = None, help = 'name of the VQC to approximate')
	parser.add_argument('--num_random', type = int, default = 1, help = "how many random unitaries you want to test")
	#parser.add_argument('--num_layer', type = int, default = 1, choices = range(1, 10), help = 'number of repetitions of the VQC')
	parser.add_argument('--only_test', type = bool, default = False, help = 'to select the test mode')
	parser.add_argument('--device', type = str, default = "sim", choices = ["sim", "hw"], help = 'to choose the execution on the pennylane simulator or on ibm real hw')
	parser.add_argument('--VQC', type = str, default = 'VQC', choices = ["VQC", 'Vidal'], help = 'parameterized quantum circuit to be executed')
	parser.add_argument('--path', type = str, default = "", help = "where to save/load the files")
	parser.add_argument('--qchem', type = bool, default=False, help = 'test the experimental matrices')

	return parser.parse_args()

def main(args):
	config.n_qubit = args.n
	dev = config.set_device(args.n, args.device) #dynamic initialization of the device
	print("Device:", dev)
	print("Device type:", type(dev))
	config.device = dev
	config.device_type = args.device
	USC.init(config.device, config.n_qubit)

	if args.qchem == True:
		ideal_module_name = 'qchem_module' #select unitaries defined within the quantum chemistry unitaries module
	else:
		ideal_module_name = f'ideal_circuitN{config.n_qubit}' #selection of the ideal unitary to approximate based on n

	ideal_module = importlib.import_module(ideal_module_name) #import the ideal module containing the premade circuits based on n

	if args.unitary == None:
		print("No unitary circuit specified!")
		#Executes this if the --unitary param is not set in the execution command
		#list of the function names of the ideal unitaries in the chosen module
		circuits = [name for name, func in inspect.getmembers(ideal_module) if isinstance(func, qml.QNode)]
		#executes a test with all ideal circuits in the module
		for c in circuits:
			print(c)
			#main_program(c, ideal_module, args, dev)
		sys.exit() #exits as no unitary is chosen
	else:
		#otherwise approximate only the choice given by the user
		#one or more random unitary matrices
		if args.unitary[0] == 'random':
			for i in range(args.num_random):
				main_program(args.unitary[0], ideal_module, args, dev)
				#searches for an eventual number in args.run_name and increments it
				match = re.match(r"(.*?)(\d+)$", args.run_name)
				if match:
					prefix, number = match.groups()
					number = int(number) + 1
					args.run_name = f"{prefix}{number}"
				else:
					args.run_name = f"{args.run_name}_r{i+1}"
		else:
			#one or more of the predefined circuits 
			for c in args.unitary:
				main_program(c, ideal_module, args, dev)

def main_program(c, ideal_module, args, dev):
	"""
	Main code: execution of training and testing.
	parameter c: string. Name of the ideal circuit/unitary to approximate.
	parameter ideal_module: string. Name of the module where the circuit c is defined.
	parameter args: arguments passed.
	parameter dev: device chosen.
	"""
	############# PRELIMINARY OPERATIONS ###############
	#define the location of the results directory and create both datasets
	#directory of the results -> subdirectories organized by qubits, run name and unitary

	results_dir=os.path.join(args.path, f"N{config.n_qubit}/{args.run_name}/{c}")
	os.makedirs(results_dir, exist_ok=True)
	pathName = args.run_name + "_" + c

	#create the dataset and the testset
	X_train, X_test = create_dataset(args)

	#the ideal circuit/unitary labeled by c variable (its name) as function

	circuit = getattr(ideal_module, c)
	
	print("\n" + "="*30)
	print("[DEBUG] Examined Circuit: " + c + "\n")	
	print("\n[DEBUG] STEP 1: UNITARY MATRIX GENERATION\n")	
	#create a/an random/empty unitary matrix for the circuit function
	#if the qchem flag is true, it executes the unitary matrices related to quantum chemistry
	if c == 'random':
		U = unitary_group.rvs(2**config.n_qubit)
	else:
		U = np.empty(shape = (2**config.n_qubit, 2**config.n_qubit), dtype=complex)

	#return the ideal density matrix generated by the circuit (random or predefined) for each training sample
	#initialise the circuit with the respective U matrix, then compute all of the ideal density matrices
	y_ideal = []
	if args.qchem == True:
		if c == 'predefined':
			#enters this branch when selecting predefined as the unitary, as it is needed to load the hamiltonian matrix
			print("List of predefined hamiltonians:\n")
			hlist = hamiltonians.getList()
			for h in hlist:
				print(h)
			funcname = input("\nName of the predefined hamiltonian matrix to import (CASE-SENSITIVE): ").strip()
			U = hamiltonians.generate_unitary(funcname)
			"""
			if os.path.exists(filename):
				H = io_utils.load_npy(filename)
				print("Loaded File\n")
			else:
				print(f"Error: file '{filename}' does not exist.")
			"""
			if checks.is_unitary(U):
				print("\n[INFO] The target matrix is unitary.")
				"""
				save_choice = input("Save matrix? (y/n): ").strip().lower()
				if save_choice == "y":
					filename = input("Insert filename (no extension): ").strip()
					io_utils.save_npy(filename, U)
					print(f"Matrix saved as {filename}.npy")
				"""
			else:
				print("[INFO] The target matrix is NOT unitary!")
		else:
			print("[INFO] Proceeding with specified ideal unitary.")
		y_ideal = [circuit(x, U) for x in X_train]
		print("[INFO] The target matrix is:\n")
		format_matrix(U)
		print("\n" + "="*30 + "\n")
	else:
		for i, x in enumerate(X_train):
			if i == 0:
				# Execute only on the first iteration, initialises the ideal U
				_ = circuit(x, U)  
				identity = np.eye(U.shape[0])
				check = np.allclose(U @ U.conj().T, identity)
				if check:
					print("[INFO] The target matrix is unitary.")
					print("[INFO] The target matrix is:\n")
					format_matrix(U)
					print("\n" + "="*30 + "\n")
				else:
					print("\n" + "="*30)
					print("\n[WARNING] The target matrix is NOT unitary!\n")
					print("\n" + "="*30)
			y_ideal.append(circuit(x, U))

	#save the target matrix U in a param file
	with open(os.path.join(results_dir, f'U_target_{pathName}.pkl'), 'wb') as f:
		pickle.dump(U, f)
	

	#calculate the determinant of the unitary that we want to approximate
	print("[DEBUG] STEP 2: SU GENERATION\n")	
	det = np.linalg.det(U)
	print("[INFO] Determinant of the chosen matrix:")
	print(det)

	#calculate the 2^n roots of the determinant, related to the 2^n possible SU
	z = sp.symbols('z', complex=True)
	equation = z**(2**config.n_qubit) - det 
	solutions = sp.solve(equation, z)
	solutions = [complex(sol.evalf()) for sol in solutions] #convert to complex number after calculating the numeric value
	print("\n[INFO] Roots of det(U) and associated phases")
	print("\nRoots of det(U):", solutions)

	#sorting the roots (unitary cfr counterclockwise starting from 0 radiant)
	polar = [cmath.polar(s) for s in solutions]
	print("\nPolar coordinates:", polar)
	polar = [p if p[1] >= 0 else (p[0], np.pi + (np.pi + p[1])) for p in polar]
	polar = bubble_sort(polar)
	print("\nPolar coordinates in order:", polar)

	#plot polar coordinates
	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
	ax.scatter([p[1] for p in polar], [p[0] for p in polar], c='blue', marker='o')
	ax.set_title("Complex roots in Polar Coordinates")
	#plt.show()
	plt.savefig(results_dir + "/" + pathName + "_root_plot.png")

	solutions = [cmath.rect(r, theta) for (r, theta) in polar]
	print("\nSorted roots:", solutions)

	#calculate each SU associated to the 2^n roots
	listSU = []
	for i in range(len(solutions)):
		SU = U / solutions[i]
		print(f"\n[INFO] SU matrix {i+1}/{len(solutions)}:")	
		format_matrix(SU)
		listSU.append(SU)
		print(f"\n[INFO] SU matrix {i+1}/{len(solutions)} determinant:")
		print(np.linalg.det(SU))
	
	#save all the ideal SU
	with open(os.path.join(results_dir, pathName + "_SU_Results.txt"), "a") as file:
		file.write("Examined unitary: " + c + "\n")
		file.write("Number of qubits: " + str(config.n_qubit) + "\n")
		file.write("Loss algorithm: " + args.loss + "\n")
		file.write("Training algorithm: " + args.opt + "\n")
		#file.write("Number of layers: " + str(args.num_layer) + "\n")
		file.write("Learning rate: " + str(args.lr) + "\n")
		file.write("Number of epochs: " + str(args.epochs) + "\n")
		file.write("Number of samples: " + str(args.num_samples) + "\n")
		file.write("Batch size: " + str(args.batch_size) + "\n")
		file.write("Test size: " + str(args.test_size) + "\n")
		file.write("Experimental Matrix: " + str(args.qchem) + "\n")
		file.write("Type of circuit: " + str(args.VQC) + "\n")
		file.write("\nDeterminant: " + str(det) + "\n")
		file.write("\nSorted roots: \n")
		file.write(str(solutions) + "\n")
		file.write("\nIDEAL SU:\n")
		for idx, SU in enumerate(listSU):
			file.write(f"[SU {idx+1}]\n")
			file.write(format_matrix_str(SU))
			file.write("\n\nDeterminant: " + str(np.linalg.det(SU)) + "\n")
			file.write("Associated root: " + str(solutions[idx]) + "\n")
			file.write("\n")
		
	x_max=2**(config.n_qubit-1)-1 #number of ProdT_factors and M_factors

	if args.only_test == False:
		"""
		Begin network training function
		"""
		######################## TRAINING OF THE NETWORK #########################
		#execute the training
		start_time = time.time()
		#the ideal U is given as argument because the vidal circuit for two qubit is used for unitary matrices and not for SU
		params, U_approx, loss, norm, SU_approx, SU_ideal, rot_count = training(X_train, y_ideal, args.batch_size, args.lr, args.epochs, args.loss, args.opt, 
																		  		#num_layer,
																				solutions, listSU, listSU[0], x_max, args.VQC, U, c, pathName)
		end_time = time.time()
		elapsed_time = end_time - start_time
		elapsed_time = str(round(elapsed_time, 3))
	else:
		print("\n[BEGIN TEST MODE]\n")
		print("[TEST MODE] Loading saved matrices for comparison...\n")

		#Rebuild pathname as defined above
		pathName = args.run_name + "_" + c
		result_dir = results_dir
		os.makedirs(result_dir, exist_ok=True)
		
		#Load approximate matrix
		with open(os.path.join(result_dir, f'U_approx_{pathName}.pkl'), 'rb') as f_app:
			U_approx = pickle.load(f_app)

    	#Load ideal matrix
		with open(os.path.join(result_dir, f'U_target_{pathName}.pkl'), 'rb') as f_target:
			U_target = pickle.load(f_target)
		
		#Calculate comparison metrics
		frob = np.linalg.norm(U_target - U_approx, ord='fro')
		fid = np.abs(np.trace(np.dot(np.conj(U_target.T), U_approx))) / (2 ** config.n_qubit)
		trace_dist = 0.5 * np.trace(scipy.linalg.sqrtm((U_target - U_approx).conj().T @ (U_target - U_approx))).real
		hilbert = hilbert_schmidt_inner_product(U_target, U_approx)
		op_norm = operator_norm(U_target, U_approx)
		#choi = choi_trace_distance(U_target, U_approx)
		geo = geodetic_distance(U_target, U_approx)
		gate_fid = average_gate_fidelity(U_target, U_approx)
		bures = bures_distance(U_target, U_approx)
		ofid = opfidelity(U_target, U_approx)
		psi = np.zeros(U_target.shape[0], dtype=complex)
		psi[0] = 1.0
		#sid = state_induced_distance(U_target, U_approx, psi)

		metrics = {
        "Frobenius": frob,
        "Fidelity": fid,
        "Trace Distance": trace_dist,
		"Hilbert-Schmidt Inner Product": hilbert,
    	"Operator Norm": op_norm,
    	"Geodetic Distance": geo,
    	"Average Gate Fidelity": gate_fid,
		"Operator Fidelity": ofid,
		"Bures Distance": bures,
    	}
		
		print("\n[TEST MODE] Loss metrics:\n")
		for k, v in metrics.items():
			print(f"{k}: {v:.6f}")

    	#Save metrics in a file
		with open(os.path.join(result_dir, f'metrics_{pathName}.pkl'), 'wb') as f_out:
			pickle.dump(metrics, f_out)

		#Save metrics in a text file
		with open(os.path.join(results_dir, pathName + '_Metrics.txt'), "a") as file:
			file.write("\n" + "="*30 + "\n")
			file.write("Examined unitary: " + c + "\n")
			file.write("Number of qubits: " + str(config.n_qubit) + "\n")
			file.write("\nExamined Metrics:\n")
			for k, v in metrics.items():
				file.write(f"{k}: {v:.6f}\n")

		print("\n[END TEST MODE]\n")	
		return
	

	print("\n[DEBUG] STEP 4: LOSS CALCULATION\n")
	#create one random state to do a test
	real_part = np.random.normal(size=2**config.n_qubit)
	imag_part = np.random.normal(size=2**config.n_qubit)
	random_complex_vector = real_part + 1j * imag_part

	#normalize the vector
	norm = np.linalg.norm(random_complex_vector)
	normalized_vector = random_complex_vector / norm

	#save the results in the general results file
	if args.only_test == False:
		"""
		print("\n" + "="*60)
		print("           ⬤⬤⬤ TESTING PHASE ⬤⬤⬤")
		print("="*60 + "\n")
		"""
		with open(os.path.join(results_dir, pathName + '_Results.txt'), "a") as file:
			file.write("\n" + "="*30 + "\n")
			file.write("Examined unitary: " + c + "\n")
			file.write("Number of qubits: " + str(config.n_qubit) + "\n")
			file.write("Loss algorithm: " + args.loss + "\n")
			file.write("Training algorithm: " + args.opt + "\n")
			#file.write("Number of layers: " + str(args.num_layer) + "\n")
			file.write("Learning rate: " + str(args.lr) + "\n")
			file.write("Number of epochs: " + str(args.epochs) + "\n")
			file.write("Number of samples: " + str(args.num_samples) + "\n")
			file.write("Batch size: " + str(args.batch_size) + "\n")
			file.write("Test size: " + str(args.test_size) + "\n")
			file.write("Type of circuit: " + str(args.VQC) + "\n")
			file.write("Experimental Matrix: " + ("yes" if args.qchem else "no") + "\n")
			file.write("Training time: " + str(elapsed_time) + ' seconds')
			file.write("\nIDEAL U:\n")
			file.write(format_matrix_str(U))
			file.write("\nAPPROX U:\n")
			file.write(format_matrix_str(U_approx))
			file.write("\nLOSS:\n")
			file.write(str(loss))	
			file.write("\nNORM: \n")
			file.write(str(norm))
		
	if args.device == "sim":
		print("[INFO] TESTING VECTOR")
		print(normalized_vector)

		#save the result
		with open(os.path.join(results_dir, pathName + '_Results.txt'), "a") as file:
			file.write("\nTESTING VECTOR:\n")
			file.write(str(normalized_vector))

		#based on if the circuit trained is the VQC or the Vidal one, calculate the associated density matrix
		if args.VQC == 'VQC':
			dm_synt = USC.circuit([], x_max, rot_count, normalized_vector, U_approx)[2]
		elif args.VQC == 'Vidal':
			dm_synt = USC.vidal([], x_max, rot_count, normalized_vector, U_approx)[2]
		
		#if the loss is the frobenius norm, calculate the trace distance between ideal density matrix and the one of the VQC
		if args.loss == 'frob':
			print("\n[INFO] TRACE DISTANCE between ideal and synth density matrix")
			
			dm_ideal = circuit(normalized_vector, U)
			print(qml.math.trace_distance(dm_synt, dm_ideal))

			with open(os.path.join(results_dir, pathName + '_Results.txt'), "a") as file:
				file.write("\nTRACE DISTANCE:\n")
				file.write(str(qml.math.trace_distance(dm_synt, dm_ideal)))

		#the reciprocal operation is performed if the loss is the trace distance
		elif args.loss == 'trace_distance':
			print("\nFROBENIUS: ")
			Msub = SU_ideal - SU_approx
			Msub_H = (np.conj(Msub)).T
		
			print(np.real(np.trace(np.dot(Msub, Msub_H)) ** 0.5))

			with open(os.path.join(results_dir, pathName + '_Results.txt'), "a") as file:
				file.write("\nFROBENIUS:\n")
				file.write(str(np.real(np.trace(np.dot(Msub, Msub_H)) ** 0.5)))

		#return the amplitude encoded density matrix
		dm_amp = USC.amplitude_density(normalized_vector)

		#calculate the ideal density matrix comprising encoding and circuit and calculate the evolution
		dm_final_ideal = U @ dm_amp @ (np.conj(U)).T

		print("\n[INFO] DENSITY MATRIX EVOLUTION COMPARISON via trace distance")
		print(qml.math.trace_distance(dm_synt, dm_final_ideal))

		with open(os.path.join(results_dir, pathName + '_Results.txt'), "a") as file:
			file.write("\nDENSITY MATRIX EVOLUTION COMPARISON between the ideal and synthesized matrices:\n")
			file.write(str(qml.math.trace_distance(dm_synt, dm_final_ideal)))
		
		if args.VQC == 'VQC':
			state, probs, rho = USC.circuit([], x_max, rot_count, normalized_vector, U_approx)
		elif args.VQC == 'Vidal':
			state, probs, rho = USC.vidal([], x_max, rot_count, normalized_vector, U_approx)
		print("\n[INFO] Synthesized statevector:")
		print(state)
		print("\n[INFO] Synthesized probabilities:")
		print(probs)
		print("\n[INFO] Synthesized density:")
		print(rho)
	print("\n" + "="*30)	


	print("\n[DEBUG] STEP 5: LOSS EVALUATION AND ERROR RATE\n")
	#testing dataset, calculate the trace distance between ideal density matrix and the synthesized one
	error_in_test = 0
	for x in X_test:
		dm_ideal = circuit(x, U)
		if args.VQC == 'VQC':
			dm_synt = USC.circuit([], x_max, rot_count, x, U_approx)[2]
		elif args.VQC == 'Vidal':
			dm_synt = USC.vidal([], x_max, rot_count, x, U_approx)[2]
		if args.device == "sim":
			error_in_test += qml.math.trace_distance(dm_synt, dm_ideal)
		elif args.device == "hw":
			print(dm_synt)
			print(dm_ideal)
			error_in_test += hellinger(dm_ideal, dm_synt)
	error_in_test = error_in_test / len(X_test)

	print("[INFO] Test Set Error Rate")
	print(error_in_test)
	print("\n[INFO] Test Complete")
	with open(os.path.join(results_dir, pathName + '_Results.txt'), "a") as file:
		file.write("\nTEST SET ERROR RATE:\n")
		file.write(str(error_in_test) + '\n')
	print("\n[INFO] Results posted in : " + args.path + pathName + '_Results.txt')

	#Create CSV file for simplified data analysis
	with open(os.path.join(results_dir, pathName + '_Results.txt'), "r") as file:
		content = file.read()

	blocks = content.split("==============================")
	detailed_data = []
	for block in blocks:
		if "Loss algorithm: " + args.loss in block:
			unitary_match = re.search(r"Examined unitary:\s*(\w+)", block)
			loss_match = re.search(r"Loss algorithm:\s*(\w+)", block)
			opt_match = re.search(r"Training algorithm:\s*(\w+)", block)
			layers_match = re.search(r"Number of layers:\s*(\d+)", block)
			circuit_match = re.search(r"Type of circuit:\s*(\w+)", block)
			lr_match = re.search(r"Learning rate:\s*([\d.]+)", block)
			epochs_match = re.search(r"Number of epochs:\s*(\d+)", block)
			samples_match = re.search(r"Number of samples:\s*(\d+)", block)
			batch_match = re.search(r"Batch size:\s*(\d+)", block)
			test_match = re.search(r"Test size:\s*(\d+)", block)
			test_csv = test_match.group(1) if test_match else ""
			exp_match = re.search(r"Experimental Matrix:\s*(yes|no)", block, re.IGNORECASE)
			exp_str = exp_match.group(1).strip().lower() if exp_match else "no"
			exp_bool = exp_str == "yes"
			exp_csv = "Yes" if exp_bool else "No"
			time_match = re.search(r"Training time:\s*([\d.]+)", block)
			error_match = re.search(r"TEST SET ERROR RATE:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", block)
			evolution_match = re.search(r"DENSITY MATRIX EVOLUTION COMPARISON.*?:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", block)
			#print(error_match.group(1))

			if all([unitary_match, loss_match, opt_match, lr_match, epochs_match, samples_match,
					batch_match, time_match, error_match, evolution_match]):
				detailed_data.append([
					len(detailed_data) + 1,  # Exec number
					unitary_match.group(1),
					loss_match.group(1),
					opt_match.group(1),
					layers_match.group(1) if layers_match else "",
					circuit_match.group(1) if circuit_match else "",
					lr_match.group(1),
					epochs_match.group(1),
					samples_match.group(1),
					batch_match.group(1),
					test_csv,
					exp_csv,
					time_match.group(1),
					error_match.group(1),
					evolution_match.group(1)
            	])

	with open(os.path.join(results_dir, args.run_name + "_" + args.loss + "_detailed_results.csv"), "w", newline="", encoding="utf-8") as csvfile:
		writer = csv.writer(csvfile, delimiter=';')
		writer.writerow([
			"Exec", "Unitary", "Loss", "Opt", "Layers", "Circuit", "Learning Rate",
			"Epochs", "Samples", "Batch Size", "Test Size", "Experimental Matrix", "Training Time (s)",
			"Error Rate", "Evolution Rate"
		])
		for row in detailed_data:
			converted = [row[0]]
			for val in row[1:]:
				if isinstance(val, str):
					if 'e' in val.lower():
						converted.append(val)
					elif re.match(r"^\d+\.\d+$", val):
						converted.append(val.replace('.', ','))
					else:
						converted.append(val)
				else:
					converted.append(str(val))
			writer.writerow(converted)
	# END OF MAIN

def create_dataset(args):
	"""
	Creates the dataset for training and testing.
	Basically, it creates a set of random complex vectors, normalizes them, and inserts them
	into the training and test sets. 
	"""
	X_train = []
	for i in range(args.num_samples):
		#step 1: generate random complex vectors
		real_part = np.random.normal(size=2**config.n_qubit)
		imag_part = np.random.normal(size=2**config.n_qubit)
		random_complex_vector = real_part + 1j * imag_part

		#step 2: normalize the vector
		norm = np.linalg.norm(random_complex_vector)
		normalized_vector = random_complex_vector / norm
		X_train.append(normalized_vector)
	
	X_test = []
	for i in range(args.test_size):
		#step 1: generate random complex vectors
		real_part = np.random.normal(size=2**config.n_qubit)
		imag_part = np.random.normal(size=2**config.n_qubit)
		random_complex_vector = real_part + 1j * imag_part

		#step 2: normalize the vector
		norm = np.linalg.norm(random_complex_vector)
		normalized_vector = random_complex_vector / norm
		X_test.append(normalized_vector)

	return X_train, X_test

def bubble_sort(polars):
	"""
	Sorts the 2^n roots of the determinant.
	
	Parameters:
		polars: the polar coordinates of the root.
	"""
	n = len(polars)
	for i in range(n):
		already_sorted = True
		for j in range(n - i - 1):
			if polars[j][1] > polars[j + 1][1]:
				polars[j], polars[j + 1] = polars[j + 1], polars[j]
				already_sorted = False
		# If there were no swaps during the last iteration, the array is sorted, ends function
		if already_sorted:
			break

	return polars

def format_matrix(matrix, decimals=3):
    """
    Pretty-print a complex matrix with aligned columns.

    Parameters:
        matrix (np.ndarray): the matrix to format
        decimals (int): number of decimals to show
    """
    rows = []
    for row in matrix:
        formatted_row = []
        for elem in row:
            real = np.round(elem.real, decimals)
            imag = np.round(elem.imag, decimals)
            formatted = f"{real:+.3f}{imag:+.3f}j"
            formatted_row.append(f"{formatted:>15}")  # aligned to the right
        rows.append(" ".join(formatted_row))
    print("\n".join(rows))

def format_matrix_str(matrix, decimals=3):
    """
    Return a formatted string of a complex matrix with aligned columns.
	
	Parameters:
        matrix (np.ndarray): the matrix to format
        decimals (int): number of decimals to show
    """
    rows = []
    for row in matrix:
        formatted_row = []
        for elem in row:
            real = np.round(elem.real, decimals)
            imag = np.round(elem.imag, decimals)
            formatted = f"{real:+.{decimals}f}{imag:+.{decimals}f}j"
            formatted_row.append(f"{formatted:>15}")
        rows.append(" ".join(formatted_row))
    return "\n".join(rows)

def traceDistanceLoss(Y_ideal, predictions):
	"""
	Calculates the trace distance between the ideal states and the states produced by the QNN

	Parameters:
		Y_ideal: List of ideal density matrices
		predictions: List of density matrices produced by the network

	return loss: the mean of the trace
	"""
	num_samples = len(predictions)
	loss = 0.0
	for i in range(len(predictions)):
		loss += qml.math.trace_distance(predictions[i], Y_ideal[i])

	return loss / num_samples

def fidelityLoss(Y_ideal, predictions):
	"""
	Calculates the fidelity between the ideal states and the states produced by the QNN

	Parameters:
		Y_ideal: List of ideal density matrices
		predictions: List of density matrices produced by the network
		device: String. can be "hw" or "sim"

	return loss: the mean of the fidelities
	
	#OLD FUNCTION
	num_samples = len(predictions)
	loss = 0.0
	for i in range(len(predictions)):
		if dev == "sim":
			loss += 1 - qml.math.fidelity(predictions[i], Y_ideal[i])
		elif dev == "real hw":
			loss += 1 - anp.int64(hellinger_fidelity(predictions[i], Y_ideal[i]))

	return loss / num_samples
	"""
	fidelities = [qml.math.fidelity(predictions[i], Y_ideal[i]) for i in range(len(predictions))]
	fidelities = qml.math.stack(fidelities)
	return 1 - qml.math.mean(fidelities)

def hellinger(prob1, prob2):
	"""
	Calculate the Hellinger distance between 2 probabilities distribution

	Parameters:	
		prob1: list of ideal probabilities
		prob2: list of probabilities produced by the QNN

	return distance: hellinger distance
	"""
	distance = (1/np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(prob1) - np.sqrt(prob2))**2))
	return distance

def frobeniusLoss(params, x_max, SU_ideal, rot_count, VQC, U_ideal):
	"""
	Calculates the frobenius Norm between the ideal matrix and the one produced by the VQC

	Parameters:
		params: List of theta parameter that are trained
		x_max: 
		SU_ideal: the ideal matrix
		rot_count: number of rotations 
		VQC: which is the trained VQC. The SRBB VQC or the 2 qubit circuit of Vidal
		U_ideal: the approximated matrix

	return loss: the frobenius norm value 
	"""

	if VQC == 'VQC':
		Msub = SU_ideal - qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
	elif VQC == 'Vidal':
		Msub = U_ideal - qml.matrix(USC.vidal)(params, x_max, rot_count)
	Msub_H = (np.conj(Msub)).T
	
	return np.real(np.trace(np.dot(Msub, Msub_H)) ** 0.5)

def qfastLoss(params, x_max, SU_ideal, rot_count, VQC):
	"""
	Calculates the qfast paper loss between the ideal matrix and the one produced by the VQC

	Parameters:
		params: List of theta parameter that are trained
		x_max: 
		SU_ideal: the ideal matrix
		rot_count: number of rotations 
		VQC: which is the trained VQC. The SRBB VQC or the 2 qubit circuit of Vidal

	return loss: the qfast loss value 
	"""
	if VQC == 'VQC':
		matrix = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
	elif VQC == 'Vidal':
		matrix = qml.matrix(USC.vidal)(params, x_max, rot_count)

	trace = np.trace(np.conj(matrix).T @ SU_ideal)
	loss = qml.math.sqrt(1 - (abs(trace) ** 2) / (SU_ideal.shape[0] ** 4))
	return loss

#Define advanced metrics
def hilbert_schmidt_inner_product(U, V):
	"""
	Calculates the Hilbert-Schmidt inner product between two operators.

	Parameters:
		U, V: two unitary operators
	"""
	d = U.shape[0]
	return np.real(np.trace(np.dot(np.conj(U.T), V))) / d

def operator_norm(U, V):
    """
    Differentiable spectral norm between two operators.

	Parameters:
		U, V: two unitary operators
    """
    diff = U - V
    s = np.linalg.svd(diff, compute_uv=False)
    return s[0]

"""
def choi_matrix(U):
	
	#Calculates the corresponding Choi matrix of a given operator U.

	#Parameters:
		#U: unitary operator
	d = U.shape[0]
	psi = np.eye(d).reshape(d * d, 1) / np.sqrt(d)  # maximally entangled state (unnormalized ket)
	U_kron = np.kron(U, np.eye(d))
	psi_out = U_kron @ psi
	return psi_out @ np.conj(psi_out).T  # density matrix
"""
"""
def choi_trace_distance(U, V, eps=0.0):

    #Trace distance between two Choi matrices using singular values.
    #Returns 0.5 * ||C1 - C2||_1 with all ops backend-safe.
    #eps: valore piccolo per stabilizzare la somma radici se vuoi una versione smoothed.

    C1 = choi_matrix(U)
    C2 = choi_matrix(V)
    diff = C1 - C2

    # usa la SVD del backend e prendi SOLO i valori singolari
    s = np.linalg.svd(diff, compute_uv=False)  # dovrebbe restituire un ArrayBox differenziabile

    if eps == 0.0:
        # norma di traccia esatta
        return 0.5 * np.sum(s)
    else:
        # versione lievemente smoothed: sum sqrt(s_i^2 + eps)
        return 0.5 * np.sum(np.sqrt(s ** 2 + eps))
"""

def geodetic_distance(U, V):
	"""
	Calculates the Geodetic (angular) distance between two operators.

	Parameters:
		U, V: two unitary operators
	"""
	d = U.shape[0]
	trace_uv = np.trace(np.dot(np.conj(U.T), V))
	return np.arccos(np.clip(np.abs(trace_uv) / d, 0.0, 1.0))

def average_gate_fidelity(U, V):
	"""
	Calculates the Average gate fidelity between two operators.

	Parameters:
		U, V: two unitary operators
	"""
	d = U.shape[0]
	fidelity = np.abs(np.trace(np.dot(np.conj(U.T), V)))**2
	return (fidelity + d) / (d * (d + 1))

def opfidelity(U, V):
	"""
	Calculates the Fidelity between two operators U and V, as defined in:
	F(U, V) = (1/d^2) * |Tr(U† V)|^2
	"""
	d = U.shape[0]
	trace_val = np.trace(np.dot(np.conj(U.T), V))
	return (np.abs(trace_val) ** 2) / (d ** 2)

def bures_distance(U, V):
	"""
	Calculates the Bures Distance between two operators, U and V as defined in:
	D_B(U, V) = sqrt(1 - Fidelity(U, V))
	"""
	F = opfidelity(U, V)
	return np.sqrt(1 - F)

"""
def state_induced_distance(U, V, psi):
	
	#Calculates the euclidean distance ||U|psi⟩ - V|psi⟩|| between two evolutions of
	#the psi state.
	
	#Parameters:
		#U, V: unitary operators (d x d)
		#psi: status vector (d)
    
	Up = U @ psi
	Vp = V @ psi
	return np.linalg.norm(Up - Vp)
"""

def cost(params, X, Y_ideal, loss, SU_ideal, VQC, x_max, rot_count, U_ideal):
	"""
	Choose which cost function to calculate, defined by the --loss parameter specified in the command line.
	This function is called inside the network training function.
	
	Parameters:
		params: List of theta parameters that are trained
		X: the analysed dataset
		Y_ideal: the ideal density matrices
		loss: which loss to use
		SU_ideal: the ideal SU matrix
		VQC: type of variational circuit: the SRBB VQC or the 2 qubit circuit of Vidal
		x_max: 
		rot_count: number of rotations
		U_ideal: the approximated matrix

	return loss: the loss value 
	"""
	if loss == 'trace_distance' or loss == 'fidelity':
		if VQC == 'VQC':
			predictions = [USC.circuit(params, x_max, rot_count, X[i]) for i in range(len(X))]
		elif VQC == 'Vidal':
			predictions = [USC.vidal(params, x_max, rot_count, X[i]) for i in range(len(X))]

		predictions, states, probs = zip(*predictions)
		predictions, states, probs = list(predictions), list(states), list(probs)
		if loss == 'trace_distance':
			return traceDistanceLoss(Y_ideal, predictions)
		elif loss == 'fidelity':
			return fidelityLoss(Y_ideal, predictions)
	elif loss == 'frob':
		return frobeniusLoss(params, x_max, SU_ideal, rot_count, VQC, U_ideal)
	elif loss == 'qfast':
		return qfastLoss(params, x_max, SU_ideal, rot_count, VQC)
	elif loss == 'hilbert':
		if VQC == 'VQC':
			approx = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
		elif VQC == 'Vidal':
			approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
		return 1 - hilbert_schmidt_inner_product(SU_ideal, approx)
	elif loss == 'opnorm':
		if VQC == 'VQC':
			approx = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
		elif VQC == 'Vidal':
			approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
		return operator_norm(SU_ideal, approx)
	#elif loss == 'choi':
		#if VQC == 'VQC':
			#approx = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
		#elif VQC == 'Vidal':
			#approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
		#return choi_trace_distance(SU_ideal, approx)
	elif loss == 'geodetic':
		if VQC == 'VQC':
			approx = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
		elif VQC == 'Vidal':
			approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
		return geodetic_distance(SU_ideal, approx)
	elif loss == 'avg_fidelity':
		if VQC == 'VQC':
			approx = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
		elif VQC == 'Vidal':
			approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
		return 1 - average_gate_fidelity(SU_ideal, approx)
	elif loss == 'bures':
		if VQC == 'VQC':
			approx = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
		elif VQC == 'Vidal':
			approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
		return bures_distance(SU_ideal, approx)
	elif loss == 'opfidelity':
		if VQC == 'VQC':
			approx = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
		elif VQC == 'Vidal':
			approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
		return 1 - opfidelity(SU_ideal, approx)
	#elif loss == 'sid':
		#psi = np.zeros(2 ** config.n_qubit, dtype=complex)
		#psi[0] = 1.0  # |0...0>
		#if VQC == 'VQC':
			#approx = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
		#elif VQC == 'Vidal':
			#approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
		#return state_induced_distance(SU_ideal, approx, psi)

def training(X_train, Y_ideal, batch_size, learning_rate, epochs, loss, optimizer, 
			#num_layer,
			listdet, listSU, SU_ideal, x_max, VQC, U_ideal, circuit, pathName):	
	"""
	Execute the training of the neural network.

	Parameters:
		X_train: training dataset
		Y_ideal: training dataset label
		batch_size: the batch size
		learning_rate: the learning rate value
		epochs: how many epochs of training to execute
		loss: which loss function to use
		optimizer: which optimizer to use (Adam or Nelder_Mead)
		num_layer: number of layers to be used by the VQC
		listdet: list of determinants of the different 2^n possibilities of SU
		listSU: list of all possible SU
		SU_ideal: the ideal SU
		x_max:
		VQC: which VQC to use (Vidal or SRBB)
		U_ideal: the ideal U
		circuit: which circuit to approximate
		pathName: String. Name given by the options chosen in arguments

	Returns:
		params: the trained parameters
		U_approx: the approximated U
		cost_new: the value of the cost function
		norm[np.argmin(norm)]: the right norm value
		SU_approx: the approximated SU
		listSU[np.argmin(norm)]: the right SU
		rot_count: number of rotations
	"""
	last_params = None
	last_update_time = time.time()
	stall_time_limit = 300

	best_params = None
	best_loss = None

	class EarlyStopException(Exception):
		pass

	print("\n" + "="*30)
	print("\n[DEBUG] STEP 3: NETWORK TRAINING\n")
	print("[BEGIN NETWORK TRAINING]")
	results_dir=os.path.join(args.path, f"N{config.n_qubit}/{args.run_name}/{circuit}")
	os.makedirs(results_dir, exist_ok=True)

	def callback(params):
		nonlocal last_params, last_update_time, best_params, best_loss	
		print(f"Norm(params): {np.linalg.norm(params)}")

		try:
			if loss == 'frob':
				current_loss = frobeniusLoss(params, x_max, SU_ideal, rot_count, VQC, U_ideal)
			else:
				current_loss = cost(params, X_train, Y_ideal, config.n_qubit, loss, SU_ideal, VQC, x_max, rot_count, U_ideal)
		except Exception as e:
			print(f"[WARNING] Impossibile calcolare la loss nella callback: {e}")
			current_loss = None

		# Update best results if improved
		if best_loss is None or (current_loss is not None and current_loss < best_loss):
			best_params = np.copy(params)
			best_loss = current_loss
		
		# Initialize if first exec
		if last_params is None:
			last_params = np.copy(params)
			last_update_time = time.time()
			return

		# Verify timer if params have not changed, tollerance is atol
		if np.allclose(params, last_params, atol=1e-8):
			if time.time() - last_update_time > stall_time_limit:
				print(f"\n[EARLY STOP] Unchanged params for {stall_time_limit}s. Stopping optimization")
				raise EarlyStopException
		else:
			# Changed params, continue
			last_params = np.copy(params)
			last_update_time = time.time()
			
	if VQC == 'VQC':
		params, rot_count = USC.Theta_array_gen(x_max)
	elif VQC == 'Vidal':
		params = np.random.randn(15, requires_grad = True)
		rot_count = [15]

	if optimizer == 'Adam':
		print("\n[TRAINING] Optimizing parameters with ADAM")
		opt = qml.AdamOptimizer(stepsize=learning_rate)
	elif optimizer == 'GDO':
		opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
	else:
		opt = qml.QNGOptimizer(stepsize=learning_rate)

	if optimizer == 'Nelder_Mead':
		try:
			print("\n[TRAINING] Optimizing parameters with NELDER MEAD")
			if loss == 'frob':
				params = scipy.optimize.fmin(func=frobeniusLoss, x0=params, args=(x_max, SU_ideal, rot_count, VQC, U_ideal), callback=callback, xtol=10**(-15), ftol=10**(-15), maxiter=10**10,maxfun=10**20)
			elif loss == 'trace_distance':
				params = scipy.optimize.fmin(func=cost, x0=params, args=(X_train, Y_ideal, loss, SU_ideal, VQC, x_max, rot_count, U_ideal), callback=callback, xtol=10**(-6), ftol=10**(-6), maxiter=10**10,maxfun=10**20)
			else:
				params = scipy.optimize.fmin(func=cost, x0=params, args=(X_train, Y_ideal, loss, SU_ideal, VQC, x_max, rot_count, U_ideal), callback=callback, xtol=10**(-6), ftol=10**(-6), maxiter=10**10,maxfun=10**20)
		except EarlyStopException:
			print("[INFO] Optimization interrupted: stalled\n")
			if best_params is not None:
				params = best_params

		cost_new = 0.0

		if VQC == 'VQC':
			SU_approx = qml.matrix(USC.circuit)(params, x_max, rot_count)
		elif VQC == 'Vidal':
			U_approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
			SU_approx = U_approx

		print("\n" + "="*30)
		print("[INFO] FINAL APPROXIMATED UNITARY")
		print("="*30 + "\n")
		format_matrix(SU_approx)	
		
		fileParams = open(os.path.join(results_dir, 'params_' + pathName + '.pkl'), 'wb')				
		pickle.dump(params, fileParams)
		fileParams.close()
		
	else:
		for e in range(epochs):
			print("\n------------------------------")
			print(f"[TRAINING] EPOCH {e+1}")
			for b in range(0, len(X_train), batch_size):
				if (b + batch_size) <= len(X_train):
					X_batch = [X_train[i] for i in range(b, b + batch_size)]
					Y_batch = [Y_ideal[i] for i in range(b, b + batch_size)]
				else:
					X_batch = [X_train[i] for i in range(b, len(X_train))]
					Y_batch = [Y_ideal[i] for i in range(b, len(X_train))]
					
				if optimizer == 'Adam':
					params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, loss, SU_ideal, VQC, x_max, rot_count, U_ideal), params)

			print("------------------------------\n")
			print(f"Cost (loss): {cost_new}\n")
			if VQC == 'VQC':
				print(f"[MATRIX] SU_approx after EPOCH {e+1}")
				print(f"------------------------------\n")
				SU_approx = qml.matrix(USC.circuit, wire_order=range(0, config.n_qubit))(params, x_max, rot_count)
				format_matrix(SU_approx)
			elif VQC == 'Vidal':
				print(f"[MATRIX] U_approx after EPOCH {e+1}")
				print(f"------------------------------\n")
				U_approx = qml.matrix(USC.vidal)(params, x_max, rot_count)
				SU_approx = U_approx
				format_matrix(U_approx)
			
			fileParams = open(os.path.join(results_dir, 'params_' + pathName + '.pkl'), 'wb')				
			pickle.dump(params, fileParams)
			fileParams.close()
			
	if VQC == 'VQC':
		with open(os.path.join(results_dir, pathName + '_SU_Results.txt'), "a") as file:
			file.write("\nAPPROX SU:\n")
			file.write(format_matrix_str(SU_approx))
			file.write("\n\n" + "="*30 + "\n\n")
	elif VQC == 'Vidal':
		with open(os.path.join(results_dir, pathName + '_SU_Results.txt'), "a") as file:
			file.write("\nAPPROX U:\n")
			file.write(format_matrix_str(U_approx))
			file.write("\n\n" + "="*30 + "\n\n")

	#choose the right SU or U
	print("\n------------------------------")
	print("\n[TRAINING] Choosing best SU\n")
	DELTA = []

	for SU in listSU:
		DELTA.append(SU - SU_approx)

	norm = []
	for d in DELTA:
		norm.append(cmath.sqrt(max(np.linalg.eig(((np.conj(d)).T) @ d)[0]))) 
		
	print("[INFO] Spectral norm distance between each ideal SU and SU_approx\n")
	for i, n in enumerate(norm):
		print(f"  Δ(SU_{i}) spectral norm: {n}")

	if VQC == 'VQC':
		best_index = np.argmin(norm)
		print(f"\n[INFO] Best matching SU index (min norm): {best_index}")
		U_approx = SU_approx * listdet[best_index]
		if loss == 'frob':
			U_approx = SU_approx * listdet[0]
		print(f"\n[INFO] Approximate Unitary:\n")
		format_matrix(U_approx)

	
	fileu_app = open(os.path.join(results_dir, 'U_approx_' + pathName + '.pkl'), 'wb')		
	pickle.dump(U_approx, fileu_app)
	fileu_app.close()

	print("\n[DEBUG] END TRAINING")
	print("\n" + "="*30)

	return params, U_approx, cost_new, norm[np.argmin(norm)], SU_approx, listSU[np.argmin(norm)], rot_count

if __name__ == "__main__":
	args = get_args()
	print(args)
	main(args)
