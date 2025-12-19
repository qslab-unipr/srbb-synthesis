import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import os
import matplotlib.pyplot as plt
from pennylane.templates.embeddings import AmplitudeEmbedding
#from pennylane_qiskit.remote import RemoteDevice

#Software implementation of the new CNOT-reduced scalability scheme for the algorithm proposed by Sarkar et al. (arXiv:2405.00012) about approximate unitary synthesis.

#0307177
def make_vidal_qnode(dev, n_qubit):
    @qml.qnode(dev)
    def vidal(params, x_max, rot_count, x = [], U_approx = []):
        if len(U_approx) == 0:
            if len(x) != 0:
                AmplitudeEmbedding(x, range(n_qubit), normalize = True)
                vidalCircuit(params)
            if len(x) == 0:
                return qml.density_matrix(range(n_qubit))
            else:
                return qml.density_matrix(range(n_qubit)), qml.state(), qml.probs(range(n_qubit))
        else:
            AmplitudeEmbedding(x, range(n_qubit), normalize = True)
            qml.QubitUnitary(U_approx, wires = range(n_qubit))
            return qml.state(), qml.probs(range(n_qubit)), qml.density_matrix(range(n_qubit))
    return vidal

def vidalCircuit(params):
	
	qml.U3(params[0], params[1], params[2], wires=0)
	qml.U3(params[3], params[4], params[5], wires=1)
	qml.CNOT(wires=[0, 1])
	qml.RY(params[6], wires=0)
	qml.RZ(params[7], wires=1)
	qml.CNOT(wires=[1, 0])
	qml.RY(params[8], wires=0)
	qml.CNOT(wires=[0, 1])
	qml.U3(params[9], params[10], params[11], wires=0)
	qml.U3(params[12], params[13], params[14], wires=1)
	"""
	qml.Rot(params[0], params[1], params[2], wires = 0)
	qml.Rot(params[3], params[4], params[5], wires = 1)
	qml.CNOT([0, 1])
	
	qml.QubitUnitary((+1j/qml.math.sqrt(2)) * (qml.PauliX.compute_matrix() + qml.PauliZ.compute_matrix()), wires = 0)
	qml.RX(params[6] + np.pi/2, wires = 0)
	qml.RZ(params[7], wires = 1)
	qml.CNOT([0, 1])

	qml.QubitUnitary((-1j/qml.math.sqrt(2)) * (qml.PauliX.compute_matrix() + qml.PauliZ.compute_matrix()), wires = 0)
	qml.RZ(params[8], wires = 1)
	qml.CNOT([0, 1])

	qml.Rot(params[9], params[10], params[11], wires = 0)
	qml.QubitUnitary((np.eye(2) -1j*qml.PauliX.compute_matrix())/qml.math.sqrt(2), wires = 0)

	qml.Rot(params[12], params[13], params[14], wires = 1)
	qml.QubitUnitary(np.linalg.inv((np.eye(2) -1j*qml.PauliX.compute_matrix())/qml.math.sqrt(2)), wires = 1)
    """

def make_circuit_qnode(dev, n_qubit):
	#fpath = "circuit_draw.png"
	#print("DEBUG in factory → dev è QNodeDevice?", isinstance(dev, qml.devices.Device))
	#assert isinstance(dev, qml.devices.Device), f"Invalid device: {type(dev)}"
	@qml.qnode(dev)
	def circuit(params, x_max, rot_count, x = [], U_approx = []):
		#The scalability scheme for ProdT_factors is based on the binary matrix which has as rows the binary representation of numbers from 1 to x_max.
		#This implementation takes into account the reduction in the CNOT-number thanks to a new simplification scheme. 
		#The scalability scheme for M_factors is based on the binary matrix built by Gray Code.
		#This implementation does not reduce the number of rotations (future work).

		#print('\nx_max={}'.format(x_max))
		binary_matrix=np.empty(shape=(x_max,n_qubit-1),dtype=int)
		for i in range(1,x_max+1):
			#print('Binary number for {}'.format(i))
			binary_number_array = dec_bin(i,n_qubit-1,n_qubit)
			binary_matrix[i-1]=binary_number_array
		binary_matrix_reverse=np.flip(binary_matrix,0)	#According to the circuit-wise order, the first element is ProdTxmax
		#print('\nBinary matrix where the ones indicate the target qubits of a CNOT gate:\n')
		#print(format_matrix(binary_matrix))
		#print('\nBinary matrix with rows flipped:\n')
		#print(format_matrix(binary_matrix_reverse))

		control_target_pairs = CNOTgate_indices_even(binary_matrix, n_qubit)

		k_array = Find_k(binary_matrix_reverse, x_max, n_qubit)

		simpli_matrix_even, simpli_count = Simplification_matrix_even(binary_matrix_reverse, x_max, n_qubit)

		simpli_matrix_odd, simpli_count, double_count = Simplification_matrix_odd(binary_matrix_reverse, x_max, n_qubit)

		Gray_matrix_collection = Gray_matrix(n_qubit-1,False)

		simpli_Gray_collection = Simplification_Gray_matrix(Gray_matrix_collection, n_qubit)

	

		if len(U_approx) == 0:
			if len(x) != 0:
				AmplitudeEmbedding(x, range(n_qubit), normalize = True)
				#qml.Barrier(wires = [0, 1])
			SU_approx_circuit(params, binary_matrix_reverse, k_array, simpli_matrix_even, simpli_matrix_odd, simpli_Gray_collection, x_max, rot_count, n_qubit)
			#if not os.path.exists(fpath):
					#qml.draw_mpl(SU_approx_circuit)(params, binary_matrix_reverse, k_array, simpli_matrix_even, simpli_matrix_odd, simpli_Gray_collection, x_max, rot_count, n_qubit)
					#plt.savefig(fpath)
			#qml.draw_mpl(SU_approx_circuit)(params, binary_matrix_reverse, k_array, simpli_matrix_even, simpli_matrix_odd, simpli_Gray_collection, x_max, rot_count, n_qubit)
			#plt.show()
			#input("A")

			if len(x) == 0:
				return qml.density_matrix(range(n_qubit))
			else:
				return qml.density_matrix(range(n_qubit)), qml.state(), qml.probs(range(n_qubit))

		else: #testing
			AmplitudeEmbedding(x, range(n_qubit), normalize = True)
			qml.QubitUnitary(U_approx, wires = range(n_qubit))
			#if isinstance(dev, RemoteDevice):
			#	return qml.counts(wires = range(n_qubit)), qml.counts(wires =range(n_qubit)), qml.probs(range(n_qubit))
			#else:
			return qml.state(), qml.probs(range(n_qubit)), qml.density_matrix(range(n_qubit))
	

		#qml.draw_mpl(SU_approx_circuit)(params, binary_matrix_reverse, k_array, simpli_matrix_even, simpli_matrix_odd, simpli_Gray_collection, x_max)
		#plt.show()
	
	return circuit

def make_amplitude_density_qnode(dev, n_qubit):
	@qml.qnode(dev)
	def amplitude_density(x):
		AmplitudeEmbedding(x, range(n_qubit), normalize = True)
		return qml.density_matrix(range(n_qubit))
	return amplitude_density

#Format useful for checks
def format_matrix(matrix):
	M=('\n'.join(''.join('{:11}'.format(np.round(item,1)) for item in row) for row in matrix))
	return M

#Decimal to binary with units on the right
def dec_bin(decimal_number, bit_number, n_qubit):
	binary_number_list=[]
	if decimal_number==0:
		binary_number_list=[0]*bit_number
	if decimal_number==1:
		b0=[0]*(bit_number-1)
		b1=[1]
		binary_number_list=b0+b1
	else:
		while decimal_number>0:
			if decimal_number%2==0:
				binary_number_list.insert(0,0)
			else:
				binary_number_list.insert(0,1)
			decimal_number=int(decimal_number/2)
	if len(binary_number_list)==n_qubit-1:
		pass
	else:
		b00=[0]*(n_qubit-1-len(binary_number_list))
		binary_number_list=b00+binary_number_list
	#print(binary_number_list)
	return np.array(binary_number_list)

#CNOTs, with appropriate parameters, are the building blocks for ProdT_factors
def CNOTgate_even(position_ones, n_qubit):
	qml.CNOT(wires=[n_qubit-1,position_ones])

#There are differences in the parameter settings between even and odd cases
def CNOTgate_odd(position_k, n_qubit):
	qml.CNOT(wires=[position_k,n_qubit-1])

#Scalability scheme for the whole sequence ProdT_factors_even 
def AllProdT_gatesequence_even(binary_matrix_reverse, n_qubit):
	for i in range(binary_matrix_reverse.shape[0]):
			for j in range(binary_matrix_reverse.shape[1]):
				if binary_matrix_reverse[i][j]==1:
					CNOTgate_even(j, n_qubit)
			qml.Identity(n_qubit-1)#Only used to separate different ProdT_factors_even

#Structure of a single ProdT_factor_even 
def ProdT_gatesequence_even(binary_array, n_qubit):
	cnot_count_even1=0
	for i in range(len(binary_array)):
				if binary_array[i]==1:
					CNOTgate_even(i, n_qubit)
					cnot_count_even1+=1
	return cnot_count_even1

#List of control-target pairs for ProdT_factors_even
def CNOTgate_indices_even(binary_matrix, n_qubit):
	control_target_pairs=[]
	for i in range(binary_matrix.shape[0]):
			pairs_for_row=[]
			for j in range(binary_matrix.shape[1]):
				if binary_matrix[i][j]==1:
					pair=[n_qubit-1,j]
					pairs_for_row.append(pair)
			#print('\nList of indices for the pairs control-target qubit in ProdT{0}_even:\n{1}'.format(i+1,pairs_for_row))
			control_target_pairs.append(pairs_for_row)
	#print('\nList of indices for the pairs control-target qubit in the whole sequence ProdT_even:\n{}'.format(control_target_pairs))
	return control_target_pairs


#Parameter for ProdT_factors_odd that represents the main difference from the even case
def Find_k(binary_matrix_reverse, x_max, n_qubit):
	k_array=np.empty(shape=(x_max),dtype=int)
	for i in range(x_max):
		for j in range(n_qubit-1):
			if binary_matrix_reverse[i][j]==1:
				k_array[i]=j
				break
			else:
				pass
	#print('\nArray with k-values: {}'.format(k_array))
	return k_array


#Scalability scheme for the whole sequence ProdT_factors_odd
def AllProdT_gatesequence_odd(binary_matrix_reverse, k_array, n_qubit):
	for i in range(binary_matrix_reverse.shape[0]):
			CNOTgate_odd(k_array[i], n_qubit)
			for j in range(binary_matrix_reverse.shape[1]):
				if binary_matrix_reverse[i][j]==1:
					CNOTgate_even(j, n_qubit)
			CNOTgate_odd(k_array[i], n_qubit)
			qml.Identity(3)#Only used to separate different ProdT_factors_odd

#Structure of a single ProdT_factor_odd
def ProdT_gatesequence_odd(binary_array, k_array_element, k_array, n_qubit):
	cnot_count_odd1,cnot_count_even1=0,0
	CNOTgate_odd(k_array[k_array_element], n_qubit)
	for j in range(len(binary_array)):
		if binary_array[j]==1:
			CNOTgate_even(j, n_qubit)
			cnot_count_even1+=1
	CNOTgate_odd(k_array[k_array_element], n_qubit)
	cnot_count_odd1=cnot_count_even1+2
	return cnot_count_odd1

#Inside ProdT_factors_even, CNOT parameters and number of simplifications depend on this matrix  
def Simplification_matrix_even(binary_matrix_reverse, x_max, n_qubit):
	simpli_matrix_even=np.empty(shape=(x_max-1,n_qubit-1),dtype=int)
	simpli_count=0
	for i in range(x_max-1):
		m1=binary_matrix_reverse[i]
		m2=binary_matrix_reverse[i+1]
		for j in range(n_qubit-1):
			if m1[j]==m2[j] and m1[j]==1:
				simpli_matrix_even[i][j]=0
				simpli_count=simpli_count+1
			elif m1[j]==m2[j] and m1[j]==0:
				simpli_matrix_even[i][j]=0
			else:
				simpli_matrix_even[i][j]=1
	#print('\nNumber of simplifications (even case): {}'.format(simpli_count))
	#print('\nBinary matrix of simplifications (even case):\n')
	#print(format_matrix(simpli_matrix_even))
	return simpli_matrix_even,simpli_count


#Inside ProdT_factors_odd, CNOT parameters and number of simplificatrions depend on this matrix
def Simplification_matrix_odd(binary_matrix_reverse, x_max, n_qubit):
	simpli_matrix_odd=np.empty(shape=(x_max-1,n_qubit-1),dtype=int)
	simpli_count=0
	for i in range(x_max-1):
		m1=binary_matrix_reverse[i]
		m2=binary_matrix_reverse[i+1]
		for j in range(n_qubit-1):
			if m1[j]==m2[j] and m1[j]==1:
				simpli_matrix_odd[i][j]=2
				simpli_count=simpli_count+1
			elif m1[j]==m2[j] and m1[j]==0:
				simpli_matrix_odd[i][j]=0
			else:
				simpli_matrix_odd[i][j]=1
	double_count=0
	for l in range(simpli_matrix_odd.shape[0]):
		for h in range(simpli_matrix_odd.shape[1]):
			if simpli_matrix_odd[l][h]==2:
				double_count=double_count+1
				break
			else:
				pass
	simpli_count=simpli_count+double_count               
	#print('\nNumber of simplifications (odd case): {}'.format(simpli_count))
	#print('\nBinary matrix of simplifications (odd case):\n')
	#print(format_matrix(simpli_matrix_odd))
	return simpli_matrix_odd,simpli_count,double_count


#Scalability scheme for the CNOT sequence of ProdT_factors_even, except for the first and the last
def ProdT_simplified_even(simpli_matrix_even,theta_collection_PSI,simpli_Gray_collection, rot_count_PSI, n_qubit):
	cnot_count_even_all=0
	for i in range(simpli_matrix_even.shape[0]):
		for j in range(simpli_matrix_even.shape[1]):
			if simpli_matrix_even[i][j]==1:
				CNOTgate_even(j, n_qubit)
				cnot_count_even_all+=1
		M_factor_even(theta_collection_PSI,i+1, simpli_Gray_collection, rot_count_PSI, n_qubit)
	#print('\nNumber of cnot in the (simplified) sequence of ProdT_factors_even, except the first and the last: {}'.format(cnot_count_even_all))
	return cnot_count_even_all

#Scalability scheme for the CNOT sequence of ProdT_factors_odd, except for the first and the last
def ProdT_simplified_odd(simpli_matrix_odd, theta_collection_PHI, k_array, cnot_count_odd1, simpli_matrix_even, simpli_Gray_collection, binary_matrix_reverse, rot_count_PHI, n_qubit):
	cnot_count_odd_all=0
	for i in range(simpli_matrix_odd.shape[0]):
		for j in range(simpli_matrix_odd.shape[1]):
			if simpli_matrix_odd[i][j]==2:
				cnot_count_center=0
				CNOTgate_odd(k_array[i], n_qubit)
				for l in range(simpli_matrix_even.shape[1]):
					if simpli_matrix_even[i][l]==1:
						CNOTgate_even(l, n_qubit)
						cnot_count_center+=1
				CNOTgate_odd(k_array[i], n_qubit)
				cnot_count_odd_all+=cnot_count_center+2
				M_factor_odd(theta_collection_PHI,i+1, simpli_Gray_collection, rot_count_PHI, n_qubit)
				break
			elif np.all(simpli_matrix_odd[i]!=np.full((n_qubit-1),2)):
				row_pre1=binary_matrix_reverse[i]
				row_pre2=binary_matrix_reverse[i+1]
				cnot_count_odd1 = ProdT_gatesequence_odd(row_pre1,i, k_array, n_qubit)
				cnot_count_odd_all+=cnot_count_odd1
				cnot_count_odd1 = ProdT_gatesequence_odd(row_pre2,i+1, k_array, n_qubit)
				cnot_count_odd_all+=cnot_count_odd1
				M_factor_odd(theta_collection_PHI,i+1, simpli_Gray_collection, rot_count_PHI, n_qubit)
				break
	#print('\nNumber of cnot in the (simplified) sequence of ProdT_factors_odd, except the first and the last: {}'.format(cnot_count_odd_all))
	return cnot_count_odd_all

#Used for M_factors and ZETA_factor, side=True for right decomposition, side=False for left decomposition
def Gray_matrix(n_bits,side=bool):
	Gray_matrix_collection=[]
	if side==True:
		Gray_matrix_1=np.array([[0],[1]])
		Gray_matrix_collection.append(Gray_matrix_1)
		Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
		for i in range(2,n_bits+1):
			Gray_matrix_next=np.empty(shape=(2**i,i),dtype=int)
			semi_rows=2**(i-1)
			Gray_matrix_next[0:semi_rows,0]=[0]*semi_rows
			Gray_matrix_next[0:semi_rows,1:i]=Gray_matrix_1
			Gray_matrix_next[semi_rows:2**i,0]=[1]*semi_rows
			Gray_matrix_next[semi_rows:2**i,1:i]=Gray_matrix_1reverse
			Gray_matrix_collection.append(Gray_matrix_next)
			Gray_matrix_1=Gray_matrix_next
			Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
		#for i in range(n_bits):
			#print('\nGray matrix right-side for {} bits:\n'.format(i+1))
			#print(format_matrix(Gray_matrix_collection[i]))
	else:
		Gray_matrix_1=np.array([[1],[0]])
		Gray_matrix_collection.append(Gray_matrix_1)
		Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
		for i in range(2,n_bits+1):
			Gray_matrix_next=np.empty(shape=(2**i,i),dtype=int)
			semi_rows=2**(i-1)
			Gray_matrix_next[0:semi_rows,0]=[1]*semi_rows
			Gray_matrix_next[0:semi_rows,1:i]=Gray_matrix_1
			Gray_matrix_next[semi_rows:2**i,0]=[0]*semi_rows
			Gray_matrix_next[semi_rows:2**i,1:i]=Gray_matrix_1reverse
			Gray_matrix_collection.append(Gray_matrix_next)
			Gray_matrix_1=Gray_matrix_next
			Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
		#for i in range(n_bits):
			#print('\nGray matrix left-side for {} bits:\n'.format(i+1))
			#print(format_matrix(Gray_matrix_collection[i]))
	return Gray_matrix_collection



#Inside M_factors and ZETA_factor, CNOT parameters depend on this matrix
def Simplification_Gray_matrix(Gray_matrix_collection, n_qubit):
	simpli_Gray_collection=[]
	for p in range(n_qubit-1):
		Gray_matrix_p=Gray_matrix_collection[p]
		extension=Gray_matrix_p[0]
		Gray_matrix_extended=np.empty(shape=(2**(p+1)+1,p+1))
		Gray_matrix_extended[0:2**(p+1),0:p+1]=Gray_matrix_p
		Gray_matrix_extended[2**(p+1),0:p+1]=extension
		simpli_Gray_matrix=np.empty(shape=(2**(p+1),p+1),dtype=int)
		for i in range(2**(p+1)):
			m1=Gray_matrix_extended[i]
			m2=Gray_matrix_extended[i+1]
			for j in range(p+1):
				if m1[j]==m2[j]:
					simpli_Gray_matrix[i][j]=0
				else:
					simpli_Gray_matrix[i][j]=1
		simpli_Gray_collection.append(simpli_Gray_matrix)
	#for l in range(n_qubit-1):
		#print('\nBinary matrix of simplifications for Gray Code in {} bits:\n'.format(l+1))
		#print(format_matrix(simpli_Gray_collection[l]))
	#simpli_Gray_collection.requires_grad = False
	return simpli_Gray_collection


#The choice of the side depends on Gray_matrix only, because the decomposition process of a uniformly CROT is the same
def Uniformly_CROT(simpli_Gray_matrix,rot_type,theta_array_ucr,target_wire,delete=bool,side=bool):
	a=simpli_Gray_matrix.shape[0]
	b=simpli_Gray_matrix.shape[1]
	if delete==False:
		if side==True:
			for i in range(a):
				for j in range(b):
					if simpli_Gray_matrix[i][j]==1 and rot_type==1:
						qml.RX(theta_array_ucr[i],wires=target_wire)
						qml.CNOT(wires=[j,target_wire])
					elif simpli_Gray_matrix[i][j]==1 and rot_type==2:
						qml.RY(theta_array_ucr[i],wires=target_wire)
						qml.CNOT(wires=[j,target_wire])
					elif simpli_Gray_matrix[i][j]==1 and rot_type==3:
						qml.RZ(theta_array_ucr[i],wires=target_wire)
						qml.CNOT(wires=[j,target_wire])
		else:
			qml.CNOT(wires=[0,target_wire])
			if rot_type==1:
				for i in range(a-1):
					for j in range(b):
						if simpli_Gray_matrix[i][j]==1:
							qml.RX(theta_array_ucr[i],wires=target_wire)
							qml.CNOT(wires=[j,target_wire])
				qml.RX(theta_array_ucr[-1],wires=target_wire)
			elif rot_type==2:
				for i in range(a-1):
					for j in range(b):
						if simpli_Gray_matrix[i][j]==1: 
							qml.RY(theta_array_ucr[i],wires=target_wire)
							qml.CNOT(wires=[j,target_wire])
				qml.RY(theta_array_ucr[-1],wires=target_wire)
			elif rot_type==3:
				for i in range(a-1):
					for j in range(b):
						if simpli_Gray_matrix[i][j]==1 and rot_type==3:
							qml.RZ(theta_array_ucr[i],wires=target_wire)
							qml.CNOT(wires=[j,target_wire])
				qml.RZ(theta_array_ucr[-1],wires=target_wire)            
	if delete==True:    
		if rot_type==1:
			for i in range(a-1):
				for j in range(b):
					if simpli_Gray_matrix[i][j]==1:
						qml.RX(theta_array_ucr[i],wires=target_wire)
						qml.CNOT(wires=[j,target_wire])
			qml.RX(theta_array_ucr[-1],wires=target_wire)
		elif rot_type==2:
			for i in range(a-1):
				for j in range(b):
					if simpli_Gray_matrix[i][j]==1: 
						qml.RY(theta_array_ucr[i],wires=target_wire)
						qml.CNOT(wires=[j,target_wire])
			qml.RY(theta_array_ucr[-1],wires=target_wire)
		elif rot_type==3:
			for i in range(a-1):
				for j in range(b):
					if simpli_Gray_matrix[i][j]==1 and rot_type==3:
						qml.RZ(theta_array_ucr[i],wires=target_wire)
						qml.CNOT(wires=[j,target_wire])
			qml.RZ(theta_array_ucr[-1],wires=target_wire)

#theta_array_ucr=np.empty(simpli_Gray_collection[n-2].shape[0])
#theta_array_ucr.fill(np.pi/2)

#Generates the total number of parameters for the circuit
def Theta_array_gen(x_max, n_qubit):
	#PHI FACTOR -> M_factor_odd (x_max times)
	rot_count_Modd=0
	if n_qubit>2:
		for i in range(n_qubit-1):
			rot_count_Modd=rot_count_Modd+2**(i+1)
		rot_count_Modd=rot_count_Modd+3*2**(n_qubit-1)
	elif n_qubit==2:
		rot_count_Modd=rot_count_Modd+3*2**(n_qubit-1)
	#print('\nNumber of rotation in one single M_factor_odd: {}'.format(rot_count_Modd))
	#cnot_count_Modd=5*2**(n-1)-6
	##print('\nNumber of cnot in one single M_factor_odd: {}'.format(cnot_count_Modd))
	rot_count_PHI=rot_count_Modd*x_max
	#print('\nNumber of rotation in the whole PHI FACTOR: {}'.format(rot_count_PHI))
	#cnot_count_PHI=cnot_count_Modd*x_max+cnot_count_odd
	##print('\nNumber of cnot in the whole PHI FACTOR: {}'.format(cnot_count_PHI))
	
	
	#theta_collection_PHI = np.random.randn(x_max, rot_count_Modd, requires_grad=True)
	#theta_collection_PHI = np.random.randn(x_max * rot_count_Modd, requires_grad = True)
	#print('\nArray dimensions for the PHI FACTOR collection: {}'.format(theta_collection_PHI.shape))
	
	#PSI FACTOR -> M_factor_even (x_max times), exp{1,2,9,12...}
	rot_count_Meven=3*2**(n_qubit-1)
	#print('\nNumber of rotation in one single M_factor_even: {}'.format(rot_count_Meven))
	#cnot_count_Meven=3*2**(n-1)-2
	##print('\nNumber of cnot in one single M_factor_even: {}'.format(cnot_count_Meven))
	rot_count_PSI=rot_count_Meven*(x_max+1)
	#print('\nNumber of rotation in the whole PSI FACTOR: {}'.format(rot_count_PSI))
	#cnot_count_PSI=cnot_count_Meven*(x_max+1)+cnot_count_even_all
	##print('\nNumber of cnot in the whole PSI FACTOR: {}'.format(cnot_count_PSI))
   
	#theta_collection_PSI = np.random.randn(x_max+1, rot_count_Meven, requires_grad=True)
	#theta_collection_PSI = np.random.randn((x_max+1) * rot_count_Meven, requires_grad=True)
	#print('\nArray dimensions for the PSI FACTOR collection: {}'.format(theta_collection_PSI.shape))
	
	#ZETA FACTOR -> exp{3,8,15,24...}
	rot_count_ZETA=2**n_qubit-1
	#print('\nNumber of rotation in the whole ZETA FACTOR: {}'.format(rot_count_ZETA))
	
	#theta_collection_ZETA = np.random.randn(rot_count_ZETA, requires_grad=True)
	##print('\ntheta_collection_ZETA elements: {}'.format(theta_collection_ZETA))
	#print('\nArray dimensions for the ZETA FACTOR collection: {}'.format(theta_collection_ZETA.shape))
	rot_count_tot=rot_count_PHI+rot_count_PSI+rot_count_ZETA
	#print('\nTotal number of rotation in the circuit: {}'.format(rot_count_tot))


	params = pnp.random.randn(x_max * rot_count_Modd + (x_max + 1) * rot_count_Meven + rot_count_ZETA, requires_grad = True)

	return params, [rot_count_Modd, rot_count_Meven, rot_count_ZETA]


#M_factors_odd definition for n>2 qubits
def M_factor_odd(theta_collection_PHI, position_index, simpli_Gray_collection, rot_count_PHI, n_qubit):
	qml.RZ(theta_collection_PHI[position_index * rot_count_PHI],wires=0)
	rot_count_pre=1
	for i in range(n_qubit-2):
		rot_count_post=rot_count_pre+2**(i+1)
		Uniformly_CROT(simpli_Gray_collection[i],3,theta_collection_PHI[position_index * rot_count_PHI + rot_count_pre: position_index * rot_count_PHI + rot_count_post],i+1,False,True)
		rot_count_pre=rot_count_post
	Uniformly_CROT(simpli_Gray_collection[n_qubit-2],3,theta_collection_PHI[position_index * rot_count_PHI + rot_count_post: position_index * rot_count_PHI + rot_count_post+2**(n_qubit-1)],n_qubit-1,True,True)    
	Uniformly_CROT(simpli_Gray_collection[n_qubit-2],2,theta_collection_PHI[position_index * rot_count_PHI + rot_count_post+2**(n_qubit-1):position_index * rot_count_PHI + rot_count_post+2**(n_qubit-1)*2],n_qubit-1,True,False)
	Uniformly_CROT(simpli_Gray_collection[n_qubit-2],3,theta_collection_PHI[position_index * rot_count_PHI + rot_count_post+2**(n_qubit-1)*2: position_index * rot_count_PHI + rot_count_post+2**(n_qubit-1)*3],n_qubit-1,False,True)
	rot_count_pre=rot_count_post+2**(n_qubit-1)*3
	for i in range(n_qubit-2,0,-1):
		rot_count_post=rot_count_pre+2**(i)
		Uniformly_CROT(simpli_Gray_collection[i-1],3,theta_collection_PHI[position_index * rot_count_PHI + rot_count_pre: position_index * rot_count_PHI + rot_count_post],i,False,False)
		rot_count_pre=rot_count_post
	qml.RZ(theta_collection_PHI[position_index * rot_count_PHI + rot_count_post],wires=0)

#M_factors_even definition for n>2 qubits
def M_factor_even(theta_collection_PSI,position_index, simpli_Gray_collection, rot_count_PSI, n_qubit):
	Uniformly_CROT(simpli_Gray_collection[n_qubit-2],3,theta_collection_PSI[position_index * rot_count_PSI + 0: position_index * rot_count_PSI + 2**(n_qubit-1)],n_qubit-1,True,True)    
	Uniformly_CROT(simpli_Gray_collection[n_qubit-2],2,theta_collection_PSI[position_index * rot_count_PSI + 2**(n_qubit-1): position_index * rot_count_PSI + 2**(n_qubit-1)*2],n_qubit-1,True,False)
	Uniformly_CROT(simpli_Gray_collection[n_qubit-2],3,theta_collection_PSI[position_index * rot_count_PSI + 2**(n_qubit-1)*2: position_index * rot_count_PSI + 2**(n_qubit-1)*3],n_qubit-1,False,True)

#ZETA_factor definition for 2 qubits (special case outside the scalability scheme)
def ZETA_factor2qubits(theta_collection_ZETA,delete=bool):
	if delete==False:
		qml.CNOT(wires=[0,1])
		qml.RZ(theta_collection_ZETA[0],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RZ(theta_collection_ZETA[1],wires=[0])
		qml.RZ(theta_collection_ZETA[2],wires=[1])
	else:
		qml.RZ(theta_collection_ZETA[0],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RZ(theta_collection_ZETA[1],wires=[0])
		qml.RZ(theta_collection_ZETA[2],wires=[1])

#ZETA_factor definition for n>2 qubits
def ZETA_factor(theta_collection_ZETA,simpli_Gray_collection, n_qubit):
	offset = 0
	for k in range(n_qubit - 1):
		for i in range(simpli_Gray_collection[n_qubit-2-k].shape[0]):
			for j in range(simpli_Gray_collection[n_qubit-2-k].shape[1]):
				if simpli_Gray_collection[n_qubit-2-k][i][j]==1:
					qml.CNOT(wires=[j,n_qubit-1-k])
					qml.RZ(theta_collection_ZETA[offset + i],wires=[n_qubit-1-k])	
		offset += simpli_Gray_collection[n_qubit-2-k].shape[0]		
	qml.RZ(theta_collection_ZETA[-1],wires=[0])

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


def SU_approx_circuit(params, binary_matrix_reverse, k_array, simpli_matrix_even, simpli_matrix_odd, simpli_Gray_collection, x_max, rot_count, n_qubit):
	#print(params)
	theta_collection_PHI = params[:rot_count[0] * x_max]
	theta_collection_PSI = params[rot_count[0] * x_max: rot_count[0] * x_max + rot_count[1] * (x_max + 1)]
	theta_collection_ZETA = params[rot_count[0] * x_max + rot_count[1] * (x_max + 1): rot_count[0] * x_max + rot_count[1] * (x_max + 1) + rot_count[2]]
	
	rot_count_PHI = rot_count[0]
	rot_count_PSI = rot_count[1]
	#print(rot_count_PSI)
	"""
	print(theta_collection_PHI)
	print(theta_collection_PSI)
	print(theta_collection_ZETA)

	print(rot_count_PHI)
	print(rot_count_PSI)
	"""
	if n_qubit==2:
		#PHI_FACTOR[Transpositions for M1_odd: (2,3) -- Related elements for M1_odd: (5,7,11,14)]
		ProdT1_odd()
		#M1_odd
		#ProdT1_odd
		#some simplifications arise, so we replace with...
		qml.RZ(theta_collection_PHI[0],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RZ(theta_collection_PHI[1],wires=[1])
		qml.RY(theta_collection_PHI[2],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RY(theta_collection_PHI[3],wires=[1])
		qml.RZ(theta_collection_PHI[4],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RZ(theta_collection_PHI[5],wires=[1])
		qml.CNOT(wires=[1,0])
		qml.CNOT(wires=[0,1])

		#PSI FACTOR [Transpositions for M1_even: (2,4) -- Related elements for M1_even: (10,13,4,6)]
		ProdT1_even()
		M1(theta_collection_PSI[0:rot_count_PSI])
		ProdT1_even()
		#Exp(1,2,9,12)
		#since Exp(1,2,9,12) has a ZYZ decomposition, it can be implemented as M1...
		M1(theta_collection_PSI[rot_count_PSI: rot_count_PSI * (x_max + 1)], True)

		#ZETA FACTOR [Elements: 3,8,15]
		qml.RZ(theta_collection_ZETA[0],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RZ(theta_collection_ZETA[1],wires=[0])
		qml.RZ(theta_collection_ZETA[2],wires=[1])
	
	else:
		
		#PHI_FACTOR_circuit
		cnot_count_odd1 = ProdT_gatesequence_odd(binary_matrix_reverse[0],0, k_array, n_qubit)
		#print('\nNumber of cnot in ProdT7_odd: {}'.format(cnot_count_odd1))
		M_factor_odd(theta_collection_PHI,0, simpli_Gray_collection, rot_count_PHI, n_qubit)       
		cnot_count_odd_all = ProdT_simplified_odd(simpli_matrix_odd, theta_collection_PHI, k_array, cnot_count_odd1, simpli_matrix_even, simpli_Gray_collection, binary_matrix_reverse, rot_count_PHI, n_qubit)
		cnot_count_odd1 = ProdT_gatesequence_odd(binary_matrix_reverse[x_max-1],x_max-1, k_array, n_qubit)
		#print('\nNumber of cnot in ProdT1_odd: {}'.format(cnot_count_odd1))
		
		#PSI_FACTOR_circuit
		cnot_count_even1 = ProdT_gatesequence_even(binary_matrix_reverse[0], n_qubit)
		#print('\nNumber of cnot in ProdT7_even: {}'.format(cnot_count_even1))
		M_factor_even(theta_collection_PSI,0, simpli_Gray_collection, rot_count_PSI, n_qubit)    
		cnot_count_even_all = ProdT_simplified_even(simpli_matrix_even, theta_collection_PSI, simpli_Gray_collection, rot_count_PSI, n_qubit)
		cnot_count_even1 = ProdT_gatesequence_even(binary_matrix_reverse[x_max-1], n_qubit)
		#print('\nNumber of cnot in ProdT1_even: {}'.format(cnot_count_even1))
		M_factor_even(theta_collection_PSI, simpli_matrix_even.shape[0] + 1, simpli_Gray_collection, rot_count_PSI, n_qubit)
		
		#Uniformly_CROT(simpli_Gray_collection[n-2],3,theta_array_ucr,n-1,False,False)
		#M_factor_odd(theta_collection_PHI,6)
		#M_factor_even(theta_collection_PSI,6)
		
		#ZETA_FACTOR_circuit
		#ZETA_factor2qubits(theta_collection_ZETA,True)
		ZETA_factor(theta_collection_ZETA,simpli_Gray_collection, n_qubit)
		
	#return [qml.expval(qml.PauliZ(i)) for i in range(n_qubit)]


