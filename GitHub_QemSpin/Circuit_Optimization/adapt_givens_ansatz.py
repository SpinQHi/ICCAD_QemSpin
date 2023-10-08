# Importing required libraries and modules
import numpy as np
import torch

from spinqkit import Circuit, CX, H, Ry, X
from spinqkit import get_basic_simulator, get_compiler, BasicSimulatorConfig, generate_hamiltonian_matrix
from spinqkit.backend.pytorch_backend import TorchSimulator
from spinqkit.algorithm.Expval import _scipy_sparse_mat_to_torch_sparse_tensor
from spinqkit.model.parameter import Parameter

import vqe_spinqkit as vqe


def initialize_state(circ: Circuit, state):
    """
    Given a circuit, it changes the initial state from |0000> to the needed state.
    
    Parameters:
    circuit (Circuit): The initial quantum circuit.
    state (list or array-like): The desired initial state.
    """
    
    for idx, s in enumerate(state):
        if s == 1:
            circ << (X, [idx])

            
def double_excitation(circ: Circuit, qubits, param_idx=0):
    """
    Implements a double excitation gate which is a 4-qubit gate. The matrix form is a Givens rotation,
    rotating |1100> to a superposition of |1100> and |0011>, or rotating |0011> 
    into a superposition of |1100> and |0011>.
    
    Parameters:
    circuit (Circuit): The quantum circuit.
    qubits (list): The list of qubits where the basic gates are applied.
    param_idx (int, optional): Index for extracting the double excitation gate parameter from the total circuit parameters, default is 0.
    
    For more details on the circuit, see the paper: Universal Quantum Chemistry Circuit.
    """
    circ << (CX, [qubits[2], qubits[3]])
    circ << (CX, [qubits[0], qubits[2]])
    circ << (H, qubits[3])
    circ << (H, qubits[0])
    circ << (CX, [qubits[2], qubits[3]])
    circ << (CX, [qubits[0], qubits[1]])
    circ << (Ry, qubits[1], lambda x: x[param_idx] / 8)
    circ << (Ry, qubits[0], lambda x: -x[param_idx] / 8)
    circ << (CX, [qubits[0], qubits[3]])
    circ << (H, [qubits[3]])
    circ << (CX, [qubits[3], qubits[1]])
    circ << (Ry, qubits[1], lambda x: x[param_idx] / 8)
    circ << (Ry, qubits[0], lambda x: -x[param_idx] / 8)
    circ << (CX, [qubits[2], qubits[1]])
    circ << (CX, [qubits[2], qubits[0]])
    circ << (Ry, qubits[1], lambda x: -x[param_idx] / 8)
    circ << (Ry, qubits[0], lambda x: x[param_idx] / 8)
    circ << (CX, [qubits[3], qubits[1]])
    circ << (H, [qubits[3]])
    circ << (CX, [qubits[0], qubits[3]])
    circ << (Ry, qubits[1], lambda x: -x[param_idx] / 8)
    circ << (Ry, qubits[0], lambda x: x[param_idx] / 8)
    circ << (CX, [qubits[0], qubits[1]])
    circ << (CX, [qubits[2], qubits[0]])
    circ << (H, qubits[0])
    circ << (H, qubits[3])
    circ << (CX, [qubits[0], qubits[2]])
    circ << (CX, [qubits[2], qubits[3]])
    
def sigle_excitation(circ: Circuit, qubits, param_idx=0):
    """
    Implements a single excitation gate which is a 2-qubit gate. The matrix form is a Givens rotation,
    rotating |10> into a superposition of |10> and |01>, or rotating |01> into a superposition of |10> and |01>.
    
    Parameters:
    circuit (Circuit): The quantum circuit.
    qubits (list): The list of qubits where the basic gates are applied.
    param_idx (int, optional): Index for extracting the single excitation gate parameter from the total circuit parameters, default is 0.
    """
    circ << (CX, [qubits[0], qubits[1]])
    circ << (Ry, qubits[0], lambda x: x[param_idx] / 2)
    circ << (CX, [qubits[1], qubits[0]])
    circ << (Ry, qubits[0], lambda x: -x[param_idx] / 2)
    circ << (CX, [qubits[1], qubits[0]])
    circ << (CX, [qubits[0], qubits[1]])


def excitations(electrons, orbitals, delta_sz=0):
    """
    Generate single and double excitations from a Hartree-Fock reference state.
    
    """

    if not electrons > 0:
        raise ValueError(
            f"The number of active electrons has to be greater than 0 \n"
            f"Got n_electrons = {electrons}"
        )

    if orbitals <= electrons:
        raise ValueError(
            f"The number of active spin-orbitals ({orbitals}) "
            f"has to be greater than the number of active electrons ({electrons})."
        )

    if delta_sz not in (0, 1, -1, 2, -2):
        raise ValueError(
            f"Expected values for 'delta_sz' are 0, +/- 1 and +/- 2 but got ({delta_sz})."
        )

    # define the spin projection 'sz' of the single-particle states
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    singles = [
        [r, p]
        for r in range(electrons)
        for p in range(electrons, orbitals)
        if sz[p] - sz[r] == delta_sz
    ]

    doubles = [
        [s, r, q, p]
        for s in range(electrons - 1)
        for r in range(s + 1, electrons)
        for q in range(electrons, orbitals - 1)
        for p in range(q + 1, orbitals)
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    return singles, doubles


def hf_state_fun(electrons, orbitals):
    """
    Generate the occupation-number vector representing the Hartree-Fock state.
    """

    if electrons <= 0:
        raise ValueError(
            f"The number of active electrons has to be larger than zero; "
            f"got 'electrons' = {electrons}"
        )

    if electrons > orbitals:
        raise ValueError(
            f"The number of active orbitals cannot be smaller than the number of active electrons;"
            f" got 'orbitals'={orbitals} < 'electrons'={electrons}"
        )

    state = np.where(np.arange(orbitals) < electrons, 1, 0)
    return np.array(state)


def double_circuit(double_list, qubit_num, hf_state, params_double=None):
    """
    Generates a circuit with only double excitation gates. This is useful for gradient computation 
    by excluding unnecessary gates.
    
    Parameters:
    double_list (list): Contains the bit positions where double excitation gates are applied.
    qubit_num (int): The number of qubits in the circuit.
    hf_state (array-like): Hartree-Fock reference state.
    params_double (array-like, optional): Parameters for the double excitation gates. If None, parameters are initialized to zero.
    
    Returns:
    Circuit: A quantum circuit with only double excitation gates.
    """
    if params_double is None:
        # Create circuit parameters
        params_double = Parameter(np.zeros(len(double_list)), trainable=True)    
    
    # Create a circuit that only includes double excitations
    cir = Circuit(params=params_double)
    cir.allocateQubits(qubit_num)
    
    # Set up the initial state
    initialize_state(circ=cir, state=hf_state)

    ## Apply double excitation gates
    for i, excitation in enumerate(double_list):
            double_excitation(circ=cir, qubits=excitation, param_idx=i)
    
    # Return the circuit
    return cir


def givens_circuit(double_list, single_list, qubit_num, hf_state, params=None,seed=15):
    if params is None:
        # Create circuit parameters
        params = Parameter(np.zeros(len(double_list)+len(single_list)), trainable=True)
        
    # Create a circuit
    cir = Circuit(params=params)
    cir.allocateQubits(qubit_num)
    
    # Set up the initial state
    initialize_state(circ=cir, state=hf_state)    
    
    ## First apply double excitation gates, then single excitation gates
    if len(double_list) != 0:
        # List must not be empty to avoid errors
        for i, excitation in enumerate(double_list):
                double_excitation(circ=cir, qubits=excitation, param_idx=i)
                
    if len(single_list) != 0:
        for i, excitation in enumerate(single_list):
                sigle_excitation(circ=cir, qubits=excitation, param_idx=i+len(double_list))

    # Return the circuit
    return cir
            

def expvalcost(circuit:Circuit, hamiltonian_matrix, qubit_num):
    """
    Input a circuit and Hamiltonian, compute the expectation value of the final state and Hamiltonian as our loss function
    """

    dtype = torch.complex64
    device = torch.device('cpu')

    # Use the spinkit PyTorch simulator and compile the circuit.
    compiler = get_compiler('native')
    exe = compiler.compile(circuit, 0)
    
    sim = TorchSimulator()

    # Convert Hamiltonian to Torc
    hamiltonian = _scipy_sparse_mat_to_torch_sparse_tensor(hamiltonian_matrix)

    
    init_states = torch.eye(1, 2 ** (qubit_num)).to(device, dtype)
    
    # Run the initial state on the circuit to obtain its final state
    
    final_state = torch.permute(sim._get_final_state(exe, init_states, torch.tensor(circuit.params), exe.qnum), dims=[1, 0])

    cost_value = torch.real(final_state.conj().T @ torch.unsqueeze(hamiltonian @ final_state, dim=0)).detach().numpy()

    return cost_value[0][0][0]


def double_params_grad(double_list, hamiltonian_list, qubit_num, hf_state):
    
    grad_params_list = []
    
    if type(hamiltonian_list) == list:
        h_mat = generate_hamiltonian_matrix(hamiltonian_list) 
    else:
        h_mat = hamiltonian_list
        
    for i in range(len(double_list)):
        params_plus = Parameter(np.zeros(len(double_list)))
        params_minus = Parameter(np.zeros(len(double_list)))
        
        params_plus[i] = np.pi/2
        params_minus[i] = -np.pi/2
        

        cir_plus = double_circuit(double_list, qubit_num, hf_state, params_double=params_plus)
        cir_minus = double_circuit(double_list, qubit_num, hf_state, params_double=params_minus)
        
        cost_plus = expvalcost(circuit=cir_plus, hamiltonian_matrix=h_mat, qubit_num=qubit_num)
        cost_minus = expvalcost(circuit=cir_minus, hamiltonian_matrix=h_mat, qubit_num=qubit_num)
        

        grad_value = (cost_plus - cost_minus)/2

        grad_params_list.append(grad_value)
    
    return grad_params_list

def single_params_grad(double_list, double_param, single_list, hamiltonian_list, qubit_num, hf_state):
    
    grad_params_list = []
    

    if type(hamiltonian_list) == list:
        h_mat = generate_hamiltonian_matrix(hamiltonian_list) 
    else:
        h_mat = hamiltonian_list
      
    for i in range(len(single_list)):
        params_plus = Parameter(np.zeros(len(double_list)+len(single_list)))
        params_minus = Parameter(np.zeros(len(double_list)+len(single_list)))
                
        params_plus[:len(double_list)] = double_param
        params_minus[:len(double_list)] = double_param
        
        params_plus[i+len(double_list)] = np.pi/2
        params_minus[i+len(double_list)] = -np.pi/2
        
        cir_plus = givens_circuit(double_list, single_list, qubit_num, hf_state, params=params_plus)
        cir_minus = givens_circuit(double_list, single_list, qubit_num, hf_state, params=params_minus)
        
        cost_plus = expvalcost(circuit=cir_plus, hamiltonian_matrix=h_mat, qubit_num=qubit_num)
        cost_minus = expvalcost(circuit=cir_minus, hamiltonian_matrix=h_mat, qubit_num=qubit_num)

        grad_value = (cost_plus - cost_minus)/2
        
        grad_params_list.append(grad_value)
    
    return grad_params_list


##########################################################################################################################################################################################
##########################################################################################################################################################################################

# 基于 givens ansatz 的 VQE 集成

def train_givens_vqe(hamiltonian_list, electrons_num, givens_type='normal', learning_rate=0.1, itr_num=100, delta_sz_list=[0], verbose=True, opt_type='torch',seed=15):
    r""""
   hamiltonian_list: The Hamiltonian of the molecule, as a list.
    electrons_num: The number of active electrons in the molecule.
    givens_type: The type of Givens ansatz. If 'normal', generate all single and double excitation gates;
                 if 'adapt', compute gradients to perform single and double excitation gate deletion operations to reduce the number of gates.
    
    learning_rate: Learning rate for the optimizer.
    opt_type: Type of optimizer, default is torch.
    verbose: Whether to output the loss value at each iteration step.
    delta_sz_list: Spin projections sz involved in generalized single excitations.
                    It can contain [0, 1, -1, 2, -2] at most. If all are selected, all possible single and double excitation gates will be generated.
    
    """
    

    if type(hamiltonian_list) == list:
        h_mat = generate_hamiltonian_matrix(hamiltonian_list) 
        qubit_num = len(hamiltonian_list[0][0])
    else:
        h_mat = hamiltonian_list
        qubit_num = int(np.log2(hamiltonian_list.shape[0]))
            
            
    singles_list = [] 
    doubles_list = []

    for delta_sz in delta_sz_list:
        singles, doubles = excitations(electrons_num, qubit_num, delta_sz=delta_sz)
        
        singles_list += singles
        doubles_list += doubles

    hf_state = [1, 1, 1, 1 ,1 ,0 ,1, 1 ,1, 1, 0 ,0]

    SEED = seed  
    cir_depth = 1 
    
    if givens_type == 'normal':    

        givens_cir = givens_circuit(doubles_list, singles_list, qubit_num, hf_state,seed=SEED)
            
        summary_loss, _, circuit_params = vqe.train_vqe(h_mat, qubit_num, cir_depth=cir_depth, 
                                        circuit=givens_cir, learning_rate=learning_rate, 
                                        itr_num=itr_num, seed=SEED, verbose=verbose, opt_type=opt_type)
        
        cir_information = {
        'rotation_num':2 * len(singles_list) + 8 * len(doubles_list), 
        'hadama_num': 6 * len(doubles_list),                          
        'cnot_num':4 * len(singles_list) + 14 * len(doubles_list),    
        'gate_param_num': len(circuit_params),                        
        'circuit_params': circuit_params,                                
        'singles_list': singles_list,
        'doubles_list': doubles_list,
        }
        
    
    elif givens_type == 'adapt':
            
 
        double_grad = double_params_grad(doubles_list, h_mat, qubit_num, hf_state)
                    
        doubles_select = [doubles_list[i] for i in range(len(doubles_list)) if abs(double_grad[i]) > 1.0e-5]
                    
        doubles_select_cir = double_circuit(doubles_select, qubit_num, hf_state)
        summary_loss_double, _, double_select_params = vqe.train_vqe(h_mat, qubit_num, cir_depth = cir_depth, 
                                                        circuit=doubles_select_cir, learning_rate=learning_rate, 
                                                        itr_num=itr_num, seed=SEED, verbose=verbose, opt_type=opt_type)

    
        single_grad = single_params_grad(doubles_select, 
                                        double_select_params, singles_list, 
                                        h_mat, qubit_num, hf_state)

        singles_select = [singles_list[i] for i in range(len(singles_list)) if abs(single_grad[i]) > 1.0e-5]

        givens_select_cir = givens_circuit(doubles_select, singles_select, qubit_num, hf_state)
            
            
        summary_loss, _, circuit_params = vqe.train_vqe(h_mat, qubit_num, cir_depth=cir_depth, 
                                circuit=givens_select_cir, learning_rate=learning_rate, 
                                itr_num=itr_num, seed=SEED, verbose=verbose, opt_type=opt_type)    
            
            
        cir_information = {
        'rotation_num':2 * len(singles_select) + 8 * len(doubles_select), 
        'hadama_num': 6 * len(doubles_select),                          
        'cnot_num':4 * len(singles_select) + 14 * len(doubles_select),    
        'gate_param_num': len(circuit_params),                        
        'circuit_params': circuit_params,                                
        'singles_list': singles_select,
        'doubles_list': doubles_select,
        }
        
    return summary_loss, cir_information
    
