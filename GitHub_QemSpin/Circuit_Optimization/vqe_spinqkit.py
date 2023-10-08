#############################################################################################

r"""
使用 spinqkit 构造 VQE 进行训练
"""

# 导入相关的包

import re
import numpy as np

from spinqkit.algorithm.optimizer.torch_optim import TorchOptimizer
from spinqkit.algorithm.optimizer.adam import ADAM
from spinqkit.algorithm.optimizer.gradientdescent import GradientDescent
from spinqkit.algorithm import VQE

from spinqkit import generate_hamiltonian_matrix


def train_vqe(hamiltonian_list, num_qubit, cir_depth, circuit=None, learning_rate=0.1, itr_num=100, seed=100, verbose=True, opt_type='torch'):
    r"""
    hamiltonian_list: The input can be a list or matrix form. The matrix form is used for time saving.
    num_qubit: Number of qubits
    cir_depth: Circuit depth
    circuit: Circuit transmitted from the outside
    learning_rate: Learning rate
    itr_num: Iteration times
    seed: Random seed number
    verbose: Whether to print training process information
    opt_type: Optimizer type. In spinq mode, the default is our basic optimizer using PyTorch's Adam.
    
    Returns:
    loss_list: Loss list
    result_state.states: Result states
    solver.circuit.params: Circuit parameters
    """
    # 固定随机种子数
    np.random.seed(seed)  
    
    # Check if the input hamiltonian_list is a list
    h_mat = generate_hamiltonian_matrix(hamiltonian_list) if isinstance(hamiltonian_list, list) else hamiltonian_list

    # Choose optimizer
    if opt_type == 'spinq_adam':
        optimizer = ADAM(maxiter=itr_num, tolerance=1e-6, learning_rate=learning_rate, verbose=verbose)
    elif opt_type == 'spinq_gd':
        # Basic gradient descent optimizer
        optimizer = GradientDescent(maxiter=itr_num, tolerance=1e-6, learning_rate=learning_rate, verbose=verbose)
    elif opt_type == 'torch':
        # Select PyTorch's optimizer
        optimizer = TorchOptimizer(maxiter=itr_num, learning_rate=learning_rate, optim_type='Adam', verbose=verbose)
        
    # If no circuit is transmitted from the outside, use the default circuit, and the parameters should also be None by default
    params = None if circuit is None else circuit.params

    # Create VQE solver
    solver = VQE(qubit_num=num_qubit, depth=cir_depth, hamiltonian=h_mat, circuit=circuit, circuit_params=params, optimizer=optimizer)

    # Use TorchOptimizer to optimize parameters, then use spinq simulator to measure, and get measurement results
    loss_list = solver.run(mode='spinq', grad_method='adjoint_differentiation')

    result_state = solver.get_optimize_result()

    return loss_list, result_state.states,  solver.circuit.params
