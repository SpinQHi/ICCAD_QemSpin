from qiskit.circuit import Gate,QuantumCircuit, Parameter, ParameterVector
import numpy as np



def double_circuit(circ, params_list, qubits_list,params_idx):
  circ.cx(qubits_list[2],qubits_list[3])
  circ.cx(qubits_list[0],qubits_list[2])
  circ.h(qubits_list[3])
  circ.h(qubits_list[0])
  circ.cx(qubits_list[2],qubits_list[3])
  circ.cx(qubits_list[0],qubits_list[1])
  circ.ry(params_list[params_idx]/8,qubits_list[1])
  circ.ry(-params_list[params_idx]/8,qubits_list[0])
  circ.cx(qubits_list[0],qubits_list[3])
  circ.h(qubits_list[3])
  circ.cx(qubits_list[3],qubits_list[1])
  circ.ry(params_list[params_idx]/8,qubits_list[1])
  circ.ry(-params_list[params_idx]/8,qubits_list[0])
  circ.cx(qubits_list[2],qubits_list[1])
  circ.cx(qubits_list[2],qubits_list[0])
  circ.ry(-params_list[params_idx]/8,qubits_list[1])
  circ.ry(params_list[params_idx]/8,qubits_list[0])
  circ.cx(qubits_list[3],qubits_list[1])
  circ.h(qubits_list[3])
  circ.cx(qubits_list[0],qubits_list[3])
  circ.ry(-params_list[params_idx]/8,qubits_list[1])
  circ.ry(params_list[params_idx]/8,qubits_list[0])
  circ.cx(qubits_list[0],qubits_list[1])
  circ.cx(qubits_list[2],qubits_list[0])
  circ.h(qubits_list[0])
  circ.h(qubits_list[3])
  circ.cx(qubits_list[0],qubits_list[2])
  circ.cx(qubits_list[2],qubits_list[3])

def single_circuit(circ, params_list, qubits_list,params_idx,doubles_list):
  circ.cx(qubits_list[0],qubits_list[1])
  circ.ry(params_list[params_idx+len(doubles_list)]/2,qubits_list[0])
  circ.cx(qubits_list[1],qubits_list[0])
  circ.ry(-params_list[params_idx+len(doubles_list)]/2,qubits_list[0])
  circ.cx(qubits_list[1],qubits_list[0])
  circ.cx(qubits_list[0],qubits_list[1])


def Adapt_Givens_Ansatz():
  
  singles_list = [[6,10]]
  doubles_list = [[1, 2, 10, 11]]


  qubits_num = 12
  
  params_list = ParameterVector("x", len(singles_list)+len(doubles_list))
  circ = QuantumCircuit(qubits_num)
  initial_list = np.array([0,1,2,3,4,6,7,8,9])
  
  # circ.x(range(electrons))
  circ.x(initial_list)
  if len(doubles_list) != 0:
    for i, excitation in enumerate(doubles_list):
      double_circuit(circ=circ, params_list = params_list, qubits_list=excitation, params_idx=i)
  if len(singles_list) != 0:
    for i, excitation in enumerate(singles_list):
      single_circuit(circ=circ, params_list = params_list, qubits_list=excitation, params_idx=i,doubles_list=doubles_list)
  return circ