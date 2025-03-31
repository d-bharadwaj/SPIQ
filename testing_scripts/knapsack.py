import numpy as np
import random
import warnings
import os
import sys
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import transpile
from qiskit.circuit import Parameter,ParameterExpression
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_optimization.applications import Knapsack
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import QAOAAnsatz

import sys
sys.path.append("../")
from clapton.clapton import claptonize
from clapton.circuit_manipulation import transform_to_allowed_gates,qiskit_to_stim, modify_circuit, multi_angle_qaoa_circuit, generate_qiskit_param_map,relax_qaoa_parameters
from testing_scripts.energy_utils import evaluate_energy
from testing_scripts.knapsack_utils import generate_knapsack_instance,evaluate_knapsack

n_items = int(sys.argv[1])  
reps = int(sys.argv[2]) 
seed =  int(sys.argv[3])

# Knapsack Formulation 
# prob = Knapsack(values=[3, 4, 5, 6, 7], weights=[2, 3, 4, 5, 6], max_weight=10)
prob = generate_knapsack_instance(num_items=n_items,seed=seed)

qp = prob.to_quadratic_program()
print(qp.prettyprint())

# intermediate QUBO form of the optimization problem
conv = QuadraticProgramToQubo()
qubo = conv.convert(qp)

# qubit Hamiltonian and offset
op, offset = qubo.to_ising()
print(f"num qubits: {op.num_qubits}, offset: {offset}\n")

cost_hamiltonian = op
paulis,coeffs = cost_hamiltonian.paulis.to_labels(),cost_hamiltonian.coeffs.real
reversed_paulis = [p[::-1] for p in paulis]

# Transform qiskit circ. to stim.
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
modified_circ = modify_circuit(circuit)
pcirc = transform_to_allowed_gates(modified_circ)

# Relax Ansatz Parameters
pcirc, dag , angle_multipliers= relax_qaoa_parameters(pcirc)
stim_circ = qiskit_to_stim(pcirc)

# Parameter Mapping
param_map = generate_qiskit_param_map(pcirc)
stim_circ.define_parameter_map(param_map)

# CAFQA 
ks_best, _, energy_best = claptonize(
    reversed_paulis,
    coeffs,
    stim_circ,
    n_proc=4,           # total number of processes in parallel
    n_starts=4,         # number of random genetic algorithm starts in parallel
    n_rounds=1,         # number of budget rounds, if None it will terminate itself
    callback=print,     # callback for internal parameter (#iteration, energies, ks) processing
    budget=1500//2        # budget per genetic algorithm instance
)
print(f"Minimum Energy found with CAFQA initalization: {energy_best}")
stim_circ.assign(ks_best)

# Solve with classical Eigensolver for comparison
eigensolver = NumPyMinimumEigensolver()
exact_solution = eigensolver.compute_minimum_eigenvalue(cost_hamiltonian).eigenvalue.real
print("Exact Energy from Eigensolver:", exact_solution)

cafqa_angles = [param * np.pi/2 for param in ks_best]

# energies = [evaluate_energy(pcirc, cost_hamiltonian, cafqa_angles) for _ in range(100)]
# average_energy = np.mean(energies)
# print(f"Average CAFQA Qiskit Energy: {average_energy}")

# Random Initalization 
random_angles = np.random.random(len(ks_best))
random_energies = [evaluate_energy(pcirc, cost_hamiltonian, random_angles) for _ in range(100)]
min_energy = min(random_energies)
print(f"Minimum Energy found with Random initialization over 100 runs: {min_energy}")

cafqa_params = [param * np.pi/2 for param in ks_best]

# Evaluate Maxcut 
max_iters = 1000
random_obj_values,random_fin_energy = evaluate_knapsack(pcirc, random_angles, cost_hamiltonian,max_iters)
cafqa_obj_values,cafqa_fin_energy = evaluate_knapsack(pcirc, cafqa_params, cost_hamiltonian,max_iters)

# Save energies to a dictionary
energies_dict = {
    "CAFQA_initial_energy": energy_best,
    "Exact_solution_energy": exact_solution,
    "Min_initial_random_energy": min_energy,
    "Final_cafqa_energy": cafqa_fin_energy,
    "Final_random_energy": random_fin_energy
}

output_dir = f"../np_data/knapsack/rep_sweep/{op.num_qubits}_qubits"
os.makedirs(output_dir, exist_ok=True); np.save(os.path.join(output_dir, f"results_{seed}.npy"), energies_dict)