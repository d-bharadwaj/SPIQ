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
from testing_scripts.qaoa_utils import evaluate_energy,run_qaoa
from testing_scripts.knapsack_utils import generate_knapsack_instance
from testing_scripts.qaoa_utils import QAOASolver,evaluate_energy

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

knapsack_qaoa = QAOASolver(cost_hamiltonian,circuit)
knapsack_qaoa.prepare_circuit()

# Run CAFQA Process
knapsack_qaoa.run_CAFQA(n_gens=1000)

# Solve with classical Eigensolver for comparison
exact_solution = knapsack_qaoa.evaluate_exact_energy()
print("Exact Energy from Eigensolver:", exact_solution)

cafqa_angles = [param * np.pi/2 for param in knapsack_qaoa.ks_best]

# Random Initalization 
random_angles = np.random.random(len(knapsack_qaoa.ks_best))
random_energies = [evaluate_energy(knapsack_qaoa.pcirc, cost_hamiltonian, random_angles) for _ in range(100)]
min_energy = min(random_energies)
print(f"Minimum Energy found with Random initialization over 100 runs: {min_energy}")

cafqa_params = [param * np.pi/2 for param in knapsack_qaoa.ks_best]

# Evaluate Maxcut 
max_iters = 1000
random_result,random_obj_values = knapsack_qaoa.run_qaoa(random_angles,max_iters,noise=True)
cafqa_result,cafqa_obj_values = knapsack_qaoa.run_qaoa(cafqa_params, max_iters,noise=True)

# Save energies to a dictionary
energies_dict = {
    "CAFQA_initial_energy": knapsack_qaoa.energy_best,
    "Exact_solution_energy": exact_solution,
    "Min_initial_random_energy": min_energy,
    "Final_cafqa_energy": cafqa_result.fun,
    "Final_random_energy": random_result.fun,
    "Random_objective_values": random_obj_values,
    "CAFQA_objective_values": cafqa_obj_values
}

output_dir = f"../np_data/knapsack/rep_sweep/{op.num_qubits}_qubits"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, f"results_single_noisy{seed}.npy"), energies_dict)