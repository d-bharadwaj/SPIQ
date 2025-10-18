import os
import sys
import warnings

import numpy as np
import rustworkx as rx

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from scipy.optimize import minimize

sys.path.append("../")
import multiprocessing

from clapton.circuit_manipulation import (generate_qiskit_param_map,
                                          modify_circuit,
                                          multi_angle_qaoa_circuit,
                                          qiskit_to_stim,
                                          transform_to_allowed_gates)
from clapton.clapton import claptonize

from testing_scripts.graphs_utils import (build_max_cut_paulis,
                                          compute_optimal_max_cut,
                                          generate_k_regular_graph,
                                          generate_random_complete_graph)
from testing_scripts.qaoa_utils import QAOASolver

# Get arguments from command line
n_qubits = int(sys.argv[1])
reps = int(sys.argv[2])
n_gens = int(sys.argv[3])
mutation_prob = tuple(map(float, sys.argv[4].split()))
elitism = int(sys.argv[5])
crossover_type = str(sys.argv[6])
seed = int(sys.argv[7])
noise = bool(int(sys.argv[8]))

n = n_qubits
G = generate_random_complete_graph(num_vertices=n, weighted=True, seed=False)

max_cut_paulis = build_max_cut_paulis(G)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
paulis, coeffs = cost_hamiltonian.paulis.to_labels(), cost_hamiltonian.coeffs.real
reversed_paulis = [p[::-1] for p in paulis]  # to respect stim ordering for hamiltonian

circuit = multi_angle_qaoa_circuit(n, G, reps)

# Create QAOA object
maxcut_qaoa = QAOASolver(cost_hamiltonian, circuit)
maxcut_qaoa.prepare_circuit()

# CAFQA Process
maxcut_qaoa.run_CAFQA(n_gens=n_gens)
print(f"{n} Qubits and {reps} reps")

# CAFQA Initialization
print(f"Minimum Energy found with CAFQA initalization: {maxcut_qaoa.energy_best}")

# Exact Ground state Energy
exact_solution = maxcut_qaoa.evaluate_exact_energy()
print("Exact Energy from Eigensolver:", exact_solution)

# QAOA Optimization
ordered_params = [param.name for param in maxcut_qaoa.pcirc.parameters]
angle_multipliers = [
    -np.pi / 4 if "gamma" in param else np.pi / 4 for param in ordered_params
]
cafqa_params = [
    param * (np.pi / 2)
    for param, multiplier in zip(maxcut_qaoa.ks_best, angle_multipliers)
]  # This has to be in the order we come across the gates.

# Evaluate Maxcut
max_iters = 1500  # TODO: change
cobyla_result, cobyla_obj_values = maxcut_qaoa.run_qaoa(
    initial_params=cafqa_params, max_iters=max_iters, opt="COBYLA"
)
spsa_result, spsa_obj_values = maxcut_qaoa.run_qaoa(
    initial_params=cafqa_params, max_iters=max_iters, opt="SPSA"
)
bfgs_result, bfgs_obj_values = maxcut_qaoa.run_qaoa(
    initial_params=cafqa_params, max_iters=max_iters, opt="COBYQA"
)


results_dict = {
    "COBYLA": {"result": cobyla_result, "obj_values": cobyla_obj_values},
    "SPSA": {"result": spsa_result, "obj_values": spsa_obj_values},
    "BFGS": {"result": bfgs_result, "obj_values": bfgs_obj_values},
    "exact_solution": exact_solution,
}

# Save results to a file using numpy
output_dir = f"../np_data/optimizer_sweep/{n_qubits}_qbs"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, f"2_optimizer_results_{seed}.npy"), results_dict)
