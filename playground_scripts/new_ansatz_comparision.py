import rustworkx as rx
import numpy as np
import warnings
import os
import sys

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import SparsePauliOp

sys.path.append("../")
from clapton.clapton import claptonize
from clapton.circuit_manipulation import (
    transform_qiskit_to_stim,
    multi_angle_qaoa_circuit,
    multi_angle_qaoa_plus_circuit,
    generate_qiskit_param_map,
)
from testing_scripts.graphs_utils import (
    generate_random_complete_graph,
    generate_k_regular_graph,
    build_max_cut_paulis,
)

# Get arguments from command line
n_qubits = int(sys.argv[1])
reps = int(sys.argv[2])
n_gens = int(sys.argv[3])
mutation_prob = tuple(map(float, sys.argv[4].split()))
elitism = int(sys.argv[5])
crossover_type = str(sys.argv[6])
seed = int(sys.argv[7])

n = n_qubits
k = 3  # for k-regular graphs

G = generate_random_complete_graph(num_vertices=n, weighted=True, seed=False)
# G = generate_k_regular_graph(num_vertices=n, k=k, weighted=True)

max_cut_paulis = build_max_cut_paulis(G)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
paulis, coeffs = cost_hamiltonian.paulis.to_labels(), cost_hamiltonian.coeffs.real
reversed_paulis = [p[::-1] for p in paulis]  # to respect stim ordering for hamiltonian

# Create circuits
ma_qaoa_circuit = multi_angle_qaoa_circuit(n, G, reps)
ma_qaoa_plus_circuit = multi_angle_qaoa_plus_circuit(n, G, reps)

# Transform circuits
ma_qaoa_stim_circ, ma_qaoa_pcirc = transform_qiskit_to_stim(ma_qaoa_circuit)
ma_qaoa_plus_stim_circ, ma_qaoa_plus_pcirc = transform_qiskit_to_stim(
    ma_qaoa_plus_circuit
)

# Param Map
ma_qaoa_param_map = generate_qiskit_param_map(ma_qaoa_pcirc)
ma_qaoa_plus_param_map = generate_qiskit_param_map(ma_qaoa_plus_pcirc)

# ma-qaoa claptonize
ma_qaoa_ks_best, _, ma_qaoa_energy_best = claptonize(
    reversed_paulis,
    coeffs,
    ma_qaoa_stim_circ,
    n_proc=4,  # total number of processes in parallel
    n_starts=4,  # number of random genetic algorithm starts in parallel
    n_rounds=1,  # number of budget rounds, if None it will terminate itself
    callback=print,  # callback for internal parameter (#iteration, energies, ks) processing
    budget=n_gens // 2,  # budget per genetic algorithm instance
    mutation_probability=mutation_prob,
    keep_elitism=elitism,
    crossover_type=crossover_type,
)

ma_qaoa_plus_ks_best, _, ma_qaoa_plus_energy_best = claptonize(
    reversed_paulis,
    coeffs,
    ma_qaoa_plus_stim_circ,
    n_proc=4,  # total number of processes in parallel
    n_starts=4,  # number of random genetic algorithm starts in parallel
    n_rounds=1,  # number of budget rounds, if None it will terminate itself
    callback=print,  # callback for internal parameter (#iteration, energies, ks) processing
    budget=n_gens // 2,  # budget per genetic algorithm instance
    mutation_probability=mutation_prob,
    keep_elitism=elitism,
    crossover_type=crossover_type,
)

print(f"Minimum Energy found with ma-QAOA: {ma_qaoa_energy_best}")
print(f"Minimum Energy found with ma-QAOA+: {ma_qaoa_plus_energy_best}")

# Exact Ground state Energy
eigensolver = NumPyMinimumEigensolver()
exact_solution = eigensolver.compute_minimum_eigenvalue(
    cost_hamiltonian
).eigenvalue.real
print("Exact Energy from Eigensolver:", exact_solution)

results = {
    "ma_QAOA_Energy": ma_qaoa_energy_best,
    "ma_QAOA+_Energy": ma_qaoa_plus_energy_best,
    "Exact_solution_energy": exact_solution,
}

output_dir = f"../np_data/ansatz_comparision"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, f"results_{seed}.npy"), results)
