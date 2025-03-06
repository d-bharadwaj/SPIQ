import rustworkx as rx
import numpy as np
import random
import warnings
warnings.simplefilter("ignore", UserWarning)

from qiskit.circuit import Parameter
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator

import sys
sys.path.append("../")
from clapton.clapton import claptonize
from clapton.circuit_manipulation import transform_to_allowed_gates,qiskit_to_stim, modify_circuit, multi_angle_qaoa_circuit, generate_qiskit_param_map
from testing_scripts.graphs_utils import generate_random_complete_graph,generate_k_regular_graph, build_max_cut_paulis

# Get arguments from command line
n_qubits = int(sys.argv[1])  # First argument: No. of qubits
reps = int(sys.argv[2])         # Second argument: Reps in ansatz
n_gens = int(sys.argv[3])         # Third Arugment : No. of Generations in GA.
mutation_prob = tuple(map(float, sys.argv[4].split()))
elitism = int(sys.argv[5])
crossover_type = str(sys.argv[6])
seed =  int(sys.argv[7])

n = n_qubits
k = 3 # for 3-regular graphs

G = generate_random_complete_graph(num_vertices=n, weighted=True)
# G = generate_k_regular_graph(num_vertices=n, k=k, weighted=True)

max_cut_paulis = build_max_cut_paulis(G)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
paulis,coeffs = cost_hamiltonian.paulis.to_labels(),cost_hamiltonian.coeffs.real
reversed_paulis = [p[::-1] for p in paulis] #to respect stim ordering for hamiltonian

gamma_params = [Parameter(f'gamma_{i}_{j}_{r}') for r in range(reps) for i, j in G.edge_list()]
beta_params = [Parameter(f'beta_{i}_{r}') for r in range(reps) for i in G.node_indexes()]
circuit = multi_angle_qaoa_circuit(gamma_params,beta_params,n,G ,reps)

# Transform qiskit circ. to stim.
modified_circ = modify_circuit(circuit)
pcirc = transform_to_allowed_gates(modified_circ)
stim_circ = qiskit_to_stim(pcirc)

param_map = generate_qiskit_param_map(pcirc)

stim_circ.define_parameter_map(param_map)

# CAFQA Process

ks_best, _, energy_best = claptonize(
    reversed_paulis,
    coeffs,
    stim_circ,
    n_proc=4,           # total number of processes in parallel
    n_starts=4,         # number of random genetic algorithm starts in parallel
    n_rounds=1,          # number of budget rounds, if None it will terminate itself
    callback=print,     # callback for internal parameter (#iteration, energies, ks) processing
    budget=n_gens//2,         # budget per genetic algorithm instance
    mutation_probability = mutation_prob,
    keep_elitism = elitism,
    crossover_type = crossover_type
)

def evaluate_energy(circuit, hamiltonian, parameters):
    estimator = Estimator(mode=AerSimulator(method='statevector'))
    isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)

    pub = (circuit, isa_hamiltonian, parameters)
    job = estimator.run([pub])

    results = job.result()[0]
    return results.data.evs

print(f"{n} Qubits and {reps} reps")

#CAFQA Initialization
print(f"Minimum Energy found with CAFQA initalization: {energy_best}")

# Random Initalization 
random_angles = np.random.random(len(ks_best))
random_energies = [evaluate_energy(pcirc, cost_hamiltonian, random_angles) for _ in range(1000)]
min_energy = min(random_energies)
print(f"Minimum Energy found with Random initialization over 100 runs: {min_energy}")

#Minimum Energy found with Angle Rounding
rounded_angles = np.random.choice(np.arange(-np.pi, np.pi + np.pi/8, np.pi/8), len(ks_best))
rounded_energies = [evaluate_energy(pcirc, cost_hamiltonian, rounded_angles) for _ in range(1000)]
min_rounded_energy = min(rounded_energies)
print(f"Minimum Energy found with Angle Rounding over 100 runs: {min_rounded_energy}")

# Exact Ground state Energy
eigensolver = NumPyMinimumEigensolver()
exact_solution = eigensolver.compute_minimum_eigenvalue(cost_hamiltonian).eigenvalue.real
print("Exact Energy from Eigensolver:", exact_solution)

# Save energies to a numpy dictionary
energies = {
    "CAFQA_initialization": energy_best,
    "Random_initialization": min_energy,
    "Angle_rounding": min_rounded_energy,
    "Exact_solution": exact_solution
}

np.save(f"../np_data/15_qbs/energies_{seed}", energies)
