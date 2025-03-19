import rustworkx as rx
import numpy as np
import warnings
import os
import sys
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.circuit import Parameter
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from scipy.optimize import minimize

sys.path.append("../")
from clapton.clapton import claptonize
from clapton.circuit_manipulation import transform_to_allowed_gates,qiskit_to_stim, modify_circuit, multi_angle_qaoa_circuit, generate_qiskit_param_map
from testing_scripts.graphs_utils import generate_random_complete_graph,generate_k_regular_graph, build_max_cut_paulis, compute_optimal_max_cut
from testing_scripts.energy_utils import evaluate_energy
from testing_scripts.maxcut_processing import evaluate_maxcut

# Get arguments from command line
n_qubits = int(sys.argv[1])  
reps = int(sys.argv[2])     
n_gens = int(sys.argv[3])        
mutation_prob = tuple(map(float, sys.argv[4].split()))
elitism = int(sys.argv[5])
crossover_type = str(sys.argv[6])   
seed =  int(sys.argv[7])

n = n_qubits
k = 3 # for k-regular graphs

G = generate_random_complete_graph(num_vertices=n, weighted=True,seed=False)
# G = generate_k_regular_graph(num_vertices=n, k=k, weighted=True)

# Evaluate Optimal Maxcut
optimal_max_cut_val = compute_optimal_max_cut(G)

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

print(f"{n} Qubits and {reps} reps")

#CAFQA Initialization
print(f"Minimum Energy found with CAFQA initalization: {energy_best}")

# Random Initalization 
random_angles = np.random.random(len(ks_best))
random_energies = [evaluate_energy(pcirc, cost_hamiltonian, random_angles) for _ in range(100)]
min_energy = min(random_energies)
print(f"Minimum Energy found with Random initialization over 100 runs: {min_energy}")

#Minimum Energy found with Angle Rounding
rounded_angles = np.random.choice(np.arange(-np.pi, np.pi + np.pi/8, np.pi/8), len(ks_best))
rounded_energies = [evaluate_energy(pcirc, cost_hamiltonian, rounded_angles) for _ in range(100)]
min_rounded_energy = min(rounded_energies)
print(f"Minimum Energy found with Angle Rounding over 100 runs: {min_rounded_energy}")

# Exact Ground state Energy
eigensolver = NumPyMinimumEigensolver()
exact_solution = eigensolver.compute_minimum_eigenvalue(cost_hamiltonian).eigenvalue.real
print("Exact Energy from Eigensolver:", exact_solution)

#QAOA Optimization 
ordered_params = [param.name for param in pcirc.parameters]
angle_multipliers = [-np.pi/4 if 'gamma' in param else np.pi/4 for param in ordered_params]
cafqa_params = [param * (multiplier) for param,multiplier in zip(ks_best,angle_multipliers)] #This has to be in the order we come across the gates.

# Evaluate Maxcut 
max_iters = 1500
random_max_cut_val,random_obj_values,random_fin_energy = evaluate_maxcut(G,pcirc, random_angles, cost_hamiltonian,max_iters)
RA_max_cut_val,RA_obj_values,RA_fin_energy = evaluate_maxcut(G,pcirc, rounded_angles, cost_hamiltonian,max_iters)
cafqa_max_cut_val,cafqa_obj_values,cafqa_fin_energy = evaluate_maxcut(G,pcirc, cafqa_params, cost_hamiltonian,max_iters)

random_approx_ratio = random_max_cut_val / optimal_max_cut_val
cafqa_approx_ratio = cafqa_max_cut_val / optimal_max_cut_val

print(f"Optimal Max Cut Value : {optimal_max_cut_val}")

print(f"Max Cut value with Random initialization after {max_iters} iteration: {random_max_cut_val}")
print(f"Max Cut value with CAFQA initialization after {max_iters} iteration: {cafqa_max_cut_val}")

print(f"Random Approx. Ratio: {random_approx_ratio}")
print(f"CAFQA Approx. Ratio: {cafqa_approx_ratio}")

results = {
    "CAFQA_initialization_energy": energy_best,
    "Random_initialization_energy": min_energy,
    "Angle_rounding_energy": min_rounded_energy,
    "Exact_solution_energy": exact_solution,
    "Random_initialization_max_cut": random_max_cut_val,
    "CAFQA_initialization_max_cut": cafqa_max_cut_val,
    "Random_approx_ratio": random_approx_ratio,
    "CAFQA_approx_ratio": cafqa_approx_ratio,
    "Final_random_energy": random_fin_energy,
    "Final_Rounded_Angle_energy":RA_fin_energy,
    "Final_cafqa_energy": cafqa_fin_energy
}

output_dir = f"../np_data/rep_sweep_2000_gens/{reps}_layers"
os.makedirs(output_dir, exist_ok=True); np.save(os.path.join(output_dir, f"results_{seed}.npy"), results)