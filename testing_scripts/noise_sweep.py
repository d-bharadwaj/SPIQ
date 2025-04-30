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
from testing_scripts.qaoa_utils import QAOASolver
from maxcut_processing import evaluate_maxcut
import multiprocessing

# Get arguments from command line
n_qubits = int(sys.argv[1])  
reps = int(sys.argv[2])     
n_gens = int(sys.argv[3])        
mutation_prob = tuple(map(float, sys.argv[4].split()))
elitism = int(sys.argv[5])
crossover_type = str(sys.argv[6])   
seed =  int(sys.argv[7])
noise = bool(int(sys.argv[8]))

n = n_qubits

G = generate_random_complete_graph(num_vertices=n, weighted=True,seed=True)

# Evaluate Optimal Maxcut
optimal_max_cut_val = compute_optimal_max_cut(G)

max_cut_paulis = build_max_cut_paulis(G)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
paulis,coeffs = cost_hamiltonian.paulis.to_labels(),cost_hamiltonian.coeffs.real
reversed_paulis = [p[::-1] for p in paulis] #to respect stim ordering for hamiltonian

circuit = multi_angle_qaoa_circuit(n,G,reps)

#Create QAOA object
maxcut_qaoa = QAOASolver(cost_hamiltonian,circuit)
maxcut_qaoa.prepare_circuit()

# CAFQA Process
maxcut_qaoa.run_CAFQA(n_gens=n_gens)
print(f"{n} Qubits and {reps} reps")

#CAFQA Initialization
print(f"Minimum Energy found with CAFQA initalization: {maxcut_qaoa.energy_best}")

# Random Initalization 
random_angles = np.random.random(len(maxcut_qaoa.ks_best))
# random_energies = [maxcut_qaoa.evaluate_energy(maxcut_qaoa.pcirc, cost_hamiltonian, random_angles,noise=noise,err=1e-3) for _ in range(100)]
# min_energy = min(random_energies)
# print(f"Minimum Energy found with Random initialization over 100 runs: {min_energy}")

# Exact Ground state Energy
exact_solution = maxcut_qaoa.evaluate_exact_energy()
print("Exact Energy from Eigensolver:", exact_solution)

#QAOA Optimization 
ordered_params = [param.name for param in maxcut_qaoa.pcirc.parameters]
angle_multipliers = [-np.pi/4 if 'gamma' in param else np.pi/4 for param in ordered_params]
cafqa_params = [param * (np.pi/2) for param,multiplier in zip(maxcut_qaoa.ks_best,angle_multipliers)] #This has to be in the order we come across the gates.

# Evaluate Maxcut 
max_iters = 200
noise_levels = [1e-3,1e-4,1e-5]
results_per_noise = {}

for noise_level in noise_levels:
    random_result, random_obj_values = maxcut_qaoa.run_qaoa(initial_params=random_angles,err=noise_level, max_iters=max_iters, noise=noise)
    cafqa_result, cafqa_obj_values = maxcut_qaoa.run_qaoa(initial_params=cafqa_params,err=noise_level, max_iters=max_iters, noise=noise)

    random_fin_energy = random_result.fun
    cafqa_fin_energy = cafqa_result.fun

    result = {
        "CAFQA_best_energy": maxcut_qaoa.energy_best,
        "noise_level": noise_level,
        "exact_energy": exact_solution,
        "Random_initialization_energy": random_fin_energy,
        "CAFQA_initialization_energy": cafqa_fin_energy,
        "Random_obj_values": random_obj_values,
        "CAFQA_obj_values": cafqa_obj_values
    }

    # Save results for this noise level
    output_dir = f"../np_data/rep_sweep_2000_gens/single/{reps}_layers"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"new_results_{seed}_noise_{noise_level}.npy"), result)

    results_per_noise[noise_level] = result

# Save the combined results_per_noise dictionary
output_file = f"../np_data/rep_sweep_2000_gens/{reps}_layers/1e-3_new_n-cafqa_single_noisy_results_{seed}.npy"
np.save(output_file, results_per_noise)
