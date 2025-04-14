import rustworkx as rx
import numpy as np
import warnings
import os
import sys

from playground_scripts.ma_qaoa_plus import RA_fin_energy
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
from testing_scripts.qaoa_utils import QAOASolver,evaluate_energy
from maxcut_processing import evaluate_maxcut

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
k = 3 # for k-regular graphs

G = generate_random_complete_graph(num_vertices=n, weighted=True,seed=False)
# G = generate_k_regular_graph(num_vertices=n, k=k, weighted=True)

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
random_energies = [evaluate_energy(maxcut_qaoa.pcirc, cost_hamiltonian, random_angles) for _ in range(100)]
min_energy = min(random_energies)
print(f"Minimum Energy found with Random initialization over 100 runs: {min_energy}")

#Minimum Energy found with Angle Rounding
rounded_angles = np.random.choice(np.arange(-np.pi, np.pi + np.pi/8, np.pi/8), len(maxcut_qaoa.ks_best))
rounded_energies = [evaluate_energy(maxcut_qaoa.pcirc, cost_hamiltonian, rounded_angles) for _ in range(100)]
min_rounded_energy = min(rounded_energies)
print(f"Minimum Energy found with Angle Rounding over 100 runs: {min_rounded_energy}")

# Exact Ground state Energy
exact_solution = maxcut_qaoa.evaluate_exact_energy()
print("Exact Energy from Eigensolver:", exact_solution)

#QAOA Optimization 
ordered_params = [param.name for param in maxcut_qaoa.pcirc.parameters]
angle_multipliers = [-np.pi/4 if 'gamma' in param else np.pi/4 for param in ordered_params]
cafqa_params = [param * (multiplier) for param,multiplier in zip(maxcut_qaoa.ks_best,angle_multipliers)] #This has to be in the order we come across the gates.

#NOTE: Change this above^ 

# Evaluate Maxcut 
max_iters = 1000

random_result,random_obj_values = maxcut_qaoa.run_qaoa(random_angles, max_iters,noise)
RA_result,RA_obj_values = maxcut_qaoa.run_qaoa(rounded_angles, max_iters,noise)
cafqa_result, cafqa_obj_values = maxcut_qaoa.run_qaoa(cafqa_params,max_iters,noise)

random_fin_energy =  random_result.fun
# RA_fin_energy = RA_result.fun
cafqa_fin_energy = cafqa_result.fun

random_max_cut_val = evaluate_maxcut(G,maxcut_qaoa.pcirc,random_result,maxcut_qaoa.backend)
RA_max_cut_val = evaluate_maxcut(G,maxcut_qaoa.pcirc,RA_result,maxcut_qaoa.backend)
cafqa_max_cut_val = evaluate_maxcut(G,maxcut_qaoa.pcirc,cafqa_result,maxcut_qaoa.backend)

random_approx_ratio = random_max_cut_val / optimal_max_cut_val
cafqa_approx_ratio = cafqa_max_cut_val / optimal_max_cut_val

print(f"Optimal Max Cut Value : {optimal_max_cut_val}")

print(f"Max Cut value with Random initialization after {max_iters} iteration: {random_max_cut_val}")
print(f"Max Cut value with CAFQA initialization after {max_iters} iteration: {cafqa_max_cut_val}")

print(f"Random Approx. Ratio: {random_approx_ratio}")
print(f"CAFQA Approx. Ratio: {cafqa_approx_ratio}")

results = {
    "CAFQA_initialization_energy": maxcut_qaoa.energy_best,
    "Random_initialization_energy": min_energy,
    "Angle_rounding_energy": min_rounded_energy,
    "Exact_solution_energy": exact_solution,
    "Random_initialization_max_cut": random_max_cut_val,
    "CAFQA_initialization_max_cut": cafqa_max_cut_val,
    "Random_approx_ratio": random_approx_ratio,
    "CAFQA_approx_ratio": cafqa_approx_ratio,
    "Final_random_energy": random_fin_energy,
    "Final_Rounded_Angle_energy":RA_fin_energy,
    "Final_cafqa_energy": cafqa_fin_energy,
    "Random_obj_values": random_obj_values,
    "RA_obj_values": RA_obj_values,
    "CAFQA_obj_values": cafqa_obj_values
}

output_dir = f"../np_data/rep_sweep_2000_gens/{reps}_layers"
os.makedirs(output_dir, exist_ok=True); np.save(os.path.join(output_dir, f"results_{seed}.npy"), results)