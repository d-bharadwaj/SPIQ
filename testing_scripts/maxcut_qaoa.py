import rustworkx as rx
import numpy as np
import warnings
import os
import sys
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.circuit import Parameter
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from scipy.optimize import minimize
from timeit import default_timer as timer


sys.path.append("../")
from clapton.clapton import claptonize
from clapton.circuit_manipulation import transform_to_allowed_gates,qiskit_to_stim, modify_circuit, multi_angle_qaoa_circuit, generate_qiskit_param_map
from testing_scripts.graphs_utils import generate_random_complete_graph,generate_k_regular_graph, build_max_cut_paulis, compute_optimal_max_cut
from testing_scripts.qaoa_utils import QAOASolver,evaluate_energy
from maxcut_processing import evaluate_maxcut
print("Command-line arguments:", sys.argv)

# # Get arguments from command line
# n_qubits = int(sys.argv[1])  
# reps = int(sys.argv[2])     
# n_gens = int(sys.argv[3])        
# mutation_prob = tuple(map(float, sys.argv[4].split()))
# elitism = int(sys.argv[5])
# crossover_type = str(sys.argv[6])   
# seed =  int(sys.argv[7])
# noise = bool(int(sys.argv[8]))


# n = n_qubits
# k = 3 # for k-regular graphs

# G = generate_random_complete_graph(num_vertices=n, weighted=True,seed=True)
# # G = generate_k_regular_graph(num_vertices=n, k=k, weighted=True)

# # Evaluate Optimal Maxcut
# optimal_max_cut_val = compute_optimal_max_cut(G)

# max_cut_paulis = build_max_cut_paulis(G)
# cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
# paulis,coeffs = cost_hamiltonian.paulis.to_labels(),cost_hamiltonian.coeffs.real
# reversed_paulis = [p[::-1] for p in paulis] #to respect stim ordering for hamiltonian

# circuit = multi_angle_qaoa_circuit(n,G,reps)

# #Create QAOA object
# maxcut_qaoa = QAOASolver(cost_hamiltonian,circuit)
# maxcut_qaoa.prepare_circuit()

# # CAFQA Process
# maxcut_qaoa.run_CAFQA(n_gens=n_gens)
# print(f"{n} Qubits and {reps} reps")

# #CAFQA Initialization
# print(f"Minimum Energy found with CAFQA initalization: {maxcut_qaoa.energy_best}")

# # Random Initalization 
# random_angles = np.random.random(len(maxcut_qaoa.ks_best))
# random_energies = [evaluate_energy(maxcut_qaoa.pcirc, cost_hamiltonian, random_angles) for _ in range(100)]
# min_energy = min(random_energies)
# print(f"Minimum Energy found with Random initialization over 100 runs: {min_energy}")

# #Minimum Energy found with Angle Rounding
# rounded_angles = np.random.choice(np.arange(-np.pi, np.pi + np.pi/8, np.pi/8), len(maxcut_qaoa.ks_best))
# rounded_energies = [evaluate_energy(maxcut_qaoa.pcirc, cost_hamiltonian, rounded_angles) for _ in range(100)]
# min_rounded_energy = min(rounded_energies)
# print(f"Minimum Energy found with Angle Rounding over 100 runs: {min_rounded_energy}")

# # Exact Ground state Energy
# exact_solution = maxcut_qaoa.evaluate_exact_energy()
# print("Exact Energy from Eigensolver:", exact_solution)

# #QAOA Optimization 
# ordered_params = [param.name for param in maxcut_qaoa.pcirc.parameters]
# angle_multipliers = [-np.pi/4 if 'gamma' in param else np.pi/4 for param in ordered_params]
# cafqa_params = [param * (np.pi/2) for param,multiplier in zip(maxcut_qaoa.ks_best,angle_multipliers)] #This has to be in the order we come across the gates.

# # Evaluate Maxcut 
# max_iters = 1000

# start = timer()

# random_result,random_obj_values = maxcut_qaoa.run_qaoa(initial_params=random_angles, max_iters=max_iters,opt="COBYLA")
# # RA_result,RA_obj_values = maxcut_qaoa.run_qaoa(rounded_angles, max_iters=max_iters)
# cafqa_result, cafqa_obj_values = maxcut_qaoa.run_qaoa(initial_params=cafqa_params, max_iters=max_iters,opt="COBYLA")
# random_fin_energy =  random_result.fun
# # RA_fin_energy = RA_result.fun
# cafqa_fin_energy = cafqa_result.fun
# end = timer()

# print(f"Total Time : {end - start}")


# random_max_cut_val = evaluate_maxcut(G,maxcut_qaoa.pcirc,random_result,maxcut_qaoa.backend)
# # RA_max_cut_val = evaluate_maxcut(G,maxcut_qaoa.pcirc,RA_result,maxcut_qaoa.backend)
# cafqa_max_cut_val = evaluate_maxcut(G,maxcut_qaoa.pcirc,cafqa_result,maxcut_qaoa.backend)

# random_approx_ratio = random_max_cut_val / optimal_max_cut_val
# cafqa_approx_ratio = cafqa_max_cut_val / optimal_max_cut_val

# # # Vanilla QAOA
# # circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
# # vanilla_teague = QAOASolver(cost_hamiltonian,circuit.decompose().decompose())
# # vanilla_random_params = np.random.random(circuit.num_parameters)
# # vanilla_teague.vanilla = True
# # vanilla_res,vanilla_obj_vals=vanilla_teague.run_qaoa(vanilla_random_params, max_iters=max_iters)
# # vanilla_qaoa_energy = vanilla_res.fun

# print(f"Optimal Max Cut Value : {optimal_max_cut_val}")

# print(f"Max Cut value with Random initialization after {max_iters} iteration: {random_max_cut_val}")
# print(f"Max Cut value with CAFQA initialization after {max_iters} iteration: {cafqa_max_cut_val}")

# print(f"Random Approx. Ratio: {random_approx_ratio}")
# print(f"CAFQA Approx. Ratio: {cafqa_approx_ratio}")

# results = {
#     "CAFQA_initialization_energy": maxcut_qaoa.energy_best,
#     "Random_initialization_energy": min_energy,
#     "Angle_rounding_energy": min_rounded_energy,
#     "Exact_solution_energy": exact_solution,
#     "Random_initialization_max_cut": random_max_cut_val,
#     "CAFQA_initialization_max_cut": cafqa_max_cut_val,
#     "Random_approx_ratio": random_approx_ratio,
#     "CAFQA_approx_ratio": cafqa_approx_ratio,
#     "Final_random_energy": random_fin_energy,
#     # "Final_Rounded_Angle_energy":RA_fin_energy,
#     "Final_cafqa_energy": cafqa_fin_energy,
#     # "Vanilla_QAOA_energy": vanilla_qaoa_energy,
#     "Random_obj_values": random_obj_values,
#     # "RA_obj_values": RA_obj_values,
#     "CAFQA_obj_values": cafqa_obj_values,
#     # "Vanilla_QAOA_values" : vanilla_obj_vals
# }

# output_dir = f"../np_data/rep_sweep_2000_gens/NM_test/{reps}_layers"
# os.makedirs(output_dir, exist_ok=True); np.save(os.path.join(output_dir, f"12_qb_single_results_{seed}.npy"), results)


from multiprocessing import Process, Queue
import traceback
import os
import numpy as np

# Define a function to handle the .run_qaoa task
def run_qaoa_task(task_name, qaoa_solver, initial_params, max_iters, opt, queue):
    try:
        print(f"Starting task: {task_name}")
        result, obj_values = qaoa_solver.run_qaoa(initial_params=initial_params, max_iters=max_iters, opt=opt)
        queue.put({
            "task_name": task_name,
            "result": result,
            "obj_values": obj_values,
            "final_energy": result.fun,
        })
        print(f"Finished task: {task_name}")


    except Exception as e:
        print(f"Error in task {task_name}: {e}")
        traceback.print_exc()
        queue.put({
            "task_name": task_name,
            "error": str(e),
        })

# Function to execute the QAOA tasks in parallel
def execute_qaoa_tasks(maxcut_qaoa, random_angles, cafqa_params, max_iters, G, optimal_max_cut_val):
    # Create a queue to collect results
    queue = Queue()

    # Create processes for each .run_qaoa task
    random_process = Process(target=run_qaoa_task, args=("random", maxcut_qaoa, random_angles, max_iters, "COBYLA", queue))
    cafqa_process = Process(target=run_qaoa_task, args=("cafqa", maxcut_qaoa, cafqa_params, max_iters, "COBYLA", queue))

    # Start the processes
    random_process.start()
    cafqa_process.start()

    # Wait for the processes to finish
    random_process.join()
    cafqa_process.join()

    # Collect results from the queue
    results = {}
    while not queue.empty():
        task_result = queue.get()
        task_name = task_result["task_name"]
        results[task_name] = task_result

    # Extract results for random and CAFQA tasks
    random_result = results["random"]["result"]
    random_obj_values = results["random"]["obj_values"]
    random_fin_energy = results["random"]["final_energy"]

    cafqa_result = results["cafqa"]["result"]
    cafqa_obj_values = results["cafqa"]["obj_values"]
    cafqa_fin_energy = results["cafqa"]["final_energy"]

    # Evaluate Maxcut and compute approximation ratios
    maxcut_qaoa._initialize_backend()
    # random_max_cut_val = evaluate_maxcut(G, maxcut_qaoa.pcirc, random_result, maxcut_qaoa.backend)
    # cafqa_max_cut_val = evaluate_maxcut(G, maxcut_qaoa.pcirc, cafqa_result, maxcut_qaoa.backend)

    # random_approx_ratio = random_max_cut_val / optimal_max_cut_val
    # cafqa_approx_ratio = cafqa_max_cut_val / optimal_max_cut_val

    # # Print results
    # print(f"Optimal Max Cut Value : {optimal_max_cut_val}")
    # print(f"Max Cut value with Random initialization after {max_iters} iteration: {random_max_cut_val}")
    # print(f"Max Cut value with CAFQA initialization after {max_iters} iteration: {cafqa_max_cut_val}")
    # print(f"Random Approx. Ratio: {random_approx_ratio}")
    # print(f"CAFQA Approx. Ratio: {cafqa_approx_ratio}")

    # # Return results
    # return {
    #     "CAFQA_initialization_energy": maxcut_qaoa.energy_best,
    #     # "Random_initialization_energy": min_energy,
    #     # "Angle_rounding_energy": min_rounded_energy,
    #     # "Exact_solution_energy": exact_solution,
    #     "Random_initialization_max_cut": random_max_cut_val,
    #     "CAFQA_initialization_max_cut": cafqa_max_cut_val,
    #     "Random_approx_ratio": random_approx_ratio,
    #     "CAFQA_approx_ratio": cafqa_approx_ratio,
    #     "Final_random_energy": random_fin_energy,
    #     "Final_cafqa_energy": cafqa_fin_energy,
    #     "Random_obj_values": random_obj_values,
    #     "CAFQA_obj_values": cafqa_obj_values,
    # }

# Main function to set up and execute the script
def main():
    # Initialize variables (replace with your actual initialization logic)
    n_qubits = int(sys.argv[1])  
    reps = int(sys.argv[2])     
    n_gens = int(sys.argv[3])        
    mutation_prob = tuple(map(float, sys.argv[4].split()))
    elitism = int(sys.argv[5])
    crossover_type = str(sys.argv[6])   
    seed = int(sys.argv[7])
    noise = bool(int(sys.argv[8]))

    n = n_qubits
    k = 3  # for k-regular graphs

    # Generate the graph
    G = generate_random_complete_graph(num_vertices=n, weighted=True, seed=True)

    # Evaluate Optimal Maxcut
    optimal_max_cut_val = compute_optimal_max_cut(G)

    # Build the cost Hamiltonian
    max_cut_paulis = build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    # Create the QAOA circuit
    circuit = multi_angle_qaoa_circuit(n, G, reps)

    # Create QAOA object
    maxcut_qaoa = QAOASolver(cost_hamiltonian, circuit)
    maxcut_qaoa.prepare_circuit()

    # Run CAFQA process
    maxcut_qaoa.run_CAFQA(n_gens=n_gens)
    print(f"{n} Qubits and {reps} reps")
    print(f"Minimum Energy found with CAFQA initialization: {maxcut_qaoa.energy_best}")

    # Random Initialization
    random_angles = np.random.random(len(maxcut_qaoa.ks_best))

    # QAOA Optimization
    ordered_params = [param.name for param in maxcut_qaoa.pcirc.parameters]
    angle_multipliers = [-np.pi / 4 if 'gamma' in param else np.pi / 4 for param in ordered_params]
    cafqa_params = [param * (np.pi / 2) for param, multiplier in zip(maxcut_qaoa.ks_best, angle_multipliers)]

    # Execute QAOA tasks
    max_iters = 1000
    start = timer()

    results = execute_qaoa_tasks(maxcut_qaoa, random_angles, cafqa_params, max_iters, G, optimal_max_cut_val)

    end = timer()
    print(f"Total Time : {end - start}")
    # # Save results
    # output_dir = f"../np_data/rep_sweep_2000_gens/NM_test/{reps}_layers"
    # os.makedirs(output_dir, exist_ok=True)
    # np.save(os.path.join(output_dir, f"12_qb_single_results_{seed}.npy"), results)

# Entry point
if __name__ == "__main__":
    main()