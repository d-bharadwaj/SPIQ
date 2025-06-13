import rustworkx as rx
import numpy as np
import warnings
import os
import sys
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from timeit import default_timer as timer

sys.path.append("../")
from clapton.circuit_manipulation import multi_angle_qaoa_circuit
import testing_scripts.graphs_utils as graph_utils
from testing_scripts.qaoa_utils import QAOASolver

import multiprocess as mp
import traceback
import os
import numpy as np

def run_qaoa_task_pool(args):

    max_iters = 10*1e+3
    task_id, maxcut_qaoa, initial_params, fitness_val, graph = args

    # QAOA Optimization
    cafqa_params = [param * (np.pi / 2) for param in initial_params]

    try:
        print(f"Starting task: {task_id} for val : {fitness_val}")
        result, obj_values = maxcut_qaoa.run_qaoa(initial_params=cafqa_params, max_iters=max_iters, opt="COBYLA")
        print(f"Finished task: {task_id}")

        # Approximation Ratio
        distribution = graph_utils.get_final_distribution(maxcut_qaoa,result.x)
        approx_ratio = graph_utils.calculate_approximation_ratio(distribution,graph)

        return {
            "task_name": task_id,
            "result": result,
            "fitness_val": fitness_val,
            "obj_values": obj_values,
            "final_energy": result.fun,
            "approx_ratio": approx_ratio
        }
    
    except Exception as e:
        print(f"Error in task {task_id}: {e}")
        traceback.print_exc()
        return {
            "task_name": task_id,
            "error": str(e),
        }

# Function to execute in parallel using Pool
def execute_qaoa_tasks(ma_qaoa_object, vanilla_qaoa_object, selected_cafqa_parameters, selected_spaced_fitness_vals, graph):

    cafqa_args = [
        (f"task_{i}", ma_qaoa_object, params, fitness , graph)
        for i, (params, fitness) in enumerate(zip(selected_cafqa_parameters, selected_spaced_fitness_vals))
    ]

    # Random Init. Params for MA-QAOA 
    # Generate 100 random parameter sets and evaluate their energies, then select the median
    random_angles_arr = np.random.uniform(0, 2 * np.pi, (100, ma_qaoa_object.pcirc.num_parameters))
    random_energies = [ma_qaoa_object.evaluate_energy(ma_qaoa_object.pcirc,ma_qaoa_object.cost_hamiltonian,params) for params in random_angles_arr]
    median_idx = np.argsort(random_energies)[len(random_energies) // 2]
    median_angles = random_angles_arr[median_idx]
    random_args = [("Random_MA-QAOA", ma_qaoa_object, median_angles, selected_spaced_fitness_vals, graph)]

    # Prepare task for Vanilla QAOA
    random_vanilla_angles = np.random.uniform(0, 2 * np.pi, vanilla_qaoa_object.circuit.num_parameters)
    vanilla_args = [("Vanilla_task", vanilla_qaoa_object, random_vanilla_angles, selected_spaced_fitness_vals, graph)]

    # Combine all tasks
    all_args = cafqa_args + random_args + vanilla_args 

    with mp.Pool(processes=len(all_args)) as pool:
        results_list = pool.map(run_qaoa_task_pool, all_args)

    results = {res["task_name"]: res for res in results_list}

    # Extract objective values for each task_id
    task_objective_values = {
        task_id: res.get("obj_values", None) for task_id, res in results.items()
    }

    # Return results
    return {
        "CAFQA_initialization_energy": ma_qaoa_object.energy_best,
        "Exact_Ground_State_Energy":ma_qaoa_object.exact_energy,
        "Task_results": results,
        "Task_objective_values": task_objective_values,
    }

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
    G = graph_utils.generate_random_complete_graph(num_vertices=n, weighted=True,seed=seed)

    # Build the cost Hamiltonian
    max_cut_paulis = graph_utils.build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    # Create the QAOA circuit
    circuit = multi_angle_qaoa_circuit(n, G, reps)

    # Create QAOA object
    maxcut_qaoa = QAOASolver(cost_hamiltonian, circuit,sim_device="GPU")
    maxcut_qaoa.prepare_circuit()
    maxcut_qaoa.err = noise

    #Evaluate Exact Ground State Energy 
    maxcut_qaoa.evaluate_exact_energy()

    # Run CAFQA process
    start_cafqa = timer()
    maxcut_qaoa.run_CAFQA(n_gens=n_gens)
    end_cafqa = timer()
    print(f"CAFQA optimization time: {end_cafqa - start_cafqa} seconds")
    print(f"{n} Qubits and {reps} reps")
    print(f"Minimum Energy found with CAFQA initialization: {maxcut_qaoa.energy_best}")

    # Best Solutions
    best_cafqa_fitness_values = maxcut_qaoa.best_cafqa_gen_fitness[::-1]
    best_cafqa_parameters = maxcut_qaoa.best_cafqa_gen_params[::-1]

    unique_fitness_values = np.unique(best_cafqa_fitness_values)

    # When choosing best out of unique energies
    selected_spaced_fitness_vals = list(unique_fitness_values[::3][:5]) #Select 5 of the best unique spaced-out solutions.
# selected_spaced_fitness_vals = list(unique_fitness_values[:5]) #Select 5 of the best unique solutions.

    selected_fitness_indices = [best_cafqa_fitness_values.index(value) for value in selected_spaced_fitness_vals]
    selected_cafqa_parameters = np.array(best_cafqa_parameters)[selected_fitness_indices]
    
    # # #When choosing just best 5 (can be same energies) 
    # selected_spaced_fitness_vals = list(best_cafqa_fitness_values)[:5] #Select 5 of the best solutions (can be repeated).
    # selected_cafqa_parameters = np.array(best_cafqa_parameters)[:5]
    
    # Vanilla QAOA
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    vanilla_maxcut = QAOASolver(cost_hamiltonian,circuit.decompose().decompose(),sim_device="GPU")
    vanilla_maxcut.vanilla = True

    start = timer()
    results = execute_qaoa_tasks(maxcut_qaoa, vanilla_maxcut, selected_cafqa_parameters, selected_spaced_fitness_vals, G)
    end = timer()
    print(f"Total Time : {end - start}")

    # Save results
    output_dir = f"../np_data/Final_Data_Collection/Maxcut/Complete_Graphs/Less_Reps/{n_qubits}_qbs"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"result_{seed}.npy"), results)

# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()