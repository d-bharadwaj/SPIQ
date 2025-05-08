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
print("Command-line arguments:", sys.argv)

import multiprocess as mp
import traceback
import os
import numpy as np

import qubovert

sys.path.append("../teague_code/code-for-gokul")
import pcbo_utils
from qiskit.quantum_info import SparsePauliOp
import os


def run_qaoa_task_pool(args):

    max_iters = 2000
    task_id, teague_qaoa, initial_params, fitness_val = args

    # QAOA Optimization
    cafqa_params = [param * (np.pi / 2) for param in initial_params]

    try:
        print(f"Starting task: {task_id} for val : {fitness_val}")
        result, obj_values = teague_qaoa.run_qaoa(initial_params=cafqa_params, max_iters=max_iters, opt="SPSA")
        print(f"Finished task: {task_id}")
        return {
            "task_name": task_id,
            "result": result,
            "fitness_val": fitness_val,
            "obj_values": obj_values,
            "final_energy": result.fun,
        }
    
    except Exception as e:
        print(f"Error in task {task_id}: {e}")
        traceback.print_exc()
        return {
            "task_name": task_id,
            "error": str(e),
        }

# Function to execute in parallel using Pool
def execute_qaoa_tasks(qaoa_object, selected_cafqa_parameters, selected_fitness_values):

    args = [
        (f"task_{i}", qaoa_object, params, fitness)
        for i, (params, fitness) in enumerate(zip(selected_cafqa_parameters, selected_fitness_values))
    ]

    with mp.Pool(processes=len(selected_cafqa_parameters)) as pool: #TODO: Change this
        results_list = pool.map(run_qaoa_task_pool, args)

    results = {res["task_name"]: res for res in results_list}

    # Extract objective values for each task_id
    task_objective_values = {
        task_id: res.get("obj_values", None) for task_id, res in results.items()
    }

    # Return results
    return {
        "CAFQA_initialization_energy": qaoa_object.energy_best,
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

    # Generate the graph
    feature_set, feature_to_idx, first_corr_arr, second_corr_arr, third_corr_arr = pcbo_utils.load_features_and_corr_files(f'../teague_code/code-for-gokul/sampled_{n_qubits}_features_subproblem_1')

    pcbo_obj = pcbo_utils.create_three_body_cubo(
        feature_set,
        first_corr_arr,
        second_corr_arr,
        third_corr_arr,
        feature_to_idx,
        select_n_features=4,
    )

    pubo = {key: float(value) for key, value in pcbo_obj.to_pubo().items()}

    def convert_pubo_to_ising(hypergraph: dict) -> list[tuple[str, float]]: #TODO: put this in util file.
        """Convert a hypergraph dictionary to a list of Pauli strings with weights."""
        n = n_qubits # Number of qubits
        pauli_list = []

        for edge, weight in hypergraph.items():
            if edge:  # Ensure the edge is not empty
                # Create a Pauli string with "I" for all qubits
                paulis = ["I"] * n
                # Replace "I" with "Z" for qubits in the edge
                for node in edge:
                    paulis[node] = "Z"
                # Append the reversed Pauli string and weight to the list
                pauli_list.append(("".join(paulis[::-1]), weight))

        return pauli_list

    max_cut_paulis = convert_pubo_to_ising(pubo)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    paulis,coeffs = cost_hamiltonian.paulis.to_labels(),cost_hamiltonian.coeffs.real

    reps=2
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    teague_qaoa = QAOASolver(cost_hamiltonian,circuit)

    teague_qaoa.prepare_circuit()

    # Run CAFQA process
    teague_qaoa.run_CAFQA(n_gens=n_gens)
    print(f"{n} Qubits and {reps} reps")
    print(f"Minimum Energy found with CAFQA initialization: {teague_qaoa.energy_best}")

    # Best Solutions
    best_cafqa_fitness_values = teague_qaoa.best_cafqa_gen_fitness
    best_cafqa_parameters = teague_qaoa.best_cafqa_gen_params

    unique_fitness_values = np.unique(best_cafqa_fitness_values)

    selected_fitness_values = list(unique_fitness_values[::3][:5])  # Select 5 of the best spaced-out solutions.

    selected_fitness_indices = [best_cafqa_fitness_values.index(value) for value in selected_fitness_values]
    selected_cafqa_parameters = np.array(best_cafqa_parameters)[selected_fitness_indices]

    start = timer()
    results = execute_qaoa_tasks(teague_qaoa, selected_cafqa_parameters, selected_fitness_values)
    end = timer()
    print(f"Total Time : {end - start}")

    # Save results
    output_dir = f"../np_data/CAFQA_Analysis/Teague_data/{n_qubits}_qbs"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"SPSA_point_analysis_{seed}.npy"), results)

# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()