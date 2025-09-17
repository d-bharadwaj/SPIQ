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
from timeit import default_timer as timer
import multiprocess as mp
import traceback

sys.path.append("../")
from clapton.clapton import claptonize
from clapton.circuit_manipulation import (
    transform_to_allowed_gates,
    qiskit_to_stim,
    modify_circuit,
    multi_angle_qaoa_circuit,
    generate_qiskit_param_map,
)
from testing_scripts.graphs_utils import (
    generate_random_complete_graph,
    generate_k_regular_graph,
    build_max_cut_paulis,
    compute_optimal_max_cut,
)
from testing_scripts.qaoa_utils import (
    QAOASolver,
    evaluate_energy,
    convert_pubo_to_ising,
)

print("Command-line arguments:", sys.argv)

sys.path.append("../teague_code/code-for-gokul")
import pcbo_utils
from qiskit.quantum_info import SparsePauliOp
import os
import pickle


def run_qaoa_task_pool(args):

    max_iters = 10 * 1e3
    task_id, teague_qaoa, initial_params, fitness_val = args

    # QAOA Optimization
    cafqa_params = [param * (np.pi / 2) for param in initial_params]

    try:
        print(f"Starting task: {task_id} for val : {fitness_val}")
        result, obj_values = teague_qaoa.run_qaoa(
            initial_params=cafqa_params, max_iters=max_iters, opt="COBYLA"
        )
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
def execute_qaoa_tasks(
    ma_qaoa_object,
    vanilla_qaoa_object,
    selected_cafqa_parameters,
    selected_fitness_vals,
):

    cafqa_args = [
        (f"task_{i}", ma_qaoa_object, params, fitness)
        for i, (params, fitness) in enumerate(
            zip(selected_cafqa_parameters, selected_fitness_vals)
        )
    ]

    # Random Init. Params for MA-QAOA
    random_angles = np.random.uniform(0, 2 * np.pi, ma_qaoa_object.pcirc.num_parameters)
    random_args = [
        ("Random_MA-QAOA", ma_qaoa_object, random_angles, selected_fitness_vals)
    ]

    # Prepare task for Vanilla QAOA
    random_vanilla_angles = np.random.uniform(
        0, 2 * np.pi, vanilla_qaoa_object.circuit.num_parameters
    )
    vanilla_args = [
        (
            "Vanilla_task",
            vanilla_qaoa_object,
            random_vanilla_angles,
            selected_fitness_vals,
        )
    ]

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
        "Task_results": results,
        "Task_objective_values": task_objective_values,
    }


# Main function to set up and execute the script
def main():
    # Initialize variables (replace with your actual initialization logic)
    n_qubits = int(sys.argv[1])
    reps = int(sys.argv[2])
    n_gens = int(sys.argv[3])
    seed = int(sys.argv[4])
    noise = bool(int(sys.argv[5]))

    n = n_qubits

    # Generate the graph
    feature_set, feature_to_idx, first_corr_arr, second_corr_arr, third_corr_arr = (
        pcbo_utils.load_features_and_corr_files(
            f"../teague_code/code-for-gokul/sampled_{n_qubits}_features_subproblem_1"
        )
    )

    pcbo_obj = pcbo_utils.create_three_body_cubo(
        feature_set,
        first_corr_arr,
        second_corr_arr,
        third_corr_arr,
        feature_to_idx,
        select_n_features=4,
    )

    pubo = {key: float(value) for key, value in pcbo_obj.to_pubo().items()}

    max_cut_paulis = convert_pubo_to_ising(pubo, n_qubits)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    reps = 2
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    teague_qaoa = QAOASolver(cost_hamiltonian, circuit, sim_device="GPU")

    teague_qaoa.prepare_circuit()

    # # Run CAFQA process
    # teague_qaoa.run_CAFQA(n_gens=n_gens,err=noise)
    # print(f"{n} Qubits and {reps} reps")
    # print(f"Minimum Energy found with CAFQA initialization: {teague_qaoa.energy_best}")

    # # Best Solutions
    # best_cafqa_fitness_values = teague_qaoa.best_cafqa_gen_fitness
    # best_cafqa_parameters = teague_qaoa.best_cafqa_gen_params

    # Importing from seperate file
    with open(
        f"../teague_code/code-for-gokul/teague_pickle_data/{n}_qb_teague_cafqa_results.pkl",
        "rb",
    ) as f:
        cafqa_data = pickle.load(f)
    best_cafqa_fitness_values = cafqa_data["best_cafqa_fitness_values"]
    best_cafqa_parameters = cafqa_data["best_cafqa_parameters"]
    print("cafqa_data keys:", list(cafqa_data.keys()))
    teague_qaoa.energy_best = cafqa_data["CAFQA_initialization_energy"]

    unique_fitness_values = np.unique(best_cafqa_fitness_values)

    # When choosing best out of unique energies
    selected_fitness_vals = list(
        unique_fitness_values[::3][:5]
    )  # Select 5 of the best unique spaced-out solutions.
    # selected_fitness_vals = list(unique_fitness_values[:5]) #Select 5 of the best unique solutions.

    selected_fitness_indices = [
        best_cafqa_fitness_values.index(value) for value in selected_fitness_vals
    ]
    selected_cafqa_parameters = np.array(best_cafqa_parameters)[
        selected_fitness_indices
    ]

    # # # # #When choosing just best 5 (can be same energies)
    # selected_fitness_vals = list(best_cafqa_fitness_values)[:5] #Select 5 of the best solutions (can be repeated).
    # selected_cafqa_parameters = np.array(best_cafqa_parameters)[:5]

    # Vanilla QAOA
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    vanilla_teague = QAOASolver(
        cost_hamiltonian, circuit.decompose().decompose(), sim_device="GPU"
    )
    vanilla_teague.vanilla = True

    if noise:
        vanilla_teague.err = 1e-4
        teague_qaoa.err = 1e-4

    start = timer()
    results = execute_qaoa_tasks(
        teague_qaoa, vanilla_teague, selected_cafqa_parameters, selected_fitness_vals
    )
    end = timer()
    print(f"Total Time : {end - start}")

    # Save results
    output_dir = f"../np_data/Comprehensive_Proof/Teague/noisy/8"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"full_run_result_{seed}.npy"), results)


# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
