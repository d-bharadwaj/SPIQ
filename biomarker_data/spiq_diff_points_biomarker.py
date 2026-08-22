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

from clapton.clapton import claptonize
from clapton.circuit_manipulation import (
    transform_to_allowed_gates,
    qiskit_to_stim,
    modify_circuit,
    multi_angle_qaoa_circuit,
    generate_qiskit_param_map,
)
from spiq.graphs import (
    generate_random_complete_graph,
    generate_k_regular_graph,
    build_max_cut_paulis,
    compute_optimal_max_cut,
)
from spiq.qaoa import QAOASolver, evaluate_energy
from biomarker_data import pcbo_utils
from biomarker_data.biomarker_utils import convert_pubo_to_ising
from biomarker_data.paths import biomarker_pickle_dir, sample_data_dir
from qiskit.quantum_info import SparsePauliOp
import os
import pickle


def run_qaoa_task_pool(args):

    max_iters = 10 * 1e3
    task_id, biomarker_qaoa, initial_params, fitness_val = args

    # QAOA Optimization
    spiq_params = [param * (np.pi / 2) for param in initial_params]

    try:
        print(f"Starting task: {task_id} for val : {fitness_val}")
        result, obj_values = biomarker_qaoa.run_qaoa(
            initial_params=spiq_params, max_iters=max_iters, opt="COBYLA"
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
    selected_spiq_parameters,
    selected_fitness_vals,
):

    spiq_args = [
        (f"task_{i}", ma_qaoa_object, params, fitness)
        for i, (params, fitness) in enumerate(
            zip(selected_spiq_parameters, selected_fitness_vals)
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
    all_args = spiq_args + random_args + vanilla_args

    with mp.Pool(processes=len(all_args)) as pool:
        results_list = pool.map(run_qaoa_task_pool, all_args)

    results = {res["task_name"]: res for res in results_list}

    # Extract objective values for each task_id
    task_objective_values = {
        task_id: res.get("obj_values", None) for task_id, res in results.items()
    }

    # Return results
    return {
        "SPIQ_initialization_energy": ma_qaoa_object.energy_best,
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
            str(sample_data_dir(n_qubits))
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
    biomarker_qaoa = QAOASolver(cost_hamiltonian, circuit, sim_device="GPU")

    biomarker_qaoa.prepare_circuit()

    # # Run SPIQ process
    # biomarker_qaoa.run_spiq(n_gens=n_gens,err=noise)
    # print(f"{n} Qubits and {reps} reps")
    # print(f"Minimum Energy found with SPIQ initialization: {biomarker_qaoa.energy_best}")

    # # Best Solutions
    # best_spiq_fitness_values = biomarker_qaoa.best_spiq_gen_fitness
    # best_spiq_parameters = biomarker_qaoa.best_spiq_gen_params

    # Importing from seperate file
    with open(
        biomarker_pickle_dir() / f"{n}_qb_biomarker_spiq_results.pkl",
        "rb",
    ) as f:
        spiq_data = pickle.load(f)
    best_spiq_fitness_values = spiq_data["best_spiq_fitness_values"]
    best_spiq_parameters = spiq_data["best_spiq_parameters"]
    print("spiq_data keys:", list(spiq_data.keys()))
    biomarker_qaoa.energy_best = spiq_data["SPIQ_initialization_energy"]

    unique_fitness_values = np.unique(best_spiq_fitness_values)

    # When choosing best out of unique energies
    selected_fitness_vals = list(
        unique_fitness_values[::3][:5]
    )  # Select 5 of the best unique spaced-out solutions.
    # selected_fitness_vals = list(unique_fitness_values[:5]) #Select 5 of the best unique solutions.

    selected_fitness_indices = [
        best_spiq_fitness_values.index(value) for value in selected_fitness_vals
    ]
    selected_spiq_parameters = np.array(best_spiq_parameters)[
        selected_fitness_indices
    ]

    # # # # #When choosing just best 5 (can be same energies)
    # selected_fitness_vals = list(best_spiq_fitness_values)[:5] #Select 5 of the best solutions (can be repeated).
    # selected_spiq_parameters = np.array(best_spiq_parameters)[:5]

    # Vanilla QAOA
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    vanilla_biomarker = QAOASolver(
        cost_hamiltonian, circuit.decompose().decompose(), sim_device="GPU"
    )
    vanilla_biomarker.vanilla = True

    if noise:
        vanilla_biomarker.err = 1e-4
        biomarker_qaoa.err = 1e-4

    start = timer()
    results = execute_qaoa_tasks(
        biomarker_qaoa, vanilla_biomarker, selected_spiq_parameters, selected_fitness_vals
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
