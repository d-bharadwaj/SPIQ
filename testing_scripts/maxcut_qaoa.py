import os
import sys
import warnings

import numpy as np
import rustworkx as rx

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from timeit import default_timer as timer

from qiskit.circuit import Parameter
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from scipy.optimize import minimize

sys.path.append("../")
from clapton.circuit_manipulation import (generate_qiskit_param_map,
                                          modify_circuit,
                                          multi_angle_qaoa_circuit,
                                          qiskit_to_stim,
                                          transform_to_allowed_gates)
from clapton.clapton import claptonize

from maxcut_processing import evaluate_maxcut
from testing_scripts.graphs_utils import (build_max_cut_paulis,
                                          compute_optimal_max_cut,
                                          generate_k_regular_graph,
                                          generate_random_complete_graph)
from testing_scripts.qaoa_utils import QAOASolver, evaluate_energy

print("Command-line arguments:", sys.argv)

import os
import traceback

import multiprocess as mp
import numpy as np


def run_qaoa_task_pool(args):
    task_name, qaoa_solver, initial_params, max_iters, opt = args
    try:
        print(f"Starting task: {task_name}")
        result, obj_values = qaoa_solver.run_qaoa(
            initial_params=initial_params, max_iters=max_iters, opt=opt
        )
        print(f"Finished task: {task_name}")
        return {
            "task_name": task_name,
            "result": result,
            "obj_values": obj_values,
            "final_energy": result.fun,
        }
    except Exception as e:
        print(f"Error in task {task_name}: {e}")
        traceback.print_exc()
        return {
            "task_name": task_name,
            "error": str(e),
        }


# Function to execute in parallel using Pool
def execute_qaoa_tasks(
    maxcut_qaoa, random_angles, cafqa_params, max_iters, G, optimal_max_cut_val
):
    tasks = [
        ("random", maxcut_qaoa, random_angles, max_iters, "COBYLA"),
        ("cafqa", maxcut_qaoa, cafqa_params, max_iters, "COBYLA"),
    ]

    with mp.Pool(processes=2) as pool:
        results_list = pool.map(run_qaoa_task_pool, tasks)

    results = {res["task_name"]: res for res in results_list}

    # Extract values (same as before)
    random_result = results["random"]["result"]
    random_obj_values = results["random"]["obj_values"]
    random_fin_energy = results["random"]["final_energy"]

    cafqa_result = results["cafqa"]["result"]
    cafqa_obj_values = results["cafqa"]["obj_values"]
    cafqa_fin_energy = results["cafqa"]["final_energy"]

    maxcut_qaoa._initialize_backend()
    random_max_cut_val = evaluate_maxcut(
        G, maxcut_qaoa.pcirc, random_result, maxcut_qaoa.backend
    )
    cafqa_max_cut_val = evaluate_maxcut(
        G, maxcut_qaoa.pcirc, cafqa_result, maxcut_qaoa.backend
    )

    random_approx_ratio = random_max_cut_val / optimal_max_cut_val
    cafqa_approx_ratio = cafqa_max_cut_val / optimal_max_cut_val

    # Print results
    print(f"Optimal Max Cut Value : {optimal_max_cut_val}")
    print(
        f"Max Cut value with Random initialization after {max_iters} iteration: {random_max_cut_val}"
    )
    print(
        f"Max Cut value with CAFQA initialization after {max_iters} iteration: {cafqa_max_cut_val}"
    )
    print(f"Random Approx. Ratio: {random_approx_ratio}")
    print(f"CAFQA Approx. Ratio: {cafqa_approx_ratio}")

    # Return results
    return {
        "CAFQA_initialization_energy": maxcut_qaoa.energy_best,
        "Random_initialization_max_cut": random_max_cut_val,
        "CAFQA_initialization_max_cut": cafqa_max_cut_val,
        "Random_approx_ratio": random_approx_ratio,
        "CAFQA_approx_ratio": cafqa_approx_ratio,
        "Final_random_energy": random_fin_energy,
        "Final_cafqa_energy": cafqa_fin_energy,
        "Random_obj_values": random_obj_values,
        "CAFQA_obj_values": cafqa_obj_values,
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
    maxcut_qaoa.run_spiq(n_gens=n_gens)
    print(f"{n} Qubits and {reps} reps")
    print(f"Minimum Energy found with CAFQA initialization: {maxcut_qaoa.energy_best}")

    # Random Initialization
    random_angles = np.random.random(len(maxcut_qaoa.ks_best))

    # QAOA Optimization
    ordered_params = [param.name for param in maxcut_qaoa.pcirc.parameters]
    angle_multipliers = [
        -np.pi / 4 if "gamma" in param else np.pi / 4 for param in ordered_params
    ]
    cafqa_params = [
        param * (np.pi / 2)
        for param, multiplier in zip(maxcut_qaoa.ks_best, angle_multipliers)
    ]

    # Execute QAOA tasks
    max_iters = 1000

    start = timer()
    results = execute_qaoa_tasks(
        maxcut_qaoa, random_angles, cafqa_params, max_iters, G, optimal_max_cut_val
    )
    end = timer()
    print(f"Total Time : {end - start}")
    # Save results
    output_dir = f"../np_data/rep_sweep_2000_gens/NM_test/{reps}_layers"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"12_qb_single_results_{seed}.npy"), results)


# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
