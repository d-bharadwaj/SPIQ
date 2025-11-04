import os
import sys
import warnings

import numpy as np
import rustworkx as rx

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle
import traceback
from timeit import default_timer as timer

import multiprocess as mp
from qiskit.circuit import Parameter
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_optimization.converters import QuadraticProgramToQubo

sys.path.append("../")
import testing_scripts.graphs_utils as graph_utils
from testing_scripts.knapsack_utils import generate_knapsack_instance
from testing_scripts.qaoa_utils import QAOASolver

print("Command-line arguments:", sys.argv)

import os

from qiskit.quantum_info import SparsePauliOp


def main():
    # Initialize variables (replace with your actual initialization logic)
    n_qubits = int(sys.argv[1])
    reps = int(sys.argv[2])
    n_gens = int(sys.argv[3])
    seed = int(sys.argv[4])
    noise = bool(int(sys.argv[5]))
    graph = str(sys.argv[6])

    print(f"Num. qubits: {n_qubits}, Num reps: {reps}")

    task_id = os.environ.get("SLURM_PROCID")
    if task_id is None:
        print("Warning: SLURM_PROCID not set — running outside Slurm?")
        task_id = 0
    else:
        task_id = int(task_id)
    print(f"Running task with SLURM_PROCID = {task_id}")
    seed = task_id

    ## Maxcut Formulation
    if graph == "k_reg":
        print("Selected graph type: k-regular")
        k = 3  # for k-regular graphs
        G = graph_utils.generate_k_regular_graph(
            num_vertices=n_qubits, weighted=False, seed=seed + 3, k=k
        )
    elif graph == "complete":  # unweighted
        print("Selected graph type: complete")
        G = graph_utils.generate_random_complete_graph(
            num_vertices=n_qubits, weighted=False, seed=seed
        )
    elif graph == "ego":
        print("Selected graph type: ego")
        G = graph_utils.generate_random_ego_graph(
            num_vertices=n_qubits, weighted=False, seed=seed
        )

    else:
        raise ValueError(f"Graph type '{graph}' not available.")

    # G = graph_utils.generate_random_erdos_renyi_graph(num_nodes=n_qubits,probability=0.4,weighted=True,seed=seed)

    # CAFQA Workflow
    max_cut_paulis = graph_utils.build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    qaoa_obj = QAOASolver(cost_hamiltonian, circuit, sim_device="CPU")

    qaoa_obj.prepare_circuit()
    results_dir = f"../np_data/Comprehensive_Proof/Maxcut/Unweighted/{graph}/{reps}_reps/{n_qubits}_qbs"
    os.makedirs(results_dir, exist_ok=True)  # Creates folder if it doesn't exist
    results_file = os.path.join(
        results_dir, f"cafqa_result_{seed}"
    )  # This is a file path, not a directory

    if n_qubits <= 20:
        exact_energy = qaoa_obj.evaluate_exact_energy()
    else:
        exact_energy = None

    # Run CAFQA process
    qaoa_obj.run_cafqa(n_gens=n_gens, out_file=results_file)

    best_cafqa_params = qaoa_obj.best_cafqa_gen_params[::-1]
    best_cafqa_fitness_values = qaoa_obj.best_cafqa_gen_fitness[::-1]

    unique_fitness_values = np.unique(best_cafqa_fitness_values)

    print("Best 20 best CAFQA fitness values:", unique_fitness_values[:20])

    results_dict = {
        "best_cafqa_parameters": best_cafqa_params,
        "best_cafqa_fitness_values": best_cafqa_fitness_values,
        "CAFQA_initialization_energy": qaoa_obj.energy_best,
        "exact_energy": exact_energy,
    }

    # Save the dictionary using numpy
    np.save(results_file, results_dict)


# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
