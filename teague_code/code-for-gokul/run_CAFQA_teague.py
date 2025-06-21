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
import pickle

sys.path.append("../")
from testing_scripts.qaoa_utils import QAOASolver
import testing_scripts.graphs_utils as graph_utils
print("Command-line arguments:", sys.argv)

sys.path.append("../teague_code/code-for-gokul")
from qiskit.quantum_info import SparsePauliOp
import pcbo_utils
import os

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

    print(f"Num. qubits: {n_qubits}, Num reps: {reps}")
    k = 3  # for k-regular graphs

    # Generate the graph
    G = graph_utils.generate_random_complete_graph(num_vertices=n, weighted=True,seed=seed)

    # Build the cost Hamiltonian
    max_cut_paulis = graph_utils.build_max_cut_paulis(G)

    #Teague Data
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

    def convert_pubo_to_ising(hypergraph: dict) -> list[tuple[str, float]]:
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

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    qaoa_obj = QAOASolver(cost_hamiltonian,circuit,sim_device="CPU")

    qaoa_obj.prepare_circuit()

    # Run CAFQA process
    qaoa_obj.run_CAFQA(n_gens=n_gens)

    best_cafqa_params = qaoa_obj.best_cafqa_gen_params[::-1]
    best_cafqa_fitness_values = qaoa_obj.best_cafqa_gen_fitness[::-1]


    unique_fitness_values = np.unique(best_cafqa_fitness_values)

    print("Best 20 best CAFQA fitness values:", unique_fitness_values[:20])
    
    exact_energy = qaoa_obj.evaluate_exact_energy()
    print("Exact energy:", exact_energy)
    
    pickle_folder = "../teague_code/code-for-gokul/teague_pickle_data"
    os.makedirs(pickle_folder, exist_ok=True)  # Creates folder if it doesn't exist

    # Save results to a file for later import
    output_data = {
        "best_cafqa_fitness_values": best_cafqa_fitness_values,
        "best_cafqa_parameters": best_cafqa_params,
        "CAFQA_initialization_energy" : qaoa_obj.energy_best
    }
    with open(f"{pickle_folder}/{n_qubits}_qb_teague_cafqa_results.pkl", "wb") as f:
        pickle.dump(output_data, f)

# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()