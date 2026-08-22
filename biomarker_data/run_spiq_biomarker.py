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

from spiq.qaoa import QAOASolver
import spiq.graphs as graph_utils

print("Command-line arguments:", sys.argv)

from biomarker_data import pcbo_utils
from biomarker_data.biomarker_utils import convert_pubo_to_ising
from biomarker_data.paths import biomarker_pickle_dir, sample_data_dir


def main():
    n_qubits = int(sys.argv[1])
    reps = int(sys.argv[2])
    n_gens = int(sys.argv[3])
    seed = int(sys.argv[4])
    noise = bool(int(sys.argv[5]))

    n = n_qubits

    print(f"Num. qubits: {n_qubits}, Num reps: {reps}")

    feature_set, feature_to_idx, first_corr_arr, second_corr_arr, third_corr_arr = (
        pcbo_utils.load_features_and_corr_files(
            str(sample_data_dir(n_qubits, use_copy=False))
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
    biomarker_paulis = convert_pubo_to_ising(pubo, n_qubits)
    cost_hamiltonian = SparsePauliOp.from_list(biomarker_paulis)

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    qaoa_obj = QAOASolver(cost_hamiltonian, circuit, sim_device="CPU")

    qaoa_obj.prepare_circuit()

    exact_energy = qaoa_obj.evaluate_exact_energy()

    # Run CAFQA process
    qaoa_obj.run_spiq(n_gens=n_gens)

    best_cafqa_params = qaoa_obj.best_cafqa_gen_params[::-1]
    best_cafqa_fitness_values = qaoa_obj.best_cafqa_gen_fitness[::-1]

    unique_fitness_values = np.unique(best_cafqa_fitness_values)

    print("Best 20 best CAFQA fitness values:", unique_fitness_values[:20])

    pickle_folder = biomarker_pickle_dir()
    os.makedirs(pickle_folder, exist_ok=True)

    output_data = {
        "best_cafqa_fitness_values": best_cafqa_fitness_values,
        "best_cafqa_parameters": best_cafqa_params,
        "CAFQA_initialization_energy": qaoa_obj.energy_best,
    }
    with open(f"{pickle_folder}/{n_qubits}_qb_biomarker_cafqa_results.pkl", "wb") as f:
        pickle.dump(output_data, f)


# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
