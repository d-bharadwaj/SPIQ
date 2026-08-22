import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import transpile
from qiskit.circuit import Parameter, ParameterExpression
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2
from qiskit.converters import circuit_to_dag, dag_to_circuit

import os

print("Current working directory:", os.getcwd())
from clapton.clapton import claptonize
from clapton.circuit_manipulation import (
    transform_to_allowed_gates,
    qiskit_to_stim,
    modify_circuit,
    multi_angle_qaoa_circuit,
    transform_qiskit_to_stim,
    generate_qiskit_param_map,
    relax_qaoa_parameters,
)
from spiq.graphs import (
    generate_random_complete_graph,
    generate_k_regular_graph,
    compute_optimal_max_cut,
    build_max_cut_paulis,
)
from spiq.qaoa import QAOASolver, evaluate_energy

import numpy as np
import qubovert

from biomarker_data import pcbo_utils
from biomarker_data.biomarker_utils import convert_pubo_to_ising
from biomarker_data.paths import sample_data_dir

n_qubits = 14
feature_set, feature_to_idx, first_corr_arr, second_corr_arr, third_corr_arr = (
    pcbo_utils.load_features_and_corr_files(str(sample_data_dir(n_qubits)))
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

ising_paulis = convert_pubo_to_ising(pubo, n_qubits)
cost_hamiltonian = SparsePauliOp.from_list(ising_paulis)
paulis, coeffs = cost_hamiltonian.paulis.to_labels(), cost_hamiltonian.coeffs.real

reps = 2
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

biomarker_qaoa = QAOASolver(cost_hamiltonian, circuit)
biomarker_qaoa.prepare_circuit()

biomarker_qaoa.run_spiq(n_gens=1000)

exact_solution = biomarker_qaoa.evaluate_exact_energy()

spiq_angles = [param * np.pi / 2 for param in biomarker_qaoa.ks_best]
random_angles = np.random.random(len(biomarker_qaoa.ks_best))

max_iters = 1000
spiq_result, spiq_iteration_vals = biomarker_qaoa.run_qaoa(
    initial_params=spiq_angles, max_iters=max_iters, opt="SPSA"
)
random_result, random_iteration_vals = biomarker_qaoa.run_qaoa(
    initial_params=random_angles, max_iters=max_iters, opt="SPSA"
)

results_dict = {
    "spiq_fin_energy": spiq_result.fun,
    "spiq_iteration_vals": spiq_iteration_vals,
    "random_result_energy": random_result.fun,
    "random_iteration_vals": random_iteration_vals,
}

output_dir = f"../np_data/biomarker_data_results/SPSA_test/{n_qubits}_qubits"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "SPSA_results.npy"), results_dict)
