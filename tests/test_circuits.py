import pytest
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
import numpy as np

import clapton.circuit_manipulation as circuit_manipulation
import testing_scripts.qaoa_utils as qaoa_utils
import testing_scripts.graphs_utils as graphs_utils


num_qubits_list = list(np.linspace(2, 50, 5, dtype=int))
reps_list = list(range(1, 6))  # same length


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("num_qubits, reps", list(zip(num_qubits_list, reps_list)))
def test_num_params_in_ma_qaoa_circuit(num_qubits, reps):
    G = graphs_utils.generate_random_complete_graph(
        num_vertices=num_qubits, weighted=True, seed=False
    )
    num_nodes = G.num_nodes()
    num_edges = G.num_edges()
    max_cut_paulis = graphs_utils.build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    qaoa_obj = qaoa_utils.QAOASolver(cost_hamiltonian, circuit, sim_device="CPU")
    qaoa_obj.prepare_circuit()

    num_ma_qaoa_params = (num_nodes + num_edges) * reps
    assert num_ma_qaoa_params == qaoa_obj.pcirc.num_parameters


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("num_qubits, reps", list(zip(num_qubits_list, reps_list)))
def test_num_params_in_vanilla_circuit(num_qubits, reps):
    G = graphs_utils.generate_random_complete_graph(
        num_vertices=num_qubits, weighted=True, seed=False
    )
    max_cut_paulis = graphs_utils.build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    qaoa_obj = qaoa_utils.QAOASolver(cost_hamiltonian, circuit, sim_device="CPU")
    qaoa_obj.vanilla = True
    qaoa_obj.prepare_circuit()

    num_ma_qaoa_params = 2 * reps
    assert num_ma_qaoa_params == qaoa_obj.circuit.num_parameters
