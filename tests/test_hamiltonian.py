import pytest
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.converters import QuadraticProgramToQubo
import numpy as np

from clapton.circuit_manipulation import (
    transform_to_allowed_gates,
    qiskit_to_stim,
    modify_circuit,
    multi_angle_qaoa_circuit,
    transform_qiskit_to_stim,
    generate_qiskit_param_map,
)
import testing_scripts.graphs_utils as graphs_utils
from testing_scripts.knapsack_utils import generate_knapsack_instance


@pytest.mark.parametrize(
    "edges,expected_terms",
    [
        ([(0, 1, 1)], 1),
        ([(0, 1, 1), (1, 2, 1)], 2),
        ([(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)], 4),
        ([(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1), (2, 3, 1)], 6),  # K4
    ],
)
def test_cost_hamiltonian_term_count_explicit_and_hermitian(edges, expected_terms):
    G = rx.PyGraph()
    # Determine how many nodes are needed
    node_count = max(max(u, v) for u, v, _ in edges) + 1
    G.add_nodes_from([None] * node_count)  # Add nodes 0 to max_index
    G.add_edges_from(edges)
    max_cut_paulis = graphs_utils.build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    assert len(cost_hamiltonian) == expected_terms
    H_matrix = cost_hamiltonian.to_matrix()
    assert np.allclose(H_matrix, H_matrix.conj().T)


def test_cost_hamiltonian_empty_graph():
    G = rx.PyGraph()
    with pytest.raises(AssertionError, match="Graph is empty"):
        graphs_utils.build_max_cut_paulis(G)


def test_cost_hamiltonian_single_node():
    G = rx.PyGraph()
    G.add_node(None)
    with pytest.raises(AssertionError, match="Single Node Graph!"):
        graphs_utils.build_max_cut_paulis(G)


def test_cost_hamiltonian_single_edge():
    G = rx.PyGraph()
    # Determine how many nodes are needed
    G.add_nodes_from(range(2))
    G.add_edge(0, 1, 1)
    max_cut_paulis = graphs_utils.build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    assert len(cost_hamiltonian) == 1
    assert (
        cost_hamiltonian.paulis[0].to_label().count("Z") == 2
    )  # Each term should be a ZZ interaction


def test_cost_hamiltonian_complete_graph():
    G = rx.generators.complete_graph(5)
    max_cut_paulis = graphs_utils.build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    assert len(cost_hamiltonian) == 10  # K5 has 10 edges
