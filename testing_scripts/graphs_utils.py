import rustworkx as rx
import random
from sage.all import Graph
from typing import Sequence
import numpy as np 
from qiskit_ibm_runtime import SamplerV2 as Sampler


def generate_k_regular_graph(num_vertices, k, weighted=False, seed=False):
    """
    Generate a k-regular graph with a specified number of vertices.

    Parameters
    ----------
    num_vertices : int
        The number of vertices in the graph.
    k : int
        Each vertex will have exactly k edges. Must be less than num_vertices and num_vertices * k must be even.
    weighted : bool, optional
        If True, the edges will have random weights between 1 and 10. If False, all edges will have a weight of 1. Default is False.
    seed : bool, optional
        If True, the random number generator will be seeded with 0 for reproducibility. Default is False.

    Returns
    -------
    graph : rx.PyGraph
        A k-regular graph with the specified number of vertices and edges.

    Raises
    ------
    ValueError
        If k is greater than or equal to num_vertices or if num_vertices * k is not even.

    Notes
    -----
    The function uses a retry mechanism to ensure that a valid k-regular graph is generated. If a deadlock is detected during edge assignment, the process restarts from scratch.
    """
    if seed:
        random.seed(0)

    if k >= num_vertices or (num_vertices * k) % 2 != 0:
        raise ValueError("Invalid parameters: k must be < n and n*k must be even")

    while True:  # Retry if we fail to complete the graph
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_vertices))

        stubs = list(range(num_vertices)) * k
        random.shuffle(stubs)

        edges = set()
        while stubs:
            u, v = stubs.pop(), stubs.pop()
            attempts = 0
            while u == v or (u, v) in edges or (v, u) in edges:
                stubs.append(v)
                random.shuffle(stubs)
                v = stubs.pop()
                attempts += 1
                if attempts > len(stubs):  # Deadlock detected, restart
                    break
            else:
                edges.add((u, v))
                weight = random.randint(1, 10) if weighted else 1
                graph.add_edge(u, v, weight)
                continue
            break  # Restart from scratch

        if len(edges) == (num_vertices * k) // 2:
            return graph  # Successfully created a k-regular graph
    
def generate_random_complete_graph(num_vertices, weighted=False, seed=False, save_path=None):
    """
    Generate a random complete graph.
    Parameters
    ----------
    num_vertices : int
        The number of vertices in the graph.
    weighted : bool, optional
        If True, edges will have random weights between 1 and 10. Default is False.
    seed : bool, optional
        If True, the random seed will be set to 0 for reproducibility. Default is False.
    save_path : str, optional
        Path to save the generated graph. Default is None.
    Returns
    -------
    G : rx.PyGraph
        The generated complete graph.
    """
    G = rx.PyGraph()
    G.add_nodes_from(range(num_vertices))

    if seed:
        random.seed(0)

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            weight = random.randint(1, 10) if weighted else 1
            G.add_edge(i, j, weight)
    return G

def compute_optimal_max_cut(graph: rx.PyGraph) -> int:
    """
    Compute the optimal Max-Cut of a graph.

    Parameters
    ----------
    graph : rx.PyGraph
        The input graph.

    Returns
    -------
    int
        The weight of the optimal Max-Cut.
    """
    # Convert rustworkx graph into SageMath Graph.
    sage_graph = Graph()
    for u, v, weight in graph.weighted_edge_list():
        sage_graph.add_edge(u, v, weight)

    # Compute the Max-Cut
    return sage_graph.max_cut(use_edge_labels=True)

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    seq =  [int(digit) for digit in result]
    seq.reverse()
    return seq

def _evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(list(graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
    return sum(w*(x[u] * (1 - x[v]) + x[v] * (1 - x[u])) for u, v,w in list(graph.weighted_edge_list()))

def get_final_distribution(qaoa_obj,final_params):
    if qaoa_obj.vanilla:
        optimized_circuit = qaoa_obj.circuit.assign_parameters(final_params)
    else:
        optimized_circuit = qaoa_obj.pcirc.assign_parameters(final_params)
    optimized_circuit.measure_all()

    sampler = Sampler(mode=qaoa_obj.backend)
    sampler.options.default_shots = 10000 #NOTE: This is duplicate?

    pub= (optimized_circuit, )
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val/shots for key, val in counts_int.items()}

    return final_distribution_int

#NOTE: can definitely make this super fast.
def calculate_approximation_ratio(fin_distribution, graph):
    """
    Calculates the approximation ratio for QAOA's MaxCut solution.
    
    Args:
        probabilities (list): List of probabilities for each measured bitstring
        cut_values (list): List of cut values corresponding to each bitstring
        optimal_cut (float): Optimal MaxCut value for the graph
        
    Returns:
        float: Approximation ratio (between 0 and 1)
    """
    # Handle edge case where optimal cut is 0 (unlikely for non-trivial graphs)

    optimal_cut = compute_optimal_max_cut(graph)
    if optimal_cut == 0:
        return 0.0
    
    # Calculate expected cut value
    bitstrings = list(fin_distribution.keys())
    probabilities = list(fin_distribution.values())
    num_bits = len(graph)
    bitstrings_seq = [to_bitstring(int(bitstring),num_bits) for bitstring in bitstrings] 

    cut_values = [_evaluate_sample(bitstring_seq,graph) for bitstring_seq in bitstrings_seq]

    expected_cut = sum(p * c for p, c in zip(probabilities, cut_values))

    return expected_cut / optimal_cut
