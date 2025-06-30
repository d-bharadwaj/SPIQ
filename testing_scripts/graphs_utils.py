import rustworkx as rx
import random
from sage.all import Graph
from typing import Sequence
import numpy as np
from qiskit_ibm_runtime import SamplerV2 as Sampler

def generate_k_regular_graph(num_vertices, k, weighted=False, seed=None):
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
    seed : int, optional
        If provided, the random number generator will be seeded with `seed` for reproducibility. Default is None.

    Returns
    -------
    graph : rustworkx.PyGraph
        A k-regular graph with the specified number of vertices and edges.

    Raises
    ------
    ValueError
        If k is greater than or equal to num_vertices or if num_vertices * k is not even.

    Notes
    -----
    The function uses a retry mechanism to ensure that a valid k-regular graph is generated.
    If a deadlock is detected during edge assignment, the process restarts from scratch.
    """
    if seed:
        random.seed(seed)

    if k >= num_vertices or (num_vertices * k) % 2 != 0:
        raise ValueError("Invalid parameters: k must be < n and n*k must be even")

    while True:
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
                if attempts > len(stubs):
                    break
            else:
                edges.add((u, v))
                weight = random.randint(1, 10) if weighted else 1
                graph.add_edge(u, v, weight)
                continue
            break

        if len(edges) == (num_vertices * k) // 2:
            return graph

def generate_random_complete_graph(num_vertices, weighted=False, seed=None, save_path=None):
    """
    Generate a random complete graph.

    Parameters
    ----------
    num_vertices : int
        The number of vertices in the graph.
    weighted : bool, optional
        If True, edges will have random weights between 1 and 10. Default is False.
    seed : int, optional
        If provided, the random seed will be set to `seed` for reproducibility. Default is None.
    save_path : str, optional
        Path to save the generated graph. Default is None.

    Returns
    -------
    G : rustworkx.PyGraph
        The generated complete graph.
    """
    G = rx.PyGraph()
    G.add_nodes_from(range(num_vertices))

    if seed:
        random.seed(seed)

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            weight = random.randint(1, 10) if weighted else 1
            G.add_edge(i, j, weight)
    return G

def generate_random_ego_graph(num_nodes, weighted=False, seed=None):
    """
    Generate a random ego network with the specified number of nodes.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the ego network (including the ego).
        Must be >= 2.
    weighted : bool, optional
        If True, assign random weights (1–10) to edges. If False, all edge weights are 1. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    graph : rustworkx.PyGraph
        A randomly generated ego network with one ego node connected to all others,
        and some random edges among the neighbors.

    Raises
    ------
    ValueError
        If num_nodes is less than 2.
    """
    if num_nodes < 2:
        raise ValueError("Ego network must have at least 2 nodes (1 ego + 1 neighbor)")
    if seed:
        random.seed(seed)

    graph = rx.PyGraph()
    graph.add_nodes_from(range(num_nodes))

    ego = 0  # node 0 is the ego
    neighbors = list(range(1, num_nodes))

    # Connect ego to all neighbors
    for n in neighbors:
        weight = random.randint(1, 10) if weighted else 1
        graph.add_edge(ego, n, weight)

    # Optionally add random edges between neighbors (like a small-world neighborhood)
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if random.random() < 0.3:  # 30% chance of adding edge
                u, v = neighbors[i], neighbors[j]
                weight = random.randint(1, 10) if weighted else 1
                graph.add_edge(u, v, weight)

    return graph

def generate_random_erdos_renyi_graph(num_nodes, probability=0.5, weighted=False, seed=None):
    if seed is not None:
        random.seed(seed)
    # Generate random Erdos-Renyi graph structure
    random_graph = rx.undirected_gnp_random_graph(num_nodes, probability, seed=seed)
    
    # Create a new PyGraph and add the same number of nodes
    graph = rx.PyGraph()
    graph.add_nodes_from([None] * num_nodes)  # Adds num_nodes empty payloads

    # Add edges with optional weights
    for u, v in random_graph.edge_list():
        weight = random.randint(1, 10) if weighted else 1
        graph.add_edge(u, v, weight)

    largest_cc = max(rx.connected_components(graph), key=len)
    G_sub = graph.subgraph(list(largest_cc)).copy()

    return G_sub

def compute_optimal_max_cut(graph: rx.PyGraph) -> int:
    """
    Compute the optimal Max-Cut of a graph.

    Parameters
    ----------
    graph : rustworkx.PyGraph
        The input graph.

    Returns
    -------
    int
        The weight of the optimal Max-Cut.
    """
    sage_graph = Graph()
    for u, v, weight in graph.weighted_edge_list():
        sage_graph.add_edge(u, v, weight)
    return sage_graph.max_cut(use_edge_labels=True)

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """
    Convert the graph to a list of Pauli strings for Max-Cut.

    Parameters
    ----------
    graph : rustworkx.PyGraph
        The input graph.

    Returns
    -------
    pauli_list : list of tuple of (str, float)
        List of tuples where each tuple contains a Pauli string and its corresponding edge weight.
    """

    if len(graph)==0:
        assert False, "Graph is empty"

    if len(graph)==1:
        assert False, "Single Node Graph!"

    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("".join(paulis)[::-1], weight))
    return pauli_list

def to_bitstring(integer, num_bits):
    """
    Convert an integer to a bitstring of given width, reversed as a list.

    Parameters
    ----------
    integer : int
        The integer to convert.
    num_bits : int
        The width of the bitstring.

    Returns
    -------
    seq : list of int
        The reversed bitstring as a list of bits.
    """
    result = np.binary_repr(integer, width=num_bits)
    seq = [int(digit) for digit in result]
    seq.reverse()
    return seq

def _evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    """
    Evaluate the Max-Cut value for a given bitstring assignment.

    Parameters
    ----------
    x : Sequence[int]
        Bitstring assignment for the nodes.
    graph : rustworkx.PyGraph
        The input graph.

    Returns
    -------
    float
        The cut value for the assignment.
    """
    assert len(x) == len(list(graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
    return sum(
        w * (x[u] * (1 - x[v]) + x[v] * (1 - x[u]))
        for u, v, w in list(graph.weighted_edge_list())
    )

def get_final_distribution(qaoa_obj, final_params):
    """
    Get the final measurement distribution from a QAOA object and parameters.

    Parameters
    ----------
    qaoa_obj : object
        QAOA object with circuit and backend attributes.
    final_params : array-like
        Optimized parameters for the QAOA circuit.

    Returns
    -------
    final_distribution_int : dict
        Dictionary mapping bitstrings (as int) to probabilities.
    """ 
    if qaoa_obj.vanilla:
        optimized_circuit = qaoa_obj.circuit.assign_parameters(final_params)
    else:
        optimized_circuit = qaoa_obj.pcirc.assign_parameters(final_params)
    optimized_circuit.measure_all()

    sampler = Sampler(mode=qaoa_obj.backend)
    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val / shots for key, val in counts_int.items()}
    return final_distribution_int

def calculate_approximation_ratio(fin_distribution, graph):
    """
    Calculate the approximation ratio for QAOA's Max-Cut solution.

    Parameters
    ----------
    fin_distribution : dict
        Dictionary mapping bitstrings (as int or str) to probabilities.
    graph : rustworkx.PyGraph
        The input graph.

    Returns
    -------
    float
        Approximation ratio (between 0 and 1).
    """
    optimal_cut = compute_optimal_max_cut(graph)
    if optimal_cut == 0:
        return 0.0

    num_bits = len(graph)
    expected_cut = 0.0
    for bitstring, prob in fin_distribution.items():
        x = to_bitstring(int(bitstring), num_bits)
        cut = _evaluate_sample(x, graph)
        expected_cut += prob * cut

    return expected_cut / optimal_cut
