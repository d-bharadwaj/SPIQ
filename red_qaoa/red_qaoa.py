from itertools import combinations
import networkx as nx
import random
import math


# Average node degree of a graph
def average_node_degree(graph):
    return sum(d for n, d in graph.degree()) / float(graph.number_of_nodes())


# Objective function for simulated annealing
# Returns the absolute difference between the average node degree of the subgraph and the original graph
def objective_function(subgraph, original_graph_and):
    return abs(average_node_degree(subgraph) - original_graph_and)


# Generate a random neighbor of a given subgraph
def generate_neighbor(subgraph, graph):
    nodes_to_replace = list(subgraph.nodes)
    nodes_not_in_subgraph = list(set(graph.nodes) - set(subgraph.nodes))
    node_to_remove = random.choice(nodes_to_replace)
    node_to_add = random.choice(nodes_not_in_subgraph)
    neighbor = subgraph.copy()
    neighbor.remove_node(node_to_remove)
    neighbor.add_node(node_to_add)
    neighbor.add_edges_from(
        [
            (node_to_add, nbr)
            for nbr in graph.neighbors(node_to_add)
            if nbr in neighbor.nodes
        ]
    )
    return neighbor


# Simulated annealing algorithm for finding a subgraph of a given size
# with cloest average node degree to the original graph
def sa(
    graph,
    subgraph_size,
    initial_temperature=100,
    cooling_rate=0.99,
    stopping_temperature=1e-6,
):
    original_graph_and = average_node_degree(graph)

    # Initialize the subgraph with random nodes
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    subgraph = graph.subgraph(nodes[:subgraph_size]).copy()

    # Initialize the best subgraph found so far
    best_subgraph = subgraph.copy()
    best_objective = objective_function(subgraph, original_graph_and)

    temperature = initial_temperature

    while temperature > stopping_temperature:
        neighbor = generate_neighbor(subgraph, graph)
        current_objective = objective_function(subgraph, original_graph_and)
        neighbor_objective = objective_function(neighbor, original_graph_and)
        delta_energy = neighbor_objective - current_objective

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            subgraph = neighbor
            current_objective = neighbor_objective

            if current_objective < best_objective:
                best_subgraph = subgraph.copy()
                best_objective = current_objective

        temperature *= cooling_rate

    mapping = {
        k: v
        for k, v in zip(best_subgraph.nodes(), range(best_subgraph.number_of_nodes()))
    }

    return nx.relabel_nodes(best_subgraph, mapping)


# Adaptive simulated annealing algorithm for finding a subgraph of a given size
# with cloest average node degree to the original graph
def sa_adapt(
    graph,
    subgraph_size,
    initial_temperature=100,
    cooling_rate=0.99,
    stopping_temperature=1e-6,
    max_rejections=10,
):
    original_graph_and = average_node_degree(graph)

    # Initialize the subgraph with random nodes
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    subgraph = graph.subgraph(nodes[:subgraph_size]).copy()

    # Initialize the best subgraph found so far
    best_subgraph = subgraph.copy()
    best_objective = objective_function(subgraph, original_graph_and)

    temperature = initial_temperature
    rejections = 0

    while temperature > stopping_temperature and rejections < max_rejections:
        neighbor = generate_neighbor(subgraph, graph)
        current_objective = objective_function(subgraph, original_graph_and)
        neighbor_objective = objective_function(neighbor, original_graph_and)
        delta_energy = neighbor_objective - current_objective

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            subgraph = neighbor
            current_objective = neighbor_objective

            if current_objective < best_objective:
                best_subgraph = subgraph.copy()
                best_objective = current_objective
                rejections = 0
            else:
                rejections += 1

        temperature *= cooling_rate

        if rejections >= max_rejections / 2:
            cooling_rate *= 0.9  # Reduce cooling rate to explore more
        elif current_objective < best_objective:
            cooling_rate *= 1.1  # Increase cooling rate to converge faster

    # relabel nodes to 0, 1, 2, ...
    mapping = {
        k: v
        for k, v in zip(best_subgraph.nodes(), range(best_subgraph.number_of_nodes()))
    }

    return nx.relabel_nodes(best_subgraph, mapping)


# Generate all possible subgraphs of a given size
def all_possible_subgraphs(graph, subgraph_size):
    subgraphs = []
    for nodes_subset in combinations(graph.nodes, subgraph_size):
        sub_g = graph.subgraph(nodes_subset).copy()
        if nx.is_connected(sub_g):
            subgraphs.append(sub_g)
    return subgraphs


# Generate random subgraphs of a given size
def random_subgraphs(graph, subgraph_size, num_samples):
    subgraphs = []
    all_nodes = list(graph.nodes)
    for _ in range(num_samples):
        nodes_subset = random.sample(all_nodes, subgraph_size)
        subgraphs.append(graph.subgraph(nodes_subset).copy())
    return subgraphs


# Red-QAOA algorithm for reducing the graph size for QAOA
def red_qaoa_exe(graph, and_ratio=0.75):
    num_nodes = graph.number_of_nodes()

    and_base = average_node_degree(graph)

    # Binary search for the minimum node count
    lower = 1
    upper = num_nodes - 1
    best_subgraph = sa_adapt(graph, upper)

    while lower <= upper:
        mid = (lower + upper) // 2
        # Use the sa_adapt function to generate the subgraph with closest average node degree to the original graph
        subgraph = sa_adapt(graph, mid)

        and_sub = average_node_degree(subgraph)
        if (and_sub / and_base) > and_ratio:
            best_subgraph = subgraph
            upper = mid - 1
        else:
            lower = mid + 1

    return best_subgraph
