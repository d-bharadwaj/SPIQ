import rustworkx as rx
import random

def generate_k_regular_graph(num_vertices, k, weighted=False, seed=False):
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
        
def generate_random_complete_graph(num_vertices, edge_prob=0.5, weighted=False, seed=False,save_path=None):
    G = rx.PyGraph()
    G.add_nodes_from(range(num_vertices))
    
    if seed:
        random.seed(0)
    
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            weight = random.randint(1, 10) if weighted else 1
            G.add_edge(i, j, weight)
    return G