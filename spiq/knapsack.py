import random

from qiskit_optimization.applications import Knapsack


def generate_knapsack_instance(num_items=10, weight_range=(1, 5), seed=None):
    if seed is not None:
        random.seed(seed)
    weights = [random.randint(*weight_range) for _ in range(num_items)]
    values = [max(1, w + random.randint(-2, 2)) for w in weights]
    # max_weight = sum(weights) // 2  # Set max_weight as half of total weight sum
    max_weight = max_weight = 15 * (num_items // 4)
    return Knapsack(values=values, weights=weights, max_weight=max_weight)
