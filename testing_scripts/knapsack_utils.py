import random 
from qiskit_optimization.applications import Knapsack

def generate_knapsack_instance(num_items=10, value_range=(1, 20), weight_range=(1, 15), seed=None):
    if seed is not None:
        random.seed(seed)
    values = [random.randint(*value_range) for _ in range(num_items)]
    weights = [random.randint(*weight_range) for _ in range(num_items)]
    max_weight = sum(weights) // 2  # Set max_weight as half of total weight sum
    return Knapsack(values=values, weights=weights, max_weight=max_weight)