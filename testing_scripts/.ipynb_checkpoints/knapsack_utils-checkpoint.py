import random

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_optimization.applications import Knapsack
from scipy.optimize import minimize


def generate_knapsack_instance(num_items=10, value_range=(1, 20), weight_range=(1, 15), seed=None):
    if seed is not None:
        random.seed(seed)
    values = [random.randint(*value_range) for _ in range(num_items)]
    weights = [random.randint(*weight_range) for _ in range(num_items)]
    max_weight = sum(weights) // 2  # Set max_weight as half of total weight sum
    return Knapsack(values=values, weights=weights, max_weight=max_weight)


def evaluate_knapsack(circuit,params,cost_hamiltonian,maxiter):

    objective_func_vals= []

    def cost_func_estimator(params, ansatz, hamiltonian, estimator,objective_func_vals):

        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost = results.data.evs
        
        objective_func_vals.append(cost)
        return cost
    
    estimator = Estimator(mode=AerSimulator(method='statevector'))
    result = minimize(
        cost_func_estimator,
        params,
        args=(circuit, cost_hamiltonian, estimator,objective_func_vals),
        method="COBYLA",
        tol=1e-4,
        options={'maxiter': maxiter}
    )
    return objective_func_vals,result.fun


