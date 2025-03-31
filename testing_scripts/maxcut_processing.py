from scipy.optimize import minimize
import numpy as np
from typing import Sequence
import rustworkx as rx

from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2


def evaluate_maxcut(G,circuit,params,cost_hamiltonian,maxiter,noise=False):

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
    
    if noise:
        noise_model = NoiseModel()
        noisy_backend = FakeMumbaiV2() # Your quantum backend
        noise_model = NoiseModel.from_backend(noisy_backend) 
        backend = AerSimulator(method='statevector',noise_model=noise_model)
    else: 
        backend = AerSimulator(method='statevector')

    estimator = Estimator(mode=backend)
    result = minimize(
        cost_func_estimator,
        params,
        args=(circuit, cost_hamiltonian, estimator,objective_func_vals),
        method="COBYLA",
        tol=1e-4,
        options={'maxiter': maxiter}
    )

    optimized_circuit = circuit.assign_parameters(result.x)
    optimized_circuit.measure_all()

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 10000

    pub= (optimized_circuit, )
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val/shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

    # auxiliary functions to sample most likely bitstring
    def to_bitstring(integer, num_bits):
        result = np.binary_repr(integer, width=num_bits)
        return [int(digit) for digit in result]

    keys = list(final_distribution_int.keys())
    values = list(final_distribution_int.values())
    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_bitstring = to_bitstring(most_likely, len(G))
    most_likely_bitstring.reverse()

    print("Result bitstring:", most_likely_bitstring)

    def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
        assert len(x) == len(list(graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
        return sum( weight * (x[u] * (1 - x[v]) + x[v] * (1 - x[u])) for u,v,weight in list(graph.weighted_edge_list()))

    cut_value= evaluate_sample(most_likely_bitstring, G)

    return cut_value,objective_func_vals,result.fun

