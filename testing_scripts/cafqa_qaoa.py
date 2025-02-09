import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import random

from qiskit import transpile
from qiskit.circuit import Parameter,ParameterExpression
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeManilaV2,FakeNairobiV2,FakeMelbourneV2,FakeGuadalupeV2

import sys
sys.path.append("../")
from clapton.clapton import claptonize
from clapton.circuit_manipulation import transform_to_allowed_gates,qiskit_to_stim, modify_circuit, multi_angle_qaoa_circuit

# Get arguments from command line
n_qubits = int(sys.argv[1])  # First argument: No. of qubits
n_reps = int(sys.argv[2])         # Second argument: Reps in ansatz

# Print to verify
print(f"Input file: {n_qubits}")
print(f"Tag: {n_reps}")

def generate_random_graph(num_vertices, edge_prob=0.5, weighted=False, save_path=None):
    G = rx.PyGraph()
    G.add_nodes_from(range(num_vertices))
    
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            # if random.random() < edge_prob:
            weight = random.randint(1, 10) if weighted else 1
            G.add_edge(i, j, weight)
    return G

n = n_qubits
G = generate_random_graph(num_vertices=n)

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

max_cut_paulis = build_max_cut_paulis(G)

cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
print("Cost Function Hamiltonian:", cost_hamiltonian)
paulis,coeffs = cost_hamiltonian.paulis.to_labels(),cost_hamiltonian.coeffs.real


reps = n_reps
gamma_params = [Parameter(f'gamma_{i}_{j}_{r}') for r in range(reps) for i, j in G.edge_list()]
beta_params = [Parameter(f'beta_{i}_{r}') for r in range(reps) for i in G.node_indexes()]
circuit = multi_angle_qaoa_circuit(gamma_params,beta_params,n,G ,reps)

# Transform qiskit circ. to stim.
modified_circ = modify_circuit(circuit)
pcirc = transform_to_allowed_gates(modified_circ)
stim_circ = qiskit_to_stim(pcirc)

def qiskit_params_map(circ):
    dag = circuit_to_dag(circ)
    param_list = [list(node.op.params[0].parameters)[0].name for node in dag.op_nodes() if node.op.params and isinstance(node.op.params[0], ParameterExpression)]
    return {k: v for v, k in enumerate(param_list)}

qiskit_param_map = qiskit_params_map(pcirc)

# Ensure the sorted names are correct
ordered_params = [param.name for param in pcirc.parameters]
assert sorted(qiskit_param_map.keys()) == ordered_params

param_map = {qiskit_param_map[param]: i for i, param in enumerate(ordered_params)}

stim_circ.define_parameter_map(param_map)

# CAFQA Process

ks_best, _, energy_best = claptonize(
    paulis,
    coeffs,
    stim_circ,
    n_proc=4,           # total number of processes in parallel
    n_starts=4,         # number of random genetic algorithm starts in parallel
    n_rounds=1,         # number of budget rounds, if None it will terminate itself
    callback=print,     # callback for internal parameter (#iteration, energies, ks) processing
    budget=20           # budget per genetic algorithm instance
)


print(f"{n} Qubits and {reps} reps")
print(f"Minimum Energy found with CAFQA initalization: {energy_best}")

# Random Initalization 

def cafqa_params_energy(circuit, hamiltonian, parameters):
    estimator = Estimator(mode=AerSimulator(method='statevector'))
    isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)

    pub = (circuit, isa_hamiltonian, parameters)
    job = estimator.run([pub])

    results = job.result()[0]
    return results.data.evs

random_params = np.random.random(len(ks_best))
energy = cafqa_params_energy(pcirc, cost_hamiltonian, random_params)
print(f"Minimum Energy found with Random initalization: {energy}")

# Solve with classical Eigensolver for comparison
eigensolver = NumPyMinimumEigensolver()
exact_solution = eigensolver.compute_minimum_eigenvalue(cost_hamiltonian).eigenvalue.real
print("Exact Energy from Eigensolver:", exact_solution)