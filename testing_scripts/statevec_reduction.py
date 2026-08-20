
import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy.stats import gmean

import warnings

import spiq.graphs as graph_utils
from spiq.knapsack import generate_knapsack_instance
from spiq.qaoa import QAOASolver

warnings.filterwarnings("ignore", category=DeprecationWarning)


def compare_cafqa_vs_random_distribution(num_qbs, seed):
    sv_data = np.load(
        f"../np_data/Comprehensive_Proof/Knapsack/2_reps/{num_qbs}_qbs/results_{seed}.npy",
        allow_pickle=True,
    ).item()
    sv_params = sv_data["best_cafqa_parameters"][0]
    cafqa_angles = [param * np.pi / 2 for param in sv_params]

    # Knapsack
    prob = generate_knapsack_instance(num_items=num_qbs, seed=seed)
    qp = prob.to_quadratic_program()
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)
    op, offset = qubo.to_ising()
    cost_hamiltonian = op

    # # k-reg Maxcut
    # k = 3  # for k-regular graphs
    # G = graph_utils.generate_k_regular_graph(
    #     num_vertices=num_qbs, weighted=True, seed=seed, k=k
    # )
    # max_cut_paulis = graph_utils.build_max_cut_paulis(G)
    # cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    qaoa_obj = QAOASolver(cost_hamiltonian, circuit, sim_device="CPU")
    qaoa_obj.prepare_circuit()
    # Need this next line to set backends
    qaoa_obj._initialize_backend()
    random_angles = np.random.uniform(0, 2 * np.pi, size=len(cafqa_angles))
    cafqa_dist_len = len(
        graph_utils.get_final_distribution(qaoa_obj, cafqa_angles, shots=1e8)
    )
    random_dist_len = len(
        graph_utils.get_final_distribution(qaoa_obj, random_angles, shots=1e8)
    )
    print(
        f"Factor of improvement (CAFQA vs Random): {random_dist_len / cafqa_dist_len:.2f}x"
    )
    return random_dist_len / cafqa_dist_len


def geomean_cafqa_vs_random_distribution(num_qbs, num_seeds=5):
    print("starting sampling for all qubits.....")
    results = []
    for seed in range(1, num_seeds + 1):
        try:
            res = compare_cafqa_vs_random_distribution(num_qbs, seed)
            results.append(res)
        except Exception as e:
            print(f"Seed {seed} failed: {e}")

    if results:
        geo_mean = gmean(results)
        print(
            f"Geometric mean factor of improvement (CAFQA vs Random) for {num_qbs} qbs over {len(results)} seeds: {geo_mean:.2f}x"
        )
        return geo_mean
    else:
        print("No valid results to compute geometric mean.")
        return None


# Results
for num_qbs in [16]:
    geomean_cafqa_vs_random_distribution(num_qbs)
