import numpy as np
from scipy.stats import gmean

from qiskit_optimization.converters import QuadraticProgramToQubo
import sys
from qiskit.circuit.library import QAOAAnsatz

sys.path.append("../")
from testing_scripts.qaoa_utils import QAOASolver
import testing_scripts.graphs_utils as graph_utils
from testing_scripts.knapsack_utils import generate_knapsack_instance 


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def compare_cafqa_vs_random_distribution(num_qbs, seed):
    sv_data = np.load(f'../np_data/Comprehensive_Proof/Knapsack/2_reps/{num_qbs}_qbs/results_{seed}.npy', allow_pickle=True).item()
    sv_params = sv_data['best_cafqa_parameters'][0]
    cafqa_angles = [param * np.pi / 2 for param in sv_params]
    prob = generate_knapsack_instance(num_items=num_qbs, seed=seed)
    qp = prob.to_quadratic_program()
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)
    op, offset = qubo.to_ising()
    cost_hamiltonian = op
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    qaoa_obj = QAOASolver(cost_hamiltonian, circuit, sim_device="CPU")
    qaoa_obj.prepare_circuit()
    #Need this next line to set backends
    qaoa_obj._initialize_backend()
    random_angles = np.random.uniform(0, 2 * np.pi, size=len(cafqa_angles))
    cafqa_dist_len = len(graph_utils.get_final_distribution(qaoa_obj, cafqa_angles,shots=1e8))
    random_dist_len = len(graph_utils.get_final_distribution(qaoa_obj, random_angles,shots=1e8))
    print(f"Factor of improvement (CAFQA vs Random): {random_dist_len / cafqa_dist_len:.2f}x")
    return random_dist_len/cafqa_dist_len

def geomean_cafqa_vs_random_distribution(num_qbs, num_seeds=8):
    results = []
    for seed in range(num_seeds):
        try:
            res = compare_cafqa_vs_random_distribution(num_qbs, seed)
            results.append(res)
        except Exception as e:
            print(f"Seed {seed} failed: {e}")

    if results: 
        geo_mean = gmean(results)
        print(f"Geometric mean factor of improvement (CAFQA vs Random) over {len(results)} seeds: {geo_mean:.2f}x")
        return geo_mean
    else:
        print("No valid results to compute geometric mean.")
        return None

# Results
for num_qbs in [4, 9, 12, 16]:
    geomean_cafqa_vs_random_distribution(num_qbs)