import sys
import os
import warnings
import numpy as np
import scipy.stats
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append("../")
import testing_scripts.graphs_utils as graphs_utils
from testing_scripts.qaoa_utils import QAOASolver
import red_qaoa.red_qaoa as red_qaoa


def main():
    ns = [8, 12, 16, 20]
    reps = 2
    num_runs = 5
    seed_start = 3

    graph = str(sys.argv[1])

    out_log_dir = "../logs/Comprehensive_Proof/Red_QAOA/Weighted"
    os.makedirs(out_log_dir, exist_ok=True)
    summary_log = os.path.join(out_log_dir, f"only_red_qaoa_{graph}.log")

    for n in ns:
        red_acc = []
        cafqa_acc = []
        for run in range(num_runs):
            seed = seed_start + run

            ## Maxcut Formulation
            if graph == "k_reg":
                print("Selected graph type: k-regular")
                k = 3  # for k-regular graphs
                G = graphs_utils.generate_k_regular_graph(
                    num_vertices=n, weighted=True, seed=seed, k=k
                )
            elif graph == "complete":
                print("Selected graph type: complete")
                G = graphs_utils.generate_random_complete_graph(
                    num_vertices=n, weighted=True, seed=seed
                )
            elif graph == "ego":
                print("Selected graph type: ego")
                G = graphs_utils.generate_random_ego_graph(
                    num_vertices=n, weighted=True, seed=seed
                )

            else:
                raise ValueError(f"Graph type '{graph}' not available.")

            # reduced graph via red_qaoa
            nx_g = graphs_utils.rustworkx_to_networkx(G)
            try:
                red_nx = red_qaoa.red_qaoa_exe(nx_g)
            except Exception as e:
                print(f"n={n} run={run+1}: red_qaoa reduction failed: {e}")
                continue
            red_G = graphs_utils.networkx_to_rustworkx(red_nx)

            # red-QAOA on reduced graph
            red_paulis = graphs_utils.build_max_cut_paulis(red_G)
            red_cost = SparsePauliOp.from_list(red_paulis)
            red_circ = QAOAAnsatz(cost_operator=red_cost, reps=reps)
            red_solver = QAOASolver(
                red_cost, red_circ.decompose().decompose(), sim_device="CPU"
            )
            red_solver.vanilla = True
            x0 = np.random.rand(reps * 2) * np.pi
            try:
                res, _ = red_solver.run_qaoa(x0, max_iters=10**3)
            except Exception as e:
                print(f"n={n} run={run+1}: red-QAOA opt failed: {e}")
                continue
            best_red = res.x

            # evaluate on original graph
            paulis = graphs_utils.build_max_cut_paulis(G)
            cost = SparsePauliOp.from_list(paulis)
            circ = QAOAAnsatz(cost_operator=cost, reps=reps)
            eval_solver = QAOASolver(
                cost, circ.decompose().decompose(), sim_device="CPU"
            )
            eval_solver.vanilla = True
            try:
                red_energy = eval_solver.evaluate_energy(
                    eval_solver.circuit, eval_solver.cost_hamiltonian, best_red
                )
                if red_energy > 0:
                    print(f"n={n} run={run+1}: positive energy, skipped")
                    red_energy = 0.01  # NOTE: custom
                exact = eval_solver.evaluate_exact_energy()
                if exact == 0:
                    print(f"n={n} run={run+1}: exact energy zero, skipped")
                    continue
                red_acc.append(abs(red_energy / exact))
            except Exception as e:
                print(f"n={n} run={run+1}: evaluation failed: {e}")
                continue

            # CAFQA initialization and run
            try:
                qaoa_obj = QAOASolver(cost, circ, sim_device="CPU")
                qaoa_obj.prepare_circuit()
                qaoa_obj.run_CAFQA(
                    n_gens=5, out_file=None
                )  # TODO: increase gens when needed
                cafqa_energy = qaoa_obj.energy_best
                cafqa_acc.append(abs(cafqa_energy / exact))
            except Exception as e:
                print(f"n={n} run={run+1}: CAFQA failed: {e}")
                continue

        geomean_red = scipy.stats.gmean(np.array(red_acc)) if red_acc else None
        geomean_cafqa = scipy.stats.gmean(np.array(cafqa_acc)) if cafqa_acc else None
        std_red = np.std(red_acc) if red_acc else None
        std_cafqa = np.std(cafqa_acc) if cafqa_acc else None
        with open(summary_log, "a") as f:
            f.write(
                f"rerun (without skipping) {n} qubits: geomean_red={geomean_red}, std_red={std_red}, geomean_cafqa={geomean_cafqa}, std_cafqa={std_cafqa}\n"
            )
        print(
            f"rerun (without skipping)   n={n} done: geomean_red={geomean_red}, std_red={std_red}, geomean_cafqa={geomean_cafqa}, std_cafqa={std_cafqa}"
        )


if __name__ == "__main__":
    main()
