import os
import sys
import warnings

import numpy as np
import scipy.stats
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append("../")
import red_qaoa.red_qaoa as red_qaoa

import testing_scripts.graphs_utils as graphs_utils
from testing_scripts.qaoa_utils import QAOASolver

def _restart_stats(values):
    """Return (best_value, geo_mean) for a list/array of positive floats."""
    arr = np.asarray(values)
    if arr.size == 0:
        return None, None
    # protect against non-positive entries for gmean
    arr = np.clip(arr, 1e-12, None)
    # return float(np.min(arr)), float(scipy.stats.gmean(arr))
    return float(np.max(arr)), float(scipy.stats.gmean(arr))


def main():
    # parameters (kept same semantics as original)
    sizes = [8]
    and_ratios = [0.25]
    p = 2
    runs_per_size = 5
    seed_base = 3
    n_restarts = 50 

    graph_type = str(sys.argv[1])

    # out_dir = "../logs/Comprehensive_Proof/Red_QAOA/restart/weighted"
    # os.makedirs(out_dir, exist_ok=True)
    # summary_path = os.path.join(out_dir, f"only_red_qaoa_{graph_type}.log")

    for n_vertices in sizes:
        for and_ratio in and_ratios:

            # accumulate per-run summaries (over restarts) so we can average across runs
            red_best_per_run = []
            red_avg_per_run = []
            cafqa_per_run = []
            for run_idx in range(runs_per_size):
                seed = seed_base + run_idx

                # build the original graph
                if graph_type == "k_reg":
                    k = 3
                    orig_G = graphs_utils.generate_k_regular_graph(
                        num_vertices=n_vertices, weighted=False, seed=seed, k=k
                    )
                elif graph_type == "complete":
                    orig_G = graphs_utils.generate_random_complete_graph(
                        num_vertices=n_vertices, weighted=False, seed=seed
                    )
                elif graph_type == "ego":
                    orig_G = graphs_utils.generate_random_ego_graph(
                        num_vertices=n_vertices, weighted=False, seed=seed
                    )
                else:
                    raise ValueError(f"Graph type '{graph_type}' not available.")

                # reduced graph (may raise)
                nx_orig = graphs_utils.rustworkx_to_networkx(orig_G)
                
                try:
                    nx_reduced = red_qaoa.red_qaoa_exe(nx_orig,and_ratio)
                except Exception as e:
                    print(f"n={n_vertices} run={run_idx+1}: reduction failed: {e}")
                    continue
                reduced_G = graphs_utils.networkx_to_rustworkx(nx_reduced)

                # build cost operators and circuits for reduced and original graphs
                reduced_paulis = graphs_utils.build_max_cut_paulis(reduced_G)
                reduced_cost = SparsePauliOp.from_list(reduced_paulis)
                reduced_circ = QAOAAnsatz(cost_operator=reduced_cost, reps=p)
                reduced_solver = QAOASolver(
                    reduced_cost, reduced_circ.decompose().decompose(), sim_device="CPU"
                )
                reduced_solver.vanilla = True

                orig_paulis = graphs_utils.build_max_cut_paulis(orig_G)
                orig_cost = SparsePauliOp.from_list(orig_paulis)
                orig_circ = QAOAAnsatz(cost_operator=orig_cost, reps=p)
                eval_solver = QAOASolver(
                    orig_cost, orig_circ.decompose().decompose(), sim_device="CPU"
                )
                eval_solver.vanilla = True

                exact_energy = eval_solver.evaluate_exact_energy()

                # perform several restarts; collect ratio = red_energy / exact_energy per restart
                per_restart_ratios = []
                for _ in range(n_restarts):
                    try:
                        x0 = np.random.rand(p * 2) * np.pi
                        res, _ = reduced_solver.run_qaoa(x0, max_iters=10**3)
                        theta = res.x
                    except Exception as e:
                        print(f"n={n_vertices} run={run_idx+1}: optimization failed: {e}")
                        continue

                    try:
                        red_energy = eval_solver.evaluate_energy(
                            eval_solver.circuit, eval_solver.cost_hamiltonian, theta
                        )
                        if red_energy > 0:
                            # skip pathological positive energies, treat as tiny positive
                            red_energy = 1e-5
                        # exact_energy = eval_solver.evaluate_exact_energy()
                        if exact_energy == 0:
                            # skip this restart if exact energy is zero
                            continue
                        per_restart_ratios.append(abs(red_energy / exact_energy))
                    except Exception as e:
                        print(f"n={n_vertices} run={run_idx+1}: evaluation failed: {e}")
                        continue

                # if we collected no successful restarts, skip this run
                if not per_restart_ratios:
                    print(f"n={n_vertices} run={run_idx+1}: no successful restarts, skipping run")
                    continue

                # compute best and (geometric) mean across restarts for this run
                run_best, run_gmean = _restart_stats(per_restart_ratios)
                red_best_per_run.append(run_best)
                red_avg_per_run.append(run_gmean)

                # run CAFQA once per run (as before)
                try:
                    cafqa_solver = QAOASolver(orig_cost, orig_circ, sim_device="CPU")
                    cafqa_solver.prepare_circuit()
                    cafqa_solver.run_cafqa(n_gens=1, out_file=None)
                    cafqa_energy = cafqa_solver.energy_best
                    if cafqa_solver.evaluate_exact_energy() != 0:
                        cafqa_per_run.append(abs(cafqa_energy / cafqa_solver.evaluate_exact_energy()))
                except Exception as e:
                    print(f"n={n_vertices} run={run_idx+1}: CAFQA failed: {e}")
                    # continue without appending

                # after all runs for this size: average the per-run statistics across runs
                def _mean_or_none(arr):
                    return float(np.mean(arr)) if arr else None

            avg_of_best = _mean_or_none(red_best_per_run)
            avg_of_means = _mean_or_none(red_avg_per_run)
            avg_cafqa = _mean_or_none(cafqa_per_run)
            std_of_best = float(np.std(red_best_per_run)) if red_best_per_run else None
            std_of_means = float(np.std(red_avg_per_run)) if red_avg_per_run else None
            std_cafqa = float(np.std(cafqa_per_run)) if cafqa_per_run else None

            # embed the and_ratio into the avg_of_best string so the existing write/print lines include it
            # avg_of_best = f"{avg_of_best} (and_ratio={and_ratio})"
            # with open(summary_path, "a") as fh:
            #     fh.write(
            #         f"n={n_vertices}: avg_best_over_runs={avg_of_best}, std_best={std_of_best}, "
            #         f"avg_mean_over_runs={avg_of_means}, std_mean={std_of_means}, "
            #         f"avg_cafqa_over_runs={avg_cafqa}, std_cafqa={std_cafqa}\n"
            #     )

            print(
                f"n={n_vertices} done: avg_best_over_runs={avg_of_best}, std_best={std_of_best}, "
                f"avg_mean_over_runs={avg_of_means}, std_mean={std_of_means}, "
                f"avg_cafqa_over_runs={avg_cafqa}, std_cafqa={std_cafqa}"
            )

if __name__ == "__main__":
    main()
