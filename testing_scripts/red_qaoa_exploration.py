import os
import sys
import warnings

import numpy as np
import scipy.stats
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import red_qaoa.red_qaoa as red_qaoa

import spiq.graphs as graphs_utils
from spiq.qaoa import QAOASolver
import pickle

def main():
    # parameters (kept same semantics as original)
    sizes = [12]
    p = 2
    runs_per_size = 5
    seed_base = 3
    n_restarts = 50 

    graph_type = str(sys.argv[1])

    for n_vertices in sizes:
    # accumulate per-run summaries (over restarts) so we can average across runs
        seed = seed_base 

        # build the original graph
        if graph_type == "k_reg":
            k = 3
            orig_G = graphs_utils.generate_k_regular_graph(
                num_vertices=n_vertices, weighted=True, seed=seed, k=k
            )
        elif graph_type == "complete":
            orig_G = graphs_utils.generate_random_complete_graph(
                num_vertices=n_vertices, weighted=True, seed=seed
            )
        elif graph_type == "ego":
            orig_G = graphs_utils.generate_random_ego_graph(
                num_vertices=n_vertices, weighted=True, seed=seed
            )
        else:
            raise ValueError(f"Graph type '{graph_type}' not available.")

        # reduced graph (may raise)
        nx_orig = graphs_utils.rustworkx_to_networkx(orig_G)
        
        try:
            nx_reduced = red_qaoa.red_qaoa_exe(nx_orig)
        except Exception as e:
            print(f"n={n_vertices}: reduction failed: {e}")
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
        best_accuracy_fitness = 0
        best_accuracy_parameters = []
        for _ in range(n_restarts):
            try:
                x0 = np.random.rand(p * 2) * np.pi
                res, _ = reduced_solver.run_qaoa(x0)
                best_red_angles = res.x
            except Exception as e:
                print(f"n={n_vertices} optimization failed: {e}")
                continue

            try:
                red_energy = eval_solver.evaluate_energy(
                    eval_solver.circuit, eval_solver.cost_hamiltonian, best_red_angles
                )
                if red_energy > 0:
                    # skip pathological positive energies, treat as tiny positive
                    red_energy = 1e-5
                elif red_energy == 0:
                    # skip this restart if exact energy is zero
                    continue
                current_accuracy = abs(red_energy / exact_energy)
                if current_accuracy > best_accuracy_fitness:
                    best_accuracy_fitness = current_accuracy
                    best_accuracy_parameters = best_red_angles

            except Exception as e:
                print(f"n={n_vertices} evaluation failed: {e}")
                continue
                
        # if we collected no successful restarts, skip this run
        if len(best_accuracy_parameters) == 0 :
            raise RuntimeError(f"no successful restarts, aborting")

        #Full QAOA run with red-qaoa points

        res,obj_values = eval_solver.run_qaoa(best_red_angles)
        results = {
            "res": res,
            "obj_values": obj_values,
            "n_vertices": n_vertices,
            'ground_energy': exact_energy,
            "seed": seed,
            "graph_type": graph_type,
        }
        output_dir = f"../np_data/Final_Data_Collection/Multi-Start/Red_QAOA_full_runs/Maxcut/{n_vertices}_qbs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"result_seed_{seed}.npy")
        np.save(output_file, results)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
