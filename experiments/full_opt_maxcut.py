"""Full MaxCut optimization: SPIQ → point selection → parallel COBYLA multi-start.
"""

from __future__ import annotations

import argparse
import traceback
import warnings
from pathlib import Path
from timeit import default_timer as timer

import multiprocess as mp
import numpy as np
from clapton.circuit_manipulation import multi_angle_qaoa_circuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from red_qaoa import red_qaoa_exe
from spiq.graphs import (
    build_max_cut_paulis,
    calculate_approximation_ratio,
    generate_random_complete_graph,
    get_final_distribution,
    networkx_to_rustworkx,
    rustworkx_to_networkx,
)
from spiq.qaoa import QAOASolver
from spiq.selection import fixed_interval_selection


def run_qaoa_task_pool(args):
    max_iters = int(10 * 1e3)
    task_id, maxcut_qaoa, initial_params, fitness_val, graph = args

    try:
        print(f"Starting task: {task_id} for val : {fitness_val}")
        result, obj_values = maxcut_qaoa.run_qaoa(
            initial_params=initial_params, max_iters=max_iters, opt="COBYLA"
        )
        print(f"Finished task: {task_id}")

        distribution = get_final_distribution(maxcut_qaoa, result.x)
        approx_ratio = calculate_approximation_ratio(distribution, graph)

        return {
            "task_name": task_id,
            "result": result,
            "fitness_val": fitness_val,
            "obj_values": obj_values,
            "final_energy": result.fun,
            "approx_ratio": approx_ratio,
        }
    except Exception as exc:
        print(f"Error in task {task_id}: {exc}")
        traceback.print_exc()
        return {"task_name": task_id, "error": str(exc)}


def find_red_qaoa_init(graph, reps, device, and_ratio, n_restarts, seed):
    """Reduce the graph with Red-QAOA and pick the best transferrable init."""
    np.random.seed(seed)
    nx_orig = rustworkx_to_networkx(graph)
    nx_reduced = red_qaoa_exe(nx_orig, and_ratio=and_ratio)
    reduced_g = networkx_to_rustworkx(nx_reduced)

    reduced_cost = SparsePauliOp.from_list(build_max_cut_paulis(reduced_g))
    reduced_circ = QAOAAnsatz(cost_operator=reduced_cost, reps=reps)
    reduced_solver = QAOASolver(
        reduced_cost, reduced_circ.decompose().decompose(), sim_device=device
    )
    reduced_solver.vanilla = True

    orig_cost = SparsePauliOp.from_list(build_max_cut_paulis(graph))
    orig_circ = QAOAAnsatz(cost_operator=orig_cost, reps=reps)
    eval_solver = QAOASolver(
        orig_cost, orig_circ.decompose().decompose(), sim_device=device
    )
    eval_solver.vanilla = True
    exact_energy = eval_solver.evaluate_exact_energy()

    best_accuracy = -np.inf
    best_angles = None
    for _ in range(n_restarts):
        try:
            x0 = np.random.rand(reps * 2) * np.pi
            res, _ = reduced_solver.run_qaoa(x0, max_iters=10**3)
            theta = res.x
            red_energy = eval_solver.evaluate_energy(
                eval_solver.circuit, eval_solver.cost_hamiltonian, theta
            )
            if red_energy > 0:
                red_energy = 1e-5
            if exact_energy == 0 or red_energy == 0:
                continue
            accuracy = abs(red_energy / exact_energy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_angles = theta
        except Exception as exc:
            print(f"Red-QAOA restart failed: {exc}")
            continue

    if best_angles is None:
        raise RuntimeError("Red-QAOA found no successful restarts")

    print(
        f"Red-QAOA init: reduced_nodes={reduced_g.num_nodes()}, "
        f"best_transfer_accuracy={best_accuracy:.6f}"
    )
    return np.asarray(best_angles), best_accuracy, reduced_g.num_nodes()


def execute_qaoa_tasks(
    ma_qaoa_object,
    vanilla_qaoa_object,
    selected_spiq_parameters,
    selected_spaced_fitness_vals,
    graph,
    red_qaoa_angles=None,
    red_qaoa_fitness=None,
):
    spiq_args = [
        (
            f"task_{i}",
            ma_qaoa_object,
            [p * (np.pi / 2) for p in params],
            fitness,
            graph,
        )
        for i, (params, fitness) in enumerate(
            zip(selected_spiq_parameters, selected_spaced_fitness_vals)
        )
    ]

    random_angles_arr = np.random.uniform(
        0, 2 * np.pi, (100, ma_qaoa_object.pcirc.num_parameters)
    )
    random_energies = [
        ma_qaoa_object.evaluate_energy(
            ma_qaoa_object.pcirc, ma_qaoa_object.cost_hamiltonian, params
        )
        for params in random_angles_arr
    ]
    median_idx = np.argsort(random_energies)[len(random_energies) // 2]
    median_angles = random_angles_arr[median_idx]
    random_args = [
        (
            "Random_MA-QAOA",
            ma_qaoa_object,
            median_angles,
            selected_spaced_fitness_vals,
            graph,
        )
    ]

    random_vanilla_angles = np.random.uniform(
        0, 2 * np.pi, vanilla_qaoa_object.circuit.num_parameters
    )
    vanilla_args = [
        (
            "Vanilla_task",
            vanilla_qaoa_object,
            random_vanilla_angles,
            selected_spaced_fitness_vals,
            graph,
        )
    ]

    all_args = spiq_args + random_args + vanilla_args
    if red_qaoa_angles is not None:
        all_args.append(
            (
                "Red_QAOA",
                vanilla_qaoa_object,
                red_qaoa_angles,
                red_qaoa_fitness,
                graph,
            )
        )

    with mp.Pool(processes=len(all_args)) as pool:
        results_list = pool.map(run_qaoa_task_pool, all_args)

    results = {res["task_name"]: res for res in results_list}
    task_objective_values = {
        task_id: res.get("obj_values", None) for task_id, res in results.items()
    }

    return {
        "SPIQ_initialization_energy": ma_qaoa_object.energy_best,
        "Exact_Ground_State_Energy": ma_qaoa_object.exact_energy,
        "Task_results": results,
        "Task_objective_values": task_objective_values,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full MaxCut SPIQ + multi-start COBYLA optimization"
    )
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--n-gens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-select", type=int, default=5)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Depolarizing error rate; 0 disables noise",
    )
    parser.add_argument(
        "--and-ratio",
        type=float,
        default=0.75,
        help="Red-QAOA average-node-degree retention ratio",
    )
    parser.add_argument(
        "--red-restarts",
        type=int,
        default=50,
        help="Random restarts on the Red-QAOA reduced graph",
    )
    parser.add_argument(
        "--skip-red-qaoa",
        action="store_true",
        help="Skip the Red-QAOA comparison arm",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    n = args.n_qubits

    graph = generate_random_complete_graph(
        num_vertices=n, weighted=True, seed=args.seed
    )
    cost_hamiltonian = SparsePauliOp.from_list(build_max_cut_paulis(graph))
    ma_circuit = multi_angle_qaoa_circuit(n, graph, args.reps)

    maxcut_qaoa = QAOASolver(cost_hamiltonian, ma_circuit, sim_device=args.device)
    maxcut_qaoa.err = args.noise if args.noise > 0 else None
    maxcut_qaoa.prepare_circuit()
    maxcut_qaoa.evaluate_exact_energy()

    start_spiq = timer()
    maxcut_qaoa.run_spiq(n_gens=args.n_gens)
    end_spiq = timer()
    print(f"SPIQ optimization time: {end_spiq - start_spiq} seconds")
    print(f"{n} Qubits and {args.reps} reps")
    print(
        f"Minimum Energy found with SPIQ initialization: {maxcut_qaoa.energy_best}"
    )

    best_spiq_parameters = maxcut_qaoa.best_spiq_gen_params[::-1]
    best_spiq_fitness = maxcut_qaoa.best_spiq_gen_fitness[::-1]
    selected_spiq_parameters, selected_spaced_fitness_vals = fixed_interval_selection(
        best_spiq_parameters,
        best_spiq_fitness,
        num_select=args.num_select,
        stride=args.stride,
    )

    vanilla_circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=args.reps)
    vanilla_maxcut = QAOASolver(
        cost_hamiltonian,
        vanilla_circuit.decompose().decompose(),
        sim_device=args.device,
    )
    vanilla_maxcut.vanilla = True
    vanilla_maxcut.err = args.noise if args.noise > 0 else None

    red_angles = None
    red_fitness = None
    red_meta = None
    if not args.skip_red_qaoa:
        start_red = timer()
        try:
            red_angles, red_fitness, reduced_nodes = find_red_qaoa_init(
                graph,
                args.reps,
                args.device,
                args.and_ratio,
                args.red_restarts,
                args.seed,
            )
            red_meta = {
                "and_ratio": args.and_ratio,
                "n_restarts": args.red_restarts,
                "reduced_nodes": reduced_nodes,
                "transfer_accuracy": red_fitness,
            }
        except Exception as exc:
            print(f"Red-QAOA comparison skipped due to error: {exc}")
            traceback.print_exc()
        print(f"Red-QAOA init search time: {timer() - start_red} seconds")

    start = timer()
    results = execute_qaoa_tasks(
        maxcut_qaoa,
        vanilla_maxcut,
        selected_spiq_parameters,
        selected_spaced_fitness_vals,
        graph,
        red_qaoa_angles=red_angles,
        red_qaoa_fitness=red_fitness,
    )
    if red_meta is not None:
        results["Red_QAOA_meta"] = red_meta
    end = timer()
    print(f"Total Time : {end - start}")

    output_dir = Path("data") / Path(__file__).stem / f"{n}_qbs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"result_{args.seed}.npy"
    np.save(out_path, results)
    print(f"saved {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
