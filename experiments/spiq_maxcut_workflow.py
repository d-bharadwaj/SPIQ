"""End-to-end SPIQ MaxCut workflow: create problem instance → SPIQ → point selection."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from spiq.graphs import (
    build_max_cut_paulis,
    generate_k_regular_graph,
    generate_random_complete_graph,
    generate_random_ego_graph,
)
from spiq.qaoa import QAOASolver
from spiq.selection import fixed_interval_selection, k_gaps_selection


def build_graph(n_qubits: int, graph: str, seed: int):
    if graph == "k_reg":
        return generate_k_regular_graph(
            num_vertices=n_qubits, k=3, weighted=False, seed=seed
        )
    if graph == "complete":
        return generate_random_complete_graph(
            num_vertices=n_qubits, weighted=False, seed=seed
        )
    if graph == "ego":
        return generate_random_ego_graph(
            num_vertices=n_qubits, weighted=False, seed=seed
        )
    raise ValueError(f"Unknown graph type: {graph}")


def parse_args():
    parser = argparse.ArgumentParser(description="SPIQ MaxCut end-to-end workflow")
    parser.add_argument("--n-qubits", type=int, required=True, help="Problem size (vertices)")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--n-gens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-select", type=int, default=3)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Depolarizing error rate; 0 disables noise",
    )
    parser.add_argument(
        "--graph",
        choices=["k_reg", "complete", "ego"],
        default="k_reg",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    graph = build_graph(args.n_qubits, args.graph, args.seed)
    cost_hamiltonian = SparsePauliOp.from_list(build_max_cut_paulis(graph))
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=args.reps)

    solver = QAOASolver(cost_hamiltonian, circuit, sim_device=args.device)
    solver.err = args.noise if args.noise > 0 else None
    solver.prepare_circuit()
    exact_energy = solver.evaluate_exact_energy()

    print(f"qubits={args.n_qubits}, reps={args.reps}, params={solver.pcirc.num_parameters}")
    print(f"exact energy={exact_energy:.6f}")

    solver.run_spiq(n_gens=args.n_gens)
    best_params = solver.best_spiq_gen_params[::-1]
    best_fitness = solver.best_spiq_gen_fitness[::-1]
    print(f"SPIQ best energy={solver.energy_best:.6f}")
    print(f"candidate points={len(best_fitness)}")

    spaced_params, spaced_fitness = fixed_interval_selection(
        best_params, best_fitness, num_select=args.num_select
    )
    for i, (params, energy) in enumerate(zip(spaced_params, spaced_fitness)):
        print(f"fixed interval {i + 1}: energy={energy:.6f}, params={params}")

    kgaps_result = k_gaps_selection(
        best_params,
        best_fitness,
        solver,
        num_select=args.num_select,
        rng=np.random.default_rng(args.seed),
    )
    if kgaps_result is None:
        print("k-gaps selection returned no points")
        kgaps_params = kgaps_fitness = kgaps_grads = None
    else:
        kgaps_params, kgaps_fitness, kgaps_grads = kgaps_result
        for i, (params, energy, grad) in enumerate(
            zip(kgaps_params, kgaps_fitness, kgaps_grads)
        ):
            print(
                f"k-gaps {i + 1}: energy={energy:.6f}, grad_norm={grad:.6f}, params={params}"
            )

    output_dir = Path("data") / Path(__file__).stem / f"{args.n_qubits}_qbs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "n_qubits": args.n_qubits,
        "reps": args.reps,
        "n_gens": args.n_gens,
        "seed": args.seed,
        "graph": args.graph,
        "best_spiq_parameters": best_params,
        "best_spiq_fitness": best_fitness,
        "energy_best": solver.energy_best,
        "exact_energy": exact_energy,
        "spaced_params": spaced_params,
        "spaced_fitness": spaced_fitness,
        "kgaps_params": kgaps_params,
        "kgaps_fitness": kgaps_fitness,
        "kgaps_grads": kgaps_grads,
    }
    out_path = output_dir / f"results_{args.seed}.npy"
    np.save(out_path, results)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
