"""End-to-end SPIQ Knapsack workflow: create problem instance → SPIQ → point selection."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.converters import QuadraticProgramToQubo

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from spiq.knapsack import generate_knapsack_instance
from spiq.qaoa import QAOASolver
from spiq.selection import fixed_interval_selection, k_gaps_selection


def build_knapsack_hamiltonian(n_items: int, seed: int):
    prob = generate_knapsack_instance(num_items=n_items, seed=seed)
    qp = prob.to_quadratic_program()
    qubo = QuadraticProgramToQubo().convert(qp)
    cost_hamiltonian, offset = qubo.to_ising()
    return cost_hamiltonian, offset, qp


def parse_args():
    parser = argparse.ArgumentParser(description="SPIQ Knapsack end-to-end workflow")
    parser.add_argument("--n-items", type=int, required=True, help="Problem size (items)")
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
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    cost_hamiltonian, offset, qp = build_knapsack_hamiltonian(args.n_items, args.seed)
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=args.reps)

    solver = QAOASolver(cost_hamiltonian, circuit, sim_device=args.device)
    solver.err = args.noise if args.noise > 0 else None
    solver.prepare_circuit()
    exact_energy = solver.evaluate_exact_energy()

    print(qp.prettyprint())
    print(
        f"items={args.n_items}, qubits={cost_hamiltonian.num_qubits}, "
        f"offset={offset:.6f}, reps={args.reps}"
    )
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

    output_dir = Path("data") / Path(__file__).stem / f"{args.n_items}_items"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "n_items": args.n_items,
        "n_qubits": cost_hamiltonian.num_qubits,
        "reps": args.reps,
        "n_gens": args.n_gens,
        "seed": args.seed,
        "offset": offset,
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
