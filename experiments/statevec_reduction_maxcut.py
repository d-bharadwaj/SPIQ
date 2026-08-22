"""Reports the factor of reduction for MaxCut QAOA instances."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from scipy.stats import gmean

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from spiq.graphs import (
    build_max_cut_paulis,
    generate_k_regular_graph,
    get_final_distribution,
)
from spiq.qaoa import QAOASolver


def artifact_path(n_qubits: int, seed: int) -> Path:
    return Path("data") / "spiq_maxcut_workflow" / f"{n_qubits}_qbs" / f"results_{seed}.npy"


def load_or_run_spiq_params(n_qubits: int, reps: int, n_gens: int, seed: int, device: str):
    path = artifact_path(n_qubits, seed)
    if path.exists():
        data = np.load(path, allow_pickle=True).item()
        params = data["best_spiq_parameters"]
        return np.asarray(params[0])

    print(f"No artifact at {path}; running SPIQ inline for seed={seed}")
    graph = generate_k_regular_graph(
        num_vertices=n_qubits, k=3, weighted=False, seed=seed
    )
    cost_hamiltonian = SparsePauliOp.from_list(build_max_cut_paulis(graph))
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    solver = QAOASolver(cost_hamiltonian, circuit, sim_device=device)
    solver.err = None
    solver.prepare_circuit()
    solver.run_spiq(n_gens=n_gens)
    best_params = solver.best_spiq_gen_params[::-1]
    return np.asarray(best_params[0])


def compare_spiq_vs_random_distribution(
    n_qubits: int,
    reps: int,
    n_gens: int,
    seed: int,
    device: str,
    shots: float,
):
    np.random.seed(seed)
    spiq_params = load_or_run_spiq_params(n_qubits, reps, n_gens, seed, device)
    spiq_angles = [param * np.pi / 2 for param in spiq_params]

    graph = generate_k_regular_graph(
        num_vertices=n_qubits, k=3, weighted=False, seed=seed
    )
    cost_hamiltonian = SparsePauliOp.from_list(build_max_cut_paulis(graph))
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    solver = QAOASolver(cost_hamiltonian, circuit, sim_device=device)
    solver.prepare_circuit()
    solver._initialize_backend()

    random_angles = np.random.uniform(0, 2 * np.pi, size=len(spiq_angles))
    spiq_dist_len = len(get_final_distribution(solver, spiq_angles, shots=shots))
    random_dist_len = len(get_final_distribution(solver, random_angles, shots=shots))
    factor = random_dist_len / spiq_dist_len
    print(
        f"seed={seed}: SPIQ support={spiq_dist_len}, "
        f"random support={random_dist_len}, factor={factor:.2f}x"
    )
    return factor


def parse_args():
    parser = argparse.ArgumentParser(
        description="MaxCut statevector support-size reduction (SPIQ vs random)"
    )
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--n-gens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--shots", type=float, default=1e8)
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        try:
            results.append(
                compare_spiq_vs_random_distribution(
                    args.n_qubits,
                    args.reps,
                    args.n_gens,
                    seed,
                    args.device,
                    args.shots,
                )
            )
        except Exception as exc:
            print(f"Seed {seed} failed: {exc}")

    if not results:
        print("No valid results to compute geometric mean.")
        return

    geo_mean = gmean(results)
    print(
        f"Geometric mean factor (SPIQ vs random) for {args.n_qubits} qubits "
        f"over {len(results)} seeds: {geo_mean:.2f}x"
    )


if __name__ == "__main__":
    main()
