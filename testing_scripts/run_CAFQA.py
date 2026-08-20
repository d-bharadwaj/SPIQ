import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import multiprocess as mp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.converters import QuadraticProgramToQubo

from spiq.knapsack import generate_knapsack_instance
from spiq.qaoa import QAOASolver

print("Command-line arguments:", sys.argv)


def main():
    # Initialize variables (replace with your actual initialization logic)
    n_qubits = int(sys.argv[1])
    reps = int(sys.argv[2])
    n_gens = int(sys.argv[3])
    mutation_prob = tuple(map(float, sys.argv[4].split()))
    elitism = int(sys.argv[5])
    crossover_type = str(sys.argv[6])
    seed = int(sys.argv[7])
    noise = bool(int(sys.argv[8]))

    print(f"Num. qubits: {n_qubits}, Num reps: {reps}")

    ## Maxcut Formulation
    # k = 3  # for k-regular graphs
    # # Generate the graph
    # G = graph_utils.generate_random_complete_graph(num_vertices=n, weighted=True,seed=seed)
    # # Build the cost Hamiltonian
    # max_cut_paulis = graph_utils.build_max_cut_paulis(G)
    # cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    ## Knapsack Formulation
    n_items = n_qubits
    prob = generate_knapsack_instance(num_items=n_items, seed=seed)
    qp = prob.to_quadratic_program()
    print(qp.prettyprint())
    # intermediate QUBO form of the optimization problem
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)
    # qubit Hamiltonian and offset
    op, offset = qubo.to_ising()
    print(f"num qubits: {op.num_qubits}, offset: {offset}\n")
    cost_hamiltonian = op

    # CAFQA Workflow
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    qaoa_obj = QAOASolver(cost_hamiltonian, circuit, sim_device="CPU")

    qaoa_obj.prepare_circuit()
    results_dir = f"../np_data/Comprehensive_Proof/Knapsack/{reps}_reps/{n_qubits}_qbs"
    os.makedirs(results_dir, exist_ok=True)  # Creates folder if it doesn't exist
    results_file = os.path.join(
        results_dir, f"results_{seed}"
    )  # This is a file path, not a directory

    # Run CAFQA process
    qaoa_obj.run_spiq(n_gens=n_gens, out_file=results_file)

    best_cafqa_params = qaoa_obj.best_cafqa_gen_params[::-1]
    best_cafqa_fitness_values = qaoa_obj.best_cafqa_gen_fitness[::-1]

    unique_fitness_values = np.unique(best_cafqa_fitness_values)

    print("Best 20 best CAFQA fitness values:", unique_fitness_values[:20])

    if n_qubits <= 20:
        exact_energy = qaoa_obj.evaluate_exact_energy()
        print("Exact energy:", exact_energy)
    else:
        exact_energy = None

    results_dict = {
        "best_cafqa_parameters": best_cafqa_params,
        "best_cafqa_fitness_values": best_cafqa_fitness_values,
        "CAFQA_initialization_energy": qaoa_obj.energy_best,
        "exact_energy": exact_energy,
    }

    # Save the dictionary using numpy
    np.save(results_file, results_dict)


# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
