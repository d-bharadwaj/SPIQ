import os
import sys
import warnings

import numpy as np
import rustworkx as rx

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from timeit import default_timer as timer

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

sys.path.append("../")
import os
import traceback

import multiprocess as mp
import numpy as np
from clapton.circuit_manipulation import multi_angle_qaoa_circuit

import testing_scripts.graphs_utils as graph_utils
from testing_scripts.qaoa_utils import QAOASolver


def compute_gradient_norm(qaoa_object, parameters):
    """
    Compute the gradient norm for given parameters using parameter shift rule.
    For Clifford angles (multiples of pi/2), this is highly efficient.
    
    Args:
        qaoa_object: QAOASolver object with circuit and Hamiltonian
        parameters: numpy array of circuit parameters
    
    Returns:
        float: L2 norm of the gradient vector
    """
    shift = np.pi / 2  # Parameter shift for gradient calculation
    gradients = []
    
    for i in range(len(parameters)):
        # Create shifted parameter arrays
        params_plus = parameters.copy()
        params_plus[i] += shift
        
        params_minus = parameters.copy()
        params_minus[i] -= shift
        
        # Evaluate energies at shifted points
        energy_plus = qaoa_object.evaluate_energy(
            qaoa_object.pcirc, 
            qaoa_object.cost_hamiltonian, 
            params_plus
        )
        energy_minus = qaoa_object.evaluate_energy(
            qaoa_object.pcirc, 
            qaoa_object.cost_hamiltonian, 
            params_minus
        )
        
        # Gradient via parameter shift rule: (E(θ+π/2) - E(θ-π/2)) / 2
        gradient = (energy_plus - energy_minus) / 2.0
        gradients.append(gradient)
    
    # Return L2 norm of gradient vector
    return np.linalg.norm(gradients)


def select_gradient_aware_parameters(
    best_cafqa_parameters, 
    best_cafqa_fitness_values, 
    qaoa_object,
    top_percent=0.1,
    num_select=5
):
    """
    Select parameters that balance low energy and high gradient norm.
    
    Strategy:
    1. Select top X% of parameters by energy
    2. Compute gradient norm for each (efficient for Clifford angles)
    3. Exclude points with zero gradient norm (local minima)
    4. Score remaining points balancing energy and gradient norm
    5. Select top 5 points
    
    Args:
        best_cafqa_parameters: List of parameter arrays from CAFQA
        best_cafqa_fitness_values: Corresponding fitness (energy) values
        qaoa_object: QAOASolver object for gradient computation
        top_percent: Percentage of best solutions to consider (default 0.1 = 10%)
        num_select: Number of final parameters to select (default 5)
    
    Returns:
        tuple: (selected_parameters, selected_fitness_values, selected_gradnorms)
    """
    # Convert to numpy arrays for easier manipulation
    parameters_array = np.array(best_cafqa_parameters)
    fitness_array = np.array(best_cafqa_fitness_values)
    
    # Step 1: Select top X% by energy (lower is better)
    n_top = max(int(len(fitness_array) * top_percent), num_select)
    top_indices = np.argsort(fitness_array)[:n_top]
    
    top_parameters = parameters_array[top_indices]
    top_fitness = fitness_array[top_indices]
    
    print(f"\nGradient-aware selection: Evaluating top {n_top} solutions ({top_percent*100:.1f}%)")
    
    # Step 2 & 3: Compute gradient norms and filter out zero gradients
    gradient_norms = []
    valid_indices = []
    
    for i, params in enumerate(top_parameters):
        # Convert CAFQA parameters to actual angles (multiply by pi/2)
        cafqa_params = [param * (np.pi / 2) for param in params]
        
        # Compute gradient norm
        grad_norm = compute_gradient_norm(qaoa_object, cafqa_params)
        
        # Only keep non-zero gradient points
        if grad_norm > 1e-10:  # Threshold for numerical zero
            gradient_norms.append(grad_norm)
            valid_indices.append(i)
    
    gradient_norms = np.array(gradient_norms)
    valid_parameters = top_parameters[valid_indices]
    valid_fitness = top_fitness[valid_indices]
    
    print(f"Found {len(valid_indices)} points with non-zero gradients")
    
    if len(valid_indices) < num_select:
        print(f"Warning: Only {len(valid_indices)} valid points found, less than requested {num_select}")
        num_select = len(valid_indices)
    
    # Step 4: Score points balancing energy and gradient norm
    # Normalize both metrics to [0, 1] range
    normalized_energy = (valid_fitness - valid_fitness.min()) / (valid_fitness.max() - valid_fitness.min() + 1e-10)
    normalized_gradnorm = (gradient_norms - gradient_norms.min()) / (gradient_norms.max() - gradient_norms.min() + 1e-10)
    
    # Composite score: prefer low energy AND high gradient
    # Weight energy more heavily (0.7) since we want good solutions
    # But also value high gradients (0.3) to avoid flat regions
    alpha = 0.7  # Weight for energy (lower is better)
    beta = 0.3   # Weight for gradient norm (higher is better)
    
    composite_scores = alpha * normalized_energy - beta * normalized_gradnorm
    
    # Step 5: Select top points by composite score
    selected_indices = np.argsort(composite_scores)[:num_select]
    
    selected_parameters = valid_parameters[selected_indices]
    selected_fitness = valid_fitness[selected_indices]
    selected_gradnorms = gradient_norms[selected_indices]
    
    print(f"Selected {num_select} parameters:")
    for i, (fitness, gradnorm) in enumerate(zip(selected_fitness, selected_gradnorms)):
        print(f"  Point {i+1}: Energy={fitness:.6f}, GradNorm={gradnorm:.6f}")
    
    return selected_parameters, selected_fitness, selected_gradnorms


def run_qaoa_task_pool(args):

    max_iters = 10 * 1e3
    task_id, maxcut_qaoa, initial_params, fitness_val, graph = args

    # QAOA Optimization
    cafqa_params = [param * (np.pi / 2) for param in initial_params]

    try:
        print(f"Starting task: {task_id} for val : {fitness_val}")
        result, obj_values = maxcut_qaoa.run_qaoa(
            initial_params=cafqa_params, max_iters=max_iters, opt="COBYLA"
        )
        print(f"Finished task: {task_id}")

        # Approximation Ratio
        distribution = graph_utils.get_final_distribution(maxcut_qaoa, result.x)
        approx_ratio = graph_utils.calculate_approximation_ratio(distribution, graph)

        return {
            "task_name": task_id,
            "result": result,
            "fitness_val": fitness_val,
            "obj_values": obj_values,
            "final_energy": result.fun,
            "approx_ratio": approx_ratio,
        }

    except Exception as e:
        print(f"Error in task {task_id}: {e}")
        traceback.print_exc()
        return {
            "task_name": task_id,
            "error": str(e),
        }


# Function to execute in parallel using Pool
def execute_qaoa_tasks(
    ma_qaoa_object,
    vanilla_qaoa_object,
    selected_cafqa_parameters,
    selected_spaced_fitness_vals,
    selected_gradient_parameters,
    selected_gradient_fitness_vals,
    graph,
):

    # Original CAFQA initialization (spaced-out selection)
    cafqa_args = [
        (f"CAFQA_Spaced_{i}", ma_qaoa_object, params, fitness, graph)
        for i, (params, fitness) in enumerate(
            zip(selected_cafqa_parameters, selected_spaced_fitness_vals)
        )
    ]

    # New gradient-aware CAFQA initialization
    gradient_cafqa_args = [
        (f"CAFQA_Gradient_{i}", ma_qaoa_object, params, fitness, graph)
        for i, (params, fitness) in enumerate(
            zip(selected_gradient_parameters, selected_gradient_fitness_vals)
        )
    ]

    # Random Init. Params for MA-QAOA
    # Generate 100 random parameter sets and evaluate their energies, then select the median
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

    # Prepare task for Vanilla QAOA
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

    # Combine all tasks (now including gradient-aware initialization)
    all_args = cafqa_args + gradient_cafqa_args + random_args + vanilla_args

    print(f"\nExecuting {len(all_args)} parallel tasks:")
    print(f"  - CAFQA Spaced: {len(cafqa_args)}")
    print(f"  - CAFQA Gradient-aware: {len(gradient_cafqa_args)}")
    print(f"  - Random MA-QAOA: {len(random_args)}")
    print(f"  - Vanilla QAOA: {len(vanilla_args)}")

    with mp.Pool(processes=len(all_args)) as pool:
        results_list = pool.map(run_qaoa_task_pool, all_args)

    results = {res["task_name"]: res for res in results_list}

    # Extract objective values for each task_id
    task_objective_values = {
        task_id: res.get("obj_values", None) for task_id, res in results.items()
    }

    # Return results
    return {
        "CAFQA_initialization_energy": ma_qaoa_object.energy_best,
        "Exact_Ground_State_Energy": ma_qaoa_object.exact_energy,
        "Task_results": results,
        "Task_objective_values": task_objective_values,
    }


# Main function to set up and execute the script
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

    n = n_qubits
    k = 3  # for k-regular graphs

    # Generate the graph
    G = graph_utils.generate_random_complete_graph(
        num_vertices=n, weighted=True, seed=seed
    )

    # Build the cost Hamiltonian
    max_cut_paulis = graph_utils.build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    # Create the QAOA circuit
    circuit = multi_angle_qaoa_circuit(n, G, reps)

    # Create QAOA object
    maxcut_qaoa = QAOASolver(cost_hamiltonian, circuit, sim_device="CPU")
    maxcut_qaoa.prepare_circuit()
    maxcut_qaoa.err = noise

    # Evaluate Exact Ground State Energy
    maxcut_qaoa.evaluate_exact_energy()

    # Run CAFQA process
    start_cafqa = timer()
    maxcut_qaoa.run_cafqa(n_gens=n_gens)
    end_cafqa = timer()
    print(f"CAFQA optimization time: {end_cafqa - start_cafqa} seconds")
    print(f"{n} Qubits and {reps} reps")
    print(f"Minimum Energy found with CAFQA initialization: {maxcut_qaoa.energy_best}")

    # Best Solutions
    best_cafqa_fitness_values = maxcut_qaoa.best_cafqa_gen_fitness[::-1]
    best_cafqa_parameters = maxcut_qaoa.best_cafqa_gen_params[::-1]

    unique_fitness_values = np.unique(best_cafqa_fitness_values)

    # ============================================================
    # Strategy 1: Original spaced-out selection
    # ============================================================
    print("\n" + "="*60)
    print("Strategy 1: Spaced-out selection (Original)")
    print("="*60)
    
    selected_spaced_fitness_vals = list(
        unique_fitness_values[::3][:5]
    )  # Select 5 of the best unique spaced-out solutions.

    selected_fitness_indices = [
        best_cafqa_fitness_values.index(value) for value in selected_spaced_fitness_vals
    ]
    selected_cafqa_parameters = np.array(best_cafqa_parameters)[
        selected_fitness_indices
    ]
    
    print(f"Selected {len(selected_spaced_fitness_vals)} spaced-out parameters with energies:")
    for i, fitness in enumerate(selected_spaced_fitness_vals):
        print(f"  Spaced {i+1}: {fitness:.6f}")

    # ============================================================
    # Strategy 2: NEW - Gradient-aware selection
    # ============================================================
    print("\n" + "="*60)
    print("Strategy 2: Gradient-aware selection (NEW)")
    print("="*60)
    
    selected_gradient_parameters, selected_gradient_fitness_vals, gradient_norms = \
        select_gradient_aware_parameters(
            best_cafqa_parameters,
            best_cafqa_fitness_values,
            maxcut_qaoa,
            top_percent=0.1,  # Consider top 10% of solutions
            num_select=5
        )

    # Vanilla QAOA
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    vanilla_maxcut = QAOASolver(
        cost_hamiltonian, circuit.decompose().decompose(), sim_device="CPU"
    )
    vanilla_maxcut.vanilla = True

    print("\n" + "="*60)
    print("Starting parallel QAOA optimization tasks")
    print("="*60)
    
    start = timer()
    results = execute_qaoa_tasks(
        maxcut_qaoa,
        vanilla_maxcut,
        selected_cafqa_parameters,
        selected_spaced_fitness_vals,
        selected_gradient_parameters,
        selected_gradient_fitness_vals,
        G,
    )
    end = timer()
    print(f"\nTotal parallel execution time: {end - start:.2f} seconds")

    # Add gradient information to results
    results["Gradient_norms"] = gradient_norms.tolist()
    results["Gradient_fitness_values"] = selected_gradient_fitness_vals.tolist()

    # Save results
    output_dir = f"../np_data/Final_Data_Collection/Maxcut/Complete_Graphs/Less_Reps/{n_qubits}_qbs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"result_{seed}_with_gradient.npy")
    np.save(output_file, results)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary comparison
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nExact Ground State Energy: {results['Exact_Ground_State_Energy']:.6f}")
    print(f"CAFQA Best Energy: {results['CAFQA_initialization_energy']:.6f}")
    
    print("\nFinal energies by initialization method:")
    for task_name, task_result in results["Task_results"].items():
        if "error" not in task_result:
            print(f"  {task_name:30s}: {task_result['final_energy']:.6f} (approx_ratio: {task_result['approx_ratio']:.4f})")
        else:
            print(f"  {task_name:30s}: ERROR")


# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
