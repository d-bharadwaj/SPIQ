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
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp

import os
import traceback

import multiprocess as mp
import numpy as np

sys.path.append("../")
import red_qaoa.red_qaoa as red_qaoa
from clapton.circuit_manipulation import multi_angle_qaoa_circuit
import testing_scripts.graphs_utils as graph_utils
from testing_scripts.qaoa_utils import QAOASolver

# Import sklearn for clustering
from sklearn.cluster import KMeans
import math


def compute_gradient_norm(qaoa_object, parameters, verbose=False):
    """
    Compute the gradient norm for given parameters using parameter shift rule.
    Uses EXPLICIT stabilizer simulator for efficient Clifford circuit computation.
    
    Args:
        qaoa_object: QAOASolver object with circuit and Hamiltonian
        parameters: numpy array of circuit parameters
        verbose: if True, print detailed gradient information
    
    Returns:
        float: L2 norm of the gradient vector
    """
    shift = np.pi / 2  # Parameter shift for gradient calculation
    gradients = []
    
    # Create estimator with EXPLICIT stabilizer simulator
    estimator = Estimator(
        options={
            "backend_options": {
                "method": "stabilizer",  # Explicit stabilizer simulator
                "device": "CPU"
            }
        }
    )
    
    if verbose:
        print(f"\n  Computing gradient using stabilizer simulator...")
        print(f"  Parameters shape: {parameters.shape}")
    
    for i in range(len(parameters)):
        # Create shifted parameter arrays
        params_plus = parameters.copy()
        params_plus[i] += shift
        
        params_minus = parameters.copy()
        params_minus[i] -= shift
        
        # Evaluate energies at shifted points using stabilizer simulator
        pub_plus = (qaoa_object.pcirc, qaoa_object.cost_hamiltonian, params_plus)
        pub_minus = (qaoa_object.pcirc, qaoa_object.cost_hamiltonian, params_minus)
        
        job = estimator.run([pub_plus, pub_minus])
        results = job.result()
        
        energy_plus = results[0].data.evs
        energy_minus = results[1].data.evs
        
        # Gradient via parameter shift rule: (E(θ+π/2) - E(θ-π/2)) / 2
        gradient = (energy_plus - energy_minus) / 2.0
        gradients.append(gradient)
    
    grad_norm = np.linalg.norm(gradients)
    
    if verbose:
        print(f"  Gradient vector L2 norm: {grad_norm:.6f}")
    
    return grad_norm


def print_parameter_details(params, label="Parameters", show_all=False):
    """
    Print detailed information about parameters.
    
    Args:
        params: numpy array of parameters
        label: Label for the parameters
        show_all: if True, print all parameter values; if False, print summary
    """
    params = np.array(params)
    
    print(f"\n    {label}:")
    print(f"      Shape: {params.shape}")
    print(f"      Mean: {params.mean():.6f}")
    print(f"      Std: {params.std():.6f}")
    print(f"      Min: {params.min():.6f}")
    print(f"      Max: {params.max():.6f}")
    
    if show_all or len(params) <= 20:
        # Print all values if requested or if small
        print(f"      Values (CAFQA space): {params}")
        # Also show in angle space
        angles = params * (np.pi / 2)
        print(f"      Values (radians): {angles}")
    else:
        # Print first and last few values
        print(f"      First 5 values (CAFQA): {params[:5]}")
        print(f"      Last 5 values (CAFQA): {params[-5:]}")
        angles_first = params[:5] * (np.pi / 2)
        angles_last = params[-5:] * (np.pi / 2)
        print(f"      First 5 values (rad): {angles_first}")
        print(f"      Last 5 values (rad): {angles_last}")


def deduplicate_parameters(parameters, fitness_values, tolerance=1e-10):
    """
    Remove duplicate parameter sets (same parameters → same quantum state).
    
    Args:
        parameters: array of parameter sets
        fitness_values: corresponding energy values
        tolerance: tolerance for considering parameters as identical
    
    Returns:
        tuple: (unique_parameters, unique_fitness, original_indices)
    """
    parameters = np.array(parameters)
    fitness_values = np.array(fitness_values)
    
    unique_params = []
    unique_fitness = []
    original_indices = []
    
    for i, (params, fitness) in enumerate(zip(parameters, fitness_values)):
        # Check if this parameter set is already in unique list
        is_duplicate = False
        for unique_param in unique_params:
            if np.allclose(params, unique_param, atol=tolerance):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_params.append(params)
            unique_fitness.append(fitness)
            original_indices.append(i)
    
    return (np.array(unique_params), 
            np.array(unique_fitness), 
            original_indices)


def select_clustering_stratified_parameters(
    best_cafqa_parameters,
    best_cafqa_fitness_values,
    qaoa_object,
    num_select=5,
    verbose=True
):
    """
    Select parameters using K-means clustering with stratified sampling and gradient filtering.
    
    Strategy:
    1. Deduplicate all parameters
    2. Cluster using K-means (n_clusters = ceil(n_unique / 10))
    3. For each cluster, stratify points into 3 energy layers (low 1/3, mid 1/3, high 1/3)
    4. Sample 2 points from low layer, 1 from mid layer per cluster
    5. Collect all sampled points, compute gradient norms
    6. Filter out gradient=0 points
    7. Select 2 lowest energy points + randomly sample 3 from remaining
    
    Args:
        best_cafqa_parameters: List of parameter arrays from CAFQA
        best_cafqa_fitness_values: Corresponding fitness (energy) values
        qaoa_object: QAOASolver object for gradient computation
        num_select: Final number of points to select (default 5)
        verbose: if True, print detailed information
    
    Returns:
        tuple: (selected_parameters, selected_fitness_values, selected_gradnorms)
    """
    parameters_array = np.array(best_cafqa_parameters)
    fitness_array = np.array(best_cafqa_fitness_values)
    
    print(f"\n{'='*70}")
    print(f"CLUSTERING-BASED STRATIFIED SELECTION WITH GRADIENT FILTERING")
    print(f"{'='*70}")
    print(f"  Total points: {len(fitness_array)}")
    print(f"  Target: {num_select} diverse points")
    
    # Step 1: Deduplicate all parameters
    print(f"\nStep 1: Deduplication...")
    unique_params, unique_fitness, _ = deduplicate_parameters(
        parameters_array, fitness_array
    )
    n_duplicates = len(parameters_array) - len(unique_params)
    print(f"  Removed {n_duplicates} duplicates")
    print(f"  Unique points: {len(unique_params)}")
    
    # Step 2: Determine n_clusters dynamically
    n_clusters = math.ceil(len(unique_params) / 10)
    n_clusters = max(1, min(n_clusters, len(unique_params)))  # At least 1, at most n_unique
    
    print(f"\nStep 2: K-means clustering...")
    print(f"  n_clusters = ceil({len(unique_params)} / 10) = {n_clusters}")
    
    # Handle periodicity: map CAFQA params {0,1,2,3} to unit circle
    X_periodic = np.column_stack([
        np.cos(unique_params * np.pi / 2),
        np.sin(unique_params * np.pi / 2)
    ])
    
    if n_clusters == 1:
        # Only one cluster - all points together
        cluster_labels = np.zeros(len(unique_params), dtype=int)
        print(f"  Using single cluster (all points together)")
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_periodic)
        print(f"  K-means completed with {n_clusters} clusters")
    
    # Step 3: Stratified sampling from each cluster
    print(f"\nStep 3: Stratified sampling from each cluster...")
    sampled_candidates = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_params = unique_params[cluster_mask]
        cluster_fitness = unique_fitness[cluster_mask]
        
        n_cluster_points = len(cluster_params)
        print(f"\n  Cluster {cluster_id + 1}:")
        print(f"    Total points: {n_cluster_points}")
        print(f"    Energy range: [{cluster_fitness.min():.6f}, {cluster_fitness.max():.6f}]")
        
        if n_cluster_points == 0:
            continue
        
        # Sort by energy
        sorted_indices = np.argsort(cluster_fitness)
        sorted_params = cluster_params[sorted_indices]
        sorted_fitness = cluster_fitness[sorted_indices]
        
        # Stratify into 3 layers based on unique energy values
        unique_cluster_energies = np.unique(sorted_fitness)
        n_unique_energies = len(unique_cluster_energies)
        
        # Define layer boundaries
        if n_unique_energies == 1:
            # Only one energy level - sample 3 points from it
            print(f"    Single energy level detected")
            n_sample = min(3, len(sorted_params))
            sample_indices = np.random.choice(len(sorted_params), n_sample, replace=False)
            
            for idx in sample_indices:
                sampled_candidates.append({
                    'params': sorted_params[idx],
                    'energy': sorted_fitness[idx],
                    'cluster': cluster_id
                })
                print(f"    ✓ Sampled: E={sorted_fitness[idx]:.6f}")
        else:
            # Multiple energy levels - stratify
            third = max(1, n_unique_energies // 3)
            low_threshold = unique_cluster_energies[third - 1] if third > 0 else unique_cluster_energies[0]
            mid_threshold = unique_cluster_energies[min(2*third - 1, n_unique_energies - 1)]
            
            layer_low_mask = sorted_fitness <= low_threshold
            layer_mid_mask = (sorted_fitness > low_threshold) & (sorted_fitness <= mid_threshold)
            
            layer_low_indices = np.where(layer_low_mask)[0]
            layer_mid_indices = np.where(layer_mid_mask)[0]
            
            print(f"    Low layer: {len(layer_low_indices)} points (E ≤ {low_threshold:.6f})")
            print(f"    Mid layer: {len(layer_mid_indices)} points ({low_threshold:.6f} < E ≤ {mid_threshold:.6f})")
            
            # Sample 2 from low layer (or all if less than 2)
            if len(layer_low_indices) > 0:
                n_sample_low = min(2, len(layer_low_indices))
                low_sample_indices = np.random.choice(layer_low_indices, n_sample_low, replace=False)
                for idx in low_sample_indices:
                    sampled_candidates.append({
                        'params': sorted_params[idx],
                        'energy': sorted_fitness[idx],
                        'cluster': cluster_id
                    })
                    print(f"    ✓ Sampled from low layer: E={sorted_fitness[idx]:.6f}")
            
            # Sample 1 from mid layer (if exists)
            if len(layer_mid_indices) > 0:
                mid_sample_idx = np.random.choice(layer_mid_indices, 1)[0]
                sampled_candidates.append({
                    'params': sorted_params[mid_sample_idx],
                    'energy': sorted_fitness[mid_sample_idx],
                    'cluster': cluster_id
                })
                print(f"    ✓ Sampled from mid layer: E={sorted_fitness[mid_sample_idx]:.6f}")
            elif len(layer_low_indices) > 2:
                # If no mid layer but have extra low layer points, sample one more from low
                extra_low = [i for i in layer_low_indices if i not in low_sample_indices]
                if len(extra_low) > 0:
                    extra_idx = np.random.choice(extra_low, 1)[0]
                    sampled_candidates.append({
                        'params': sorted_params[extra_idx],
                        'energy': sorted_fitness[extra_idx],
                        'cluster': cluster_id
                    })
                    print(f"    ✓ Additional sample from low layer: E={sorted_fitness[extra_idx]:.6f}")
    
    print(f"\n  Total sampled candidates: {len(sampled_candidates)}")
    
    # Step 4: Compute gradient norms and filter
    print(f"\nStep 4: Computing gradient norms and filtering...")
    
    valid_candidates = []
    for i, candidate in enumerate(sampled_candidates):
        cafqa_params = candidate['params'] * (np.pi / 2)
        grad_norm = compute_gradient_norm(qaoa_object, cafqa_params, verbose=False)
        
        if grad_norm > 1e-10:
            candidate['grad_norm'] = grad_norm
            valid_candidates.append(candidate)
            print(f"  Candidate {i+1} (Cluster {candidate['cluster']+1}): "
                  f"E={candidate['energy']:.6f}, Grad={grad_norm:.6f} ✓")
        else:
            print(f"  Candidate {i+1} (Cluster {candidate['cluster']+1}): "
                  f"E={candidate['energy']:.6f}, Grad≈0 (filtered)")
    
    print(f"\n  Valid candidates after filtering: {len(valid_candidates)}")
    
    if len(valid_candidates) == 0:
        raise ValueError("No valid candidates with non-zero gradients found!")
    
    # Step 5: Select 2 lowest energy + randomly sample 3 from remaining
    print(f"\nStep 5: Final selection (2 lowest energy + 3 random)...")
    
    # Sort by energy
    valid_candidates.sort(key=lambda x: x['energy'])
    
    # Select 2 lowest energy points (or all if less than 2)
    n_lowest = min(2, len(valid_candidates))
    selected = valid_candidates[:n_lowest]
    print(f"  ✓ Selected {n_lowest} lowest energy point(s):")
    for i, point in enumerate(selected):
        print(f"    Point {i+1}: E={point['energy']:.6f}, Grad={point['grad_norm']:.6f} (Cluster {point['cluster']+1})")
    
    # Randomly sample 3 from remaining (or all if less than 3)
    remaining = valid_candidates[n_lowest:]
    if len(remaining) > 0:
        n_random = min(3, len(remaining))
        random_indices = np.random.choice(len(remaining), n_random, replace=False)
        
        print(f"  ✓ Randomly sampled {n_random} additional point(s):")
        for idx in random_indices:
            point = remaining[idx]
            selected.append(point)
            print(f"    Point {len(selected)}: E={point['energy']:.6f}, Grad={point['grad_norm']:.6f} (Cluster {point['cluster']+1})")
    
    # Ensure we have exactly num_select points (or fewer if not enough valid candidates)
    selected = selected[:num_select]
    
    print(f"\n{'='*70}")
    print(f"FINAL SELECTION: {len(selected)} points")
    print(f"{'='*70}")
    
    for i, point in enumerate(selected):
        print(f"\nPoint {i+1} (from Cluster {point['cluster']+1}):")
        print(f"  Energy: {point['energy']:.6f}")
        print(f"  Gradient Norm: {point['grad_norm']:.6f}")
        print_parameter_details(point['params'], f"Selected parameters", 
                              show_all=(len(point['params']) <= 20))
    
    # Extract results
    selected_parameters = np.array([p['params'] for p in selected])
    selected_fitness = np.array([p['energy'] for p in selected])
    selected_gradnorms = np.array([p['grad_norm'] for p in selected])
    
    print(f"\n{'='*70}\n")
    
    return selected_parameters, selected_fitness, selected_gradnorms


def run_qaoa_task_pool(args):

    max_iters = 1 *1e3
    task_id, maxcut_qaoa, initial_params, fitness_val = args

    # QAOA Optimization
    cafqa_params = [param * (np.pi / 2) for param in initial_params]

    try:
        print(f"Starting task: {task_id} for val : {fitness_val}")
        result, obj_values = maxcut_qaoa.run_qaoa(
            initial_params=cafqa_params, max_iters=max_iters, opt="COBYLA"
        )
        print(f"Finished task: {task_id}")

        return {
            "task_name": task_id,
            "result": result,
            "fitness_val": fitness_val,
            "obj_values": obj_values,
            "final_energy": result.fun,
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
    red_qaoa_object,
    selected_cafqa_parameters,
    selected_spaced_fitness_vals,
    selected_clustering_parameters,
    selected_clustering_fitness_vals,
):

    # Original CAFQA initialization (spaced-out selection)
    cafqa_args = [
        (f"CAFQA_Spaced_{i}", ma_qaoa_object, params, fitness)
        for i, (params, fitness) in enumerate(
            zip(selected_cafqa_parameters, selected_spaced_fitness_vals)
        )
    ]

    # Clustering-based stratified selection with gradient filtering
    clustering_cafqa_args = [
        (f"CAFQA_Clustering_{i}", ma_qaoa_object, params, fitness)
        for i, (params, fitness) in enumerate(
            zip(selected_clustering_parameters, selected_clustering_fitness_vals)
        )
    ]

    # Random Init. Params for MA-QAOA
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
        )
    ]

    #Prepare Red-QAOA 

    x0 =  np.random.uniform(
        0, 2 * np.pi, red_qaoa_object.circuit.num_parameters
    )
    res, _ = red_qaoa_object.run_qaoa(x0)
    red_qaoa_angles = res.x / (np.pi / 2)

    red_qaoa_args = [
        (
            "Red_qaoa_task",
            vanilla_qaoa_object,
            red_qaoa_angles,
            selected_spaced_fitness_vals,
        )
    ]


    # Combine all tasks
    all_args = cafqa_args + clustering_cafqa_args + random_args + vanilla_args + red_qaoa_args

    print(f"\nExecuting {len(all_args)} parallel tasks:")
    print(f"  - CAFQA Spaced: {len(cafqa_args)}")
    print(f"  - CAFQA Clustering (with gradient filtering): {len(clustering_cafqa_args)}")
    print(f"  - Random MA-QAOA: {len(random_args)}")
    print(f"  - Vanilla QAOA: {len(vanilla_args)}")
    print(f"  - Red QAOA: {len(red_qaoa_args)}")

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


# Main function
def main():
    # Initialize variables
    n_qubits = int(sys.argv[1])
    reps = int(sys.argv[2])
    n_gens = int(sys.argv[3])
    seed = int(sys.argv[4])
    noise = float(sys.argv[5])

    n = n_qubits
    # k = 3

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate the graph
    G = graph_utils.generate_k_regular_graph(
        num_vertices=n, k=3, weighted=True, seed=seed
    )

    # Build the cost Hamiltonian
    max_cut_paulis = graph_utils.build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)


    # Create the QAOA circuit
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

    # Create QAOA object
    maxcut_qaoa = QAOASolver(cost_hamiltonian, circuit, sim_device="CPU")
    maxcut_qaoa.prepare_circuit()
    maxcut_qaoa.err = noise

    # Evaluate Exact Ground State Energy
    exact_energy = maxcut_qaoa.evaluate_exact_energy()
    # Store and print the exact ground state energy for later use
    maxcut_qaoa.exact_energy = exact_energy
    print(f"\nExact Ground State Energy: {exact_energy:.6f}")

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
    # Strategy 1: Original spaced-out selection WITH GRADIENT COMPUTATION
    # ============================================================
    print("\n" + "="*60)
    print("Strategy 1: Spaced-out selection (Original)")
    print("="*60)
    
    selected_spaced_fitness_vals = list(
        unique_fitness_values[::3][:5]
    )

    selected_fitness_indices = [
        best_cafqa_fitness_values.index(value) for value in selected_spaced_fitness_vals
    ]
    selected_cafqa_parameters = np.array(best_cafqa_parameters)[
        selected_fitness_indices
    ]
    
    print(f"Selected {len(selected_spaced_fitness_vals)} spaced-out parameters with energies:")
    
    # Compute gradient norms for spaced selection
    spaced_gradient_norms = []
    for i, (params, fitness) in enumerate(zip(selected_cafqa_parameters, selected_spaced_fitness_vals)):
        print(f"\nSpaced {i+1}: Energy={fitness:.6f}")
        
        # Compute gradient norm
        cafqa_params = params * (np.pi / 2)
        grad_norm = compute_gradient_norm(maxcut_qaoa, cafqa_params, verbose=False)
        spaced_gradient_norms.append(grad_norm)
        
        print(f"  Gradient Norm: {grad_norm:.6f}")
        print_parameter_details(params, f"Spaced {i+1} parameters", show_all=False)

    # ============================================================
    # Strategy 2: Clustering-based stratified selection with gradient filtering
    # ============================================================
    print("\n" + "="*60)
    print("Strategy 2: Clustering + Gradient Filtering")
    print("="*60)
    
    selected_clustering_parameters, selected_clustering_fitness_vals, clustering_gradient_norms = \
        select_clustering_stratified_parameters(
            best_cafqa_parameters,
            best_cafqa_fitness_values,
            maxcut_qaoa,
            num_select=5,
            verbose=True
        )

    # Vanilla QAOA
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    vanilla_maxcut = QAOASolver(
        cost_hamiltonian, circuit.decompose().decompose(), sim_device="CPU"
    )
    vanilla_maxcut.vanilla = True
    vanilla_maxcut.err = noise


    print("\n" + "="*60)
    print("Starting parallel QAOA optimization tasks")
    print("="*60)


    #Red-QAOA Initialized Points

    # reduced graph (may raise)
    nx_orig = graph_utils.rustworkx_to_networkx(G)
    nx_reduced = red_qaoa.red_qaoa_exe(nx_orig)

    reduced_G = graph_utils.networkx_to_rustworkx(nx_reduced)

    # build cost operators and circuits for reduced and original graphs
    reduced_paulis = graph_utils.build_max_cut_paulis(reduced_G)
    reduced_cost = SparsePauliOp.from_list(reduced_paulis)
    reduced_circ = QAOAAnsatz(cost_operator=reduced_cost, reps=reps)
    red_qaoa_solver = QAOASolver(
        reduced_cost, reduced_circ.decompose().decompose(), sim_device="CPU"
    )
    red_qaoa_solver.vanilla = True
    red_qaoa_solver.err = noise


    # orig_paulis = graph_utils.build_max_cut_paulis(G)
    # orig_cost = SparsePauliOp.from_list(orig_paulis)
    # orig_circ = QAOAAnsatz(cost_operator=orig_cost, reps=reps)
    # eval_solver = QAOASolver(
    #     orig_cost, orig_circ.decompose().decompose(), sim_device="CPU"
    # )
    # eval_solver.vanilla = True

    # x0 = np.random.rand(reps * 2) * np.pi
    # res, _ = red_qaoa_solver.run_qaoa(x0)
    # best_red_angles = res.x

    
    start = timer()
    results = execute_qaoa_tasks(
        maxcut_qaoa,
        vanilla_maxcut,
        red_qaoa_solver,
        selected_cafqa_parameters,
        selected_spaced_fitness_vals,
        selected_clustering_parameters,
        selected_clustering_fitness_vals,
    )
    end = timer()
    print(f"\nTotal parallel execution time: {end - start:.2f} seconds")

    # Add all gradient information to results
    results["Spaced_gradient_norms"] = [float(g) for g in spaced_gradient_norms]
    results["Spaced_fitness_values"] = selected_spaced_fitness_vals
    results["Spaced_selected_parameters"] = selected_cafqa_parameters.tolist()
    
    results["Clustering_gradient_norms"] = clustering_gradient_norms.tolist()
    results["Clustering_fitness_values"] = selected_clustering_fitness_vals.tolist()
    results["Clustering_selected_parameters"] = selected_clustering_parameters.tolist()

    # Save results
    output_dir = f"../np_data/Final_Data_Collection/Multi-Start/Gradient_Norm/Maxcut/noisy/{n_qubits}_qbs"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = f"result_{seed}_clustering_k_means.npy"
    output_file = os.path.join(output_dir, base_name)

    if os.path.exists(output_file):
        stem, ext = os.path.splitext(base_name)
        idx = 1
        while True:
            candidate = os.path.join(output_dir, f"{stem}_{idx}{ext}")
            if not os.path.exists(candidate):
                output_file = candidate
                break
            idx += 1

    np.save(output_file, results)
    print(f"\nResults saved to: {output_file}")
    
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nExact Ground State Energy: {results['Exact_Ground_State_Energy']:.6f}")
    print(f"CAFQA Best Energy: {results['CAFQA_initialization_energy']:.6f}")
    
    print("\nInitial gradient norms by strategy:")
    print(f"\n  Spaced strategy:")
    for i, grad in enumerate(spaced_gradient_norms):
        print(f"    Point {i+1}: {grad:.6f}")
    
    print(f"\n  Clustering + Gradient strategy:")
    for i, grad in enumerate(clustering_gradient_norms):
        print(f"    Point {i+1}: {grad:.6f}")
    
    print("\nFinal energies by initialization method:")
    for task_name, task_result in results["Task_results"].items():
        if "error" not in task_result:
            print(f"  {task_name:30s}: {task_result['final_energy']:.6f}")
        else:
            print(f"  {task_name:30s}: ERROR")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
