import rustworkx as rx
import numpy as np
import warnings
import os
import sys
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.converters import QuadraticProgramToQubo
from timeit import default_timer as timer

sys.path.append("../")
from testing_scripts.qaoa_utils import QAOASolver
from testing_scripts.knapsack_utils import generate_knapsack_instance


import multiprocess as mp
import traceback
import os
import numpy as np

def run_qaoa_task_pool(args):

    max_iters = 10*1e+3
    task_id, knapsack_qaoa, initial_params, fitness_val = args

    # QAOA Optimization
    cafqa_params = [param * (np.pi / 2) for param in initial_params]

    try:
        print(f"Starting task: {task_id} for val : {fitness_val}")
        result, obj_values = knapsack_qaoa.run_qaoa(initial_params=cafqa_params, max_iters=max_iters, opt="COBYLA")
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
def execute_qaoa_tasks(ma_qaoa_object, vanilla_qaoa_object, selected_cafqa_parameters, selected_fitness_vals):

    cafqa_args = [
        (f"task_{i}", ma_qaoa_object, params, fitness)
        for i, (params, fitness) in enumerate(zip(selected_cafqa_parameters, selected_fitness_vals))
    ]

    # Random Init. Params for MA-QAOA 
    random_angles = np.random.random(ma_qaoa_object.pcirc.num_parameters)
    random_args = [("Random_MA-QAOA",ma_qaoa_object,random_angles,selected_fitness_vals)]

    # Prepare task for Vanilla QAOA
    random_vanilla_angles = np.random.random(vanilla_qaoa_object.circuit.num_parameters)
    vanilla_args = [("Vanilla_task", vanilla_qaoa_object, random_vanilla_angles, selected_fitness_vals)]

    # Combine all tasks
    all_args = cafqa_args + random_args + vanilla_args 

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
        "Exact_Ground_State_Energy":ma_qaoa_object.exact_energy,
        "Task_results": results,
        "Task_objective_values": task_objective_values,
    }

# Main function to set up and execute the script
def main():
    # Initialize variables (replace with your actual initialization logic)
    n_items = int(sys.argv[1])
    reps = int(sys.argv[2])     
    n_gens = int(sys.argv[3])        
    mutation_prob = tuple(map(float, sys.argv[4].split()))
    elitism = int(sys.argv[5])
    crossover_type = str(sys.argv[6])   
    seed = int(sys.argv[7])
    noise = bool(int(sys.argv[8]))

    # Knapsack Formulation 
    prob = generate_knapsack_instance(num_items=n_items,seed=seed)

    qp = prob.to_quadratic_program()
    print(qp.prettyprint())

    # intermediate QUBO form of the optimization problem
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)

    # qubit Hamiltonian and offset
    op, offset = qubo.to_ising()
    print(f"num qubits: {op.num_qubits}, offset: {offset}\n")

    cost_hamiltonian = op
    # Transform qiskit circ. to stim.
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

    knapsack_qaoa = QAOASolver(cost_hamiltonian, circuit)
    knapsack_qaoa.prepare_circuit()

    #Evaluate Exact Ground State Energy 
    knapsack_qaoa.evaluate_exact_energy()

    # Run CAFQA process
    knapsack_qaoa.run_CAFQA(n_gens=n_gens)
    print(f"{op.num_qubits} Qubits and {reps} reps")
    print(f"Minimum Energy found with CAFQA initialization: {knapsack_qaoa.energy_best}")

    # Best Solutions
    best_cafqa_fitness_values = knapsack_qaoa.best_cafqa_gen_fitness[::-1]
    best_cafqa_parameters = knapsack_qaoa.best_cafqa_gen_params[::-1]

    unique_fitness_values = np.unique(best_cafqa_fitness_values)

    # # When choosing best out of unique energies
    selected_fitness_vals = list(unique_fitness_values[::3][:5]) #Select 5 of the best unique spaced-out solutions.
    # selected_fitness_vals = list(unique_fitness_values[:5]) #Select 5 of the best unique solutions.

    selected_fitness_indices = [best_cafqa_fitness_values.index(value) for value in selected_fitness_vals]
    selected_cafqa_parameters = np.array(best_cafqa_parameters)[selected_fitness_indices]
    
    # #When choosing just best 5 (can be same energies) 
    # selected_fitness_vals = list(best_cafqa_fitness_values)[:5] #Select 5 of the best solutions (can be repeated).
    # selected_cafqa_parameters = np.array(best_cafqa_parameters)[:5]
    
    # Vanilla QAOA
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    vanilla_knapsack = QAOASolver(cost_hamiltonian,circuit.decompose().decompose())
    vanilla_knapsack.vanilla = True

    start = timer()
    results = execute_qaoa_tasks(knapsack_qaoa, vanilla_knapsack, selected_cafqa_parameters, selected_fitness_vals)
    end = timer()
    print(f"Total Time : {end - start}")

    # Save results
    output_dir = f"../np_data/Final_Data_Collection/Knapsack/{op.num_qubits}_qbs"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"result_{seed}.npy"), results)

# Entry point
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()