import numpy as np
import os
import warnings
from scipy.optimize import minimize
from skquant.opt import minimize as skquant_minimize
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
# from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2
from qiskit_aer.noise import  (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)
from qiskit_aer.primitives import EstimatorV2 as Estimator

from clapton.clapton import claptonize
from clapton.circuit_manipulation import (
    transform_to_allowed_gates,
    qiskit_to_stim,
    modify_circuit,
    relax_qaoa_parameters,
    generate_qiskit_param_map,
)
from clapton.depolarization import GateGeneralDepolarizationModel

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def evaluate_energy(circuit, hamiltonian, parameters):
    estimator = Estimator(options={"backend_options": 
                                            {"method": 'statevector', "device": 'CPU'}})
    isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)

    pub = (circuit, isa_hamiltonian, parameters)
    job = estimator.run([pub])

    results = job.result()[0]
    return results.data.evs

def convert_pubo_to_ising(hypergraph: dict, n: int) -> list[tuple[str, float]]:
    """Convert a hypergraph dictionary to a list of Pauli strings with weights.

    Args:
        hypergraph: Dictionary where keys are tuples of node indices and values are weights.
        n: Total number of qubits (nodes).

    Returns:
        List of tuples (Pauli string, weight).
    """
    pauli_list = []

    for edge, weight in hypergraph.items():
        if edge:  # Ensure the edge is not empty
            # Create a Pauli string with "I" for all qubits
            paulis = ["I"] * n
            # Replace "I" with "Z" for qubits in the edge
            for node in edge:
                paulis[node] = "Z"
            # Append the reversed Pauli string and weight to the list
            pauli_list.append(("".join(paulis[::-1]), weight))

    return pauli_list

class QAOASolver:
    def __init__(self, cost_hamiltonian, qaoa_ansatz, sim_device):
        """
        Initialize the QAOA object.

        Args:
            cost_hamiltonian: The cost hamiltonian for the problem.
            qaoa_ansatz: The Qiskit QAOA ansatz for the problem.
        """
        # Attributes for QAOA setup
        self.cost_hamiltonian = cost_hamiltonian
        self.exact_energy = None
        self.circuit = qaoa_ansatz
        self.pcirc = None
        self.stim_circ = None
        self.param_map = None
        self.vanilla = False

        # Attributes for CAFQA results
        self.ks_best = None
        self.energy_best = None

        # Backend and Estimator
        self.backend = None
        self.estimator = None

        #Noise Model 
        self.err = None

        #CPU or GPU simulation 
        self.sim_device = sim_device
        assert self.sim_device, "sim_device must be provided as either 'cpu' or 'gpu'"

    def prepare_circuit(self):
        """
        Prepare the QAOA circuit and relax its parameters.

        Args:
            circuit: The QAOA circuit to be prepared.
        """
        modified_circ = modify_circuit(self.circuit)
        self.pcirc = transform_to_allowed_gates(modified_circ)
        self.pcirc, _, _ = relax_qaoa_parameters(self.pcirc)
        self.stim_circ = qiskit_to_stim(self.pcirc)
        self.param_map = generate_qiskit_param_map(self.pcirc)
        self.stim_circ.define_parameter_map(self.param_map)

    def run_CAFQA(self, n_gens, out_file=None):
        """
        Run the CAFQA initialization.

        Args:
            n_gens: Number of generations for the genetic algorithm.
        """
        paulis, coeffs = self.cost_hamiltonian.paulis.to_labels(), self.cost_hamiltonian.coeffs.real
        reversed_paulis = [p[::-1] for p in paulis]

        self.best_cafqa_gen_params = None
        self.best_cafqa_gen_fitness = None

        if self.err:
                # let's add a noise model where we specify global 1q and 2q gate errors
                nm = GateGeneralDepolarizationModel(p1=self.err, p2=10*self.err)
                self.stim_circ.add_depolarization_model(nm)

        self.ks_best, self.noisy_energy_best, self.energy_best, self.best_cafqa_gen_params, self.best_cafqa_gen_fitness = claptonize(
            reversed_paulis,
            coeffs,
            self.stim_circ,
            n_proc=128,
            n_starts=4,
            n_rounds=1,
            callback=None, #NOTE: usually print
            budget=n_gens // 2,
            out_file = out_file
        )

        print(f"Minimum Energy found with CAFQA initialization: {self.energy_best}")
        self.stim_circ.assign(self.ks_best)

    def evaluate_exact_energy(self):
        """
        Solve the problem classically using the NumPyMinimumEigensolver.

        Returns:
            The exact energy value.
        """
        eigensolver = NumPyMinimumEigensolver()
        exact_solution = eigensolver.compute_minimum_eigenvalue(self.cost_hamiltonian).eigenvalue.real
        print("Exact Energy from Eigensolver:", exact_solution)
        self.exact_energy  = exact_solution
        return exact_solution

    def _create_noise_model(self):
        noise_model = NoiseModel()
        single_qb_error = depolarizing_error(self.err, 1)
        double_qb_error = depolarizing_error(10*self.err, 2)
        noise_model.add_all_qubit_quantum_error(single_qb_error, ["h","rz","s","sx","id"])
        noise_model.add_all_qubit_quantum_error(double_qb_error, ["cx"])
        return noise_model

    def _initialize_backend(self):
        """
        Initialize the quantum backend with or without noise.

        Args:
            noise: Boolean indicating whether to include noise in the simulation.

        Returns:
            A configured quantum backend.
        """

        noise_model=None

        #NOTE: This is not used anywhere
        self.backend = AerSimulator(method='statevector',device=self.sim_device)
        if self.err:
            # noise_model = self._create_noise_model()
            noise_model = NoiseModel.from_backend(FakeMumbaiV2())
            self.backend.set_options(noise_model=noise_model)

            
        #Change to use density matrix simulator for noisy sims. 
        self.estimator = Estimator(options={"backend_options": 
                                            {"method": 'statevector',
                                             "device": self.sim_device,
                                             "noise_model":noise_model}})
        
    def _cost_function(self, params, objective_func_vals):
        """
        Cost function to be minimized.

        Args:
            params: Parameters for the quantum circuit.
            objective_func_vals: List to store the cost function values.

        Returns:
            Cost value.
        """

        if self.vanilla:
            circuit = self.circuit
        else:
            circuit = self.pcirc
        pub = (circuit, self.cost_hamiltonian, params) 
        job = self.estimator.run([pub])
        results = job.result()[0]
        cost = results.data.evs
        objective_func_vals.append(cost)
        return cost

    def  evaluate_energy(self, qiskit_circuit, hamiltonian, parameters):

        if not self.backend:
            self._initialize_backend()

        pub = (qiskit_circuit, hamiltonian, parameters)
        job = self.estimator.run([pub])

        results = job.result()[0]
        return results.data.evs

    def _run_qaoa(self, initial_params, maxiter=1000,opt="COBYLA"):
        """
        Run the QAOA optimization.

        Args:
            initial_params: Initial parameters for the optimization.
            maxiter: Maximum number of iterations for the optimizer.

        Returns:
            Optimization result and the list of objective function values.
        """
        objective_func_vals = []

        def _cost_function_with_vals(x):
            return self._cost_function(x, objective_func_vals)

        if opt == "SPSA":

            # Auto-configure based on problem scale
            lr, pert = SPSA.calibrate(
                _cost_function_with_vals, 
                initial_params,
                target_magnitude=0.01, # Controls step aggressiveness
                c=0.1,
                alpha=0.202,            # Learning rate decay
                gamma=0.101             # Perturbation decay
            )

            spsa = SPSA(maxiter=maxiter,
                        learning_rate=lr,
                        perturbation=pert)

            # spsa = SPSA(maxiter=maxiter, #more accurate
            #             learning_rate=0.001,
            #             perturbation=0.005)

            result = spsa.minimize(_cost_function_with_vals, 
                                   x0=initial_params,
                                   bounds=[(0,2*np.pi)*len(initial_params)])
    
        elif opt in ['imfil', 'snobfit', 'orbit', 'nomad', 'bobyqa']:
            bounds=np.array([[0,2*np.pi]]*len(initial_params), dtype=float)

            # method can be ImFil, SnobFit, Orbit, NOMAD, or Bobyqa
            result, history = \
                skquant_minimize(_cost_function_with_vals, 
                        x0=initial_params, 
                        bounds=bounds, 
                        budget=maxiter,
                        method=opt)

        else:
            result = minimize(
                self._cost_function,
                initial_params,
                args=(objective_func_vals,),
                method=opt,
                tol=1e-3,
                options={'maxiter': maxiter,
                         'initial_tr_radius': 5.0},
                )
        return result, objective_func_vals

    def run_qaoa(self, initial_params, max_iters=1000 ,opt = "COBYLA"):
        """
        Run QAOA with custom initial angles.

        Args:
            initial_params: Initial parameters for the optimization.
            max_iters: Maximum number of iterations for the optimizer.
            noise: Boolean indicating whether to include noise in the simulation.

        Returns:
            Optimization result and the list of objective function values.
        """
        self._initialize_backend()
        result, obj_values = self._run_qaoa(initial_params, max_iters,opt=opt)
        return result, obj_values