import numpy as np
import os
import warnings
from scipy.optimize import minimize
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2

from clapton.clapton import claptonize
from clapton.circuit_manipulation import (
    transform_to_allowed_gates,
    qiskit_to_stim,
    modify_circuit,
    relax_qaoa_parameters,
    generate_qiskit_param_map,
)

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def evaluate_energy(circuit, hamiltonian, parameters):
    estimator = Estimator(mode=AerSimulator(method='statevector'))
    isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)

    pub = (circuit, isa_hamiltonian, parameters)
    job = estimator.run([pub])

    results = job.result()[0]
    return results.data.evs

class QAOASolver:
    def __init__(self, cost_hamiltonian, qaoa_ansatz):
        """
        Initialize the QAOA object.

        Args:
            cost_hamiltonian: The cost hamiltonian for the problem.
            qaoa_ansatz: The Qiskit QAOA ansatz for the problem.
        """
        # Attributes for QAOA setup
        self.cost_hamiltonian = cost_hamiltonian
        self.circuit = qaoa_ansatz
        self.pcirc = None
        self.stim_circ = None
        self.param_map = None

        # Attributes for CAFQA results
        self.ks_best = None
        self.energy_best = None

        # Backend and Estimator
        self.backend = None
        self.estimator = None

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

    def run_CAFQA(self, n_gens):
        """
        Run the CAFQA initialization.

        Args:
            n_gens: Number of generations for the genetic algorithm.
        """
        paulis, coeffs = self.cost_hamiltonian.paulis.to_labels(), self.cost_hamiltonian.coeffs.real
        reversed_paulis = [p[::-1] for p in paulis]

        self.ks_best, _, self.energy_best = claptonize(
            reversed_paulis,
            coeffs,
            self.stim_circ,
            n_proc=4,
            n_starts=4,
            n_rounds=1,
            callback=print,
            budget=n_gens // 2,
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
        return exact_solution

    def _initialize_backend(self, noise=False):
        """
        Initialize the quantum backend with or without noise.

        Args:
            noise: Boolean indicating whether to include noise in the simulation.

        Returns:
            A configured quantum backend.
        """
        self.backend = AerSimulator(method='statevector')
        if noise:
            noisy_backend = FakeMumbaiV2()
            noise_model = NoiseModel.from_backend(noisy_backend)
            self.backend.set_options(noise_model=noise_model)

        self.estimator = Estimator(mode=self.backend)

    def _cost_function(self, params, objective_func_vals):
        """
        Cost function to be minimized.

        Args:
            params: Parameters for the quantum circuit.
            objective_func_vals: List to store the cost function values.

        Returns:
            Cost value.
        """
        isa_hamiltonian = self.cost_hamiltonian.apply_layout(self.pcirc.layout)
        pub = (self.pcirc, isa_hamiltonian, params)
        job = self.estimator.run([pub])
        results = job.result()[0]
        cost = results.data.evs
        objective_func_vals.append(cost)
        return cost

    def _run_qaoa(self, initial_params, maxiter=1000):
        """
        Run the QAOA optimization.

        Args:
            initial_params: Initial parameters for the optimization.
            maxiter: Maximum number of iterations for the optimizer.

        Returns:
            Optimization result and the list of objective function values.
        """
        objective_func_vals = []
        result = minimize(
            self._cost_function,
            initial_params,
            args=(objective_func_vals,),
            method="COBYLA",
            tol=1e-4,
            options={'maxiter': maxiter},
        )
        return result, objective_func_vals

    def run_qaoa(self, initial_params, max_iters=1000, noise=False):
        """
        Run QAOA with custom initial angles.

        Args:
            initial_params: Initial parameters for the optimization.
            max_iters: Maximum number of iterations for the optimizer.
            noise: Boolean indicating whether to include noise in the simulation.

        Returns:
            Optimization result and the list of objective function values.
        """
        self._initialize_backend(noise)
        result, obj_values = self._run_qaoa(initial_params, max_iters)
        return result, obj_values