import numpy as np
import os
import warnings
from scipy.optimize import minimize
from qiskit import transpile
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.applications import Knapsack
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2

from testing_scripts.qaoa_utils import run_qaoa
from clapton.clapton import claptonize
from clapton.circuit_manipulation import (
    transform_to_allowed_gates,
    qiskit_to_stim,
    modify_circuit,
    relax_qaoa_parameters,
    generate_qiskit_param_map,
)

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class QAOASolver:
    def __init__(self, n_items, reps, seed):
        """
        Initialize the QAOA object.

        Args:
            n_items: Number of items in the knapsack problem.
            reps: Number of QAOA repetitions.
            seed: Random seed for reproducibility.
        """
        self.cost_hamiltonian = None
        self.pcirc = None
        self.stim_circ = None
        self.ks_best = None
        self.energy_best = None
        self.param_map = None

    def prepare_circuit(self,circuit):
        """Prepare the QAOA circuit and relax its parameters."""
        modified_circ = modify_circuit(circuit)
        self.pcirc = transform_to_allowed_gates(modified_circ)
        self.pcirc, _, _ = relax_qaoa_parameters(self.pcirc)
        self.stim_circ = qiskit_to_stim(self.pcirc)
        self.param_map = generate_qiskit_param_map(self.pcirc)
        self.stim_circ.define_parameter_map(self.param_map)

    def run_cafqa(self,n_gens):
        """Run the CAFQA initialization."""
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
            budget=n_gens//2,
        )
        print(f"Minimum Energy found with CAFQA initialization: {self.energy_best}")
        self.stim_circ.assign(self.ks_best)

    def exact_energy(self):
        """Solve the problem classically using the NumPyMinimumEigensolver."""
        eigensolver = NumPyMinimumEigensolver()
        exact_solution = eigensolver.compute_minimum_eigenvalue(self.cost_hamiltonian).eigenvalue.real
        print("Exact Energy from Eigensolver:", exact_solution)
        return exact_solution

    def run_qaoa(self,initial_params, max_iters=1000,noise=False):
        """Run QAOA with both random and CAFQA initialization."""

        result, obj_values = self._run_qaoa(
            self.pcirc, initial_params, self.cost_hamiltonian, max_iters, noise
        )

        return result,obj_values

    def _cost_function(self, params, objective_func_vals):
        """
        Cost function to be minimized.

        Args:
            params: Parameters for the quantum circuit.
            objective_func_vals: List to store the cost function values.

        Returns:
            Cost value.
        """
        isa_hamiltonian = self.cost_hamiltonian.apply_layout(self.circuit.layout)
        pub = (self.circuit, isa_hamiltonian, params)
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
            options={'maxiter': maxiter}
        )
        return result, objective_func_vals


