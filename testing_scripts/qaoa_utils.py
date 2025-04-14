import numpy as np
import os
import warnings
from scipy.optimize import minimize
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import  (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)
from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2

from clapton.clapton import claptonize,group_claptonize
from clapton.circuit_manipulation import (
    transform_to_allowed_gates,
    qiskit_to_stim,
    modify_circuit,
    relax_qaoa_parameters,
    generate_qiskit_param_map,)
from clapton.depolarization import GateGeneralDepolarizationModel


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

    def prepare_circuit(self,multi_angle=True):
        """
        Prepare the QAOA circuit and relax its parameters.

        Args:
            circuit: The QAOA circuit to be prepared.
        """
        modified_circ = modify_circuit(self.circuit) #TODO: bring all helper functions into this file
        self.pcirc = transform_to_allowed_gates(modified_circ)
        if multi_angle:
            self.pcirc, _, _ = relax_qaoa_parameters(self.pcirc)    
        self.stim_circ = qiskit_to_stim(self.pcirc)
        self.param_map = generate_qiskit_param_map(self.pcirc)
        self.stim_circ.define_parameter_map(self.param_map)

    def run_CAFQA(self, n_gens , noise=None):
        """
        Run the CAFQA initialization.

        Args:
            n_gens: Number of generations for the genetic algorithm.
        """
        paulis, coeffs = self.cost_hamiltonian.paulis.to_labels(), self.cost_hamiltonian.coeffs.real
        reversed_paulis = [p[::-1] for p in paulis]

        if noise: 
                # let's add a noise model where we specify global 1q and 2q gate errors
                nm = GateGeneralDepolarizationModel(p1=noise, p2=noise)
                self.stim_circ.add_depolarization_model(nm)

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

    def run_group_CAFQA(self,n_gens,group_dict):
        """
        Run the CAFQA initialization.

        Args:
            n_gens: Number of generations for the genetic algorithm.
        """
        paulis, coeffs = self.cost_hamiltonian.paulis.to_labels(), self.cost_hamiltonian.coeffs.real
        reversed_paulis = [p[::-1] for p in paulis]
        
        self.stim_circ.group_dict = group_dict

        self.ks_best, _, self.energy_best = group_claptonize(
            reversed_paulis,
            coeffs,
            self.stim_circ,
            n_proc=4,
            n_starts=4,
            n_rounds=1,
            callback=print,
            budget=n_gens // 2,
            group_dict = group_dict
        )

        print(f"Minimum Energy found with CAFQA initialization: {self.energy_best}")

        # group_dict = {'γ[0]': [0], 'β[0]': [1,2]}
        # group_dict = {'β[0]': [0, 1, 3], 'γ[0]': [2, 4, 5]}
        placeholder_list = [0] * sum(map(len, group_dict.values()))

        for i,j in enumerate(group_dict.items()):
            val = self.ks_best[i]
            for k in j[1]:
                placeholder_list[k] = val
        self.ks_best = placeholder_list

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
    
    def evaluate_energy(self, qiskit_circuit, hamiltonian, parameters,noise=False,err=None):

        self._initialize_backend(err=err,noise=noise)
        isa_hamiltonian = hamiltonian.apply_layout(qiskit_circuit.layout)

        pub = (qiskit_circuit, isa_hamiltonian, parameters)
        job = self.estimator.run([pub])

        results = job.result()[0]
        return results.data.evs

    def _create_noise_model(self,err):
        noise_model = NoiseModel()
        error = depolarizing_error(err, 1)
        cx_err = depolarizing_error(err, 2)
        noise_model.add_all_qubit_quantum_error(error, ["rx", "rz"])
        noise_model.add_all_qubit_quantum_error(cx_err, ["cx"])
        return noise_model


    def _initialize_backend(self, err=1e-3, noise=False,):
        """
        Initialize the quantum backend with or without noise.

        Args:
            noise: Boolean indicating whether to include noise in the simulation.

        Returns:
            A configured quantum backend.
        """
        self.backend = AerSimulator(method='statevector')
        if noise:
            # noisy_backend = FakeMumbaiV2()
            noise_model = self._create_noise_model(err)
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

    def _run_qaoa(self, initial_params, maxiters=1000):
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
            options={'maxiter': maxiters},
        )
        return result, objective_func_vals

    def run_qaoa(self, initial_params, err=None, max_iters=1000, noise=False):
        """
        Run QAOA with custom initial angles.

        Args:
            initial_params: Initial parameters for the optimization.
            max_iters: Maximum number of iterations for the optimizer.
            noise: Boolean indicating whether to include noise in the simulation.

        Returns:
            Optimization result and the list of objective function values.
        """
        self._initialize_backend(err=err,noise=noise)
        result, obj_values = self._run_qaoa(initial_params, max_iters)
        return result, obj_values