# This module defines a Qiskit transpiler pass from the official blog (https://docs.quantum.ibm.com/guides/custom-transpiler-pass) for adding Pauli twirls to two-qubit gates.
# The PauliTwirl class inherits from TransformationPass and introduces Pauli twirling to 
# specified two-qubit gates in a quantum circuit to mitigate errors.
#
# The PauliTwirl class:
# - Initializes with a list of gates to twirl (defaulting to CX and ECR gates).
# - Builds a set of Pauli twirl pairs for each gate.
# - Applies the twirling transformation to the specified gates in a given DAGCircuit.
#
# Imports necessary components from Qiskit and uses numpy for random number generation.

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister, Gate
from qiskit.circuit.library import CXGate, ECRGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import Operator, pauli_basis
import numpy as np
from typing import Iterable, Optional

class PauliTwirl(TransformationPass):
    """Add Pauli twirls to two-qubit gates."""
 
    def __init__(self, gates_to_twirl: Optional[Iterable[Gate]] = None):
        if gates_to_twirl is None:
            gates_to_twirl = [CXGate(), ECRGate()]
        self.gates_to_twirl = gates_to_twirl
        self.build_twirl_set()
        super().__init__()
 
    def build_twirl_set(self):
        self.twirl_set = {}
        for twirl_gate in self.gates_to_twirl:
            twirl_list = []
            for pauli_left in pauli_basis(2):
                for pauli_right in pauli_basis(2):
                    if (Operator(pauli_left) @ Operator(twirl_gate)).equiv(
                        Operator(twirl_gate) @ pauli_right
                    ):
                        twirl_list.append((pauli_left, pauli_right))
            self.twirl_set[twirl_gate.name] = twirl_list
 
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        twirling_gate_classes = tuple(gate.base_class for gate in self.gates_to_twirl)
        for node in dag.op_nodes():
            if not isinstance(node.op, twirling_gate_classes):
                continue
            pauli_index = np.random.randint(0, len(self.twirl_set[node.op.name]))
            twirl_pair = self.twirl_set[node.op.name][pauli_index]
            mini_dag = DAGCircuit()
            register = QuantumRegister(2)
            mini_dag.add_qreg(register)
            mini_dag.apply_operation_back(twirl_pair[0].to_instruction(), [register[0], register[1]])
            mini_dag.apply_operation_back(node.op, [register[0], register[1]])
            mini_dag.apply_operation_back(twirl_pair[1].to_instruction(), [register[0], register[1]])
            dag.substitute_node_with_dag(node, mini_dag)
        return dag
