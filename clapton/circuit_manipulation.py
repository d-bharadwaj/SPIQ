from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
import numpy as np
import stim

from clapton.clifford import ParametrizedCliffordCircuit


def transform_to_allowed_gates(circuit, **kwargs):
    """
    circuit (QuantumCircuit): Circuit with only Clifford gates (1q rotations Ry, Rz must be k*pi/2).
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    
    Returns:
    (QuantumCircuit) Logically equivalent circuit but with gates in required format (no Ry, Rz gates; only S, Sdg, H, X, Z).
    """
    dag = circuit_to_dag(circuit)
    threshold = 1e-3
    # we will substitute nodes inplace
    for node in dag.op_nodes():
            if node.name == "ry" and not isinstance(node.op.params[0],ParameterExpression):
                angle = float(node.op.params[0])
                # substitute gates
                if abs(angle - 0) < threshold:
                    dag.remove_op_node(node)
                elif abs(angle - np.pi/2) < threshold:
                    qc_loc = QuantumCircuit(1)
                    qc_loc.sdg(0)
                    qc_loc.sx(0)
                    qc_loc.s(0)
                    qc_loc_instr = qc_loc.to_instruction()
                    dag.substitute_node(node, qc_loc_instr, inplace = True)
                elif abs(angle - np.pi) < threshold:
                    qc_loc = QuantumCircuit(1)
                    qc_loc.y(0)
                    qc_loc_instr = qc_loc.to_instruction()
                    dag.substitute_node(node, qc_loc_instr, inplace=True)
                elif abs(angle - 1.5*np.pi) < threshold:
                    qc_loc = QuantumCircuit(1)
                    qc_loc.sdg(0)
                    qc_loc.sxdg(0)
                    qc_loc.s(0)
                    qc_loc_instr = qc_loc.to_instruction()
                    dag.substitute_node(node, qc_loc_instr, inplace = True)
            elif node.name == 'rz' and not isinstance(node.op.params[0],ParameterExpression):
                angle = float(node.op.params[0])
                #substitute gates
                if abs(angle - 0) < threshold:
                    dag.remove_op_node(node)
                elif abs(angle - np.pi/2) < threshold:
                    qc_loc = QuantumCircuit(1)
                    qc_loc.s(0)
                    qc_loc_instr = qc_loc.to_instruction()
                    dag.substitute_node(node, qc_loc_instr, inplace=True)
                elif abs(angle - np.pi) < threshold:
                    qc_loc = QuantumCircuit(1)
                    qc_loc.z(0)
                    qc_loc_instr = qc_loc.to_instruction()
                    dag.substitute_node(node, qc_loc_instr, inplace=True)
                elif abs(angle - 1.5*np.pi) < threshold:
                    qc_loc = QuantumCircuit(1)
                    qc_loc.sdg(0)
                    qc_loc_instr = qc_loc.to_instruction()
                    dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif node.name == 'rx' and not isinstance(node.op.params[0],ParameterExpression): # NOTE: Added this 
                angle = float(node.op.params[0])
                # substitute gates
                if abs(angle - 0) < threshold:
                    dag.remove_op_node(node)
                elif abs(angle - np.pi/2) < threshold:
                    qc_loc = QuantumCircuit(1)
                    qc_loc.sx(0)
                    qc_loc_instr = qc_loc.to_instruction()
                    dag.substitute_node(node, qc_loc_instr, inplace = True)
                elif abs(angle - np.pi) < threshold:
                    qc_loc = QuantumCircuit(1)
                    qc_loc.x(0)
                    qc_loc_instr = qc_loc.to_instruction()
                    dag.substitute_node(node, qc_loc_instr, inplace=True)
                elif abs(angle - 1.5*np.pi) < threshold:
                    qc_loc = QuantumCircuit(1)
                    qc_loc.sxdg(0)
                    qc_loc_instr = qc_loc.to_instruction()
                    dag.substitute_node(node, qc_loc_instr, inplace = True)
            elif node.name == "x":
                qc_loc = QuantumCircuit(1)
                qc_loc.x(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
    return dag_to_circuit(dag).decompose()

def qiskit_to_stim(circuit): # TODO: need to change this to accomodate .fix
    """
    Transform Qiskit QuantumCircuit into stim circuit.
    circuit (QuantumCircuit): Clifford-only circuit.

    Returns:
    (stim._stim_sse2.Circuit) stim circuit.
    """
    assert isinstance(circuit, QuantumCircuit), f"Circuit is not a Qiskit QuantumCircuit."
    allowed_gates = ["X", "Y", "Z", "H", "CX", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG"]


    # stim_circ = stim.Circuit()
    # # make sure right number of qubits in stim circ
    # for i in range(circuit.num_qubits):
    #     stim_circ.append("I", [i])
    # for instruction in circuit:
    #     gate_lbl = instruction.operation.name.upper()
    #     if gate_lbl == "BARRIER":
    #         continue
    #     elif gate_lbl == "SDG":
    #         gate_lbl = "S_DAG"
    #     elif gate_lbl == "SX":
    #         gate_lbl = "SQRT_X"
    #     elif gate_lbl == "SXDG":
    #         gate_lbl = "SQRT_X_DAG"
    #     assert gate_lbl in allowed_gates, f"Invalid gate {gate_lbl}."
    #     qubit_idc = [qb.index for qb in instruction.qubits]
    #     stim_circ.append(gate_lbl, qubit_idc)

    stim_circ = ParametrizedCliffordCircuit()
    # make sure right number of qubits in stim circ
    # for i in range(circuit.num_qubits):
    #     stim_circ.append("I", [i])

    #NOTE : You can optimize this code very easily
    for instruction in circuit:
        gate_lbl = instruction.operation.name.upper()
        if gate_lbl == "BARRIER" or gate_lbl == "MEASURE":
            continue
        elif gate_lbl in ["X", "Y", "Z", "H", "S"]:
            single_gate_dict = {"X": 1, "Y": 2, "Z": 3, "H": 4,"S" : 8}
            qb = instruction.qubits[0]._index
            stim_circ.C1(qb).fix(single_gate_dict[gate_lbl])
        elif gate_lbl == "CX":
            control_qb = instruction.qubits[0]._index
            target_qb = instruction.qubits[1]._index
            stim_circ.Q2(control_qb, target_qb).fix(1)
        elif gate_lbl == "SDG":
            qb = instruction.qubits[0]._index
            stim_circ.RZ(qb).fix(3)
            gate_lbl = "S_DAG"
        elif gate_lbl == "SX":
            qb = instruction.qubits[0]._index
            stim_circ.RX(qb).fix(1)
            gate_lbl = "SQRT_X"
        elif gate_lbl == "SXDG":
            qb = instruction.qubits[0]._index
            stim_circ.RX(qb).fix(3)
            gate_lbl = "SQRT_X_DAG"
        elif gate_lbl == "U1":
            qb = instruction.qubits[0]._index
            if isinstance(instruction.operation.params[0],ParameterExpression):
                stim_circ.RZ(qb) #keep it not fixed
                gate_lbl = "Z"
            else:
                angle = float(instruction.operation.params[0])
                gate_index = int((angle % (2 * np.pi)) // (np.pi / 2))
                stim_circ.RZ(qb).fix(gate_index) #NOTE: need to check
                gate_lbl = ["I", "S", "Z", "S_DAG"][gate_index]

        assert gate_lbl in allowed_gates, f"Invalid gate {gate_lbl}."
        # qubit_idc = [qb._index for qb in instruction.qubits]
        # stim_circ.append(gate_lbl, qubit_idc)

    return stim_circ
