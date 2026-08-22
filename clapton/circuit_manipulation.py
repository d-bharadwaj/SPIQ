from clapton.clifford import ParametrizedCliffordCircuit

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
import numpy as np


# Function to build the QAOA circuit
def multi_angle_qaoa_circuit(num_qubits, G, reps, gamma_params=None, beta_params=None):
    if not gamma_params and not beta_params:
        gamma_params = [
            Parameter(f"gamma_{i}_{j}_{r}")
            for r in range(reps)
            for i, j in G.edge_list()
        ]
        beta_params = [
            Parameter(f"beta_{i}_{r}") for r in range(reps) for i in G.node_indexes()
        ]

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    for rep in range(reps):
        gamma_offset = rep * len(G.edge_list())
        beta_offset = rep * len(G.node_indexes())

        for idx, (i, j, weight) in enumerate(G.weighted_edge_list()):
            gamma = gamma_params[gamma_offset + idx]  # Get the gamma parameter
            qc.cx(i, j)
            qc.rz(
                -2 * weight * gamma, j
            )  # if we add weights here, then we need to use the weights for circuit conversion.
            qc.cx(i, j)

        for idx, i in enumerate(G.node_indexes()):
            beta = beta_params[beta_offset + idx]  # Get the beta parameter
            qc.rx(2 * beta, i)

    return qc


def multi_angle_qaoa_plus_circuit(num_qubits, G, reps=1):
    qc = QuantumCircuit(num_qubits)

    gamma_params = [
        Parameter(f"gamma_{i}_{j}_{r}") for r in range(reps) for i, j in G.edge_list()
    ]
    beta_params = [
        Parameter(f"beta_{i}_{r}") for r in range(reps) for i in G.node_indexes()
    ]

    gamma_plus_params = [
        Parameter(f"gamma_plus_{i}_{i+1}_{r}")
        for r in range(reps)
        for i in range(num_qubits - 1)
    ]
    beta_plus_params = [
        Parameter(f"beta_plus_{i}_{r}") for r in range(reps) for i in range(num_qubits)
    ]

    for i in range(reps):
        ma_qaoa_circ = multi_angle_qaoa_circuit(
            num_qubits, G, reps, gamma_params=gamma_params, beta_params=beta_params
        )
        qc = qc.compose(ma_qaoa_circ)

        for j in range(num_qubits - 1):
            gamma_plus = gamma_plus_params[
                i * (num_qubits - 1) + j
            ]  # Get the gamma plus parameter
            a, b = j, j + 1
            qc.cx(a, b)
            qc.rz(gamma_plus, b)
            qc.cx(a, b)

        for j in range(num_qubits):
            beta_plus = beta_plus_params[
                i * (num_qubits - 1) + j
            ]  # Get the beta plus parameter
            qc.rx(beta_plus, j)

    return qc


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
        if node.name == "ry" and not isinstance(node.op.params[0], ParameterExpression):
            angle = float(node.op.params[0])
            # substitute gates
            if abs(angle - 0) < threshold:
                dag.remove_op_node(node)
            elif abs(angle - np.pi / 2) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.sdg(0)
                qc_loc.sx(0)
                qc_loc.s(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.y(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - 1.5 * np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.sdg(0)
                qc_loc.sxdg(0)
                qc_loc.s(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
        elif node.name == "rz" and not isinstance(
            node.op.params[0], ParameterExpression
        ):
            angle = float(node.op.params[0])
            # substitute gates
            if abs(angle - 0) < threshold:
                dag.remove_op_node(node)
            elif abs(angle - np.pi / 2) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.s(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.z(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - 1.5 * np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.sdg(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
        elif node.name == "rx" and not isinstance(
            node.op.params[0], ParameterExpression
        ):  # NOTE: Added this
            angle = float(node.op.params[0])
            # substitute gates
            if abs(angle - 0) < threshold:
                dag.remove_op_node(node)
            elif abs(angle - np.pi / 2) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.sx(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.x(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - 1.5 * np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.sxdg(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
        elif node.name == "x":
            qc_loc = QuantumCircuit(1)
            qc_loc.x(0)
            qc_loc_instr = qc_loc.to_instruction()
            dag.substitute_node(node, qc_loc_instr, inplace=True)
    return dag_to_circuit(dag)  # .decompose()


def qiskit_to_stim(circuit):
    """
    Transform Qiskit QuantumCircuit into stim circuit.
    circuit (QuantumCircuit): Clifford-only circuit.

    Returns:
    (stim._stim_sse2.Circuit) stim circuit.
    """
    assert isinstance(
        circuit, QuantumCircuit
    ), f"Circuit is not a Qiskit QuantumCircuit."
    allowed_gates = ["X", "Y", "Z", "H", "CX", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG"]

    stim_circ = ParametrizedCliffordCircuit()
    # make sure right number of qubits in stim circ
    # for i in range(circuit.num_qubits):
    #     stim_circ.append("I", [i])

    for instruction in circuit:
        gate_lbl = instruction.operation.name.upper()
        if gate_lbl == "BARRIER" or gate_lbl == "MEASURE":
            continue
        elif gate_lbl in ["X", "Y", "Z", "H", "S"]:
            single_gate_dict = {"X": 1, "Y": 2, "Z": 3, "H": 4, "S": 8}
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
        elif gate_lbl == "RZ":
            qb = instruction.qubits[0]._index
            if isinstance(instruction.operation.params[0], ParameterExpression):
                stim_circ.RZ(qb)  # keep it not fixed
                gate_lbl = "Z"
            else:
                angle = float(instruction.operation.params[0])
                gate_index = int((angle % (2 * np.pi)) // (np.pi / 2))
                stim_circ.RZ(qb).fix(gate_index)  # NOTE: need to check
                gate_lbl = ["I", "S", "Z", "S_DAG"][gate_index]

        assert gate_lbl in allowed_gates, f"Invalid gate {gate_lbl}."
        # qubit_idc = [qb._index for qb in instruction.qubits]
        # stim_circ.append(gate_lbl, qubit_idc)

    return stim_circ


def modify_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    # Step 1: Transpile the circuit to use Clifford + Rz gates
    transpiled_circuit = transpile(circuit, basis_gates=["rz", "s", "sx", "cx", "h"])

    # Step 2: Create a new circuit to store the modified version
    modified_circuit = QuantumCircuit(transpiled_circuit.num_qubits)

    # Step 3: Iterate through the gates in the transpiled circuit
    for instr in transpiled_circuit.data:
        gate = instr[0]
        qubits = instr[1]

        # Handle the specified Rz modifications
        if gate.name == "rz":
            angle = gate.params[0]
            if angle == -np.pi or angle == np.pi:
                # Replace "rz(-pi)" or "rz(pi)" with "z"
                modified_circuit.z(*qubits)
            elif angle == -np.pi / 2 or angle == 3 * np.pi / 2:
                # Replace "rz(-pi/2)" or "rz(3*pi/2)" with "s" followed by "z"
                modified_circuit.s(*qubits)
                modified_circuit.z(*qubits)
            elif angle == -3 * np.pi / 2 or angle == np.pi / 2:
                # Replace "rz(-3*pi/2)" or "rz(pi/2)" with "s"
                modified_circuit.s(*qubits)
            else:
                # Add the Rz gate if it does not match any modification
                modified_circuit.append(gate, qubits)
        else:
            # Add the gate to the modified circuit if it's not an Rz gate
            modified_circuit.append(gate, qubits)

    return modified_circuit


def create_cost_hamiltonian_stim(G):
    coeffs = []
    paulis = []

    for i, j, w in G.edges(data="weight"):
        weight = w if w is not None else 1
        # Store the weight (coefficient) in coeffs
        coeffs.append(weight)

        # Create the Pauli term in string format for ZZ on qubits i and j
        pauli_term = ["I"] * G.number_of_nodes()  # Initialize as identity on all qubits
        pauli_term[i] = "Z"  # Z on the first qubit of the edge
        pauli_term[j] = "Z"  # Z on the second qubit of the edge
        paulis.append("".join(pauli_term))  # Join into a single string
    print(coeffs, paulis)
    return coeffs, paulis


def generate_qiskit_param_map(circuit):
    dag = circuit_to_dag(circuit)
    param_list = [
        list(node.op.params[0].parameters)[0].name
        for node in dag.op_nodes()
        if node.op.params and isinstance(node.op.params[0], ParameterExpression)
    ]
    qiskit_param_map = {k: v for v, k in enumerate(param_list)}

    # Ensure the sorted names are correct
    ordered_params = [param.name for param in circuit.parameters]
    assert sorted(qiskit_param_map.keys()) == ordered_params

    param_map = {i: qiskit_param_map[param] for i, param in enumerate(ordered_params)}
    return param_map


def relax_qaoa_parameters(circ):
    dag = circuit_to_dag(circ)
    gamma_counter, beta_counter = 0, 0
    angle_multipliers = {}
    for node in dag.op_nodes():
        if node.op.params and isinstance(node.op.params[0], ParameterExpression):
            param_name = list(node.op.params[0].parameters)[0].name
            if "β" in param_name or "beta" in param_name:

                multiplier = float(str(node.op.params[0]).split("*")[0])

                beta = Parameter(f"{multiplier}*beta_{beta_counter}")
                new_params = [
                    beta if (isinstance(p, ParameterExpression)) else p
                    for p in node.op.params
                ]
                new_op = node.op.copy()
                new_op.params = new_params  # Create a modified version of the operation
                dag.substitute_node(node, new_op)  # Replace the node in the DAG
                beta_counter += 1

                angle_multipliers[beta.name] = multiplier

            elif "γ" in param_name or "gamma" in param_name:

                multiplier = float(str(node.op.params[0]).split("*")[0])

                gamma = Parameter(f"{multiplier}*gamma_{gamma_counter}")
                new_params = [
                    gamma if (isinstance(p, ParameterExpression)) else p
                    for p in node.op.params
                ]
                new_op = node.op.copy()
                new_op.params = new_params  # Create a modified version of the operation
                dag.substitute_node(node, new_op)  # Replace the node in the DAG
                gamma_counter += 1

                angle_multipliers[gamma.name] = multiplier

    # Convert DAG back to a circuit
    new_qc = dag_to_circuit(dag)
    return new_qc, circuit_to_dag(new_qc), angle_multipliers


def qaoa_relax_parameters(circuit, opt_val=0.2):
    # QAOA Relax Initalization method chosen from https://arxiv.org/pdf/2409.12104 paper.
    constant_vals = {}
    for param in circuit.parameters:
        param_name = param.name
        if "beta" in param_name:
            val = opt_val
        else:
            val = -1 * opt_val
        constant_vals[param_name] = val
    return list(constant_vals.values())


def transform_qiskit_to_stim(qiskit_circuit):
    # Transform qiskit circuit to stim.
    modified_circ = modify_circuit(qiskit_circuit)
    pcirc = transform_to_allowed_gates(modified_circ)
    pcirc, dag, angle_multipliers = relax_qaoa_parameters(pcirc)
    stim_circ = qiskit_to_stim(pcirc)
    return stim_circ, pcirc
