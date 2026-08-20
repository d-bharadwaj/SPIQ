def convert_pubo_to_ising(hypergraph: dict, n_qubits: int) -> list[tuple[str, float]]:
    """Convert a PUBO hypergraph dictionary to Pauli strings with weights."""
    pauli_list = []

    for edge, weight in hypergraph.items():
        if not edge:
            continue
        paulis = ["I"] * n_qubits
        for node in edge:
            paulis[node] = "Z"
        pauli_list.append(("".join(paulis[::-1]), weight))

    return pauli_list
