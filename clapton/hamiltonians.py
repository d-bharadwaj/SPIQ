import numpy as np
from numbers import Number


def ising_model(N, Jx, h, Jy=0., periodic=False):
    """
    Constructs qubit Hamiltonian for linear Ising model.
    H = sum_{i=0...N-2} (Jx_i X_i X_{i+1} + Jy_i Y_i Y_{i+1}) + sum_{i=0...N-1}  h_i Z_i

    N (Int): # sites/qubits.
    Jx (Float, Iterable[Float]): XX strength, either constant value or list (values for each pair of neighboring sites).
    h (Float, Iterable[Float]): Z self-energy, either constant value or list (values for each site).
    Jy (Float, Iterable[Float]): YY strength, either constant value or list (values for each pair of neighboring sites).
    periodic: If periodic boundary conditions. If True, include term X_0 X_{N-1} and Y_0 Y_{N-1}.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, "0"*N)
    """
    if isinstance(Jx, Number):
        if periodic:
            Jx = [Jx] * N
        else:
            Jx = [Jx] * (N-1)
    if isinstance(Jy, Number):
        if periodic:
            Jy = [Jy] * N
        else:
            Jy = [Jy] * (N-1)
    if isinstance(h, Number):
        h = [h] * N        
    if N > 1:
        assert len(Jx) == N if periodic else len(Jx) == N-1, "Jx has wrong length"
        assert len(Jy) == N if periodic else len(Jy) == N-1, "Jy has wrong length"
        assert len(h) == N, "h has wrong length"
    coeffs = []
    paulis = []
    # add XX terms
    for j in range(N-1):
        if np.abs(Jx[j]) > 1e-12:
            coeffs.append(Jx[j])
            paulis.append("I"*j+"XX"+"I"*(N-j-2))
    if N > 2 and periodic and np.abs(Jx[N-1]) > 1e-12:
        coeffs.append(Jx[N-1])
        paulis.append("X"+"I"*(N-2)+"X")
    # add YY terms
    for j in range(N-1):
        if np.abs(Jy[j]) > 1e-12:
            coeffs.append(Jy[j])
            paulis.append("I"*j+"YY"+"I"*(N-j-2))
    if N > 2 and periodic and np.abs(Jy[N-1]) > 1e-12:
        coeffs.append(Jy[N-1])
        paulis.append("Y"+"I"*(N-2)+"Y")
    # add Z terms
    for j in range(N):
        if np.abs(h[j]) > 1e-12:
            coeffs.append(h[j])
            paulis.append("I"*j+"Z"+"I"*(N-j-1))
    return coeffs, paulis, "1"*N