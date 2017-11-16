import numpy as np
from qsim.helpers import I, Sx, Sy, Sz, dagger
from .hamiltonians import make_generic_2_qubit_hamiltonian


def create_magnetisation_unitary_mpo(qubit_num, qubit_index, delta_t,
                                     axis='Z'):
    identity = np.reshape(1, 1, 2, 2)
    if axis == 'Z':
        S = Sz.reshape(1, 1, 2, 2)
    elif axis == 'X':
        S = Sx.reshape(1, 1, 2, 2)
    elif axis == 'Y':
        S = Sy.reshape(1, 1, 2, 2)
    D, U = np.linalg.eig(S)
    unitary = np.dot(U, np.dot(np.diag(np.exp(-1j * delta_t * D)), dagger(U)))
    mpo = [[]] * qubit_num
    for i in range(qubit_num):
        mpo[i] = unitary if i == qubit_index else identity
    return mpo


def create_heisenberg_unitary_mpo(qubit_num=1, J=0, h=0, t=0, even=True):
    """
    Creates MPO of Unitary for evolution under
        H = J/2(S+S- + S-S+) + JSzSz + hSz
    on pairs of nearest neighbours
    Args:
        qubit_num
        J
        h
        t time step to use
        even (default True)
    """
    # make up matrices
    S_middle = make_generic_2_qubit_unitary(t, end=0, XX=J, YY=J, ZZ=J, Z=h)
    S_first = make_generic_2_qubit_unitary(t, end=-1, XX=J, YY=J, ZZ=J, Z=h)
    S_last = make_generic_2_qubit_unitary(t, end=1, XX=J, YY=J, ZZ=J, Z=h)
    S_middle_r = np.swapaxes(S_middle.reshape(2, 2, 2, 2), 1, 2).reshape(4, 4)
    S_first_r = np.swapaxes(S_first.reshape(2, 2, 2, 2), 1, 2).reshape(4, 4)
    S_last_r = np.swapaxes(S_last.reshape(2, 2, 2, 2), 1, 2).reshape(4, 4)

    # do svd to get mpos
    U, s, V = np.linalg.svd(S_middle_r, full_matrices=False)
    a0 = np.dot(U, np.diag(np.sqrt(s)))
    a1 = np.dot(np.diag(np.sqrt(s)), V)
    M0 = np.swapaxes(a0, 0, 1).reshape(1, 4, 2, 2)
    M1 = a1.reshape(4, 1, 2, 2)

    U, s, V = np.linalg.svd(S_first_r, full_matrices=False)
    a0 = np.dot(U, np.diag(np.sqrt(s)))
    a1 = np.dot(np.diag(np.sqrt(s)), V)
    M0_first = np.swapaxes(a0, 0, 1).reshape(1, 4, 2, 2)
    M1_first = a1.reshape(4, 1, 2, 2)

    U, s, V = np.linalg.svd(S_last_r, full_matrices=False)
    a0 = np.dot(U, np.diag(np.sqrt(s)))
    a1 = np.dot(np.diag(np.sqrt(s)), V)
    M0_last = np.swapaxes(a0, 0, 1).reshape(1, 4, 2, 2)
    M1_last = a1.reshape(4, 1, 2, 2)

    mpo = [[]] * qubit_num
    if even:
        for i in range(qubit_num):
            if i % 2 == 0:
                mpo[i] = M0
            else:
                mpo[i] = M1
        mpo[0] = M0_first
        mpo[1] = M1_first
        if qubit_num % 2 == 0:
            mpo[-1] = M1_last
            mpo[-2] = M0_last
        else:
            mpo[-1] = I.reshape(1, 1, 2, 2)
    else:
        for i in range(qubit_num):
            if i % 2 != 0:
                mpo[i] = M0
            else:
                mpo[i] = M1
        mpo[0] = I.reshape(1, 1, 2, 2)
        if qubit_num % 2 != 0:
            mpo[-1] = M1_last
            mpo[-2] = M0_last
        else:
            mpo[-1] = I.reshape(1, 1, 2, 2)
    return mpo


def make_2_qubit_unitary(delta_t, end=0, **kwargs):
    H = make_generic_2_qubit_hamiltonian(end=end, **kwargs)
    D, U = np.linalg.eig(H)
    unitary = np.dot(U, np.dot(np.diag(np.exp(-1j * delta_t * D)), dagger(U)))
    return unitary


# def make_S(delta_t, J=0, h=0, end=0):
#     """
#     Makes 2 * 2 matrix for unitary evolution under Heisenberg
#         H = J/2(S+S- + S-S+) + JSzSz + hSz
#     Args:
#         delta_t
#         J
#         j
#         end [-1, 0, 1] corresponds to first, middle, last (default 0)
#     """
#     if end == -1:
#         alpha = np.sqrt((h / 2) ** 2 + J**2)
#         beta = delta_t * alpha / 2
#         S = np.exp(1j * delta_t * J / 4) * np.array([
#             [np.exp(-1j * delta_t * (2 * J + 3 * h) / 4), 0, 0, 0],
#             [0, np.cos(beta) - 1j * h * np.sin(beta) / (2 * alpha), -
#              1j * J * np.sin(beta) / alpha, 0],
#             [0, - 1j * J * np.sin(beta) / alpha, np.cos(beta) +
#              1j * h * np.sin(beta) / (2 * alpha), 0],
#             [0, 0, 0, np.exp(-1j * delta_t * (2 * J - 3 * h) / 4)]])
#     elif end == 0:
#         Sp_Sm = np.array([
#             [1, 0, 0, 0],
#             [0, np.cos(J * delta_t / 2), -1j * np.sin(J * delta_t / 2), 0],
#             [0, -1j * np.sin(J * delta_t / 2), np.cos(J * delta_t / 2), 0],
#             [0, 0, 0, 1]])
#         Sz_Sz = np.exp(1j * J * delta_t / 4) * np.array([
#             [np.exp(-1j * J * delta_t / 2), 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [0, 0, 0, np.exp(-1j * J * delta_t / 2)]])
#         Sz = np.array([
#             [np.exp(-1j * h * delta_t / 2), 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [0, 0, 0, np.exp(1j * h * delta_t / 2)]])
#         S = np.dot(Sp_Sm, np.dot(Sz_Sz, Sz))
#     elif end == 1:
#         alpha = np.sqrt((h / 2) ** 2 + J**2)
#         beta = delta_t * alpha / 2
#         S = np.exp(1j * delta_t * J / 4) * np.array([
#             [np.exp(-1j * delta_t * (2 * J + 3 * h) / 4), 0, 0, 0],
#             [0, np.cos(beta) + 1j * h * np.sin(beta) / (2 * alpha), -
#              1j * J * np.sin(beta) / alpha, 0],
#             [0, - 1j * J * np.sin(beta) / alpha, np.cos(beta) -
#              1j * h * np.sin(beta) / (2 * alpha), 0],
#             [0, 0, 0, np.exp(-1j * delta_t * (2 * J - 3 * h) / 4)]])
#     else:
#         raise RuntimeError("you're an idiot")
#     return S


