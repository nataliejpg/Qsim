import numpy as np
from qsim.helpers import Sx, Sy, Sz, I, Splus, Sminus


def create_specific_mpo(matrix):
    raise NotImplementedError


def create_heisenberg_mpo(qubit_num, J, h):
    """
    Creates MPO of Hamoltonian for
        H = J/2(S+S- + S-S+) + JSzSz + hSz
    Args:
        qubit_num number of qubits
        J value for coupling
        h value for longitudinal field
    Returns:
        mpo with shape (qubit_num, b0, b1, 2, 2)
    """
    mpo = [[]] * qubit_num
    M = np.zeros((5, 5, 2, 2), dtype=complex)

    M[0, 0, :, :] = I
    M[1, 0, :, :] = Splus
    M[2, 0, :, :] = Sz
    M[3, 0, :, :] = Sminus
    M[4, 0, :, :] = h * Sz

    M[4, 1, :, :] = J / 2 * Sminus
    M[4, 2, :, :] = J * Sz
    M[4, 3, :, :] = J / 2 * Splus
    M[4, 4, :, :] = I

    for i in range(1, qubit_num - 1):
        mpo[i] = M

    mpo[0] = np.zeros((1, 5, 2, 2), dtype=complex)
    mpo[-1] = np.zeros((5, 1, 2, 2), dtype=complex)
    mpo[0][0, :, :, :] = M[4, :, :, :]
    mpo[-1][:, 0, :, :] = M[:, 0, :, :]

    return mpo


def create_magnetisation_mpo(qubit_num, qubit_index, axis='Z'):
    """
    Creates MPO of Hamiltonian for
        H = Sxyz
    Args:
        qubit_num number of qubits
        qubit_index to apply magnetisation onto
        axis of magnetisation ('X', 'Y', 'Z')
    Returns:
        mpo with shape (qubit_num, b0, b1, 2, 2)
    """
    mpo = [[]] * qubit_num
    M = np.zeros((1, 1, 2, 2))
    M[0, 0, :, :] = np.identity(2)
    S = np.zeros((1, 1, 2, 2))
    if axis.upper() == 'Z':
        S[0, 0, :, :] = Sz
    elif axis.upper() == 'X':
        S[0, 0, :, :] = Sx
    elif axis.upper() == 'Y':
        S[0, 0, :, :] = Sy
    else:
        raise RuntimeError(
            'axis must be in [X, Y, Z], recieved {}'.format(axis))
    for i in range(qubit_num):
        if i == qubit_index:
            mpo[i] = S
        else:
            mpo[i] = M
    return mpo