import numpy as np


def create_specific_mpo(matrix):
    raise NotImplementedError


def create_heisenberg_mpo(qubit_num, J, h, Jz):
    mpo = [[]] * qubit_num
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    Splus = Sx + 1j * Sy
    Sminus = Sx - 1j * Sy
    I = np.eye(2)

    M = np.zeros((5, 5, 2, 2), dtype=complex)

    M[0, 0, :, :] = I
    M[1, 0, :, :] = Splus
    M[2, 0, :, :] = Sminus
    M[3, 0, :, :] = Sz
    M[4, 0, :, :] = h * Sz

    M[4, 1, :, :] = J / 2 * Sminus
    M[4, 2, :, :] = J / 2 * Splus
    M[4, 3, :, :] = Jz * Sz
    M[4, 4, :, :] = I

    for i in range(1, qubit_num - 1):
        mpo[i] = M

    mpo[0] = np.zeros((1, 5, 2, 2), dtype=complex)
    mpo[-1] = np.zeros((5, 1, 2, 2), dtype=complex)
    mpo[0][0, :, :, :] = M[4, :, :, :]
    mpo[-1][:, 0, :, :] = M[:, 0, :, :]

    return mpo


def create_magnetisation_mpo(qubit_num, qubit_index):
    mpo = [[]] * qubit_num
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    M = np.zeros((1, 1, 2, 2))
    M[0, 0, :, :] = np.identity(2)
    S = np.zeros((1, 1, 2, 2))
    S[0, 0, :, :] = Sz
    for i in range(qubit_num):
        if i == qubit_index:
            mpo[i] = S
        else:
            mpo[i] = M
    return mpo
