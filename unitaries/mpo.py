import numpy as np


def create_even_heisenberg_unitary_mpo(qubit_num, J, h, Jz, delta_t):
    mpo = [[]] * qubit_num
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    Splus = Sx + 1j * Sy
    Sminus = Sx - 1j * Sy
    I = np.eye(2)

    M0 = np.zeros((1, 5, 2, 2), dtype=complex)
    M1 = np.zeros((5, 1, 2, 2), dtype=complex)
    Mend = np.zeros((1, 1, 2, 2), dtype=complex)

    M0[0, 0, :, :] = -1j * delta_t * h * Sz + 0.5 * I
    M0[0, 1, :, :] = -1j * delta_t * J / 2 * Sminus
    M0[0, 2, :, :] = -1j * delta_t * J / 2 * Splus
    M0[0, 3, :, :] = -1j * delta_t * Jz * Sz
    M0[0, 4, :, :] = I

    M1[0, 0, :, :] = -1j * delta_t * I
    M1[1, 0, :, :] = -1j * delta_t * Splus
    M1[2, 0, :, :] = -1j * delta_t * Sminus
    M1[3, 0, :, :] = -1j * delta_t * Sz
    M1[4, 0, :, :] = -1j * delta_t * -h * Sz + 0.5 * I

    Mend[0, 0, :, :] = I

    for i in range(qubit_num):
        if (i % 2 == 0):
            mpo[i] = M0
        else:
            mpo[i] = M1
    if qubit_num % 2 == 1:
        mpo[-1] = Mend
    return mpo


def create_odd_heisenberg_unitary_mpo(qubit_num, J, h, Jz, delta_t):
    mpo = [[]] * qubit_num
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    Splus = Sx + 1j * Sy
    Sminus = Sx - 1j * Sy
    I = np.eye(2)

    M0 = np.zeros((1, 5, 2, 2), dtype=complex)
    M1 = np.zeros((5, 1, 2, 2), dtype=complex)
    Mend = np.zeros((1, 1, 2, 2), dtype=complex)

    M0[0, 0, :, :] = -1j * delta_t * h * Sz + 0.5 * I
    M0[0, 1, :, :] = -1j * delta_t * J / 2 * Sminus
    M0[0, 2, :, :] = -1j * delta_t * J / 2 * Splus
    M0[0, 3, :, :] = -1j * delta_t * Jz * Sz
    M0[0, 4, :, :] = I

    M1[0, 0, :, :] = -1j * delta_t * I
    M1[1, 0, :, :] = -1j * delta_t * Splus
    M1[2, 0, :, :] = -1j * delta_t * Sminus
    M1[3, 0, :, :] = -1j * delta_t * Sz
    M1[4, 0, :, :] = -1j * delta_t * -h * Sz + 0.5 * I

    Mend[0, 0, :, :] = I

    mpo[0] = Mend
    for i in range(1, qubit_num):
        if (i % 2 == 1):
            mpo[i] = M0
        else:
            mpo[i] = M1
    if qubit_num % 2 == 0:
        mpo[-1] = Mend
    return mpo


def create_heisenberg_unitary_mpo(qubit_num, J, h, Jz, delta_t):
    mpo = [[]] * qubit_num
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    Splus = Sx + 1j * Sy
    Sminus = Sx - 1j * Sy
    I = np.eye(2)

    M = np.zeros((5, 5, 2, 2), dtype=complex)

    M[0, 0, :, :] = I
    M[1, 0, :, :] = -1j * delta_t * Splus
    M[2, 0, :, :] = -1j * delta_t * Sminus
    M[3, 0, :, :] = -1j * delta_t * Sz
    M[4, 0, :, :] = -1j * delta_t * h * Sz + 0.5 * I

    M[4, 1, :, :] = -1j * delta_t * J / 2 * Sminus
    M[4, 2, :, :] = -1j * delta_t * J / 2 * Splus
    M[4, 3, :, :] = -1j * delta_t * Jz * Sz
    M[4, 4, :, :] = I

    for i in range(1, qubit_num - 1):
        mpo[i] = M

    mpo[0] = np.zeros((1, 5, 2, 2), dtype=complex)
    mpo[-1] = np.zeros((5, 1, 2, 2), dtype=complex)
    mpo[0][0, :, :, :] = M[4, :, :, :]
    mpo[-1][:, 0, :, :] = M[:, 0, :, :]

    return mpo


def create_magnetisation_unitary_mpo(qubit_num, qubit_index, delta_t):
    mpo = [[]] * qubit_num
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    M = np.zeros((1, 1, 2, 2))
    M[0, 0, :, :] = I
    S = np.zeros((1, 1, 2, 2))
    S[0, 0, :, :] = -1j * delta_t * Sz + I
    for i in range(qubit_num):
        if i == qubit_index:
            mpo[i] = S
        else:
            mpo[i] = M
    return mpo
