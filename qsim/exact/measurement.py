import numpy as np
from qsim.helpers import dagger, Sx, Sy, Sz, I


def find_overlap(state_vector1, state_vector2):
    """
    Function which finds overlap of two states (basically dot product)
    Args:
        state_vector1 shape 2**L
        state_vector2 shape 2**L
    Returns
        overlap
    """
    state_vector1_dagger = dagger(state_vector1)
    return np.dot(state_vector1_dagger, state_vector2)


def projection(state_array, axis='Z'):
    """
    Function which finds the projections for a set of state vectors onto an
    axis

    Args:
        state_array: state vectors array with shape (m, 2**qubit_num) where
            m is the number of states and n is the number of qubits
        axis for the qubits to be projected onto 'X', 'Y' or 'Z' (default Z)

    Returns:
        projection onto axis shape (m, qubit_num)
    """
    state_array = np.array(state_array)
    qubit_num = int(np.log2(state_array.shape[1]))
    projections = np.zeros((state_array.shape[0], qubit_num), dtype=complex)
    if axis.upper() == 'Z':
        S = Sz
    elif axis.upper() == 'X':
        S = Sx
    elif axis.upper() == 'Y':
        S = Sy
    else:
        raise Exception('axis be X, Y or Z, received {}'.format(axis.upper()))
    projection_matrices = [[]] * qubit_num
    for i in range(qubit_num):
        mat = 1
        for j in range(qubit_num):
            if i == j:
                mat = np.kron(mat, S)
            else:
                mat = np.kron(mat, I)
        projection_matrices[i] = mat
    for i, state in enumerate(state_array):
        for j in range(qubit_num):
            projections[i][j] = 2 * find_overlap(
                state, np.dot(projection_matrices[j], state))

    return projections
