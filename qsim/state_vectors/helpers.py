import numpy as np


def dagger(A):
    return np.conjugate(np.transpose(A))


def nth_root(number, root):
    coeff = [0] * (root + 1)
    coeff[-1] = -1 * number
    coeff[0] = 1
    return round(np.roots(coeff)[0], 5)


def normalise_state_vector(state_vector):
    state_vector = np.array(state_vector)
    if len(state_vector.shape) > 2:
        raise RuntimeError('state_vector provided has shape '
                           '{}'.format(state_vector.shape))
    mag = np.linalg.norm(np.array(state_vector))
    state_vector = state_vector / mag
    global_phase = np.angle(state_vector[0])
    state_vector = state_vector * np.exp(-global_phase * 1j)
    return state_vector


def state_vector_index_to_binary(index, qubit_num):
    """
    Function which given the number of qubits and the index in the state
    vector prints the state of the system that this index corresponds to
    in the computational basis

    Args:
        index: index to find the computational state for (eg 1)
        qubit_num: number of qubits in the system (eg 2)

    Returns:
        string of qubit states (eg '01')
    """
    return("{d:0{num}b}".format(num=qubit_num, d=index))


def binary_to_state_vector_index(binary_state):
    """
    Function which given a list of qubit states will generate the index in
    the state vector which this corresponds to

    Args:
        binary_state: list of qubit states (eg [1, 0, 0])

    Return:
        index of state vector this state corresponds to (eg 4)
    """
    n = 0
    for i, q in enumerate(binary_state):
        n += q * 2**(len(binary_state) - i - 1)
    return n


def state_vectors_many_to_one(qubit_states):
    """
    Function which given a list of state vectors per qubit returns the multi
    qubit state vector

    Args:
        qubit_states: list of state vecors for each qubit in the system
            (eg [[0.7, 0.7], [1, 0]])

    Returns:
        state vector of system (eg [0.7, 0, 0.7, 0])
    """
    s = np.ones(1)
    for q in qubit_states:
        s = np.kron(s, q)
    return s


def state_vectors_one_to_many(state_vector, as_str=False):
    """
    Function which given a state vector breaks it down into the
    composite states of each qubit

    Args:
        state_vector (eg [0.7, 0, 0.7, 0])

    Returns:
        qubit_states shape (m, n, 2) where m is the number of states in
            superposition and n is the number of qubits
            (eg [[0.7, 0], [0.7, 0],
                 [0, 0.7], [0.7, 0]])
    """
    qubit_num = int(np.log2(len(state_vector)))
    non_zero_indices = np.argwhere(
        abs(np.array(state_vector)) > 0.001).flatten()
    s = np.zeros((len(non_zero_indices), qubit_num, 2), dtype=complex)
    binary_s = ''
    for i, index in enumerate(non_zero_indices):
        binary = state_vector_index_to_binary(index, qubit_num)
        binary_s += '+ {0:.1f} |{1}>'.format(state_vector[index], binary)
        for j in range(qubit_num):
            factor = nth_root(state_vector[index], qubit_num)
            if binary[j] is '0':
                s[i, j, 0] = factor
            else:
                s[i, j, 1] = factor
    if as_str:
        return(binary_s[2:])
    else:
        return s


def projection(state_array, axis='Z'):
    """
    Function which finds the projections for a set of state vectors onto an
    axis

    Args:
        state_array: state vectors array with shape (m, 2**n) where
            m is the number of states and n is the number of qubits
        axis for the qubits to be projected onto 'X', 'Y' or 'Z' (default Z)

    Returns:
        projection onto axis shape (m, n)
    """
    state_array = np.array(state_array)
    qubit_num = int(np.log2(state_array.shape[1]))
    projections = np.zeros((state_array.shape[0], qubit_num))
    if axis.upper() == 'Z':
        S = np.array([[1, 0], [0, -1]])
    elif axis.upper() == 'X':
        S = np.array([[0, 1], [1, 0]])
    elif axis.upper() == 'Y':
        S = np.array([[0, -1j], [1j, 0]])
    else:
        raise Exception('axis be X, Y or Z, received {}'.format(axis.upper()))
    projection_matrices = [[]] * qubit_num
    identity = np.identity(2)
    for i in range(qubit_num):
        mat = 1
        for j in range(qubit_num):
            if i == j:
                mat = np.kron(mat, S)
            else:
                mat = np.kron(mat, identity)
        projection_matrices[i] = mat
    for i, state in enumerate(state_array):
        state_dagger = dagger(state)
        for j in range(qubit_num):
            projections[i][j] = np.dot(state_dagger, np.dot(projection_matrices[j], state))

    return projections

# def projection(state_array, axis='Z'):
#     """
#     Function which finds the projections for a set of state vectors onto an
#     axis

#     Args:
#         state_array: state vectors array with shape (m, 2**n) where
#             m is the number of states and n is the number of qubits
#         axis for the qubits to be projected onto 'X', 'Y' or 'Z' (default Z)

#     Returns:
#         projection onto axis shape (m, n)
#     """
#     state_array = np.array(state_array)
#     qubit_num = int(np.log2(state_array.shape[1]))
#     projections = np.zeros((state_array.shape[0], qubit_num))
#     if axis is 'Z':
#         S = np.array([[1, 0], [0, -1]])
#     elif axis is 'X':
#         S = np.array([[0, 1], [1, 0]])
#     elif axis is 'Y':
#         S = np.array([[0, -1j], [1j, 0]])
#     else:
#         raise Exception('axis nums be X, Y or Z, received {}'.format(axis))
#     for i, state in enumerate(state_array):
#         superposed_states = state_vectors_one_to_many(state)
#         q_states = np.zeros((qubit_num, 2), dtype=complex)
#         for j, superposed_state in enumerate(superposed_states):
#             for k, q_state in enumerate(superposed_state):
#                 # if q_state[0] > 0:
#                 # mag = np.sqrt(q_states[k]**2 + q_state**(2 * qubit_num))
#                 # phase = np.angle(q_states[k])
#                 q_states[k] = np.sqrt(
#                     q_states[k]**2 + q_state**(2 * qubit_num))
#         q_states_dag = [dagger(st) for st in q_states]
#         for l, q in enumerate(q_states):
#             proj = np.dot(q_states_dag[l], np.dot(S, q))
#             print(proj)
#             projections[i, l] += np.real(proj)
#     return projections
