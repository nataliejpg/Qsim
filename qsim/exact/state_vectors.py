import numpy as np
from qsim.helpers import nth_root


def normalise_state_vector(state_vector):
    """
    Given a state vector returns vector with unit length and |00>
    coefficient only real.
    Args:
        state vector 2**L long where L is numebr of qubits
    Returns:
        normalised state vector 2**L long
    """
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
        binary_state: list of qubit states (eg '100')

    Return:
        index of state vector this state corresponds to (eg 4)
    """
    n = 0
    binary_list = [int(q) for q in binary_state]
    for i, q in enumerate(binary_list):
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
    s = 1
    for q in qubit_states:
        s = np.kron(s, q)
    return s


def create_random_state(qubit_num):
    state = np.random.rand(2**qubit_num)
    state = normalise_state_vector(state)
    return state


def create_random_unentangled_state(qubit_num):
    state_list = []
    for q in range(int(qubit_num)):
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        state_list.append(np.array([alpha, beta]))
    return state_vectors_many_to_one(state_list)


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
