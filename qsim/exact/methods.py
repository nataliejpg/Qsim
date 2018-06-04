import numpy as np
from qsim.helpers import make_higher_d_mat, dagger, Sx, Sy, Sz, I
from .state_vectors import state_vectors_one_to_many, normalise_state_vector
from decimal import Decimal


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


def do_matrix_on_state_vector(matrix, vector, axes=None):
    """
    Args:
        matrix shape (abc, 2**qubit_num, 2**qubit_num)
        vector shape (def, 2**qubit_num)
        axes to sum over, defaults to (-1, -1)
    Returns:
        vector shape (abcdef, 2**qubit_num)
    """
    vec_axes = len(vector.shape)
    result = np.tensordot(matrix, vector, axes=axes or (-1, -1))
    return np.moveaxis(result, -vec_axes, -1)


def time_evolution(x0, create_mat_fn, time_step, time,
                   print_out=False, keep_intermediary=True, **kwargs):
    """
    Unitary evolution with initial state x0, total time and time_step
    time defined.

    Args:
        x0 (numpy array shape (m, 2**qubit_num))): initial state
        create_mat_fn to create unitary function for evolution
        time_step (float): time step size
        time (float): total time for evolution
        print_out (bool default False), print intermediary states
        keep_intermediary (bool default True) keep intermediary states and
            return them
        **kwargs to be passed to unitary function

    Returns:
        t (numpy array shape (n,)) (omitted if keep_intermediary False)
        x1 (numpy array shape (m, n, 2**qubit_num)): state vectors at
            these times evolved by unitary where m number of initial_states
            and n is number of_time steps
            (n is omitted if keep_intermediary False)
    """
    x0 = np.array(x0)
    qubit_num = int(np.log2(x0.shape[1]))
    if float(Decimal(str(time)) % Decimal(str(time_step))) != 0:
        raise RuntimeError('Could not find int number of points to fill time')
    points = round(time / time_step + 1)
    time_array = np.linspace(0, time, points, endpoint=True)
    inputs = x0.shape[0]
    if keep_intermediary:
        outputs = np.zeros((x0.shape[0], points, 2**qubit_num), dtype=complex)
    else:
        outputs = np.zeros((x0.shape[0], 2**qubit_num), dtype=complex)

    if print_out:
        for i in range(inputs):
            print('input {}, initial_state: {}'.format(
                i, x0[i]))
    if keep_intermediary:
        higher_d_mat = make_higher_d_mat(
            create_mat_fn, ['t'], [time_array], qubit_num=qubit_num, **kwargs)
        for i in range(inputs):
            a = do_matrix_on_state_vector(higher_d_mat, x0[i])
            # a = np.array([normalise_state_vector(j) for j in a])
            outputs[i] = a
            if print_out:
                print('input {}, time {}, state, {}'.format(
                      i, time_array[-1], a[-1]))
        return time_array, outputs
    else:
        mat = create_mat_fn(t=time, qubit_num=qubit_num, **kwargs)
        for i in range(inputs):
            a = do_matrix_on_state_vector(mat, x0[i])
            # outputs[i] = normalise_state_vector(a)
            outputs[i] = a
        return None, outputs


def time_evolution_trotter(x0, unitary_method_list, time_step, time,
                           print_out=False, keep_intermediary=True, **kwargs):
    """
    Args:
        x0 (numpy array shape (m, 2**qubit_num)): initial state
        unitary_method_list: list of functions to create unitaries to
            apply at each step
        time_step (float): time step size
        time (float): total time for evolution
        print_out (bool default False), print intermediary states
        keep_intermediary (bool default True) keep intermediary states and
            return them
        periodic
        **kwargs to be passed to mpo_methods

    Returns
        t (numpy array shape (n,)) (omitted if keep_intermediary False)
        x1 (numpy array shape (m, n, 2**qubit_num)): state vectors at
            these times evolved by unitary where m number of initial_states
            and n is number of_time steps
            (n is omitted if keep_intermediary False)
    """
    x0 = np.array(x0)
    qubit_num = int(np.log2(x0.shape[1]))
    if float(Decimal(str(time)) % Decimal(str(time_step))) != 0:
        raise RuntimeError('Could not find int number of points to fill time')
    points = int(round(time / time_step + 1))
    time_array = np.linspace(0, time, points, endpoint=True)
    inputs = x0.shape[0]
    if keep_intermediary:
        outputs = np.zeros((x0.shape[0], points, 2**qubit_num), dtype=complex)
        # outputs[:, 0, :] = np.array([normalise_state_vector(x) for x in x0])
        outputs[:, 0, :] = x0
    else:
        outputs = np.zeros((x0.shape[0], 2**qubit_num), dtype=complex)
        # outputs[:] = np.array([normalise_state_vector(x) for x in x0])
        outputs[:] = x0
    # unitary_list = []
    # for i, unitary_method in unitary_method_list:
    #     u = unitary_method(t=time_step, qubit_num=2, periodic=False, **kwargs)
    #     mat = np.eye(2**(qubit_num))
    #     for j in range(qubit_num - 1):
    #         mat = np.dot(mat, kron(np.eye(2**(j)),
    #                                np.kron(u, np.eye(2**(qubit_num - j - 2)))))
    #     if periodic:
    #         end_mat =
    #         mat = np.dot(mat, end_mats[i])
    #     unitary_list.append(mat)
    unitary_list = [u_method(t=time_step, qubit_num=qubit_num, **kwargs)
                    for u_method in unitary_method_list]
    if print_out:
        for i in range(inputs):
            print('input {}, initial_state: {}'.format(
                i, x0[i]))

    for j in range(1, points):
        for i in range(inputs):
            if keep_intermediary:
                state = outputs[i, j - 1]
            else:
                state = outputs[i]
            for unitary in unitary_list:
                state = do_matrix_on_state_vector(unitary, state)
            if keep_intermediary:
                # outputs[i, j] = normalise_state_vector(state)
                outputs[i, j] = state
            else:
                # outputs[i] = normalise_state_vector(state)
                outputs[i] = state
            if print_out:
                print('input {}, step {}, time {}, state, {}'.format(
                    i, j + 1, time_array[j], state))
    if keep_intermediary:
        return time_array, outputs
    else:
        return None, outputs
