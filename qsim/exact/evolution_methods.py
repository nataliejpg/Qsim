import numpy as np
from qsim.helpers import make_higher_d_mat
from .state_vectors import state_vectors_one_to_many


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
                   print_out=True, keep_intermediary=True, **kwargs):
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
    points = round(time / time_step + 1)
    time_array = np.linspace(0, time, points)
    inputs = np.arange(x0.shape[0])
    if keep_intermediary:
        outputs = np.zeros((x0.shape[0], points, 2**qubit_num), dtype=complex)
    else:
        outputs = np.zeros((x0.shape[0], 2**qubit_num), dtype=complex)
    if keep_intermediary:
        higher_d_mat = make_higher_d_mat(
            create_mat_fn, ['t'], [time_array], qubit_num=qubit_num, **kwargs)
        for i in inputs:
            a = do_matrix_on_state_vector(higher_d_mat, x0[i])
            outputs[i] = a
        return time_array, outputs
    else:
        mat = create_mat_fn(t=time, qubit_num=qubit_num, **kwargs)
        for i in inputs:
            a = do_matrix_on_state_vector(mat, x0[i])
            outputs[i] = a
        return outputs


def time_evolution_trotter(x0, unitary_method_list, time_step, time,
                           print_out=False, keep_intermediary=True, **kwargs):
    """
    Args:
        x0 (numpy array shape (m, 2**qubit_num)): initial state
        unitary_method_list: list of functions to create unitaries to
            apply at each step, each should have shape (2)
        time_step (float): time step size
        time (float): total time for evolution
        print_out (bool default False), print intermediary states
        keep_intermediary (bool default True) keep intermediary states and
            return them
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
    points = round(time / time_step + 1)
    time_array = np.linspace(0, time, points)
    inputs = np.arange(x0.shape[0])
    if keep_intermediary:
        outputs = np.zeros((x0.shape[0], points, 2**qubit_num), dtype=complex)
        outputs[:, 0, :] = x0
    else:
        outputs = np.zeros((x0.shape[0], 2**qubit_num), dtype=complex)
        outputs = x0
    unitary_list = [u_method(t=time_step, qubit_num=qubit_num, **kwargs)
                    for u_method in unitary_method_list]
    if print_out:
        for i in inputs:
            print('input {}, initial_state: '.format(
                i, state_vectors_one_to_many(x0[i], as_str=True)))

    for j in range(1, points):
        for i in inputs:
            if keep_intermediary:
                state = outputs[i, j - 1]
            else:
                state = outputs[i]
            for unitary in unitary_list:
                state = do_matrix_on_state_vector(
                    unitary, state)
            if keep_intermediary:
                outputs[i, j] = state
            else:
                outputs[i] = state
            if print_out:
                print('input {}, step {}, time,  state, {}'.format(
                    i, j + 1, time_array[j],
                    state_vectors_one_to_many(state, as_str=True)))
    if keep_intermediary:
        return time_array, outputs
    else:
        return outputs
