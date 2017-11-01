import numpy as np


def unitary_evolution(x0, unitary, time_step, time, **kwargs):
    """
    Unitary evolution with initial state x0, total time and time_step
    time defined.

    Args:
        x0 (numpy array shape (m, 2)): initial state
        unitary function for evolution
        time_step (float): time step size
        time (float): total time for evolution
        **kwargs to be passed to unitary function

    Returns:
        t (numpy array shape (n,)): array of times
        x (numpy array shape (m, n, 2)): state vectors at these times evolved
            by unitary where m number of initial_states and n is number
            of_time steps
    """
    x0 = np.array(x0)
    points = round(time / time_step + 1)
    time_array = np.linspace(0, time, points)
    inputs = np.arange(x0.shape[0])
    higher_d_mat = make_higher_d_mat(
        unitary, ['inputs', 't'], [inputs, time_array], **kwargs)
    x1 = do_matrix_on_state_vector(higher_d_mat, x0)
    return time_array, x1


def detuning_sweep(x0, unitary, start, stop, step, time_step, time,
                   action_before=None, action_after=None, **kwargs):
    x0 = np.array(x0)
    detuning_points = int(round(abs(stop - start) / step + 1))
    detuning_array = np.linspace(start, stop, num=detuning_points)
    time_points = int(round(time / time_step + 1))
    time_array = np.linspace(0, time, time_points)
    if action_before is not None:
        before_mat = make_higher_d_mat(
            unitary, ['detuning'], [detuning_array], **kwargs)
        x = do_matrix_on_state_vector(higher_d_mat, x0)

    higher_d_mat = make_higher_d_mat(
        unitary, ['detuning', 't'], [detuning_array, time_array], **kwargs)
    x1 = do_matrix_on_state_vector(higher_d_mat, x0)
    return time_array, x1


def do_matrix_on_state_vector(matrix, vector):
    return np.tensordot(matrix, vector, axes=1)


def make_higher_d_mat(mat_creation_fn, kwargnames, kwargvalues,
                      qubit_num=1, **kwargs):
    shape_extension = [np.array(l).shape[0] for l in kwargvalues]
    new_shape = shape_extension + [2**qubit_num, 2**qubit_num]
    new_mat = np.zeros(new_shape)
    it = np.nditer(np.zeros(shape_extension), flags=['multi_index'])
    kw_d = dict.fromkeys(kwargnames)
    kw_d.update(kwargs)
    while not it.finished:
        for i, j in enumerate(list(it.multi_index)):
            kw_d[kwargnames[i]] = kwargvalues[i][j]
        new_mat[it.multi_index] = mat_creation_fn(**kw_d)
        it.iternext()
    return new_mat
