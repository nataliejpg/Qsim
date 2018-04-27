import numpy as np
from qsim.helpers import make_higher_d_mat
from qsim.exact.evolution_methods import do_matrix_on_state_vector


def time_evolution(x0, create_mat_fn, time_step, time, **kwargs):
    """
    Funtion which finds the fourth order Runge Kutta numerical solution to the
    Schrodinger equation.

    Args:
        x0 (numpy array shape (m, 2)): initial state
        create_mat_fn to create hamiltonian function to be used in TDSE
        time_step (float): time step size
        time (float): total time for evolution
        **kwargs to be passed to hadamard function

    Returns:
        t (numpy array shape (n)): array of times
        x1 (numpy array shape (m, n, 2)): state vectors at these times evolved
            under TDSE where m number of initial_states and n is number
            of_time steps
    """
    x0 = np.array(x0)
    points = int(round(time / time_step + 1))
    x1 = np.zeros((x0.shape[0], points, 2), dtype=np.complex)
    inputs = np.arange(x0.shape[0])
    time_array = np.zeros(points)
    x1[:, 0] = x0
    for i in range(1, points):
        k1_mat = make_higher_d_mat(create_mat_fn, ['inputs'],
                                   [inputs], t=time_array[i - 1], **kwargs)
        k1_x = x0[:, i - 1]
        k1 = do_matrix_on_state_vector(k1_mat, k1_x)
        k2_mat = make_higher_d_mat(
            create_mat_fn, ['inputs'], [inputs],
            t=(time_array[i - 1] + time_step / 2), **kwargs)
        k2_x = x0[:, i - 1] + (time_step / 2) * k1
        k2 = do_matrix_on_state_vector(k2_mat, k2_x)
        k3_mat = k2_mat
        k3_x = x0[:, i - 1] + (time_step / 2) * k2
        k3 = do_matrix_on_state_vector(k3_mat, k3_x)
        k4_mat = make_higher_d_mat(
            create_mat_fn, ['inputs'], [inputs],
            t=(time_array[i - 1] + time_step), **kwargs)
        k4_x = x0[:, i - 1] + time_step * k3
        k4 = do_matrix_on_state_vector(k4_mat, k4_x)
        x1[:, i] = x1[:, i - 1] + (time_step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        time_array[i] = time_array[i - 1] + time_step
    return time_array, x1


def detuning_sweep(x0, create_mat_fn, start, stop, step, time_step, time,
                   action_before=None, action_after=None, **kwargs):
    """
    np uses unitary action not rk4 as no rk4 action is defined
    Args:
        x0 shape (2,)
        create_mat_fn for hamiltonian
        start demod_freq
        stop demod_freq
        step demod_freq
        time_step
        time
        action_before (fn which takes vector shape (2,) and returns same shape)
        action_after (as above)
        **kwargs to pass to create_mat_fn
    Returns:
        detuning_array
        time_array
        result shape (detuning_num, time_num, 2): state after action_before,
            evolution, action_after
    """
    x0 = np.array(x0)
    detuning_points = int(round(abs(stop - start) / step + 1))
    detuning_array = np.linspace(start, stop, num=detuning_points)
    time_points = int(round(time / time_step + 1))
    time_array = np.linspace(0, time, time_points)
    result = np.zeros((detuning_points, time_points, 2))
    for i, d in enumerate(detuning_array):
        if action_before is not None:
            x1 = action_before(x0, detuning=d)
        else:
            x1 = x0  # shape (2,)
        x2 = time_evolution([x1], create_mat_fn, time_step, time,
                            detuning=d, **kwargs)[0]  # shape (time, 2)
        if action_after is not None:
            for j in range(time_points):
                x3 = action_after(x2[j], detuning=d)
                result[i, j] = x3  # shape (2,)
        else:
            result[i] = x2
    return detuning_array, time_array, result
