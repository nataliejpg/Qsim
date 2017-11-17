import numpy as np
from .methods import time_evolution, do_matrix_on_state_vector


def action(create_mat_fn, x0, **kwargs):
    """
    Args:
        create_mat_fn to create the unitary
        x0 shape (2,)
    Returns:
        x1 shape (2,)
    """
    x0 = np.array(x0)
    mat = create_mat_fn(**kwargs)
    x1 = do_matrix_on_state_vector(mat, x0)
    return x1


def detuning_sweep(x0, create_mat_fn, start, stop, step, time_step, time,
                   action_before=None, action_after=None, **kwargs):
    """
    Args:
        x0 shape (2,)
        create_mat_fn for unitary
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
    time_points = int(round(time / time_step) + 1)
    result = np.zeros((detuning_points, time_points, 2), dtype=complex)
    for i, d in enumerate(detuning_array):
        if action_before is not None:
            x1 = action_before(x0, detuning=d)
        else:
            x1 = x0  # shape (2,)
        time_array, x2 = time_evolution(
            [x1], create_mat_fn, time_step, time, detuning=d, **kwargs)
        x2 = x2[0]  # shape (time, 2)
        if action_after is not None:
            for j, t in enumerate(time_array):
                x3 = action_after(x2[j], detuning=d)
                result[i, j] = x3
        else:
            result[i] = x2
    return time_array, detuning_array, result
