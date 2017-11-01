import numpy as np


def unitary_evolution(x0, unitary, step, time, **kwargs):
    """
    Unitary evolution with initial state x0, total time and step time defined.

    Args:
        x0 (numpy array shape (m, 2)): initial state
        unitary function for evolution
        step (float): time step size
        time (float): total time for evolution
        **kwargs to be passed to unitary function

    Returns:
        t (numpy array shape (n,)): array of times
        x (numpy array shape (m, n, 2)): state vectors at these times evolved
            by unitary where m number of initial_states and n is number
            of_time steps
    """
    points = round(time / step + 1)
    t = np.linspace(0, time, points)
    x = unitary(t, x0, **kwargs)
    return t, x
