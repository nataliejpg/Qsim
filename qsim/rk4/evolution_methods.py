import numpy as np


def RK4(x0, hamiltonian, step, time, **kwargs):
    """
    Funtion which finds the fourth order Runge Kutta numerical solution to the
    Schrodinger equation.

    Args:
        x0 (numpy array shape (m, 2)): initial state
        hamiltonian function to be used in TDSE
        step (float): time step size
        time (float): total time for evolution
        **kwargs to be passed to hadamard function

    Returns:
        t (numpy array shape (n)): array of times
        x (numpy array shape (m, n, 2)): state vectors at these times evolved
            under TDSE where m number of initial_states and n is number
            of_time steps
    """
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    time_step_num = int(round(time / step + 1))
    x = np.zeros((x0.shape[0], time_step_num, 2), dtype=np.complex)
    t = np.zeros(time_step_num)
    x[:, 0] = x0
    for i in range(1, time_step_num):
        k1 = -1j * hamiltonian(t[i - 1], x[:, i - 1], **kwargs)
        k2 = -1j * hamiltonian(t[i - 1], x[:, i - 1] +
                               (step / 2) * k1, **kwargs)
        k3 = -1j * hamiltonian(t[i - 1], x[:, i - 1] +
                               (step / 2) * k2, **kwargs)
        k4 = -1j * hamiltonian(t[i - 1], x[:, i - 1] + step * k3, **kwargs)
        x[:, i] = x[:, i - 1] + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i] = t[i - 1] + step
    return t, x
