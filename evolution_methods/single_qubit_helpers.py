import numpy as np
from helpers import projection


def action(evolution_fn, matrix, step, time, x0, **kwargs):
    """
    Evolves an initial state x0 with a matrix (hamiltonian or unitary) for
    time with steps size step using a function for evolution
    (rK4 or unitary_evolution).

    Args:
        evolution_fn: rk4 or unitary_evolution
        matrix: hamiltonian or unitary function
        step: step size to do evolution, this only matters for rk4 evolution
            as unitary is exact
        x0 (shape (m, n, 2): initial state(s) where m number of initial_states
            and n is number of_time steps
        **kwargs to be passed to matrix via evolution_fn

    Returns:
        x1: final state
    """
    t, x = evolution_fn(x0, matrix, step, time, **kwargs)
    return x[-1]


def detuning_sweep(start, stop, step, x0, execution_fn, execution_mat,
                   time_step, time,
                   action_before=None,
                   action_after=None, **kwargs):
    """
    Function which emulates the sweeps typically done in experiment
    to perform unitary evolution for a variety of detuning values.

    Args:
        start (float): initial detuning in Hz
        stop (float): final detuning in Hz
        step (float): step size for detuning 'sweep'
        x0 (numpy array shape (2,)): initial state
        exacution_fn (function): this should be one from unitary_evolution,
            rk_4 or similar which can be used to evaluate the dynamics
        execution_fn (function): this is the function which is used in the
            execution_fn for evolution, could be a hamiltonian or a unitary
            depending on the execution_fn
        time_step (float): time step size
        time (float): total time for evolution
        action_before: function to do to initial state before evolving with RK$
        action_after: function to do to final state after RK4 evolution
        **kwargs to be passed to exacution_fn

    Returns:
        t (numpy array shape (n,)): array of times
        detuning_array (numpy array shape (n,)): values for detuning
        p(1) (numpy array shape (m, n)): expectation of the qubit being
            in the 1 state for m detunings at n duration of unitary evolution
    """
    detuning_points = int(round(abs(stop - start) / step + 1))
    detuning_array = np.linspace(start, stop, num=detuning_points)
    time_points = int(round(time / time_step + 1))
    z = np.zeros((detuning_points, time_points))
    for i, d in enumerate(detuning_array):
        if action_before is not None:
            x = action_before(x0, detuning=d)
        else:
            x = x0
        t, x = execution_fn(x, execution_mat, time_step, time, detuning=d,
                            **kwargs)
        if action_after is not None:
            x = action_after(x, detuning=d)
        z[i] = projection(x, axis='Z')
    return t, detuning_array, (-z + 1) / 2
