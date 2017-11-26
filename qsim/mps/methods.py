import numpy as np
from collections import defaultdict
from qsim.mps.hamiltonians import create_magnetisation_mpo
from qsim.exact.state_vectors import state_vectors_one_to_many
from qsim.mps.state_vectors import normalise_mps, evaluate_mps
import copy


def do_mpo_on_mps(mpo, mps):
    """
    Args:
        mpo with shape (qubit_num, b0, b1, 2, 2)
        mps with shape (qubit_num, 2, a0, a1)
    Returns:
        mps with shape (qubit_num, 2, a0*b0, a1*b1)
    """
    if len(mpo) != len(mps):
        raise Exception('mpo length != mps length')
    new_mps = [[]] * len(mps)
    for i, o in enumerate(mpo):
        a1, a2 = mps[i].shape[1:]
        b1, b2 = o.shape[:2]
        s = np.tensordot(o, mps[i], axes=1)
        s = np.moveaxis(s, [2, 0, 3, 1, 4], [0, 1, 2, 3, 4])
        new_mps[i] = s.reshape(2, b1 * a1, b2 * a2)
    return new_mps


def projection(mps, axis='Z'):
    """
    Function which finds the projections for a set of mps states onto an
    axis

    Args:
        mps_list: state vectors list with shape (qubit_num, 2, a0, a1)
        axis for the qubits to be projected onto 'X', 'Y' or 'Z' (default Z)

    Returns:
        projection onto axis shape (qubit_num)
    """
    qubit_num = len(mps)
    projections = np.zeros(qubit_num)
    for q in range(qubit_num):
        projection_mat = create_magnetisation_mpo(qubit_num, q, axis=axis)
        final_state = do_mpo_on_mps(projection_mat, mps)
        projections[q] = 2 * np.real(find_overlap(final_state, mps))
    return projections


def find_overlap(mps1, mps2):
    """
    Function which finds overlap of two states in mps form
    Args:
        mps1 shape (qubit_num, 2, a0, a1)
        mps2 shape (qubit_num, 2, a0', a1')
    Returns
        overlap
    """
    if len(mps1) != len(mps2):
        raise Exception('mps1 length != mps2 length')
    new_mps = [[]] * len(mps1)
    for i, s2 in enumerate(mps2):
        s1 = np.conjugate(np.moveaxis(mps1[i], 0, 2))
        new_mps[i] = np.moveaxis(np.tensordot(s1, s2, axes=1), 1, 3)
    state = new_mps[0]
    for s in new_mps[1:]:
        state = np.tensordot(state, s, axes=[[-1, -2], [0, 1]])
    return state[0, 0, 0, 0]


def find_entropy(mps, k=None):
    if k is None:
        k = int(np.floor(len(mps) / 2 + 1))
    mps, sing_vals = normalise_mps(mps, direction='M', k=k)
    entropy = -1 * sum([i**2 * np.log2(i**2) for i in sing_vals if abs(i) > 0])
    return entropy


def time_evolution(initial_mps, mpo_method_list, time_step, time, max_d=None,
                   print_out=False, measurements=None, **kwargs):
    """
    Args:
        list of mps to evolve with dimensions (qubit_num, 2, a0, a1)
        list of mpo creation methods to create mpos to be applied at each step,
            each with with shape (qubit_num, b0, b1, 2, 2)
        time_step size for each method to be applied,
        total time
        max_d (optional) dimension to truncate mps to at each step,
            if not specified mps will grow with each step
        print_out (bool default False), print intermediary states
        measurements (list), what to keep at each time step
        **kwargs to be passed to mpo_methods

    Returns
        outputs dict
    """
    meas = defaultdict(lambda: False)
    meas.update(dict.fromkeys(measurements, True))
    qubit_num = len(initial_mps)
    points = round(time / time_step + 1)
    time_array = np.linspace(0, time, points)
    outputs = {}
    outputs['initial_state'] = initial_mps
    outputs['time'] = time_array
    if meas['intermediary_states']:
        outputs['intermediary_states'] = [[]] * points
        outputs['intermediary_states'][0] = initial_mps
    if meas['entanglement_entropy']:
        outputs['entanglement_entropy'] = [[]] * points
        outputs['entanglement_entropy'][0] = find_entropy(initial_mps)
    if meas['X_projection']:
        outputs['X'] = [[]] * points
        outputs['X'][0] = projection(initial_mps, axis='X')
    if meas['Y']:
        outputs['Y'] = [[]] * points
        outputs['Y'][0] = projection(initial_mps, axis='Y')
    if meas['Z']:
        outputs['Z'] = [[]] * points
        outputs['Z'][0] = projection(initial_mps, axis='Z')
    mpo_list = [mpo_method(qubit_num=qubit_num, t=time_step, **kwargs)
                for mpo_method in mpo_method_list]
    if print_out:
        print('initial_state: ' + state_vectors_one_to_many(
            evaluate_mps(initial_mps), as_str=True))
    mps = copy.deepcopy(initial_mps)
    for i in range(1, points):
        for mpo in mpo_list:
            mps = do_mpo_on_mps(mpo, mps)
            mps = normalise_mps(mps, direction='R')
            mps = normalise_mps(mps, direction='L', max_d=max_d)
            if meas['intermediary_states']:
                outputs['intermediary_states'][i] = mps
            if meas['entanglement_entropy']:
                outputs['entanglement_entropy'][i] = find_entropy(mps)
            if meas['X']:
                outputs['X'][i] = projection(mps, axis='X')
            if meas['Y']:
                outputs['Y'][i] = projection(mps, axis='Y')
            if meas['Z']:
                outputs['Z'][i] = projection(mps, axis='Z')
            if print_out:
                print('time,  state, {}'.format(
                    time_array[i],
                    state_vectors_one_to_many(evaluate_mps(mps), as_str=True)))
    outputs['final_state'] = mps
    return outputs
