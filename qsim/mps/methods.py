import numpy as np
from collections import defaultdict
from qsim.mps.hamiltonians import create_magnetisation_mpo
from qsim.exact.state_vectors import state_vectors_one_to_many
from qsim.mps.state_vectors import normalise_mps, evaluate_mps
import copy


def do_mpo_on_mps(mpo, mps):
    """
    Applies the mpo to the mps and returns the new mps without truncation
    Args:
        mpo: mpo array of length "qubit_num" with (local)
            shape (b_{k-1}, b_{k}, sigma_{k}^{'}, sigma_{k})
        mps: mps array of length "qubit_num" with (local)
            shape (sigma_{k}, a_{k-1}, a_{k})
    Returns:
        new_mps: mps array of length "qubit_num" with (local)
            shape (sigma_{k}, b_{k-1}*a_{k-1}, b_{k}*a_{k})
    """
    if len(mpo) != len(mps):
        raise Exception('mpo length != mps length')
    new_mps = [[]] * len(mps)
    for i, o in enumerate(mpo):
        d = mps[i].shape[0]
        a1, a2 = mps[i].shape[1:]
        b1, b2 = o.shape[:2]
        s = np.tensordot(o, mps[i], axes=1)
        s = np.moveaxis(s, [2, 0, 3, 1, 4], [0, 1, 2, 3, 4])
        new_mps[i] = s.reshape(d, b1 * a1, b2 * a2)
    return new_mps


def projection(mps, axis='Z'):
    """
    Function which finds the projections for a set of mps states onto an axis
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
        mps1: mps array of length "qubit_num" with (local)
            shape (sigma_{k}, a_{k-1}, a_{k})
        mps2: mps array of length "qubit_num" with (local)
            shape (sigma_{k}, a_{k-1}, a_{k})
    Returns:
        norm/overlap
    """
    if len(mps1) != len(mps2):
        raise Exception('mps1 length != mps2 length')
    norm = np.ones([1, 1])
    for i in range(len(mps1)):
        norm = np.tensordot(np.conjugate(mps1[i]), np.tensordot(
            norm, mps2[i], axes=([1], [1])), axes=([1, 0], [0, 1]))
    return norm[0, 0]


def find_entropy(mps, k=None):
    """
    Finds the entanglement entropy of the mps when cutting the system
    to left of site k
    Args:
        mps: mps array of length "qubit_num" with (local)
            shape (sigma_{k}, a_{k-1}, a_{k})
        k: cutting to the left of site k
            (default: middle of the system => entanglement entropy)
    Returns:
        entanglement entropy
    """
    if k is None:
        k = int(np.floor(len(mps) / 2))
    mps, sing_vals = normalise_mps(mps, direction='S', k=k)
    entropy = -1 * sum([i**2 * np.log2(i**2) for i in sing_vals if abs(i) > 0])
    return entropy


def time_evolution(initial_mps, mpo_method_list, time_step, time, max_d=None,
                   print_out=False, measurements=None, **kwargs):
    """
    Evolve mps over a certain time ("time") with finite time-step ("time_step")
    with unitary mpo specified by mpo_method_list
    Args:
        initial_mps: mps array of length "qubit_num" with (local)
            shape (sigma_{k}, a_{k-1}, a_{k})
        mpo_metod_list: list of mpo methods to create mpo array of length
            "qubit_num" with (local) shape
            (b_{k-1}, b_{k}, sigma_{k}^{'}, sigma_{k})
            to be applied at each time step
        time_step: size for each method to be applied
        total time
        max_d: (optional) dimension to truncate mps to at each step,
            if not specified mps will grow with each step
        print_out: (bool default False), print intermediary states
        measurements: (list), what to keep at each time step
        **kwargs to be passed to mpo_methods

    Returns
        outputs: dict
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
