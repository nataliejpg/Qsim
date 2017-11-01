import numpy as np
from qsim.helpers import state_vectors_one_to_many
from state_vectors import normalise_mps, evaluate_mps


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


def do_evolution(mpo_list, mps, step_num, max_d=None):
    """
    Args:
        list of mpos to apply at each step,
            each with with shape (qubit_num, b0, b1, 2, 2)
        mps with dimensions (qubit_num, 2, a0, a1)
        number of steps to apply
        max_d (optional) dimension to truncate mps to at each step,
            if not specified mps will grow with each step
    Returns
        mps with dimensions (qubit_num, 2, max_d, max_d)
    """
    print('initial_state ' + state_vectors_one_to_many(evaluate_mps(mps),
                                                       as_str=True))
    for i in range(step_num):
        for mpo in mpo_list:
            mps = do_mpo_on_mps(mpo, mps)
        mps = normalise_mps(mps, direction='R', max_d=max_d)
        mps = normalise_mps(mps, direction='L')
        print('{} state is {}'.format(
            i, state_vectors_one_to_many(evaluate_mps(mps), as_str=True)))
