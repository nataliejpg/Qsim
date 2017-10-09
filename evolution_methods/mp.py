import numpy as np
from helpers import state_vectors_one_to_many
from mps import normalise_mps, evaluate_mps


def do_mpo_on_mps(mpo, mps):
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


def do_evolution(mps, mpo_list, step_num, max_d=None):
    print('initial_state ' + state_vectors_one_to_many(evaluate_mps(mps),
                                                       as_str=True))
    for i in range(step_num):
        for mpo in mpo_list:
            mps = do_mpo_on_mps(mpo, mps)
        mps = normalise_mps(mps, direction='R', max_d=max_d)
        mps = normalise_mps(mps, direction='L')
        print('{} state is {}'.format(
            i, state_vectors_one_to_many(evaluate_mps(mps), as_str=True)))
