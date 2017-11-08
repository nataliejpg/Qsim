import numpy as np
from qsim.exact.state_vectors import state_vectors_one_to_many
from qsim.mps.state_vectors import normalise_mps, evaluate_mps


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


def time_evolution(mps_list, mpo_method_list, time_step, time, max_d=None,
                   print_out=False, keep_intermediary=True, **kwargs):
    """
    Args:
        list of mps to evolve with dimensions (m, qubit_num, 2, a0, a1)
        list of mpo creation methods to create mpos to be applied at each step,
            each with with shape (qubit_num, b0, b1, 2, 2)
        time_step size for each method to be applied,
        total time
        max_d (optional) dimension to truncate mps to at each step,
            if not specified mps will grow with each step
        print_out (bool default False), print intermediary states
        keep_intermediary (bool default True) keep intermediary states and
            return them
        **kwargs to be passed to mpo_methods

    Returns
        t (numpy array shape (n,)) (omitted if keep_intermediary False)
        list of mps with dimensions (m, n, qubit_num, 2, max_d, max_d) where
            m number of initial_states and n is number of_time steps
            (n is omitted if keep_intermediary False)
    """
    qubit_num = len(mps_list[0])
    points = round(time / time_step + 1)
    time_array = np.linspace(0, time, points)
    inputs = np.arange(len(mps_list))
    if keep_intermediary:
        outputs = [[None for _ in range(points)] for _ in range(len(mps_list))]
        for i in inputs:
            outputs[i][0] = mps_list[i]
    else:
        outputs = [None for _ in range(len(mps_list))]
        outputs = mps_list
    mpo_list = [mpo_method(qubit_num=qubit_num, t=time_step, **kwargs)
                for mpo_method in mpo_method_list]
    if print_out:
        for i in inputs:
            print('input {}, initial_state: '.format(
                i, state_vectors_one_to_many(
                    evaluate_mps(mps_list[i]), as_str=True)))
    for j in range(1, points):
        for i in inputs:
            if keep_intermediary:
                mps = outputs[i][j - 1]
            else:
                mps = outputs[i]
            for mpo in mpo_list:
                mps = do_mpo_on_mps(mpo, mps)
            mps = normalise_mps(mps, direction='R', max_d=max_d)
            mps = normalise_mps(mps, direction='L')
            if keep_intermediary:
                outputs[i][j] = mps
            else:
                outputs[i] = mps
            if print_out:
                print('input {}, step {}, time,  state, {}'.format(
                    i, j + 1, time_array[j],
                    state_vectors_one_to_many(evaluate_mps(mps), as_str=True)))
    if keep_intermediary:
        return time_array, outputs
    else:
        return outputs
