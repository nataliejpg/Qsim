import numpy as np
from qsim.mps.hamiltonians import create_magnetisation_mpo
from qsim.mps.evolution_methods import do_mpo_on_mps


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


def projection(mps_list, axis='Z'):
    """
    Function which finds the projections for a set of mps states onto an
    axis

    Args:
        mps_list: state vectors list with shape (m, qubit_num, 2, a0, a1)
            where m is the number of states
        axis for the qubits to be projected onto 'X', 'Y' or 'Z' (default Z)

    Returns:
        projection onto axis shape (m, qubit_num)
    """
    mps_array = np.array(mps_list)
    qubit_num = mps_array.shape[1]
    projections = np.zeros((mps_array.shape[0], qubit_num))
    for i, mps in enumerate(mps_array):
        for q in range(qubit_num):
            projection_mat = create_magnetisation_mpo(qubit_num, q, axis=axis)
            final_state = do_mpo_on_mps(projection_mat, mps)
            projections[i, q] = 2 * find_overlap(final_state, mps)
    return projections


def find_entropy():
    pass
