import numpy as np
import copy


def normalise_mps_site(mps, direction, k, max_d=None):
    mps = copy.deepcopy(mps)
    if direction == 'L':
        ld, rd = 2 * mps[k].shape[1], mps[k].shape[2]
        site = mps[k].reshape((ld, rd))
        U, s, V = np.linalg.svd(site, full_matrices=False)
        if max_d is not None:
            U = U[:, :max_d]
            s = s[:max_d]
            V = V[:max_d, :]
        mps[k] = U.reshape(2, int(U.shape[0] / 2), U.shape[1])
        if k < len(mps) - 1:
            next_state = np.tensordot(np.dot(np.diag(s), V),
                                      mps[k + 1], axes=(1, 1))
            mps[k + 1] = np.moveaxis(next_state, 1, 0)
    elif direction == 'R':
        ld, rd = mps[k].shape[1], 2 * mps[k].shape[2]
        site = np.moveaxis(mps[k], 0, 2).reshape(ld, rd)
        U, s, V = np.linalg.svd(site, full_matrices=False)
        if max_d is not None:
            U = U[:, :max_d]
            s = s[:max_d]
            V = V[:max_d, :]
        reshaped_v = np.moveaxis(
            V.reshape(V.shape[0], int(V.shape[1] / 2), 2), 2, 0)
        mps[k] = reshaped_v
        if k > 0:
            mps[k - 1] = np.tensordot(mps[k - 1], np.dot(U, np.diag(s)),
                                      axes=1)
    else:
        raise RuntimeError(
            'direction must be L or R, recieved: {}'.format(direction))
    return mps


def normalise_mps(mps, direction='L', max_d=None, k=None):
    norm_mps = copy.deepcopy(mps)
    qubit_num = len(mps)
    if direction == 'L':
        for i in range(qubit_num):
            norm_mps = normalise_mps_site(norm_mps, 'L', i, max_d=max_d)
        return norm_mps
    elif direction == 'R':
        for i in range(qubit_num - 1, -1, -1):
            norm_mps = normalise_mps_site(norm_mps, 'R', i, max_d=max_d)
        return norm_mps
    elif direction == 'M':
        for i in range(k):
            norm_mps = normalise_mps_site(norm_mps, 'L', i, max_d=max_d)
        for i in range(qubit_num - 1, k, -1):
            norm_mps = normalise_mps_site(norm_mps, 'R', i, max_d=max_d)
        ld, rd = norm_mps[k].shape[1], 2 * norm_mps[k].shape[2]
        reshaped_k = np.swapaxes(norm_mps[k], 0, 2).reshape(ld, rd)
        U, s, V = np.linalg.svd(reshaped_k)
        if max_d is not None:
            U = U[:, :max_d]
            s = s[:max_d]
            V = V[:max_d, :]
        norm_mps[k - 1] = np.tensordot(norm_mps[k - 1], U, axes=1)
        norm_mps[k] = np.swapaxes(
            V.reshape(V.shape[0], int(V.shape[1] / 2), 2), 2, 0)
        return norm_mps, s
        # reshaped_k = norm_mps[k].reshape(
        #     2 * norm_mps[k].shape[1], norm_mps[k].shape[2])
        # ld, rd = norm_mps[k + 1].shape[1], 2 * norm_mps[k + 1].shape[2]
        # reshaped_kp1 = np.moveaxis(norm_mps[k + 1], 0, 2).reshape(ld, rd)
        # C = np.tensordot(reshaped_k, reshaped_kp1, axes=1)
        # U, s, V = np.linalg.svd(C)
        # if max_d is not None:
        #     U = U[:, :max_d]
        #     s = s[:max_d]
        #     V = V[:max_d, :]
        # norm_mps[k] = U.reshape(2, int(U.shape[0] / 2), U.shape[1])
        # norm_mps[k + 1] = np.moveaxis(V.reshape(V.shape[0],
        #                                         int(V.shape[1] / 2), 2), 2, 0)




# def normalise_mps(mps, direction='L', max_d=None, k=0):
#     qubit_num = len(mps)
#     normalised_mps = []
#     if direction is 'L':
#         ld, rd = 2 * mps[0].shape[1], mps[0].shape[2]
#         st = mps[0].reshape((ld, rd))
#         for i in range(qubit_num):
#             U, s, V = np.linalg.svd(st, full_matrices=False)
#             if max_d is not None:
#                 U = U[:, :max_d]
#                 s = s[:max_d]
#                 V = V[:max_d, :]
#             normalised_mps.append(
#                 U.reshape(2, int(U.shape[0] / 2)(np.diag(s), V), n_st, axes=1)
#                 st = np.moveaxis(st, 2, 0)
#                 st = st.reshape((2 * st.shape[1], st.shape[2]))
#     elif direction is 'R':, U.shape[1]))
#             if i is not (qubit_num - 1):
#                 n_st = np.moveaxis(mps[i + 1], 0, 2)
#                 st = np.tensordot(np.dot
#         ld, rd = mps[-1].shape[1], 2 * mps[-1].shape[2]
#         st = np.moveaxis(mps[-1], 0, 2).reshape(ld, rd)
#         for i in range(qubit_num):
#             U, s, V = np.linalg.svd(st, full_matrices=False)
#             if max_d is not None:
#                 U = U[:, :max_d]
#                 s = s[:max_d]
#                 V = V[:max_d, :]
#             reshaped_v = np.moveaxis(
#                 V.reshape(V.shape[0], int(V.shape[1] / 2), 2), 2, 0)
#             normalised_mps.insert(0, reshaped_v)
#             if i is not (qubit_num - 1):
#                 st = np.tensordot(mps[-i - 2], np.dot(U, np.diag(s)), axes=1)
#                 ld, rd = st.shape[1], st.shape[2]
#                 st = np.moveaxis(st, 0, 2).reshape(ld, rd * 2)
#     return normalised_mps


def create_specific_mps(state, direction='L', max_d=None):
    mps = []
    qubit_num = int(np.log2(len(state)))
    qubits_pulled_out = 0
    for i in range(qubit_num):
        left_d = 2**(min([qubits_pulled_out + 1,
                          qubit_num - qubits_pulled_out + 1]))
        right_d = 2**(qubit_num - qubits_pulled_out - 1)
        if max_d is not None and left_d > 2 * max_d:
            left_d = 2 * max_d
        m = state.reshape(left_d, right_d)
        U, s, V = np.linalg.svd(m, full_matrices=False)
        if max_d is not None:
            U = U[:, :max_d]
            s = s[:max_d]
            V = V[:max_d, :]
        state = np.dot(np.diag(s), V)
        mps.append(U.reshape(2, int(U.shape[0] / 2), U.shape[1], order='F'))
        qubits_pulled_out += 1
    return mps


def create_random_mps(qubit_num, direction='L', k=None, max_d=None):
    if max_d is None:
        max_d = 2 ** int(np.floor(qubit_num / 2))
    mps = [[]] * qubit_num
    for i in range(qubit_num):
        if i == 0:
            mps[i] = np.random.rand(2, 1, max_d) + 1j * \
                np.random.rand(2, 1, max_d)
        elif i == (qubit_num - 1):
            mps[i] = np.random.rand(2, max_d, 1) + 1j * \
                np.random.rand(2, max_d, 1)
        else:
            mps[i] = np.random.rand(2, max_d, max_d) + \
                np.random.rand(2, max_d, max_d)
    mps = normalise_mps(mps, direction=direction, k=k, max_d=max_d)
    return mps


def evaluate_mps(mps, indices=None, direction='L'):
    if indices is not None:
        if len(indices) != len(mps):
            raise Exception('indices length != mps length')
        if any(s not in [0, 1] for s in indices):
            raise Exception('indices must be 0 or 1')
        state = mps[0][indices[0]]
        for q, s in enumerate(indices[1:]):
            state = np.tensordot(state, mps[q + 1][s], axes=1)
        return state[0, 0]
    else:
        state = mps[0]
        for s in mps[1:]:
            state = np.tensordot(state, s, axes=[[-1], [1]])
        return state.flatten()
