import numpy as np


def normalise_mps(mps, direction='L', max_d=None):
    qubit_num = len(mps)
    normalised_mps = []
    if direction is 'L':
        ld, rd = 2 * mps[0].shape[1], mps[0].shape[2]
        st = mps[0].reshape((ld, rd))
        for i in range(qubit_num):
            U, s, V = np.linalg.svd(st, full_matrices=False)
            if max_d is not None:
                U = U[:, :max_d]
                s = s[:max_d]
                V = V[:max_d, :]
            normalised_mps.append(
                U.reshape(2, int(U.shape[0] / 2), U.shape[1]))
            if i is not (qubit_num - 1):
                n_st = np.moveaxis(mps[i + 1], 0, 2)
                st = np.tensordot(np.dot(np.diag(s), V), n_st, axes=1)
                st = np.moveaxis(st, 2, 0)
                st = st.reshape((2 * st.shape[1], st.shape[2]))
    elif direction is 'R':
        ld, rd = mps[-1].shape[1], 2 * mps[-1].shape[2]
        st = np.moveaxis(mps[-1], 0, 2).reshape(ld, rd)
        for i in range(qubit_num):
            U, s, V = np.linalg.svd(st, full_matrices=False)
            if max_d is not None:
                U = U[:, :max_d]
                s = s[:max_d]
                V = V[:max_d, :]
            reshaped_v = np.moveaxis(
                V.reshape(V.shape[0], int(V.shape[1] / 2), 2), 2, 0)
            normalised_mps.insert(0, reshaped_v)
            if i is not (qubit_num - 1):
                st = np.tensordot(mps[-i - 2], np.dot(U, np.diag(s)), axes=1)
                ld, rd = st.shape[1], st.shape[2]
                st = np.moveaxis(st, 0, 2).reshape(ld, rd * 2)
    return normalised_mps


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


def create_random_mps(qubit_num, direction='L', max_d=None):
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
    mps = normalise_mps(mps, direction=direction, max_d=max_d)
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
