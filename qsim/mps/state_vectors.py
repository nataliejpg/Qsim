import numpy as np
import copy


def normalise_mps_site(mps, direction, k, max_d=None):
    """
    Normalises (and possibly truncates) the mps on the given site k
    Args:
        mps: unnormalised matrix product state with shape
            (sigma_{k}, a_{k-1}, a_{k})
        k: site index
        direction: direction of normalisation "left" (L) or "right" (R)
        max_D: maximum bond dimension the mps will have
    Returns:
        mps with site k left/right canonically normalised
    """
    mps = copy.deepcopy(mps)
    # d[0] = simga_{k} # d[1] = a_{k-1} # d[2] = a_{k}
    d = mps[k].shape
    if direction == 'L':
        site = mps[k].reshape(d[0] * d[1], d[2])
        U, s, V = np.linalg.svd(site, full_matrices=False)
        if max_d is not None:
            U = U[:, :max_d]
            s = s[:max_d]
            V = V[:max_d, :]
        mps[k] = U.reshape(d[0], d[1], U.shape[1])
        if k < len(mps) - 1:
            next_state = np.tensordot(
                np.dot(np.diag(s), V), mps[k + 1], axes=(1, 1))
            mps[k + 1] = np.moveaxis(next_state, 1, 0)
    elif direction == 'R':
        site = np.moveaxis(mps[k], 0, 2).reshape(d[1], d[2] * d[0])
        U, s, V = np.linalg.svd(site, full_matrices=False)
        if max_d is not None:
            U = U[:, :max_d]
            s = s[:max_d]
            V = V[:max_d, :]
        reshaped_v = np.moveaxis(V.reshape(V.shape[0], d[2], d[0]), 2, 0)
        mps[k] = reshaped_v
        if k > 0:
            mps[k - 1] = np.tensordot(mps[k - 1], np.dot(U, np.diag(s)),
                                      axes=1)
    else:
        raise RuntimeError(
            'direction must be L or R, recieved: {}'.format(direction))
    return mps


def normalise_mps(mps, direction='L', max_d=None, k=None):
    """
    normalises the entire mps into left/right canonical form
    Args:
        mps: unnormalised matrix product state with shape
            (sigma_{k}, a_{k-1}, a_{k})
        direction: choose left ('L'), right ('R'), mixed ('M') normalisation
            or "Schmidt state" ('S') (default: left)
        max_D: maximum bond dimension the normalised mps will have
        k: "hot site" for mixed canonical mps or place the cut of the Schmidt
            state to the left of site k
    Returns:
        normalised_mps
    """
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
        return norm_mps
    elif direction == 'S':
        for i in range(k):
            norm_mps = normalise_mps_site(norm_mps, 'L', i, max_d=max_d)
        for i in range(qubit_num - 1, k, -1):
            norm_mps = normalise_mps_site(norm_mps, 'R', i, max_d=max_d)
        # d[0] = simga_{k} # d[1] = a_{k-1} # d[2] = a_{k}
        d = mps[k].shape
        reshaped_k = np.moveaxis(norm_mps[k], 0, 2).reshape(d[1], d[2] * d[0])
        U, s, V = np.linalg.svd(reshaped_k)
        if max_d is not None:
            U = U[:, :max_d]
            s = s[:max_d]
            V = V[:max_d, :]
        norm_mps[k - 1] = np.tensordot(norm_mps[k - 1], U, axes=1)
        norm_mps[k] = np.moveaxis(V.reshape(V.shape[0], d[2], d[0]), 2, 0)
        return norm_mps, s


def create_specific_mps(state, direction='L', max_d=None):
    """
    Creates the specific mps representation for a given input state
    Args:
        state: specific state given as vector on the entire hilbert space of
            the system
        direction: choose left ('L'), right ('R'), mixed ('M') normalisation or
            "Schmidt state" ('S') (default: left)
        k: "hot site" for mixed canonical mps or place the cut of the Schmidt
            state to the left of site k
        max_D: maximum bond dimension the normalised mps will have
    Returns:
        mps: specific matrix product state (normalised)
    """
    mps = []
    qubit_num = int(np.log2(len(state)))
    qubits_pulled_out = 0
    if direction != 'L':
        raise NotImplementedError
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
    """
    Creates mps state filled with random (complex) numbers
    Args:
        qubit_num: number of qubits
        direction: choose left ('L'), right ('R'), mixed ('M') normalisation
            or "Schmidt state" ('S') (default: left)
        k: "hot site" for mixed canonical mps or place the cut of the Schmidt
            state to the left of site k
        max_D: maximum bond dimension the normalised mps will have
    Returns:
        mps: random matrix product state (normalised)
    """
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


def evaluate_mps(mps, indices=None):
    """
    Creates the vector representation for a given mps input state
    Args:
        mps: specific state given as an mps
        indices: optional reference state to find amplitude of, length must
            be same as site number
    Returns:
        state vector on entire hilbert space corresponding to the mps
    """
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
