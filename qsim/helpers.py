import numpy as np


def dagger(A):
    """
    Args:
        state vector A
    Returns:
        conjugate of transpose of A
    """
    return np.conjugate(np.transpose(A))


def nth_root(number, root):
    """
    eg given 8, 3 returns 2
    Args:
        number to find root of, nth root to find
    Return:
        root
    """
    coeff = [0] * (root + 1)
    coeff[-1] = -1 * number
    coeff[0] = 1
    return round(np.roots(coeff)[0], 5)


Sx = 0.5 * np.array([[0, 1], [1, 0]])
Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
Sz = 0.5 * np.array([[1, 0], [0, -1]])
Splus = Sx + 1j * Sy
Sminus = Sx - 1j * Sy
I = np.eye(2)


def make_higher_d_mat(mat_creation_fn, kwargnames, kwargvalues,
                      qubit_num=1, **kwargs):
    shape_extension = [np.array(l).shape[0] for l in kwargvalues]
    new_shape = shape_extension + [2**qubit_num, 2**qubit_num]
    new_mat = np.zeros(new_shape, dtype=complex)
    it = np.nditer(np.zeros(shape_extension), flags=['multi_index'])
    kw_d = dict.fromkeys(kwargnames)
    kw_d.update(kwargs)
    kw_d['qubit_num'] = qubit_num
    while not it.finished:
        for i, j in enumerate(list(it.multi_index)):
            kw_d[kwargnames[i]] = kwargvalues[i][j]
        new_mat[it.multi_index] = mat_creation_fn(**kw_d)
        it.iternext()
    return new_mat
