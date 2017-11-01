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


Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])
Splus = Sx + 1j * Sy
Sminus = Sx - 1j * Sy
I = np.eye(2)
