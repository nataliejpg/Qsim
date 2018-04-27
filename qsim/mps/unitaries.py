import numpy as np
from qsim.helpers import I, Sx, Sy, Sz, dagger
from .hamiltonians import make_generic_2_qubit_hamiltonian


def create_magnetisation_unitary_mpo(qubit_num, qubit_index, delta_t,
                                     axis='Z'):
    """
    Creates the unitary time-evolution mpo for H = S_{qubit_index}^{axis}
    Hamiltonian
    Args:
        qubit_num: number of qubits in the system
        qubit_index: site on which the Hamiltonian acts non-trivially
        delta_t: time-step
        axis: axis which spin-operator is applied to site qubit_index
            (default: Sz)
    Returns:
        mpo array of length "qubit_num" with (local)
            shape (b_{k-1}, b_{k}, sigma_{k}^{'}, sigma_{k})
    """
    identity = np.reshape(1, 1, 2, 2)
    if axis == 'Z':
        S = Sz.reshape(1, 1, 2, 2)
    elif axis == 'X':
        S = Sx.reshape(1, 1, 2, 2)
    elif axis == 'Y':
        S = Sy.reshape(1, 1, 2, 2)
    D, U = np.linalg.eig(S)
    unitary = np.dot(U, np.dot(np.diag(np.exp(-1j * delta_t * D)), dagger(U)))
    mpo = [[]] * qubit_num
    for i in range(qubit_num):
        mpo[i] = unitary if i == qubit_index else identity
    return mpo


def make_odd_even_unitary_mpo(first_mat, middle_mat, last_mat,
                              qubit_num=1, even=True):
    """
    Creates the unitary time-evolution mpo for odd/even bonds
    Args:
        first_mat: 2-site unitary for the 1st bond
        middle_mat: 2-site unitary for the middle bonds
        last_mat: 2-site unitary for the last bond
        qubit_num: number of qubits
        even (bool): specifies whether to generate the even ("True") or
        odd ("False") mpo
    Returns:
        odd/even time evolution mpo array of length "qubit_num" with (local)
            shape (b_{k-1}, b_{k}, sigma_{k}^{'}, sigma_{k})
    """
    # make up matrices
    S_middle = middle_mat
    S_first = first_mat
    S_last = last_mat
    S_middle_r = np.swapaxes(S_middle.reshape(2, 2, 2, 2), 1, 2).reshape(4, 4)
    S_first_r = np.swapaxes(S_first.reshape(2, 2, 2, 2), 1, 2).reshape(4, 4)
    S_last_r = np.swapaxes(S_last.reshape(2, 2, 2, 2), 1, 2).reshape(4, 4)

    # do svd to get mpos
    U, s, V = np.linalg.svd(S_middle_r, full_matrices=False)
    a0 = np.dot(U, np.diag(np.sqrt(s)))
    a1 = np.dot(np.diag(np.sqrt(s)), V)
    M0 = np.swapaxes(a0, 0, 1).reshape(1, a0.shape[1], 2, 2)
    M1 = a1.reshape(a1.shape[0], 1, 2, 2)

    U, s, V = np.linalg.svd(S_first_r, full_matrices=False)
    a0 = np.dot(U, np.diag(np.sqrt(s)))
    a1 = np.dot(np.diag(np.sqrt(s)), V)
    M0_first = np.swapaxes(a0, 0, 1).reshape(1, a0.shape[1], 2, 2)
    M1_first = a1.reshape(a1.shape[0], 1, 2, 2)

    U, s, V = np.linalg.svd(S_last_r, full_matrices=False)
    a0 = np.dot(U, np.diag(np.sqrt(s)))
    a1 = np.dot(np.diag(np.sqrt(s)), V)
    M0_last = np.swapaxes(a0, 0, 1).reshape(1, a0.shape[1], 2, 2)
    M1_last = a1.reshape(a1.shape[0], 1, 2, 2)

    mpo = [[]] * qubit_num
    if even:
        for i in range(qubit_num):
            if i % 2 == 0:
                mpo[i] = M0
            else:
                mpo[i] = M1
        mpo[0] = M0_first
        mpo[1] = M1_first
        if qubit_num % 2 == 0:
            mpo[-1] = M1_last
            mpo[-2] = M0_last
        else:
            mpo[-1] = I.reshape(1, 1, 2, 2)
    else:
        for i in range(qubit_num):
            if i % 2 != 0:
                mpo[i] = M0
            else:
                mpo[i] = M1
        mpo[0] = I.reshape(1, 1, 2, 2)
        if qubit_num % 2 != 0:
            mpo[-1] = M1_last
            mpo[-2] = M0_last
        else:
            mpo[-1] = I.reshape(1, 1, 2, 2)
    return mpo


def make_2_qubit_unitary(delta_t, end=0, **kwargs):
    """
    Creates the 2qubit unitary time-evolution operator for hamiltonian
    specified in kwargs
    Args:
        delta_t: time-step
        end =  -1 (first), 0 (middle), 1 (last), 10 (both)
        kwargs: parameter list of the XX,YY,ZZ,X,Y,Z coupling strengths
    Returns:
        2-site unitary time-evolution operator
    """
    H = make_generic_2_qubit_hamiltonian(end=end, **kwargs)
    D, U = np.linalg.eig(H)
    unitary = np.dot(U, np.dot(np.diag(np.exp(-1j * delta_t * D)), dagger(U)))
    return unitary


def create_heisenberg_unitary_mpo(qubit_num=1, J=0, h=0, g=0, t=0,
                                  even=True):
    """
    Creates the unitary time-evolution mpo for Heisenberg model
    H = J/2(S+S- + S-S+) + JSzSz + hSz + gSx
    Args:
        qubit_num: number of qubits in the system
        J: isotropic heisenberg coupling
        h: longitudinal (on-site) field
        g: transverse (on-site) field
        t: time step
        even: even bonds (or odd bonds)
    Returns:
        mpo array of unitary time evolution on odd/even bonds of length
            "qubit_num" with (local) shape (b_{k-1}, b_{k}, sigma_{k}^{'},
            sigma_{k}) for heisenberg model
    """
    middle_mat = make_2_qubit_unitary(
        t, end=0, XX=J, YY=J, ZZ=J, Z=h, X=g)
    first_mat = make_2_qubit_unitary(
        t, end=-1, XX=J, YY=J, ZZ=J, Z=h, X=g)
    last_mat = make_2_qubit_unitary(t, end=1, XX=J, YY=J, ZZ=J, Z=h, X=g)
    return make_odd_even_unitary_mpo(first_mat, middle_mat, last_mat,
                                     qubit_num=qubit_num, even=even)


def create_Huse_unitary_mpo(qubit_num=1, J=0, h=0, g=0, t=0,
                            even=True):
    middle_mat = make_2_qubit_unitary(t, end=0, X=g, Z=h, ZZ=J)
    first_mat = make_end_Huse_2_qubit_unitary(t, end=-1, J=J, h=J, g=g)
    last_mat = make_end_Huse_2_qubit_unitary(t, end=1, J=J, h=J, g=g)
    return make_odd_even_unitary_mpo(first_mat, middle_mat, last_mat,
                                     qubit_num=qubit_num, even=even)


def create_Huse_unitary_mpo_x(qubit_num=1, J=0, h=0, g=0, t=0):
    middle_mat = g * np.kron(I, Sx) + g * np.kron(Sx, I)
    first_mat = middle_mat
    last_mat = middle_mat
    return make_odd_even_unitary_mpo(first_mat, middle_mat, last_mat,
                                     qubit_num=qubit_num, even=True)


def create_Huse_unitary_mpo_z(qubit_num=1, J=0, h=0, g=0, t=0, even=True):
    middle_mat = h / 2 * np.kron(I, Sz) + h / 2 * \
        np.kron(Sz, I) + J * np.kron(Sz, Sz)
    first_mat = (h - J) * np.kron(Sz, I) + h / 2 * \
        np.kron(I, Sz) + J * np.kron(Sz, Sz)
    last_mat = (h - J) * np.kron(I, Sz) + h / 2 * \
        np.kron(Sz, I) + J * np.kron(Sz, Sz)
    return make_odd_even_unitary_mpo(first_mat, middle_mat, last_mat,
                                     qubit_num=qubit_num, even=even)


def make_end_Huse_2_qubit_unitary(delta_t, end=None, **kwargs):
    H = make_generic_2_qubit_hamiltonian(
        end=end, ZZ=kwargs['J'], X=kwargs['g'])
    if end in [-1, 10]:
        H += (kwargs['h'] - kwargs['J']) * np.kron(Sz, I)
        H += kwargs['h'] / 2 * np.kron(I, Sz)
    if end in [1, 10]:
        H += (kwargs['h'] - kwargs['J']) * np.kron(I, Sz)
        H += kwargs['h'] / 2 * np.kron(Sz, I)
    if end not in [1, -1, 10]:
        raise RuntimeError('fuuuuck')
    D, U = np.linalg.eig(H)
    unitary = np.dot(U, np.dot(np.diag(np.exp(-1j * delta_t * D)), dagger(U)))
    return unitary
