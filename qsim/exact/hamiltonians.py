import numpy as np
# from scipy.special import jv as bessel
from qsim.helpers import Sz, Sx, Splus, Sminus, I


# def create_h1(detuning=0, **kwargs):
#     """
#     Function which creates a Hamiltonian corresponding to an optional Z
#     rotation or the identity. Z rotation frequency modulated by'detuning'.

#     Args:
#         detuning (default 0): detuning of drive from resonant freq in Hz,
#             corresponds to Z rotation.
#     Returns:
#         hamiltonian matrix shape (2, 2)
#     """
#     return create_h3(amp=0, detuning=detuning)


# def create_h3(amp=1, detuning=0, **kwargs):
#     """
#     Function which creates Hamiltonian corresponding to an X rotation.
#     Magnitude amp. Off resonance by detuning.

#     Args:
#         amp (default 1): magnitude of drive around X
#         detuning (default 0): detuning of drive from resonant freq in Hz,
#             corresponds to Z rotation.
#     Returns:
#         hamiltonian matrix shape (n, 2, 2) (n=1 by befault)
#     """
#     omega_d = 2 * np.pi * detuning
#     mat = 0.5 * (omega_d * Sz + amp * Sx)
#     return mat


# def create_h4(t=0, amp=1, detuning=0, mod_freq=0, **kwargs):
#     """
#     Function which creates Hamiltonian corresponding to an X rotation
#     modulated by cosine with frequency mod_freq. Magnitude amp. Off resonance
#     by detuning.

#     Args:
#         t time which together with mod_freq determines amplitude of x drive
#         amp (default 1): magnitude of drive around X
#         detuning (default 0): detuning of drive from resonant freq in Hz,
#             corresponds to Z rotation.
#         mod_freq (default 0): frequency of the cosine modulation of the
#             X rotation in Hz
#     Returns:
#         hamiltonian matrix shape (2, 2)
#     """
#     omega_f = 2 * np.pi * mod_freq
#     omega_d = 2 * np.pi * detuning
#     mat = 0.5 * (omega_d * Sz + amp * np.cos(omega_f * t) * Sx)
#     return mat


# def create_h5(t=0, amp=1, detuning=0, mod_freq=0, **kwargs):
#     """
#     Function which creates Hamiltonian corresponding to an X rotation
#     modulated by sine with frequency mod_freq. Magnitude amp. Off resonance
#     by detuning.

#     Args:
#         t time which together with mod_freq determines amplitude of x drive
#         amp (default 1): magnitude of drive around X
#         detuning (default 0): detuning of drive from resonant freq in Hz,
#             corresponds to Z rotation
#         mod_freq (default 0): frequency of the sine modulation of the
#             X rotation
#     Returns:
#         hamiltonian matrix shape (2, 2)
#     """
#     omega_f = 2 * np.pi * mod_freq
#     omega_d = 2 * np.pi * detuning
#     mat = 0.5 * (omega_d * Sz + amp * np.sin(omega_f * t) * Sx)
#     return mat


# def create_h6(detuning=0, mod_freq=0, amp=0, **kwargs):
#     """
#     Function which creates Hamiltonian corresponding to a Bessel function
#     rotation around the Z axis. This only works when measured at stroboscopic
#     times of the drive as it is in the floquet frame.
#     Floquet modulation cosine with frequency mod_freq. Magnitude amp. Off
#     resonance by detuning.

#     Args:
#         amp (default 1): magnitude of drive around X
#         detuning (default 0): detuning of drive from resonant freq in Hz,
#             corresponds to Z rotation
#         mod_freq (default 0): frequency of the cosine modulation of the
#             X rotation
#     Returns:
#         hamiltonian matrix shape (2, 2)
#     """
#     omega_f = 2 * np.pi * mod_freq
#     omega_d = 2 * np.pi * detuning
#     J = bessel(0, amp / omega_f)
#     mat = 0.5 * omega_d * omega_f * J * Sz
#     return mat


def create_heisenberg_h(qubit_num=1, J=0, Jz=None, Jxy=None, h=0, g=0,
                        periodic=False, **kwargs):
    Jz = Jz or J
    Jxy = Jxy or J
    H = np.zeros((2**qubit_num, 2**qubit_num), dtype=complex)
    for i in range(qubit_num - 1):
        sp_sm_mat = 1
        sm_sp_mat = 1
        sz_sz_mat = 1
        sz_mat = 1
        sx_mat = 1
        for j in range(qubit_num - 1):
            if j == i:
                sp_sm_mat = np.kron(sp_sm_mat, Splus)
                sp_sm_mat = np.kron(sp_sm_mat, Sminus)
                sm_sp_mat = np.kron(sm_sp_mat, Sminus)
                sm_sp_mat = np.kron(sm_sp_mat, Splus)
                sz_sz_mat = np.kron(sz_sz_mat, Sz)
                sz_sz_mat = np.kron(sz_sz_mat, Sz)
                sz_mat = np.kron(sz_mat, Sz)
                sx_mat = np.kron(sx_mat, Sx)
            else:
                sp_sm_mat = np.kron(sp_sm_mat, I)
                sm_sp_mat = np.kron(sm_sp_mat, I)
                sz_sz_mat = np.kron(sz_sz_mat, I)
                sz_mat = np.kron(sz_mat, I)
                sx_mat = np.kron(sx_mat, I)
        sz_mat = np.kron(sz_mat, I)
        sx_mat = np.kron(sx_mat, I)
        H += (Jz * sz_sz_mat + Jxy * (sp_sm_mat +
                                      sm_sp_mat) + h * sz_mat + g * sx_mat)
    # last term for single qubit terms
    sz_mat = np.kron(np.eye(2**(qubit_num - 1)), Sz)
    sx_mat = np.kron(np.eye(2**(qubit_num - 1)), Sx)
    H += h * sz_mat + g * sx_mat
    if periodic:
        if qubit_num > 2:
            sz_sz_mat = np.kron(np.eye(2**(qubit_num - 2)), Sz)
            sz_sz_mat = np.kron(Sz, sz_sz_mat)
            sp_sm_mat = np.kron(np.eye(2**(qubit_num - 2)), Splus)
            sp_sm_mat = np.kron(Sminus, sp_sm_mat)
            sm_sp_mat = np.kron(np.eye(2**(qubit_num - 2)), Sminus)
            sm_sp_mat = np.kron(Splus, sm_sp_mat)
            H += (Jz * sz_sz_mat + Jxy * (sp_sm_mat + sm_sp_mat))
        elif qubit_num == 2:
            sz_sz_mat = np.kron(Sz, Sz)
            sp_sm_mat = np.kron(Sminus, Splus)
            sm_sp_mat = np.kron(Splus, Sminus)
            H += (Jz * sz_sz_mat + Jxy * (sp_sm_mat + sm_sp_mat))
    return H


def create_ising_h(qubit_num=1, J=0, Jz=None, h=0, g=0,
                   **kwargs):
    Jz = Jz or J
    return create_heisenberg_h(qubit_num=qubit_num, Jz=Jz, h=h, g=g,
                               **kwargs)


def create_ising_w_next_nearest_neighbour_h(
        qubit_num=1, J=0, Jz=None, J2=0, h=0, g=0, periodic=False, **kwargs):
    H = create_heisenberg_h(qubit_num=qubit_num, Jz=Jz, h=h, g=g,
                            periodic=periodic, **kwargs)
    for i in range(qubit_num - 2):
        sz_I_sz_mat = 1
        for j in range(qubit_num - 2):
            if j == i:
                sz_I_sz_mat = np.kron(sz_I_sz_mat, Sz)
                sz_I_sz_mat = np.kron(sz_I_sz_mat, I)
                sz_I_sz_mat = np.kron(sz_I_sz_mat, Sz)
            else:
                sz_I_sz_mat = np.kron(sz_I_sz_mat, I)
        H += J2 * sz_I_sz_mat
    if periodic:
        if qubit_num > 2:
            sz_I_sz_mat = np.kron(np.kron(Sz, np.eye(2**(qubit_num - 3))),
                                  np.kron(Sz, I))
            H += J2 * sz_I_sz_mat
            sz_I_sz_mat = np.kron(np.kron(I, Sz),
                                  np.kron(np.eye(2**(qubit_num - 3)), Sz))
            H += J2 * sz_I_sz_mat
    return H


def create_ising_w_all_h(
        qubit_num=1, J=0, Jz=None, h=0, g=0, periodic=False, **kwargs):
    H = create_heisenberg_h(qubit_num=qubit_num, Jz=Jz, h=h, g=g,
                            periodic=periodic, **kwargs)
    coupling_term = np.zeros((2**qubit_num, 2**qubit_num), dtype=complex)
    for i in range(qubit_num - 2):
        for k in range(i + 2, qubit_num):
            sz_sz_mat = 1
            for j in range(qubit_num):
                if j == i:
                    sz_sz_mat = np.kron(sz_sz_mat, Sz)
                elif j == k:
                    sz_sz_mat = np.kron(sz_sz_mat, Sz)
                else:
                    sz_sz_mat = np.kron(sz_sz_mat, I)
            prepender = Jz / (k - i)
            coupling_term += prepender * sz_sz_mat
    if periodic:
        coupling_term += coupling_term
        # if qubit_num > 2:
        #     for k in range(qubit_num - 2):
        #         mat = np.kron(np.kron(np.eye(k), Sz), np.kron(np.eye(qubit_num -k - 3), Sz))
        #         mat = np.kron(mat, I)
        #         coupling_term += mat
        #     for k in range(qubit_num - 1):
        #         mat = np.kron(np.kron(np.eye(k), Sz), np.kron(np.eye(qubit_num -k - 2), Sz))
        #         coupling_term += mat
    H += coupling_term
    return H


def create_Huse_h(qubit_num=1, J=0, h=0, g=0):
    H = np.zeros((2**qubit_num, 2**qubit_num), dtype=complex)
    for i in range(qubit_num):
        if i < qubit_num - 1:
            mat = 1
            for j in range(qubit_num - 1):
                if j == i:
                    mat = np.kron(mat, Sz)
                    mat = np.kron(mat, Sz)
                else:
                    mat = np.kron(mat, I)
            H += J * mat
        mat = 1
        for j in range(qubit_num):
            if j == i:
                mat = np.kron(mat, Sx)
            else:
                mat = np.kron(mat, I)
        H += g * mat
        mat = 1
        for j in range(qubit_num):
            if j == i and j not in [0, qubit_num - 1]:
                mat = np.kron(mat, Sz)
            else:
                mat = np.kron(mat, I)
        H += h * mat
        mat = 1
        for j in range(qubit_num):
            if j == i and j in [0, qubit_num - 1]:
                mat = np.kron(mat, Sz)
            else:
                mat = np.kron(mat, I)
        H += (h - J) * mat
    return H
