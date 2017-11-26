import numpy as np
from scipy.special import jv as bessel
from qsim.helpers import dagger
from qsim.exact.hamiltonians import create_heisenberg_h, create_Huse_h


####################################################################
# Unitary Evolution
####################################################################


def create_u1(t=0, detuning=0, **kwargs):
    """
    Function which creates a unitay corresponding to an optional Z
    rotation or the identity. Z rotation frequency modulated by'detuning'.

    Args:
        t duration of evolution
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation.

    Returns:
        mat (2, 2) corresponding to unitary perfoming this evolution for time t
    """
    omega_d = 2 * np.pi * detuning
    a0 = np.cos(omega_d * t / 2) - 1j * np.sin(omega_d * t / 2)
    b1 = np.cos(omega_d * t / 2) + 1j * np.sin(omega_d * t / 2)
    a1 = 0
    b0 = 0
    mat = np.array([[a0, a1], [b0, b1]])
    return mat


def create_u3(t=0, amp=1, detuning=0, **kwargs):
    """
    Function which creates a unitay corresponding to an X rotation with
    optional z rotation.
    X drive Magnitude V. Off resonance by detuning.

    Args:
        t duration of evolution
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation.

    Returns:
        mat (2, 2) corresponding to unitary perfoming this evolution for time t
    """
    omega_d = 2 * np.pi * detuning
    alpha = np.sqrt(omega_d**2 + amp**2)
    a0 = (np.cos(alpha * t / 2) - 1j *
          (omega_d / alpha) * np.sin(alpha * t / 2))
    a1 = -1j * (amp / alpha) * np.sin(alpha * t / 2)
    b0 = - 1j * (amp / alpha) * np.sin(alpha * t / 2)
    b1 = np.cos(alpha * t / 2) + 1j * \
        (omega_d / alpha) * np.sin(alpha * t / 2)
    mat = np.array([[a0, a1], [b0, b1]])
    return mat


def create_u6(t=0, amp=0, detuning=0, mod_freq=0, **kwargs):
    """
    Function which creates a unitay corresponding to an X rotation
    modulated by cosine with frequency mod_freq. Magnitude amp. Off resonance
    by detuning.

    Args:
        t duration of evolution
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation
        mod_freq (default 0): frequency of the cosine modulation of the
            X rotation

    Returns:
        mat (2, 2) corresponding to unitary perfoming this evolution for time t
    """
    omega_d = 2 * np.pi * detuning
    omega_f = 2 * np.pi * mod_freq
    J = bessel(0, amp / omega_f)
    a0 = (np.cos(0.5 * omega_d * omega_f * J * t) -
          1j * np.sin(0.5 * omega_d * omega_f * J * t))
    b1 = (np.cos(0.5 * omega_d * omega_f * J * t) +
          1j * np.sin(0.5 * omega_d * omega_f * J * t))
    a1 = 0
    b0 = 0
    mat = np.array([[a0, a1], [b0, b1]])
    return mat


def create_heisenberg_u(qubit_num=1, t=0, J=0, h=0, g=0, **kwargs):
    H = create_heisenberg_h(qubit_num=qubit_num, J=J, h=h, g=g)
    l, u = np.linalg.eig(H)
    U = np.dot(u, np.dot(np.diag(np.exp(-1j * l * t)), dagger(u)))
    return U


def create_Huse_u(qubit_num=1, t=0, J=0, h=0, g=0, **kwargs):
    H = create_Huse_h(qubit_num=qubit_num, J=J, h=h, g=g)
    l, u = np.linalg.eig(H)
    U = np.dot(u, np.dot(np.diag(np.exp(-1j * l * t)), dagger(u)))
    return U
