import numpy as np
from scipy.special import jv as bessel


####################################################################
# Unitary Evolution
####################################################################


def u_1(t, x, detuning=0):
    """
    Function which performs a unitay corresponding to an optional Z
    rotation or the identity. Z rotation frequency modulated by'detuning'.

    Args:
        t (numpy array shape (n,)): time at which to calculate
        x (numpy array shape (m, 2)): starting state vector
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation.

    Returns:
        evolved state vector (numpy array shape (m, n, 2))
    """
    omega_d = 2 * np.pi * detuning
    t = np.array(t)
    x = np.array(x).moveaxis(0, 2)
    a0 = np.cos(omega_d * t / 2) - 1j * np.sin(omega_d * t / 2)
    b1 = np.cos(omega_d * t / 2) + 1j * np.sin(omega_d * t / 2)
    if len(t.shape) == 0:
        a1 = 0
        b0 = 0
        mat = np.array([[a0, a1],
                        [b0, b1]])
    else:
        a1 = np.zeros(t.shape[0])
        b0 = np.zeros(t.shape[0])
        mat = np.array([[a0, a1],
                        [b0, b1]]).transpose()
    return np.dot(mat, x).moveaxis(2, 0)


def u_3(t, x, amp=1, detuning=0):
    """
    Function which performs a unitay corresponding to an X rotation with
    optional z rotation.
    X drive Magnitude V. Off resonance by detuning.

    Args:
        t (numpy array shape (n,)): time at which to calculate
        x (numpy array shape (m, 2)): starting state vector
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation.

    Returns:
        evolved state vector (numpy array shape (m, n, 2))
    """
    omega_d = 2 * np.pi * detuning
    alpha = np.sqrt(omega_d**2 + amp**2)
    t = np.array(t)
    x = np.array(x).moveaxis(0, 2)
    a0 = (np.cos(alpha * t / 2) - 1j *
          (omega_d / alpha) * np.sin(alpha * t / 2))
    a1 = -1j * (amp / alpha) * np.sin(alpha * t / 2)
    b0 = - 1j * (amp / alpha) * np.sin(alpha * t / 2)
    b1 = np.cos(alpha * t / 2) + 1j * \
        (omega_d / alpha) * np.sin(alpha * t / 2)

    if len(t.shape) == 0:
        mat = np.array([[a0, a1],
                        [b0, b1]])
    else:
        mat = np.array([[a0, a1],
                        [b0, b1]]).transpose()
    return np.dot(mat, x).moveaxis(2, 0)


def u_6(t, x, amp=0, detuning=0, mod_freq=0):
    """
    Function which performs a unitay corresponding to an X rotation
    modulated by cosine with frequency mod_freq. Magnitude amp. Off resonance
    by detuning.

    Args:
        t (numpy array shape (n,)): time to be used in forming hamiltonian
        x (numpy array shape (m, 2)): starting state vector
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation
        mod_freq (default 0): frequency of the cosine modulation of the
            X rotation

    Returns:
        evolved state vector (numpy array shape (m, n, 2))
    """
    t = np.array(t)
    x = np.array(x).moveaxis(0, 2)
    omega_d = 2 * np.pi * detuning
    omega_f = 2 * np.pi * mod_freq
    J = bessel(0, amp / omega_f)
    a0 = (np.cos(0.5 * omega_d * omega_f * J * t) -
          1j * np.sin(0.5 * omega_d * omega_f * J * t))
    b1 = (np.cos(0.5 * omega_d * omega_f * J * t) +
          1j * np.sin(0.5 * omega_d * omega_f * J * t))
    if len(t.shape) == 0:
        a1 = 0
        b0 = 0
        mat = np.array([[a0, a1],
                        [b0, b1]])
    else:
        a1 = np.zeros(t.shape[0])
        b0 = np.zeros(t.shape[0])
        mat = np.array([[a0, a1],
                        [b0, b1]]).transpose()
    return np.dot(mat, x).moveaxis(2, 2)
