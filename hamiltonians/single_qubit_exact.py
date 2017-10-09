import numpy as np
from scipy.special import jv as bessel


def h_1(t, x, detuning=0):
    """
    Function which performs a hamiltonian corresponding to an optional Z
    rotation or the identity. Z rotation frequency modulated by'detuning'.

    Args:
        t (numpy array shape (n,)): time at which to calculate
        x (numpy array shape (m, 2)): starting state vectors
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation.

    Returns:
        evolved state vector (numpy array shape (m, n, 2))
    """
    omega_d = 2 * np.pi * detuning
    t = np.array(t)
    x = np.array(x).moveaxis(0, 2)
    if len(t.shape) == 0:
        mat = 0.5 * omega_d * np.array([[1, 0], [0, -1]])
    else:
        a0 = np.ones(t.shape[0]) * 0.5 * omega_d
        a1 = np.zeros(t.shape[0])
        b0 = np.zeros(t.shape[0])
        b1 = np.ones(t.shape[0]) * -0.5 * omega_d
        mat = np.array([[a0, a1],
                        [b0, b1]]).transpose()
    return np.dot(mat, x).moveaxis(2, 0)


def h_3(t, x, amp=1, detuning=0):
    """
    Function which does the Hamiltonian corresponding to an X rotation.
    Magnitude amp. Off resonance by detuning.

    Args:
        t (numpy array shape (n,)): time to be used in forming hamiltonian
        x (numpy array shape (2, m)): starting state vector
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation

    Returns:
        matrix product (numpy array (n, 2, m))
    """
    omega_d = 2 * np.pi * detuning
    t = np.array(t)
    x = np.array(x).moveaxis(0, 2)
    if len(t.shape) == 0:
        mat = 0.5 * np.array([[omega_d, amp],
                              [amp, - omega_d]])
    else:
        a0 = np.ones(t.shape[0]) * 0.5 * omega_d
        a1 = np.ones(t.shape[0]) * 0.5 * amp
        b0 = np.ones(t.shape[0]) * 0.5 * amp
        b1 = np.ones(t.shape[0]) * -0.5 * omega_d
        mat = np.array([[a0, a1],
                        [b0, b1]]).transpose()
    return np.dot(mat, np.array(x)).moveaxis(2, 0)


def h_4(t, x, amp=1, detuning=0, mod_freq=0):
    """
    Function which does the Hamiltonian corresponding to an X rotation
    modulated by cosine with frequency mod_freq. Magnitude amp. Off resonance
    by detuning.

    Args:
        t (numpy array shape (n,)): time to be used in forming hamiltonian
        x (numpy array shape (2, m)): starting state vector
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation
        mod_freq (default 0): frequency of the cosine modulation of the
            X rotation

    Returns:
        matrix product (numpy array (n, 2, m))
    """
    t = np.array(t)
    x = np.array(x).moveaxis(0, 2)
    omega_f = 2 * np.pi * mod_freq
    omega_d = 2 * np.pi * detuning
    a1 = 0.5 * amp * np.cos(omega_f * t)
    b0 = 0.5 * amp * np.cos(omega_f * t)
    if len(t.shape) == 0:
        a0 = 0.5 * omega_d
        b1 = -0.5 * omega_d
        mat = np.array([[a0, a1],
                        [b0, b1]])
    else:
        a0 = np.ones(t.shape[0]) * 0.5 * omega_d
        b1 = np.ones(t.shape[0]) * -0.5 * omega_d
        mat = np.array([[a0, a1],
                        [b0, b1]]).transpose()
    return np.dot(mat, np.array(x)).moveaxis(2, 0)


def h_5(t, x, amp=1, detuning=0, mod_freq=0):
    """
    Function which does the Hamiltonian corresponding to an X rotation
    modulated by sine with frequency mod_freq. Magnitude amp. Off resonance
    by detuning.

    Args:
        t (numpy array shape (n,)): time to be used in forming hamiltonian
        x (numpy array shape (2,)): starting state vector
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation
        mod_freq (default 0): frequency of the sine modulation of the
            X rotation

    Returns:
        matrix product (numpy array (n, 2, m))
    """
    t = np.array(t)
    x = np.array(x).moveaxis(2, 0)
    omega_f = 2 * np.pi * mod_freq
    omega_d = 2 * np.pi * detuning
    a1 = 0.5 * amp * np.sin(omega_f * t)
    b0 = 0.5 * amp * np.sin(omega_f * t)
    if len(t.shape) == 0:
        a0 = 0.5 * omega_d
        b1 = -0.5 * omega_d
        mat = np.array([[a0, a1],
                        [b0, b1]])
    else:
        a0 = np.ones(t.shape[0]) * 0.5 * omega_d
        b1 = np.ones(t.shape[0]) * -0.5 * omega_d
        mat = np.array([[a0, a1],
                        [b0, b1]]).transpose()
    return np.dot(mat, np.array(x)).moveaxis(2, 0)


def h_6(t, x, detuning=0, mod_freq=0, amp=0):
    """
    Function which does the Hamiltonian corresponding to a Bessel function
    rotation around the Z axis. This only works when measured at stroboscopic
    times of the drive as it is in the floquet frame.
    Floquet modulation cosine with frequency mod_freq. Magnitude amp. Off
    resonance by detuning.

    Args:
        t (numpy array shape (n,)): time to be used in forming hamiltonian
        x (numpy array shape (2,)): starting state vector
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation
        mod_freq (default 0): frequency of the cosine modulation of the
            X rotation

    Returns:
        matrix product (numpy array (n, 2, m))
    """
    t = np.array(t)
    x = np.array(x).moveaxis(0, 2)
    omega_f = 2 * np.pi * mod_freq
    omega_d = 2 * np.pi * detuning
    J = bessel(0, amp / omega_f)
    if len(t.shape) == 0:
        a0 = 0.5 * omega_d * omega_f * J
        a1 = 0
        b0 = 0
        b1 = -0.5 * omega_d * omega_f * J
        mat = np.array([[a0, a1],
                        [b0, b1]])
    else:
        a0 = np.ones(t.shape[0]) * 0.5 * omega_d * omega_f * J
        a1 = np.zeros(t.shape[0])
        b0 = np.zeros(t.shape[0])
        b1 = np.ones(t.shape[0]) * -0.5 * omega_d * omega_f * J
        mat = np.array([[a0, a1],
                        [b0, b1]]).transpose()
    return np.dot(mat, np.array(x)).moveaxis(0, 2)
