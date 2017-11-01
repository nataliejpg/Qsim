import numpy as np
from scipy.special import jv as bessel
from qsim.helpers import Sz, Sx


def create_h1(detuning=0, **kwargs):
    """
    Function which creates a Hamiltonian corresponding to an optional Z
    rotation or the identity. Z rotation frequency modulated by'detuning'.

    Args:
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation.
    Returns:
        hamiltonian matrix shape (2, 2)
    """
    return create_h3(amp=0, detuning=detuning)


def create_h3(amp=1, detuning=0, **kwargs):
    """
    Function which creates Hamiltonian corresponding to an X rotation.
    Magnitude amp. Off resonance by detuning.

    Args:
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation.
    Returns:
        hamiltonian matrix shape (n, 2, 2) (n=1 by befault)
    """
    omega_d = 2 * np.pi * detuning
    mat = 0.5 * (omega_d * Sz + amp * Sx)
    return mat


def create_h4(t=0, amp=1, detuning=0, mod_freq=0, **kwargs):
    """
    Function which creates Hamiltonian corresponding to an X rotation
    modulated by cosine with frequency mod_freq. Magnitude amp. Off resonance
    by detuning.

    Args:
        t time which together with mod_freq determines amplitude of x drive
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation.
        mod_freq (default 0): frequency of the cosine modulation of the
            X rotation in Hz
    Returns:
        hamiltonian matrix shape (2, 2)
    """
    omega_f = 2 * np.pi * mod_freq
    omega_d = 2 * np.pi * detuning
    mat = 0.5 * (omega_d * Sz + amp * np.cos(omega_f * t) * Sx)
    return mat


def create_h5(t=0, amp=1, detuning=0, mod_freq=0, **kwargs):
    """
    Function which creates Hamiltonian corresponding to an X rotation
    modulated by sine with frequency mod_freq. Magnitude amp. Off resonance
    by detuning.

    Args:
        t time which together with mod_freq determines amplitude of x drive
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation
        mod_freq (default 0): frequency of the sine modulation of the
            X rotation
    Returns:
        hamiltonian matrix shape (2, 2)
    """
    omega_f = 2 * np.pi * mod_freq
    omega_d = 2 * np.pi * detuning
    mat = 0.5 * (omega_d * Sz + amp * np.sin(omega_f * t) * Sx)
    return mat


def create_h6(detuning=0, mod_freq=0, amp=0, **kwargs):
    """
    Function which creates Hamiltonian corresponding to a Bessel function
    rotation around the Z axis. This only works when measured at stroboscopic
    times of the drive as it is in the floquet frame.
    Floquet modulation cosine with frequency mod_freq. Magnitude amp. Off
    resonance by detuning.

    Args:
        amp (default 1): magnitude of drive around X
        detuning (default 0): detuning of drive from resonant freq in Hz,
            corresponds to Z rotation
        mod_freq (default 0): frequency of the cosine modulation of the
            X rotation
    Returns:
        hamiltonian matrix shape (2, 2)
    """
    omega_f = 2 * np.pi * mod_freq
    omega_d = 2 * np.pi * detuning
    J = bessel(0, amp / omega_f)
    mat = 0.5 * omega_d * omega_f * J * Sz
    return mat
