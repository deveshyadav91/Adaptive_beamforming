"""Amplitude tapers (windows) for sidelobe control."""
import numpy as np

TAPER_NAMES = ("Uniform", "Hann", "Hamming", "Blackman", "Kaiser")


def taper_weights(name: str, n_elements: int, kaiser_beta: float = 6.0) -> np.ndarray:
    """Real, positive, symmetric taper with peak 1.

    Hann and Blackman are evaluated on n+2 points with the two zero end points
    dropped, so no element is switched off completely (a 12-element array keeps
    12 active elements). Hamming and Kaiser have non-zero ends already.
    """
    if n_elements < 1:
        raise ValueError("Need at least one element.")
    key = str(name).strip().lower()
    if key == "uniform":
        t = np.ones(n_elements)
    elif key == "hann":
        t = np.hanning(n_elements + 2)[1:-1]
    elif key == "hamming":
        t = np.hamming(n_elements)
    elif key == "blackman":
        t = np.blackman(n_elements + 2)[1:-1]
    elif key == "kaiser":
        if kaiser_beta < 0:
            raise ValueError("Kaiser beta must be >= 0.")
        t = np.kaiser(n_elements, kaiser_beta)
    else:
        raise ValueError(f"Unknown taper '{name}'. Choose from {TAPER_NAMES}.")
    return t / t.max()


def tapered_weights(a_desired, taper) -> np.ndarray:
    """Conventional beamformer with amplitude taper, distortionless: w^H a = 1."""
    a = np.asarray(a_desired, complex)
    t = np.asarray(taper, float)
    if a.shape != t.shape:
        raise ValueError("Taper and steering vector must have the same length.")
    return (t * a) / np.vdot(a, t * a)


def taper_efficiency(taper) -> float:
    """(sum t)^2 / (N sum t^2): 1 for uniform; <1 means gain given up for sidelobes."""
    t = np.asarray(taper, float)
    return float(np.sum(t) ** 2 / (len(t) * np.sum(t ** 2)))


def white_noise_gain_db(weights, a_desired) -> float:
    """|w^H a|^2 / (w^H w) in dB (the SNR gain against spatially white noise)."""
    w, a = np.asarray(weights, complex), np.asarray(a_desired, complex)
    return float(10 * np.log10(abs(np.vdot(w, a)) ** 2 / max(np.vdot(w, w).real, 1e-30)))
