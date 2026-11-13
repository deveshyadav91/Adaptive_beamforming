import numpy as np
from .array import steering_vector

def simulate_snapshots(positions_lambda, desired_angle, interferer_angles,
                       n_snapshots=2048, desired_power=1.0,
                       interference_powers=None, noise_power=0.1,
                       seed=7, desired_phi=0.0, interferer_phis=None):
    rng = np.random.default_rng(seed)
    n = len(positions_lambda)
    if interference_powers is None:
        interference_powers = [10.0] * len(interferer_angles)
    if interferer_phis is None:
        interferer_phis = [0.0] * len(interferer_angles)
    def complex_noise(power, size):
        return np.sqrt(power/2) * (rng.standard_normal(size) + 1j*rng.standard_normal(size))
    s = complex_noise(desired_power, n_snapshots)
    X = steering_vector(positions_lambda, desired_angle, desired_phi)[:, None] * s[None, :]
    for angle, phi, power in zip(interferer_angles, interferer_phis, interference_powers):
        q = complex_noise(power, n_snapshots)
        X += steering_vector(positions_lambda, angle, phi)[:, None] * q[None, :]
    X += complex_noise(noise_power, (n, n_snapshots))
    # Clean desired reference is a simulation-only training reference for LMS.
    return X, s, {"desired": s}


def simulate_snapshots_model(model, desired_angle, interferer_angles,
                             n_snapshots=2048, desired_power=1.0,
                             interference_powers=None, noise_power=0.1,
                             seed=7, desired_phi=0.0, interferer_phis=None):
    """Like simulate_snapshots, but sources reach the array through ArrayModel.response.

    With an ideal model and the same seed the output equals simulate_snapshots exactly.
    Noise is added after the receiver chains (spatially white).
    """
    rng = np.random.default_rng(seed)
    n = len(model.positions)
    if interference_powers is None:
        interference_powers = [10.0] * len(interferer_angles)
    if interferer_phis is None:
        interferer_phis = [0.0] * len(interferer_angles)

    def complex_noise(power, size):
        return np.sqrt(power / 2) * (rng.standard_normal(size) + 1j * rng.standard_normal(size))

    s = complex_noise(desired_power, n_snapshots)
    X = model.vector(desired_angle, desired_phi)[:, None] * s[None, :]
    for angle, phi, power in zip(interferer_angles, interferer_phis, interference_powers):
        q = complex_noise(power, n_snapshots)
        X += model.vector(angle, phi)[:, None] * q[None, :]
    X += complex_noise(noise_power, (n, n_snapshots))
    return X, s
