"""Non-ideal array model: element pattern, calibration errors, mutual coupling.

Actual response of the array to a plane wave from direction (theta, phi):

    v(theta, phi) = Gamma @ M @ ( e(theta) * a(theta, phi) )

  a      ideal steering vector (project convention, exp(+j 2 pi r.u))
  e      cosine-power element field pattern, max(cos(theta_polar), 0)**p
  M      mutual-coupling matrix (banded Toeplitz: nearest + next neighbour)
  Gamma  diag(per-channel complex gain/phase calibration error), applied after
         coupling because it models the receiver chains.

Receiver noise is added after Gamma and stays spatially white. The coupling model
assumes elements are indexed along a line (ULA); it is not a full-wave model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .array import steering_matrix


def cosine_power_element_pattern(theta_deg, exponent: float):
    """Field pattern cos(theta)^p; zero in the back hemisphere. p = 0 is isotropic.

    theta is the angle from boresight (+z): signed scan angle for a ULA in the
    x-z plane, polar angle for a planar array.
    """
    c = np.clip(np.cos(np.deg2rad(np.asarray(theta_deg, float))), 0.0, None)
    return c ** exponent


@dataclass(frozen=True)
class ImpairmentConfig:
    element_exponent: float = 0.0
    gain_std_db: float = 0.0
    phase_std_deg: float = 0.0
    coupling_nn_db: Optional[float] = None       # nearest neighbour, None = off
    coupling_nnn_db: Optional[float] = None      # next neighbour, None = off
    coupling_nn_phase_deg: float = 0.0
    coupling_nnn_phase_deg: float = 0.0
    scale_coupling_with_spacing: bool = False    # heuristic: |c| ~ ref_spacing / spacing
    coupling_ref_spacing_lambda: float = 0.5

    def __post_init__(self):
        if self.element_exponent < 0 or self.gain_std_db < 0 or self.phase_std_deg < 0:
            raise ValueError("Element exponent and error standard deviations must be >= 0.")
        if self.coupling_ref_spacing_lambda <= 0:
            raise ValueError("Coupling reference spacing must be positive.")

    @property
    def is_ideal(self) -> bool:
        return (self.element_exponent == 0 and self.gain_std_db == 0 and self.phase_std_deg == 0
                and self.coupling_nn_db is None and self.coupling_nnn_db is None)


def calibration_error_variance(gain_std_db: float, phase_std_deg: float) -> float:
    """Var of Gamma_n = 10^(eps/20) exp(j phi), eps~N(0,gain_std_db^2), phi~N(0,phase_std^2).

    Exact: E|G|^2 - |E G|^2 with log-normal amplitude and Gaussian phase.
    """
    k = np.log(10) / 20
    s2 = (k * gain_std_db) ** 2
    sp2 = np.deg2rad(phase_std_deg) ** 2
    return float(np.exp(2 * s2) - np.exp(s2 - sp2))


def predicted_error_floor_db(weights, gain_std_db, phase_std_deg,
                             theta_deg=None, element_exponent=0.0):
    """Expected incoherent (random-error) response power in dB, 0 dB = ideal peak
    of a distortionless beamformer. Equals Var(Gamma) * ||w||^2, scaled by the
    element pattern when angles are given. Ignores coupling."""
    p = calibration_error_variance(gain_std_db, phase_std_deg) * float(np.vdot(weights, weights).real)
    base = 10 * np.log10(max(p, 1e-30))
    if theta_deg is None:
        return base
    e = cosine_power_element_pattern(theta_deg, element_exponent)
    return base + 20 * np.log10(np.maximum(e, 1e-12))


def coupling_matrix(n: int, c_nn: complex = 0.0, c_nnn: complex = 0.0) -> np.ndarray:
    """Symmetric banded Toeplitz coupling matrix with unit diagonal."""
    M = np.eye(n, dtype=complex)
    for k, c in ((1, c_nn), (2, c_nnn)):
        if c != 0 and k < n:
            idx = np.arange(n - k)
            M[idx, idx + k] = c
            M[idx + k, idx] = c
    if np.max(np.sum(np.abs(M), axis=1) - 1.0) >= 1.0:
        raise ValueError("Coupling too strong: off-diagonal row sum >= 1 "
                         "(matrix not diagonally dominant). Reduce coupling or spacing scaling.")
    return M


class ArrayModel:
    """One realisation of a (possibly impaired) linear array."""

    def __init__(self, positions_lambda, config: Optional[ImpairmentConfig] = None, seed=None):
        self.positions = np.asarray(positions_lambda, float)
        self.config = config or ImpairmentConfig()
        n = len(self.positions)
        rng = np.random.default_rng(seed)
        # Both streams are always drawn so that the same seed gives proportionally
        # scaled errors when the standard deviations change (smooth sweeps).
        z_gain, z_phase = rng.standard_normal(n), rng.standard_normal(n)
        self.gain_error_db = z_gain * self.config.gain_std_db
        self.phase_error_deg = z_phase * self.config.phase_std_deg
        self.calibration = (10 ** (self.gain_error_db / 20)
                            * np.exp(1j * np.deg2rad(self.phase_error_deg)))
        self.coupling = self._build_coupling()

    def _build_coupling(self) -> Optional[np.ndarray]:
        """Coupling matrix, or None when coupling is off (avoids an N x N identity)."""
        cfg, n = self.config, len(self.positions)
        if cfg.coupling_nn_db is None and cfg.coupling_nnn_db is None:
            return None
        scale = 1.0
        if cfg.scale_coupling_with_spacing and n > 1:
            d = float(np.linalg.norm(self.positions[1] - self.positions[0]))
            scale = cfg.coupling_ref_spacing_lambda / d

        def coef(db, phase_deg):
            return 0.0 if db is None else scale * 10 ** (db / 20) * np.exp(1j * np.deg2rad(phase_deg))

        return coupling_matrix(n, coef(cfg.coupling_nn_db, cfg.coupling_nn_phase_deg),
                               coef(cfg.coupling_nnn_db, cfg.coupling_nnn_phase_deg))

    def response(self, theta_deg, phi_deg=0.0) -> np.ndarray:
        """Actual array response, shape (n_elements, n_angles)."""
        th, ph = np.broadcast_arrays(np.atleast_1d(np.asarray(theta_deg, float)),
                                     np.atleast_1d(np.asarray(phi_deg, float)))
        A = steering_matrix(self.positions, th, ph)
        e = cosine_power_element_pattern(th, self.config.element_exponent)
        if self.coupling is not None:
            A = self.coupling @ A
        return self.calibration[:, None] * A * e[None, :]

    def coupling_dense(self) -> np.ndarray:
        n = len(self.positions)
        return np.eye(n, dtype=complex) if self.coupling is None else self.coupling

    def vector(self, theta_deg, phi_deg=0.0) -> np.ndarray:
        return self.response(theta_deg, phi_deg)[:, 0]
