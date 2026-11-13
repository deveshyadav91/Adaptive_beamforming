"""Robustness experiments: ideal vs non-ideal comparison, sweeps, Monte Carlo, CSV export.

Design rule: beamformers are designed with the *nominal* steering vector (the processor
does not know the calibration errors, coupling or element pattern), while snapshots and
all performance metrics use the *actual* array response. Ideal baselines are paired with
the non-ideal runs (same snapshot seed), so differences isolate the hardware effects.
All dB patterns here are absolute: 0 dB = distortionless gain of an ideal array.
Peak sidelobe level, HPBW and null depths are relative to the main-lobe peak.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field, fields, replace
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .adaptive import lms_train, mvdr_weights
from .array import db20, steering_vector, ula_positions
from .impairments import ArrayModel, ImpairmentConfig
from .metrics import (hpbw_deg_robust, main_lobe_peak_index,
                      peak_sidelobe_level_null_based, visible_grating_lobes_ula)
from .simulation import simulate_snapshots_model
from .tapers import (TAPER_NAMES, taper_efficiency, taper_weights, tapered_weights,
                     white_noise_gain_db)

DEFAULT_ANGLES = np.linspace(-90, 90, 1801)
BEAMFORMERS = ("Conventional", "MVDR", "LMS")
KEY_METRICS = ("abs_pointing_error_deg", "target_gain_db", "hpbw_deg", "psl_db", "sinr_db")
SWEEP_METRICS = KEY_METRICS + ("sinr_loss_db", "psl_degradation_db")


@dataclass(frozen=True)
class Scenario:
    n_elements: int = 12
    spacing_lambda: float = 0.5
    target_deg: float = 10.0
    interferer_deg: tuple = (-35.0, 50.0)
    desired_power: float = 1.0
    interference_power: float = 10.0
    noise_power: float = 0.1
    n_snapshots: int = 2048
    mvdr_loading: float = 1e-3
    lms_mu: float = 1e-3
    lms_epochs: int = 3
    taper: str = "Uniform"
    kaiser_beta: float = 6.0

    def __post_init__(self):
        object.__setattr__(self, "interferer_deg", tuple(float(a) for a in self.interferer_deg))
        if self.n_elements < 2 or self.spacing_lambda <= 0:
            raise ValueError("Need at least 2 elements and positive spacing.")
        taper_weights(self.taper, self.n_elements, self.kaiser_beta)  # validates name/beta

    def positions(self) -> np.ndarray:
        return ula_positions(self.n_elements, self.spacing_lambda)


@dataclass
class MonteCarloResult:
    trials: pd.DataFrame
    angles: np.ndarray
    patterns: Dict[str, np.ndarray] = field(default_factory=dict)        # (n_trials, n_angles), abs dB
    ideal_patterns: Dict[str, np.ndarray] = field(default_factory=dict)  # trial-0 ideal, abs dB


# ------------------------------------------------------------------ design ---
def design_beamformers(model: ArrayModel, scenario: Scenario, snapshot_seed,
                       beamformers: Sequence[str] = ("Conventional", "MVDR")) -> Dict[str, np.ndarray]:
    unknown = [b for b in beamformers if b not in BEAMFORMERS]
    if unknown:
        raise ValueError(f"Unknown beamformer(s) {unknown}; choose from {BEAMFORMERS}.")
    a0 = steering_vector(scenario.positions(), scenario.target_deg)   # nominal
    w_conv = tapered_weights(a0, taper_weights(scenario.taper, scenario.n_elements,
                                               scenario.kaiser_beta))
    X = s = None
    if any(b in ("MVDR", "LMS") for b in beamformers):
        powers = [scenario.interference_power] * len(scenario.interferer_deg)
        X, s = simulate_snapshots_model(model, scenario.target_deg, scenario.interferer_deg,
                                        scenario.n_snapshots, scenario.desired_power, powers,
                                        scenario.noise_power, seed=snapshot_seed)
    out = {}
    for b in beamformers:
        if b == "Conventional":
            out[b] = w_conv
        elif b == "MVDR":
            out[b] = mvdr_weights(X, a0, scenario.mvdr_loading)
        else:
            out[b], _ = lms_train(X, s, scenario.lms_mu, scenario.lms_epochs, w_conv)
    return out


# -------------------------------------------------------------- evaluation ---
def evaluate_weights(model: ArrayModel, weights, scenario: Scenario,
                     angles=DEFAULT_ANGLES, search_half_width_deg: float = 6.0,
                     return_pattern: bool = False):
    """Pattern + SINR metrics of `weights` on the actual array `model`."""
    w = np.asarray(weights, complex)
    angles = np.asarray(angles, float)
    abs_db = db20(w.conj() @ model.response(angles))
    pk = main_lobe_peak_index(angles, abs_db, scenario.target_deg, search_half_width_deg)
    rel = abs_db - abs_db[pk]

    v0 = model.vector(scenario.target_deg)
    sig = scenario.desired_power * abs(np.vdot(w, v0)) ** 2
    interf = sum(scenario.interference_power * abs(np.vdot(w, model.vector(a))) ** 2
                 for a in scenario.interferer_deg)
    noise = scenario.noise_power * np.vdot(w, w).real

    peak_angle = float(angles[pk])
    m = {
        "peak_angle_deg": peak_angle,
        "pointing_error_deg": peak_angle - scenario.target_deg,
        "abs_pointing_error_deg": abs(peak_angle - scenario.target_deg),
        "peak_gain_db": float(abs_db[pk]),
        "target_gain_db": float(db20(np.vdot(w, v0))),
        "hpbw_deg": hpbw_deg_robust(angles, rel, peak_angle),
        "psl_db": peak_sidelobe_level_null_based(angles, rel, pk),
        "sinr_db": float(10 * np.log10(max(sig, 1e-15) / max(interf + noise, 1e-15))),
    }
    for k, ang in enumerate(scenario.interferer_deg, 1):
        mask = abs(angles - ang) <= 1.0
        m[f"null_int{k}_db"] = float(rel[mask].min() if mask.any()
                                     else rel[np.argmin(abs(angles - ang))])
    return (m, abs_db) if return_pattern else m


# ------------------------------------------------------- single comparison ---
def compare_ideal_vs_nonideal(scenario: Scenario, imp: ImpairmentConfig, seed: int = 0,
                              beamformers: Sequence[str] = BEAMFORMERS, angles=DEFAULT_ANGLES):
    """One calibration realisation vs the ideal array (same snapshot noise)."""
    cal_ss, snap_ss = np.random.SeedSequence(seed).spawn(2)
    pos = scenario.positions()
    ideal, real = ArrayModel(pos, ImpairmentConfig()), ArrayModel(pos, imp, seed=cal_ss)
    w_i = design_beamformers(ideal, scenario, snap_ss, beamformers)
    w_r = design_beamformers(real, scenario, snap_ss, beamformers)
    patterns, rows = {}, []
    for b in beamformers:
        mi, pi = evaluate_weights(ideal, w_i[b], scenario, angles, return_pattern=True)
        mr, pr = evaluate_weights(real, w_r[b], scenario, angles, return_pattern=True)
        patterns[b] = {"ideal": pi, "nonideal": pr}
        for key in mi:
            rows.append({"beamformer": b, "metric": key, "ideal": mi[key],
                         "nonideal": mr[key], "change": mr[key] - mi[key]})
    return {"angles": np.asarray(angles), "patterns": patterns,
            "metrics": pd.DataFrame(rows), "model": real}


def taper_comparison(scenario: Scenario, imp: Optional[ImpairmentConfig] = None,
                     seed: int = 0, angles=DEFAULT_ANGLES):
    """Conventional beamformer for every taper (ideal array unless `imp` is given)."""
    pos = scenario.positions()
    model = ArrayModel(pos, imp or ImpairmentConfig(), seed=seed)
    a0 = steering_vector(pos, scenario.target_deg)
    rows, patterns = [], {}
    for name in TAPER_NAMES:
        sc = replace(scenario, taper=name)
        t = taper_weights(name, sc.n_elements, sc.kaiser_beta)
        w = tapered_weights(a0, t)
        m, p = evaluate_weights(model, w, sc, angles, return_pattern=True)
        patterns[name] = p
        rows.append({"taper": name, "taper_efficiency": taper_efficiency(t),
                     "taper_loss_db": -10 * np.log10(taper_efficiency(t)),
                     "wng_db": white_noise_gain_db(w, a0), "hpbw_deg": m["hpbw_deg"],
                     "psl_db": m["psl_db"], "target_gain_db": m["target_gain_db"],
                     "sinr_db": m["sinr_db"]})
    return pd.DataFrame(rows), patterns, np.asarray(angles)


# ------------------------------------------------------------- Monte Carlo ---
def monte_carlo(scenario: Scenario, imp: ImpairmentConfig, n_trials: int = 100, seed: int = 0,
                beamformers: Sequence[str] = ("Conventional", "MVDR"),
                angles=DEFAULT_ANGLES, keep_patterns: bool = True) -> MonteCarloResult:
    """Random calibration errors (and snapshot noise) per trial, paired ideal baseline."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    angles = np.asarray(angles, float)
    pos = scenario.positions()
    ideal = ArrayModel(pos, ImpairmentConfig())
    rows, pats = [], {b: [] for b in beamformers}
    ideal_pats = {}
    for k, child in enumerate(np.random.SeedSequence(seed).spawn(n_trials)):
        cal_ss, snap_ss = child.spawn(2)
        real = ArrayModel(pos, imp, seed=cal_ss)
        w_r = design_beamformers(real, scenario, snap_ss, beamformers)
        w_i = design_beamformers(ideal, scenario, snap_ss, beamformers)
        for b in beamformers:
            mr, pr = evaluate_weights(real, w_r[b], scenario, angles, return_pattern=True)
            mi, pi = evaluate_weights(ideal, w_i[b], scenario, angles, return_pattern=True)
            row = {"trial": k, "beamformer": b, **mr}
            for key in KEY_METRICS:
                row[f"{key}_ideal"] = mi[key]
            row["sinr_loss_db"] = mi["sinr_db"] - mr["sinr_db"]
            row["psl_degradation_db"] = mr["psl_db"] - mi["psl_db"]
            rows.append(row)
            if keep_patterns:
                pats[b].append(pr.astype(np.float32))
                if k == 0:
                    ideal_pats[b] = pi
    return MonteCarloResult(
        pd.DataFrame(rows), angles,
        {b: np.vstack(v) for b, v in pats.items() if v} if keep_patterns else {}, ideal_pats)


def summarize_trials(trials: pd.DataFrame, metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Tidy statistics (mean/std/percentiles) per beamformer and metric."""
    if metrics is None:
        metrics = list(SWEEP_METRICS) + [c for c in trials.columns if c.startswith("null_int")]
    rows = []
    for b, g in trials.groupby("beamformer", sort=False):
        for m in metrics:
            x = g[m].dropna()
            rows.append({"beamformer": b, "metric": m, "n_valid": len(x),
                         "mean": x.mean(), "std": x.std(ddof=1) if len(x) > 1 else np.nan,
                         "p05": x.quantile(0.05), "p50": x.quantile(0.5), "p95": x.quantile(0.95)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sweeps ---
def parameter_sweep(scenario: Scenario, imp: ImpairmentConfig, sweep: Dict[str, Sequence],
                    n_trials: int = 50, seed: int = 0,
                    beamformers: Sequence[str] = ("Conventional",)) -> pd.DataFrame:
    """Cartesian sweep over Scenario and/or ImpairmentConfig fields, each point a Monte Carlo
    average. Every point uses the same trial seeds (common random numbers)."""
    sc_names = {f.name for f in fields(Scenario)}
    im_names = {f.name for f in fields(ImpairmentConfig)}
    bad = [k for k in sweep if k not in sc_names | im_names]
    if bad:
        raise ValueError(f"Unknown sweep parameter(s): {bad}")
    keys = list(sweep)
    rows = []
    for combo in itertools.product(*[sweep[k] for k in keys]):
        upd = dict(zip(keys, combo))
        sc = replace(scenario, **{k: v for k, v in upd.items() if k in sc_names})
        im = replace(imp, **{k: v for k, v in upd.items() if k in im_names})
        res = monte_carlo(sc, im, n_trials, seed, beamformers, keep_patterns=False)
        n_gl = len(visible_grating_lobes_ula(sc.spacing_lambda, sc.target_deg))
        for b, g in res.trials.groupby("beamformer", sort=False):
            row = {**upd, "beamformer": b, "n_trials": n_trials, "grating_lobes": n_gl}
            for m in SWEEP_METRICS:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
            for m in KEY_METRICS:
                row[f"{m}_ideal_mean"] = g[f"{m}_ideal"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- CSV ---
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def export_csv(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    return path
