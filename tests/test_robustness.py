import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from beamforming.adaptive import conventional_weights
from beamforming.array import steering_matrix, steering_vector, ula_positions
from beamforming.impairments import (ArrayModel, ImpairmentConfig, calibration_error_variance,
                                     coupling_matrix, cosine_power_element_pattern,
                                     predicted_error_floor_db)
from beamforming.metrics import output_sinr_db, pattern_db, peak_sidelobe_level_null_based
from beamforming.robustness import (Scenario, compare_ideal_vs_nonideal, evaluate_weights,
                                    export_csv, monte_carlo, parameter_sweep,
                                    summarize_trials, taper_comparison)
from beamforming.simulation import simulate_snapshots, simulate_snapshots_model
from beamforming.tapers import (TAPER_NAMES, taper_efficiency, taper_weights, tapered_weights,
                                white_noise_gain_db)

ANGLES = np.linspace(-90, 90, 1801)


# ------------------------------------------------------------------ tapers ---
@pytest.mark.parametrize("name", TAPER_NAMES)
def test_taper_is_positive_symmetric_peak_one(name):
    t = taper_weights(name, 12)
    assert t.shape == (12,) and np.all(t > 0)
    assert np.allclose(t, t[::-1]) and t.max() == pytest.approx(1.0)


def test_unknown_taper_raises():
    with pytest.raises(ValueError):
        taper_weights("triangle", 8)


def test_uniform_taper_equals_conventional_and_all_are_distortionless():
    pos = ula_positions(12, 0.5)
    a = steering_vector(pos, 20)
    assert np.allclose(tapered_weights(a, taper_weights("uniform", 12)), conventional_weights(a))
    for name in TAPER_NAMES:
        w = tapered_weights(a, taper_weights(name, 12))
        assert np.vdot(w, a) == pytest.approx(1.0)


def test_taper_efficiency_and_wng():
    assert taper_efficiency(np.ones(10)) == pytest.approx(1.0)
    pos = ula_positions(12, 0.5)
    a = steering_vector(pos, 0)
    for name in TAPER_NAMES:
        t = taper_weights(name, 12)
        w = tapered_weights(a, t)
        assert white_noise_gain_db(w, a) == pytest.approx(10 * np.log10(12 * taper_efficiency(t)))
    assert taper_efficiency(taper_weights("hann", 12)) < 1


def test_tapers_trade_sidelobes_for_beamwidth():
    df, _, _ = taper_comparison(Scenario(n_elements=24, target_deg=0.0))
    d = df.set_index("taper")
    assert d.loc["Uniform", "psl_db"] == pytest.approx(-13.2, abs=0.5)
    assert d.loc["Uniform", "psl_db"] > d.loc["Hann", "psl_db"] > d.loc["Hamming", "psl_db"] > d.loc["Blackman", "psl_db"]
    assert d.loc["Kaiser", "psl_db"] < d.loc["Hann", "psl_db"]
    assert (d.drop("Uniform")["hpbw_deg"] > d.loc["Uniform", "hpbw_deg"]).all()
    assert (d.drop("Uniform")["taper_loss_db"] > 0).all()


def test_null_based_psl_of_large_uniform_array():
    pos = ula_positions(64, 0.5)
    w = conventional_weights(steering_vector(pos, 0))
    p = pattern_db(pos, w, ANGLES)
    assert peak_sidelobe_level_null_based(ANGLES, p, int(np.argmax(p))) == pytest.approx(-13.26, abs=0.3)


# ------------------------------------------------------------ element/model ---
def test_cosine_power_element_pattern():
    assert np.allclose(cosine_power_element_pattern([-80, 0, 120], 0.0), 1.0)
    assert cosine_power_element_pattern(60.0, 1.0) == pytest.approx(0.5)
    assert cosine_power_element_pattern(60.0, 2.0) == pytest.approx(0.25)
    assert cosine_power_element_pattern(120.0, 1.0) == 0.0


def test_steering_matrix_matches_steering_vector():
    pos = ula_positions(8, 0.6)
    A = steering_matrix(pos, [-30, 0, 45])
    for k, th in enumerate([-30, 0, 45]):
        assert np.allclose(A[:, k], steering_vector(pos, th))


def test_ideal_model_reproduces_reference_implementation():
    pos = ula_positions(12, 0.5)
    m = ArrayModel(pos, ImpairmentConfig(), seed=1)
    assert np.allclose(m.response(ANGLES), steering_matrix(pos, ANGLES))
    w = conventional_weights(steering_vector(pos, 10))
    p_new = 20 * np.log10(np.abs(w.conj() @ m.response(ANGLES)))
    assert np.allclose(p_new, pattern_db(pos, w, ANGLES), atol=1e-6)


def test_ideal_model_snapshots_equal_reference_snapshots():
    pos = ula_positions(12, 0.5)
    X0, s0, _ = simulate_snapshots(pos, 10, [-35, 50], 256, 1.0, [10.0, 10.0], 0.1, seed=5)
    X1, s1 = simulate_snapshots_model(ArrayModel(pos), 10, [-35, 50], 256, 1.0, [10.0, 10.0], 0.1, seed=5)
    assert np.array_equal(X0, X1) and np.array_equal(s0, s1)


def test_actual_sinr_matches_reference_for_ideal_array():
    sc = Scenario()
    pos = sc.positions()
    w = conventional_weights(steering_vector(pos, sc.target_deg))
    m = evaluate_weights(ArrayModel(pos), w, sc)
    ref = output_sinr_db(w, sc.target_deg, sc.interferer_deg, sc.desired_power,
                         [sc.interference_power] * 2, sc.noise_power, pos)
    assert m["sinr_db"] == pytest.approx(ref)


# ------------------------------------------------------------- calibration ---
def test_calibration_errors_reproducible_and_zero_when_off():
    pos = ula_positions(12, 0.5)
    cfg = ImpairmentConfig(gain_std_db=1.0, phase_std_deg=5.0)
    assert np.array_equal(ArrayModel(pos, cfg, seed=3).calibration, ArrayModel(pos, cfg, seed=3).calibration)
    assert not np.array_equal(ArrayModel(pos, cfg, seed=3).calibration, ArrayModel(pos, cfg, seed=4).calibration)
    assert np.allclose(ArrayModel(pos, ImpairmentConfig(), seed=3).calibration, 1.0)


def test_calibration_error_statistics_match_request():
    pos = ula_positions(50000, 0.5)
    m = ArrayModel(pos, ImpairmentConfig(gain_std_db=1.0, phase_std_deg=6.0), seed=0)
    assert m.gain_error_db.std() == pytest.approx(1.0, rel=0.03)
    assert m.phase_error_deg.std() == pytest.approx(6.0, rel=0.03)


def test_random_error_power_matches_theory_at_a_null():
    # At a null of the uniform pattern the coherent term is zero, so the mean power equals
    # Var(Gamma) * ||w||^2 exactly in expectation.
    pos = ula_positions(12, 0.5)
    w = conventional_weights(steering_vector(pos, 0))
    th_null = np.rad2deg(np.arcsin(1 / 6))
    a = steering_vector(pos, th_null)
    assert abs(np.vdot(w, a)) < 1e-12
    cfg = ImpairmentConfig(gain_std_db=1.0, phase_std_deg=8.0)
    ss = np.random.SeedSequence(11).spawn(4000)
    power = np.mean([abs(np.vdot(w, ArrayModel(pos, cfg, seed=s).vector(th_null))) ** 2 for s in ss])
    expected = 10 ** (predicted_error_floor_db(w, 1.0, 8.0) / 10)
    assert power == pytest.approx(expected, rel=0.1)
    assert expected == pytest.approx(calibration_error_variance(1.0, 8.0) / 12, rel=1e-9)


def test_same_seed_gives_scaled_errors():
    pos = ula_positions(12, 0.5)
    a = ArrayModel(pos, ImpairmentConfig(phase_std_deg=2.0), seed=9).phase_error_deg
    b = ArrayModel(pos, ImpairmentConfig(phase_std_deg=4.0), seed=9).phase_error_deg
    assert np.allclose(2 * a, b)


# ---------------------------------------------------------------- coupling ---
def test_coupling_matrix_structure():
    M = coupling_matrix(6, 0.1 - 0.05j, 0.02)
    assert np.allclose(M, M.T) and np.allclose(np.diag(M), 1)
    assert M[0, 1] == 0.1 - 0.05j and M[0, 2] == 0.02 and M[0, 3] == 0
    assert np.allclose(coupling_matrix(5), np.eye(5))


def test_coupling_too_strong_raises():
    with pytest.raises(ValueError):
        coupling_matrix(8, 0.6, 0.0)


def test_coupling_effect_on_interior_elements_is_analytic():
    d, th = 0.5, 25.0
    pos = ula_positions(10, d)
    c1, c2 = 10 ** (-15 / 20), 10 ** (-25 / 20)
    m = ArrayModel(pos, ImpairmentConfig(coupling_nn_db=-15, coupling_nnn_db=-25))
    ratio = m.vector(th) / steering_vector(pos, th)
    k = 2 * np.pi * d * np.sin(np.deg2rad(th))
    expected = 1 + 2 * c1 * np.cos(k) + 2 * c2 * np.cos(2 * k)
    assert np.allclose(ratio[2:-2], expected)
    assert not np.isclose(ratio[0], expected)       # edge elements have fewer neighbours


def test_coupling_spacing_scaling():
    cfg = ImpairmentConfig(coupling_nn_db=-20, scale_coupling_with_spacing=True)
    m1 = ArrayModel(ula_positions(6, 0.5), cfg).coupling[0, 1]
    m2 = ArrayModel(ula_positions(6, 1.0), cfg).coupling[0, 1]
    assert abs(m1) == pytest.approx(0.1) and abs(m2) == pytest.approx(0.05)


# ------------------------------------------------------------ experiments ---
def test_ideal_vs_nonideal_with_ideal_config_has_no_change():
    out = compare_ideal_vs_nonideal(Scenario(), ImpairmentConfig(), seed=1, beamformers=("Conventional", "MVDR"))
    assert np.allclose(out["metrics"]["change"].to_numpy(dtype=float), 0.0, atol=1e-9)


def test_monte_carlo_reproducible_and_paired_baseline():
    sc, imp = Scenario(), ImpairmentConfig(gain_std_db=0.5, phase_std_deg=5)
    a = monte_carlo(sc, imp, 8, seed=2)
    b = monte_carlo(sc, imp, 8, seed=2)
    c = monte_carlo(sc, imp, 8, seed=3)
    pd.testing.assert_frame_equal(a.trials, b.trials)
    assert not a.trials["sinr_db"].equals(c.trials["sinr_db"])
    assert a.patterns["MVDR"].shape == (8, len(ANGLES))
    conv = a.trials[a.trials.beamformer == "Conventional"]
    assert conv["sinr_db_ideal"].nunique() == 1       # deterministic ideal baseline
    assert np.allclose(conv["sinr_loss_db"], conv["sinr_db_ideal"] - conv["sinr_db"])


def test_calibration_errors_hurt_mvdr_far_more_than_conventional():
    sc, imp = Scenario(), ImpairmentConfig(gain_std_db=0.5, phase_std_deg=5)
    t = monte_carlo(sc, imp, 30, seed=1, keep_patterns=False).trials
    loss = t.groupby("beamformer")["sinr_loss_db"].mean()
    assert loss["MVDR"] > 5 and abs(loss["Conventional"]) < 1.5
    heavy = monte_carlo(replace(sc, mvdr_loading=0.1), imp, 30, seed=1, beamformers=("MVDR",),
                        keep_patterns=False).trials
    assert heavy["sinr_loss_db"].mean() < loss["MVDR"] / 4     # diagonal loading restores robustness


def test_low_sidelobe_taper_floor_is_error_limited():
    sc = Scenario(n_elements=24, target_deg=0.0, taper="Blackman")
    imp = ImpairmentConfig(gain_std_db=1.0, phase_std_deg=10.0)
    t = monte_carlo(sc, imp, 20, seed=0, beamformers=("Conventional",), keep_patterns=False).trials
    assert t["psl_degradation_db"].mean() > 15      # Blackman's -58 dB is destroyed by errors


def test_parameter_sweep_shape_columns_and_grating_lobes():
    sc, imp = Scenario(), ImpairmentConfig(gain_std_db=0.5, phase_std_deg=5)
    df = parameter_sweep(sc, imp, {"spacing_lambda": [0.5, 1.0]}, n_trials=4, seed=0)
    assert len(df) == 2 and list(df["grating_lobes"]) == [0, 1]
    assert {"psl_db_mean", "psl_db_std", "sinr_loss_db_mean", "hpbw_deg_ideal_mean"} <= set(df.columns)
    grid = parameter_sweep(sc, imp, {"gain_std_db": [0.0, 1.0], "phase_std_deg": [0.0, 10.0]},
                           n_trials=4, beamformers=("Conventional", "MVDR"))
    assert len(grid) == 8
    z = grid[(grid.gain_std_db == 0) & (grid.phase_std_deg == 0) & (grid.beamformer == "MVDR")]
    assert z["sinr_loss_db_mean"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        parameter_sweep(sc, imp, {"nonsense": [1]})


def test_sinr_loss_grows_with_calibration_error_for_mvdr():
    sc, imp = Scenario(), ImpairmentConfig()
    df = parameter_sweep(sc, imp, {"phase_std_deg": [0.0, 2.0, 8.0]}, n_trials=15, seed=4,
                         beamformers=("MVDR",))
    assert np.all(np.diff(df["sinr_loss_db_mean"].to_numpy()) > 0)


def test_summary_and_csv_roundtrip(tmp_path):
    r = monte_carlo(Scenario(), ImpairmentConfig(gain_std_db=0.5, phase_std_deg=3), 6, seed=0,
                    keep_patterns=False)
    s = summarize_trials(r.trials)
    assert {"mean", "std", "p05", "p50", "p95"} <= set(s.columns)
    path = export_csv(r.trials, str(tmp_path / "sub" / "trials.csv"))
    back = pd.read_csv(path)
    assert len(back) == len(r.trials) and "sinr_loss_db" in back.columns
