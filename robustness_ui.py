"""Streamlit rendering for the robustness tabs (real-world array effects)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from beamforming.array import steering_vector
from beamforming.impairments import (ArrayModel, ImpairmentConfig, cosine_power_element_pattern,
                                     predicted_error_floor_db)
from beamforming.robustness import (BEAMFORMERS, Scenario, compare_ideal_vs_nonideal,
                                    evaluate_weights, monte_carlo, parameter_sweep,
                                    summarize_trials, taper_comparison, to_csv_bytes)
from beamforming.tapers import TAPER_NAMES, taper_weights, tapered_weights

LABELS = {"psl_db": "Peak sidelobe level (dB)", "sinr_db": "Output SINR (dB)",
          "sinr_loss_db": "SINR loss vs ideal (dB, negative = gain)", "hpbw_deg": "HPBW (°)",
          "abs_pointing_error_deg": "|Pointing error| (°)",
          "target_gain_db": "Target-direction gain (dB)",
          "psl_degradation_db": "PSL degradation vs ideal (dB)"}
YLAB = "Response (dB, 0 dB = ideal distortionless gain)"


def sidebar_controls():
    """Array-effect widgets; returns (taper, kaiser_beta, ImpairmentConfig, calibration_seed)."""
    with st.sidebar:
        st.divider()
        st.header("Real-world array effects")
        taper = st.selectbox("Amplitude taper (conventional / LMS start)", list(TAPER_NAMES), 0)
        beta = st.slider("Kaiser β", 0.0, 12.0, 6.0, 0.5, disabled=taper != "Kaiser")
        p_el = st.slider("Element pattern exponent p  (cos^p, 0 = isotropic)", 0.0, 3.0, 1.0, 0.1)
        g_std = st.slider("Gain error σ (dB)", 0.0, 3.0, 0.5, 0.05)
        ph_std = st.slider("Phase error σ (°)", 0.0, 30.0, 5.0, 0.5)
        cpl = st.checkbox("Mutual coupling", True)
        c1 = st.slider("Nearest-neighbour coupling (dB)", -40, -6, -20, disabled=not cpl)
        c2 = st.slider("Next-neighbour coupling (dB)", -50, -10, -30, disabled=not cpl)
        with st.expander("Coupling details"):
            ph1 = st.slider("Nearest-neighbour phase (°)", -180, 180, -90, 15, disabled=not cpl)
            ph2 = st.slider("Next-neighbour phase (°)", -180, 180, -90, 15, disabled=not cpl)
            scale = st.checkbox("Scale coupling ∝ 1/spacing (heuristic)", False, disabled=not cpl,
                                help="Coupling values above are for 0.5λ spacing.")
        seed = st.number_input("Calibration-error seed", 0, 10**6, 1)
    imp = ImpairmentConfig(
        element_exponent=float(p_el), gain_std_db=float(g_std), phase_std_deg=float(ph_std),
        coupling_nn_db=float(c1) if cpl else None, coupling_nnn_db=float(c2) if cpl else None,
        coupling_nn_phase_deg=float(ph1), coupling_nnn_phase_deg=float(ph2),
        scale_coupling_with_spacing=bool(scale))
    return taper, float(beta), imp, int(seed)


def _pick_beamformers(key, default=("Conventional", "MVDR")):
    return tuple(st.multiselect("Beamformers", BEAMFORMERS, default=list(default), key=key))


def _remember(key, data, scn, imp):
    st.session_state[key] = {"data": data, "meta": f"{scn}\n{imp}"}


def _show_meta(key):
    with st.expander("Settings used for these results"):
        st.code(st.session_state[key]["meta"])
        st.caption("Results stay on screen until you re-run; change settings then press Run again.")


# ----------------------------------------------------------------- tab 1 ---
def render_taper_tab(scn, imp):
    st.subheader("Amplitude tapering (ideal array)")
    df, pats, ang = taper_comparison(scn)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, p in pats.items():
        ax.plot(ang, p, lw=2.4 if name == scn.taper else 1.1, label=name)
    ax.axvline(scn.target_deg, color="green", ls="--", lw=0.8)
    ax.set(xlim=(-90, 90), ylim=(-90, 3), xlabel="Angle from broadside (°)", ylabel=YLAB)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(df.round(2), hide_index=True)
    st.download_button("Download taper comparison CSV", to_csv_bytes(df), "taper_comparison.csv", "text/csv")
    st.caption("Taper loss / WNG quantify the gain given up; HPBW grows as sidelobes fall. "
               "Hann and Blackman use n+2-point windows with the zero ends dropped.")

    st.subheader("Element radiation pattern")
    if imp.element_exponent == 0:
        st.info("Set the element-pattern exponent above 0 in the sidebar to see its effect.")
        return
    pos = scn.positions()
    w = tapered_weights(steering_vector(pos, scn.target_deg),
                        taper_weights(scn.taper, scn.n_elements, scn.kaiser_beta))
    iso, el = ArrayModel(pos), ArrayModel(pos, ImpairmentConfig(element_exponent=imp.element_exponent))
    m0, p0 = evaluate_weights(iso, w, scn, ang, return_pattern=True)
    m1, p1 = evaluate_weights(el, w, scn, ang, return_pattern=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(ang, p0, label="Array factor (isotropic)")
    ax.plot(ang, p1, label=f"With cos^{imp.element_exponent:g} element pattern")
    ax.plot(ang, 20 * np.log10(np.maximum(cosine_power_element_pattern(ang, imp.element_exponent), 1e-12)),
            "k:", label="Element pattern alone")
    ax.axvline(scn.target_deg, color="green", ls="--", lw=0.8)
    ax.set(xlim=(-90, 90), ylim=(-70, 3), xlabel="Angle from broadside (°)", ylabel=YLAB)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    c = st.columns(3)
    c[0].metric("Scan loss at target (dB)", f"{m1['target_gain_db']:.2f}")
    c[1].metric("Peak shift (°)", f"{m1['peak_angle_deg'] - m0['peak_angle_deg']:+.1f}")
    c[2].metric("PSL change (dB)", f"{m1['psl_db'] - m0['psl_db']:+.2f}")


# ----------------------------------------------------------------- tab 2 ---
@st.cache_data(show_spinner=False)
def _compare(scn, imp, seed, beams):
    out = compare_ideal_vs_nonideal(scn, imp, seed, beams)
    model = out.pop("model")
    out["errors"] = pd.DataFrame({"element": np.arange(1, len(model.positions) + 1),
                                  "gain_error_db": model.gain_error_db,
                                  "phase_error_deg": model.phase_error_deg})
    return out


def render_compare_tab(scn, imp, seed):
    st.subheader("Ideal vs non-ideal array (one hardware realisation)")
    beams = _pick_beamformers("s5_cmp_beams", BEAMFORMERS)
    if not beams:
        st.info("Select at least one beamformer."); return
    out = _compare(scn, imp, seed, beams)
    ang = out["angles"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, b in enumerate(beams):
        col = f"C{i}"
        ax.plot(ang, out["patterns"][b]["ideal"], color=col, label=f"{b} – ideal")
        ax.plot(ang, out["patterns"][b]["nonideal"], color=col, ls="--", label=f"{b} – non-ideal")
    ax.axvline(scn.target_deg, color="green", ls=":", lw=1)
    for a in scn.interferer_deg:
        ax.axvline(a, color="red", ls=":", lw=1)
    ax.set(xlim=(-90, 90), ylim=(-80, 12), xlabel="Angle from broadside (°)", ylabel=YLAB)
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(out["metrics"].round(2), hide_index=True)
    st.download_button("Download comparison CSV", to_csv_bytes(out["metrics"]),
                       "ideal_vs_nonideal.csv", "text/csv")
    st.caption("Beamformers are designed with the nominal steering vector; snapshots and metrics use the "
               "actual array. Pointing error saturates at the ±6° peak-search window, so a value of 6° "
               "means the main lobe was not found there. PSL/HPBW are less meaningful for a distorted adaptive pattern.")
    with st.expander("Realised calibration errors"):
        st.dataframe(out["errors"].round(3), hide_index=True)


# ----------------------------------------------------------------- tab 3 ---
def _line_plot(df, xcol, metric, title):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, (b, g) in enumerate(df.groupby("beamformer", sort=False)):
        g = g.sort_values(xcol)
        x, m, s = g[xcol].to_numpy(), g[f"{metric}_mean"].to_numpy(float), g[f"{metric}_std"].to_numpy(float)
        ax.plot(x, m, marker="o", color=f"C{i}", label=f"{b} (non-ideal mean)")
        if np.isfinite(s).any():
            ax.fill_between(x, m - np.nan_to_num(s), m + np.nan_to_num(s), color=f"C{i}", alpha=.15)
        if f"{metric}_ideal_mean" in g:
            ax.plot(x, g[f"{metric}_ideal_mean"], color=f"C{i}", ls="--", label=f"{b} (ideal)")
    gl = df.loc[df["grating_lobes"] > 0, xcol]
    if xcol == "spacing_lambda" and len(gl):
        ax.axvline(gl.min(), color="k", ls=":", lw=1, label="first predicted grating lobe")
    ax.set(xlabel=xcol, ylabel=LABELS[metric], title=title)
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    return fig


def render_sweep_tab(scn, imp, seed):
    st.subheader("Robustness sweeps (each point = Monte Carlo mean ± σ)")
    c = st.columns(3)
    trials = c[0].slider("Trials per point", 5, 200, 40, 5, key="s5_sw_trials")
    metric = c[1].selectbox("Metric to plot", list(LABELS), 0, format_func=LABELS.get, key="s5_sw_metric")
    with c[2]:
        beams = _pick_beamformers("s5_sw_beams")
    if not beams:
        st.info("Select at least one beamformer."); return

    st.markdown("**Element-spacing sweep** (non-ideal array vs its paired ideal baseline)")
    sp_all = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    spacings = st.multiselect("Spacing values (λ)", sp_all, [0.3, 0.4, 0.5, 0.6, 0.7, 1.0], key="s5_sp_vals")
    if st.button("Run spacing sweep", disabled=not spacings):
        try:
            with st.spinner("Running spacing sweep…"):
                _remember("s5_spacing", parameter_sweep(scn, imp, {"spacing_lambda": sorted(spacings)},
                                                        trials, seed, beams), scn, imp)
        except ValueError as exc:
            st.error(str(exc))
    if "s5_spacing" in st.session_state:
        df = st.session_state["s5_spacing"]["data"]
        if f"{metric}_mean" in df:
            fig = _line_plot(df, "spacing_lambda", metric, "Spacing sweep"); st.pyplot(fig); plt.close(fig)
        st.dataframe(df.round(3), hide_index=True)
        st.download_button("Download spacing sweep CSV", to_csv_bytes(df), "sweep_spacing.csv", "text/csv")
        _show_meta("s5_spacing")

    st.divider()
    st.markdown("**Calibration-error grid** (gain σ × phase σ)")
    c = st.columns(2)
    gv = c[0].multiselect("Gain σ values (dB)", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0], [0.0, 0.25, 0.5, 1.0, 2.0], key="s5_g_vals")
    pv = c[1].multiselect("Phase σ values (°)", [0.0, 1.0, 2.0, 5.0, 10.0, 20.0], [0.0, 2.0, 5.0, 10.0, 20.0], key="s5_p_vals")
    if st.button("Run calibration grid", disabled=not (gv and pv)):
        try:
            with st.spinner("Running calibration grid…"):
                _remember("s5_grid", parameter_sweep(scn, imp, {"gain_std_db": sorted(gv), "phase_std_deg": sorted(pv)},
                                                     trials, seed, beams), scn, imp)
        except ValueError as exc:
            st.error(str(exc))
    if "s5_grid" in st.session_state:
        df = st.session_state["s5_grid"]["data"]
        pick = st.selectbox("Beamformer for heatmap", list(df["beamformer"].unique()), key="s5_grid_b")
        piv = df[df.beamformer == pick].pivot(index="gain_std_db", columns="phase_std_deg",
                                              values=f"{metric}_mean")
        fig, ax = plt.subplots(figsize=(7, 4.4))
        im = ax.imshow(piv.to_numpy(float), origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)), [f"{v:g}" for v in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{v:g}" for v in piv.index])
        for (i, j), v in np.ndenumerate(piv.to_numpy(float)):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="w", fontsize=8)
        ax.set(xlabel="Phase error σ (°)", ylabel="Gain error σ (dB)", title=f"{pick}: {LABELS[metric]}")
        fig.colorbar(im, ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.dataframe(df.round(3), hide_index=True)
        st.download_button("Download calibration grid CSV", to_csv_bytes(df), "sweep_calibration.csv", "text/csv")
        _show_meta("s5_grid")


# ----------------------------------------------------------------- tab 4 ---
def render_mc_tab(scn, imp, seed):
    st.subheader("Monte Carlo over random hardware-error realisations")
    c = st.columns(4)
    n_trials = c[0].slider("Trials", 10, 500, 100, 10, key="s5_mc_n")
    mc_seed = c[1].number_input("Monte Carlo seed", 0, 10**6, 2026, key="s5_mc_seed")
    with_lms = c[2].checkbox("Include LMS (slower)", False, key="s5_mc_lms")
    beams = ("Conventional", "MVDR") + (("LMS",) if with_lms else ())
    if imp.gain_std_db == 0 and imp.phase_std_deg == 0:
        st.info("Gain/phase σ are 0, so trials differ only through snapshot noise.")
    if st.button("Run Monte Carlo", type="primary"):
        try:
            with st.spinner("Running trials…"):
                _remember("s5_mc", monte_carlo(scn, imp, n_trials, int(mc_seed), beams), scn, imp)
        except ValueError as exc:
            st.error(str(exc))
    if "s5_mc" not in st.session_state:
        return
    res = st.session_state["s5_mc"]["data"]
    summ = summarize_trials(res.trials)
    st.dataframe(summ.round(2), hide_index=True)
    d = st.columns(2)
    d[0].download_button("Download all trials CSV", to_csv_bytes(res.trials), "monte_carlo_trials.csv", "text/csv")
    d[1].download_button("Download summary CSV", to_csv_bytes(summ), "monte_carlo_summary.csv", "text/csv")

    left, right = st.columns(2)
    with left:
        pick = st.selectbox("Pattern envelope for", list(res.patterns), key="s5_mc_pick")
        p5, p50, p95 = np.percentile(res.patterns[pick], [5, 50, 95], axis=0)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.fill_between(res.angles, p5, p95, alpha=.25, label="5–95 % (non-ideal)")
        ax.plot(res.angles, p50, label="Median (non-ideal)")
        ax.plot(res.angles, res.ideal_patterns[pick], "k--", lw=1, label="Ideal (trial 1)")
        meta = st.session_state["s5_mc"]["meta"]
        if pick == "Conventional" and imp.coupling_nn_db is None and imp.coupling_nnn_db is None:
            w = tapered_weights(steering_vector(scn.positions(), scn.target_deg),
                                taper_weights(scn.taper, scn.n_elements, scn.kaiser_beta))
            ax.plot(res.angles, predicted_error_floor_db(w, imp.gain_std_db, imp.phase_std_deg,
                                                         res.angles, imp.element_exponent),
                    "r:", label="Predicted random-error floor")
        ax.axvline(scn.target_deg, color="green", ls=":", lw=1)
        ax.set(xlim=(-90, 90), ylim=(-80, 12), xlabel="Angle from broadside (°)", ylabel=YLAB)
        ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for b, g in res.trials.groupby("beamformer", sort=False):
            ax.hist(g["sinr_loss_db"], bins=20, alpha=.55, label=b)
        ax.set(xlabel="SINR loss vs paired ideal (dB; negative = gain)", ylabel="Trials",
               title="SINR-loss distribution")
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    _show_meta("s5_mc")
