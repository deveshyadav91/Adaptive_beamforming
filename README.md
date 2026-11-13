# Adaptive Digital Beamforming and Null-Steering Engine

Narrowband far-field ULA/UPA simulation with conventional, MVDR and LMS beamformers, quantitative pattern
metrics, grating-lobe analysis, and robustness analysis for real-world array effects (tapering, element patterns,
calibration errors, mutual coupling, parameter sweeps and Monte Carlo).

## Features
- Conventional (optionally tapered), MVDR (diagonal loading) and LMS beamformers; output-SINR comparison.
- HPBW (interpolated -3 dB crossings), null-based peak sidelobe level, interference-direction response.
- Theoretical grating-lobe prediction for a ULA; UPA azimuth/elevation heatmap and 3D pattern.
- Amplitude tapers: Uniform, Hann, Hamming, Blackman, Kaiser(β).
- Cosine-power element pattern `cos(θ)^p` (also on the UPA plots).
- Random per-element gain/phase calibration errors; nearest + next-neighbour mutual coupling.
- Ideal vs non-ideal comparison, parameter sweeps (spacing, error grid, taper), Monte Carlo with paired ideal baseline.
- CSV export from every analysis tab and from a headless runner.

## Project layout
```
app.py                    Streamlit app (tabs: patterns, planar scan, spacing sweep, tapers & element pattern,
                          ideal vs non-ideal, robustness sweeps, Monte Carlo, assumptions)
robustness_ui.py          Streamlit rendering for the robustness tabs
run_experiments.py        Headless runner that writes all result tables as CSV
beamforming/
  array.py                geometry, steering vectors/matrices
  adaptive.py             conventional, MVDR, LMS
  simulation.py           snapshot generation (ideal and impaired arrays)
  metrics.py              HPBW, sidelobe level, null depth, grating lobes, SINR
  tapers.py               amplitude windows, taper efficiency, white-noise gain
  impairments.py          element pattern, calibration errors, mutual coupling (ArrayModel)
  robustness.py           comparison, sweeps, Monte Carlo, CSV export
  visualization.py        UPA heatmap / 3D pattern
tests/                    pytest suite
```

## Run on Windows
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Headless experiments (CSV files go to `results/`):
```powershell
python run_experiments.py --out results --trials 100 --seed 1 [--lms] [--taper Hann]
```
Tests: `python -m pytest -q`

## Non-ideal array model
Actual response: `v(θ) = Γ · M · ( e(θ) · a(θ) )`
- `a` ideal steering vector, `e` element field pattern, `M` coupling matrix (symmetric banded Toeplitz with
  user-set complex coefficients), `Γ = diag(10^(ε/20)·e^{jφ})` with `ε~N(0,σ_g²) dB`, `φ~N(0,σ_φ²) deg`, applied after coupling.
- Receiver noise is added after `Γ` and stays spatially white.
- **Beamformers are designed with the nominal steering vector** (errors unknown to the processor); snapshots and
  metrics use the actual response. Ideal baselines use the same snapshot seed (paired).
- Tapers apply to the conventional beamformer and the LMS initial weights (always distortionless, `w^H a = 1`).
  MVDR is not tapered.
- The same seed gives the same underlying random draws, so sweeping σ scales the *same* errors; Monte Carlo is
  reproducible from its seed.

## Metrics
- Robustness plots are absolute: 0 dB = distortionless gain of an ideal array. HPBW, peak sidelobe level (PSL) and null
  depths are relative to the main-lobe peak.
- **PSL is null-based**: the highest response outside the main lobe's first nulls, so it adapts to tapers, steering and
  spacing. A visible grating lobe appears as about 0 dB.
- The main-lobe peak is searched within ±6° of the target, so pointing error saturates at 6° if the lobe is destroyed.
  PSL/HPBW are less meaningful for strongly distorted adaptive patterns.
- `sinr_loss_db = SINR_ideal - SINR_actual` (negative = the impairment helped, e.g. an element pattern that attenuates
  off-axis interferers more than the target). `psl_degradation_db = PSL_actual - PSL_ideal`.
- `predicted_error_floor_db` gives the expected random-error sidelobe power `Var(Γ)·||w||²` (exact for gain/phase
  errors without coupling); a unit test checks Monte Carlo against it.

## Conventions
- ULA is along x; scan angle is signed from broadside in the x-z plane.
- UPA lies in x-y; theta is polar angle from +z, phi is azimuth from +x toward +y.
- Positions in wavelengths. Steering `exp(+j 2π r·u)`; weights applied as `w^H a`.
- LMS training uses a clean desired reference (simulation only).

## Interpretation notes
- Coupling is a simplified model, not full-wave. The coupling coefficient *phase* decides whether coupling mainly
  changes gain or phase of the effective steering vector (the app defaults to -90°). "Scale ∝ 1/spacing" is a heuristic.
- MVDR is trained with the desired signal present in the snapshots, so it is very sensitive to steering mismatch
  (calibration error, coupling). Diagonal loading is the main protection and is visible in the sweeps.
- The UPA tab demonstrates conventional steering only; adaptive planar-array nulling is not included.
- Not modelled: polarization, near-field effects, wideband beam squint, coupled noise, nonlinear RF effects.
