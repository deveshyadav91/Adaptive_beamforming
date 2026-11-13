"""Headless experiment runner: writes all result tables as CSV.

    python run_experiments.py --out results --trials 100 --seed 1 [--lms]
"""
import argparse
import os

from beamforming.impairments import ImpairmentConfig
from beamforming.robustness import (Scenario, compare_ideal_vs_nonideal, export_csv, monte_carlo,
                                    parameter_sweep, summarize_trials, taper_comparison)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lms", action="store_true", help="include LMS (slow)")
    ap.add_argument("--taper", default="Uniform")
    a = ap.parse_args()

    sc = Scenario(taper=a.taper)
    imp = ImpairmentConfig(element_exponent=1.0, gain_std_db=0.5, phase_std_deg=5.0,
                           coupling_nn_db=-20, coupling_nnn_db=-30,
                           coupling_nn_phase_deg=-90, coupling_nnn_phase_deg=-90)
    beams = ("Conventional", "MVDR") + (("LMS",) if a.lms else ())
    path = lambda name: os.path.join(a.out, name)

    df, _, _ = taper_comparison(sc, imp, seed=a.seed)
    export_csv(df, path("taper_comparison_nonideal.csv"))
    df, _, _ = taper_comparison(sc)
    export_csv(df, path("taper_comparison_ideal.csv"))
    export_csv(compare_ideal_vs_nonideal(sc, imp, a.seed, beams)["metrics"], path("ideal_vs_nonideal.csv"))
    export_csv(parameter_sweep(sc, imp, {"spacing_lambda": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]},
                               a.trials, a.seed, beams), path("sweep_spacing.csv"))
    export_csv(parameter_sweep(sc, imp, {"gain_std_db": [0, 0.25, 0.5, 1, 2],
                                         "phase_std_deg": [0, 2, 5, 10, 20]},
                               a.trials, a.seed, beams), path("sweep_calibration.csv"))
    export_csv(parameter_sweep(sc, imp, {"taper": list(df["taper"])}, a.trials, a.seed, ("Conventional",)),
               path("sweep_taper.csv"))
    res = monte_carlo(sc, imp, a.trials, a.seed, beams, keep_patterns=False)
    export_csv(res.trials, path("monte_carlo_trials.csv"))
    summary = summarize_trials(res.trials)
    export_csv(summary, path("monte_carlo_summary.csv"))
    print(summary.round(2).to_string(index=False))
    print(f"\nCSV files written to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
