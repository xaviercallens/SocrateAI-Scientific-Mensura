"""Stage 1 dyadic laboratory: does T-dual regularization arrest the cascade?

Reproduces the three controlled experiments behind the paper's Stage 1 claims
and writes `figures/stage1_cascade.png`.

Run:  python scripts/experiment_stage1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from socrates.dualscale.geometry import effective_wavenumber  # noqa: E402
from socrates.dualscale.shell import (  # noqa: E402
    convergence_study,
    dyadic_wavenumbers,
    enstrophy_bound,
    simulate_shell_model,
)

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def control_timestep_independence() -> None:
    """Separate physical blow-up from integrator artifact."""
    print("\n[Control A] timestep refinement -- energy drift must fall with cfl")
    print(f"  {'cfl':>7} {'terminated':>14} {'t_end':>8} {'E drift':>11}")
    for row in convergence_study(n_shells=18, cfl_values=(0.4, 0.2, 0.1, 0.05), t_max=4.0):
        print(
            f"  {row['cfl']:>7} {row['terminated']:>14} "
            f"{row['t_final']:>8.4f} {row['energy_drift']:>11.2e}"
        )


def hypothesis_u_scaling(
    n_shells: int = 30,
    alphas: np.ndarray | None = None,
    t_max: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Peak enstrophy as alpha' -> 0: the shell-model analogue of Hypothesis U.

    A bounded limit would support Hypothesis U; a power law alpha'^(-p) with
    p > 0 means it fails, at a rate the exponent quantifies.
    """
    if alphas is None:
        alphas = np.logspace(-2, -10, 9)

    k_raw = dyadic_wavenumbers(n_shells)
    peaks, converged = [], []

    print("\n[Experiment] peak enstrophy vs alpha'")
    print(f"  {'alpha':>10} {'k_eff_max':>12} {'peak_enstrophy':>16} {'ceiling 2E/a':>14}")
    for alpha in alphas:
        k = effective_wavenumber(alpha, k_raw)
        result = simulate_shell_model(k, t_max=t_max, cfl=0.05, max_steps=300_000)
        ceiling = enstrophy_bound(alpha, result.energy[0])
        peaks.append(result.max_enstrophy)
        converged.append(result.terminated == "t_max")
        print(
            f"  {alpha:>10.0e} {k.max():>12.4g} {result.max_enstrophy:>16.5g} "
            f"{ceiling:>14.4g}"
        )

    peaks_arr = np.array(peaks)
    mask = np.array(converged)
    slope = float(np.polyfit(np.log10(alphas[mask]), np.log10(peaks_arr[mask]), 1)[0])

    print(f"\n  fit: peak enstrophy ~ alpha'^({slope:.4f})")
    print("  Kolmogorov K41 prediction with cutoff k_max = 1/sqrt(alpha'):")
    print("    E(k) ~ k^(-5/3)  =>  enstrophy ~ k_max^(4/3) ~ alpha'^(-2/3) = alpha'^(-0.6667)")
    print(f"  Theorem 4.2 ceiling exponent: -1.0 (not attained; bound is not tight)")
    return alphas, peaks_arr, slope


def make_figure(alphas: np.ndarray, peaks: np.ndarray, slope: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: trajectories, classical vs regularized.
    k_raw = dyadic_wavenumbers(24)
    classical = simulate_shell_model(k_raw, t_max=6.0, cfl=0.05, max_steps=300_000)
    ax1.semilogy(classical.times, classical.enstrophy, color="#c0392b", lw=2,
                 label="classical (unregularized)")

    for alpha, color in [(1e-4, "#16a085"), (1e-6, "#2980b9"), (1e-8, "#8e44ad")]:
        k = effective_wavenumber(alpha, k_raw)
        reg = simulate_shell_model(k, t_max=6.0, cfl=0.05, max_steps=300_000)
        ax1.semilogy(reg.times, reg.enstrophy, color=color, lw=2,
                     label=rf"T-dual $\alpha'={alpha:g}$")
        ax1.axhline(enstrophy_bound(alpha, reg.energy[0]), color=color,
                    ls=":", lw=1, alpha=0.7)

    ax1.set_xlabel("time")
    ax1.set_ylabel("enstrophy  $\\sum k^2 u_n^2$")
    ax1.set_title("Cascade: classical diverges, T-dual saturates\n(dotted = Thm 4.2 ceiling $2E/\\alpha'$)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # Right: the Hypothesis U scaling test.
    ax2.loglog(alphas, peaks, "o-", color="#2c3e50", lw=2, ms=7, label="measured peak enstrophy")
    ref = peaks[0] * (alphas / alphas[0]) ** (-2 / 3)
    ax2.loglog(alphas, ref, "--", color="#e67e22", lw=2,
               label="K41 prediction $\\alpha'^{-2/3}$")
    ax2.loglog(alphas, peaks[0] * (alphas / alphas[0]) ** (-1.0), ":", color="#7f8c8d",
               lw=2, label="Thm 4.2 ceiling slope $\\alpha'^{-1}$")
    ax2.set_xlabel(r"$\alpha'$")
    ax2.set_ylabel("peak enstrophy")
    ax2.set_title(f"Hypothesis U test: fitted exponent {slope:.3f}\n(bounded would require slope 0)")
    ax2.invert_xaxis()
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = FIGURES / "stage1_cascade.png"
    fig.savefig(path, dpi=160)
    return path


if __name__ == "__main__":
    control_timestep_independence()
    alphas, peaks, slope = hypothesis_u_scaling()
    path = make_figure(alphas, peaks, slope)
    print(f"\nfigure written to {path}")
