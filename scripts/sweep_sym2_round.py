"""Sweep script: Sym2-constrained shell model at n_shells=24.

Uses the same sweep pattern as hypothesis_u_scaling():
- alpha-prime from 1e-2 to 1e-10 over 9 points
- t_max=12, cfl=0.05
- Calls simulate_sym2_shell_model with lock="sym2" instead of plain shell.py

Records:
- Peak enstrophy
- Whether each run terminated at t_max
- Energy drift (to check convergence threshold < 1e-6)

Fits exponent via log-log linear regression over runs that converged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from socrates.dualscale.geometry import effective_wavenumber  # noqa: E402
from socrates.dualscale.shell_sym2 import (  # noqa: E402
    dyadic_wavenumbers,
    sweep_lock_exponent,
)


def main() -> dict:
    """Run the sweep and return results."""
    print("\n" + "="*70)
    print("Sym2-Constrained Shell Model Sweep (n_shells=24)")
    print("="*70)

    # Use same sweep pattern as hypothesis_u_scaling
    alphas = np.logspace(-2, -10, 9)
    n_shells = 24
    t_max = 12.0
    cfl = 0.05

    print(f"\nParameters:")
    print(f"  n_shells: {n_shells}")
    print(f"  t_max: {t_max}")
    print(f"  cfl: {cfl}")
    print(f"  lock: sym2 (Sym2-constrained)")
    print(f"  alphas: {len(alphas)} points from 1e-2 to 1e-10")

    # Call the sweep function
    sweep = sweep_lock_exponent(
        alphas=alphas,
        lock="sym2",
        closure="galerkin",
        n_shells=n_shells,
        t_max=t_max,
        cfl=cfl,
        base=2.0,
    )

    # Build results table
    print(f"\n{'alpha_prime':>12} {'k_eff_max':>12} {'peak_enstr':>14} {'terminated':>12} {'energy_drift':>12}")
    print("-" * 70)

    rows = sweep.table()
    k_raw = dyadic_wavenumbers(n_shells)

    table_lines = []
    for i, row in enumerate(rows):
        alpha_p = row["alpha_prime"]
        k_eff = effective_wavenumber(float(alpha_p), k_raw)
        k_eff_max = float(k_eff.max())
        peak_en = row["peak_enstrophy"]
        term = row["terminated"]
        edrift = row["energy_drift"]

        print(f"{alpha_p:>12.2e} {k_eff_max:>12.4g} {peak_en:>14.5g} {term:>12} {edrift:>12.2e}")
        table_lines.append(
            f"{alpha_p:>12.2e} {k_eff_max:>12.4g} {peak_en:>14.5g} {term:>12} {edrift:>12.2e}"
        )

    # Check convergence
    converged_runs = [
        (term == "t_max" and edrift < 1e-6)
        for term, edrift in zip(sweep.terminated, sweep.energy_drift)
    ]
    all_converged = all(converged_runs)
    n_converged = sum(converged_runs)

    print(f"\nConvergence check:")
    print(f"  Runs reaching t_max with drift < 1e-6: {n_converged}/{len(alphas)}")
    print(f"  all_converged: {all_converged}")

    # Report exponent
    print(f"\nFitted exponent (log-log regression over converged runs):")
    print(f"  p = {sweep.exponent:.6f} ± {sweep.exponent_stderr:.6f}")
    print(f"  Verdict: {sweep.verdict}")

    # Compact table summary
    table_summary = "\n".join(table_lines)

    return {
        "sweep": sweep,
        "all_converged": all_converged,
        "n_converged": n_converged,
        "table_summary": table_summary,
    }


if __name__ == "__main__":
    result = main()
    print("\n" + "="*70)
    print("Sweep complete")
    print("="*70)
