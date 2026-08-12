"""The validation ladder: one solver, problems of increasing complexity.

Each level solves a problem whose answer is known independently (analytic
solution, conserved quantity, or public ephemeris data), measures the error,
and GATES: the next level only unlocks if the current one passes. A solver
that has not climbed the ladder has no business near an open problem.

    Level 1  harmonic oscillator      oracle: analytic cos(t), order-2 convergence
    Level 2  nonlinear pendulum       oracle: exact elliptic-integral period
    Level 3  Kepler two-body          oracle: energy/angular momentum/Kepler III
    Level 4  Mars vs JPL Horizons     oracle: real open ephemeris data (network)
    Level 5  dyadic cascade           frontier: ties into the Stage 1 programme
    EIU demo exact-rational mode      oracle: bit-growth vs collapse error bound

Run:  python scripts/solver_ladder.py [--skip-network]
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from socrates.eiu import collapse_error_bound, denominator_bits, state_collapse  # noqa: E402
from socrates.solvers import convergence_order, leapfrog, leapfrog_exact  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def level1_harmonic() -> dict[str, object]:
    """x'' = -x from x(0)=1, v(0)=0; exact solution cos(t)."""
    force = lambda q: -q  # noqa: E731

    order = convergence_order(
        force, [1.0], [0.0], reference=lambda t: np.array([np.cos(t)]), t_end=10.0
    )
    run = leapfrog(force, [1.0], [0.0], dt=1e-3, n_steps=int(20 * np.pi * 1000))
    max_err = float(np.max(np.abs(run.positions[:, 0] - np.cos(run.times))))
    drift = run.energy_drift(lambda q: 0.5 * float(q @ q))

    passed = 1.8 <= order <= 2.2 and max_err < 1e-4 and drift < 1e-6
    return {
        "level": "1 harmonic oscillator",
        "measured_order": round(order, 3),
        "max_error_vs_analytic": max_err,
        "energy_drift": drift,
        "gate": "order in [1.8,2.2], err<1e-4, drift<1e-6",
        "passed": passed,
    }


def level2_pendulum(theta0: float = 2.0) -> dict[str, object]:
    """theta'' = -sin(theta); oracle is the exact period 4*K(m), m=sin^2(theta0/2)."""
    from scipy.special import ellipk

    force = lambda q: -np.sin(q)  # noqa: E731
    exact_period = float(4.0 * ellipk(np.sin(theta0 / 2.0) ** 2))

    dt = 1e-3
    run = leapfrog(force, [theta0], [0.0], dt=dt, n_steps=int(3 * exact_period / dt))
    theta = run.positions[:, 0]

    # First zero crossing is at T/4; linear interpolation between samples.
    idx = int(np.argmax(theta < 0))
    t_quarter = run.times[idx - 1] + dt * theta[idx - 1] / (theta[idx - 1] - theta[idx])
    measured_period = 4.0 * float(t_quarter)

    rel_err = abs(measured_period - exact_period) / exact_period
    passed = rel_err < 1e-5
    return {
        "level": "2 nonlinear pendulum",
        "exact_period_4K": exact_period,
        "measured_period": measured_period,
        "relative_error": rel_err,
        "gate": "period error < 1e-5",
        "passed": passed,
    }


def level3_kepler(eccentricity: float = 0.6) -> dict[str, object]:
    """Two-body orbit, GM=a=1: period 2*pi, L and E conserved (L exactly, by KDK)."""
    gm = 1.0

    def force(q: np.ndarray) -> np.ndarray:
        r = float(np.linalg.norm(q))
        return -gm * q / r**3

    r_peri = 1.0 - eccentricity
    v_peri = np.sqrt(gm * (1 + eccentricity) / r_peri)
    dt = 5e-4  # perihelion of an e=0.6 orbit is the fastest phase; resolve it
    n = int(10 * 2 * np.pi / dt)
    run = leapfrog(force, [r_peri, 0.0], [0.0, v_peri], dt=dt, n_steps=n)

    drift = run.energy_drift(lambda q: -gm / float(np.linalg.norm(q)))
    angular = (
        run.positions[:, 0] * run.velocities[:, 1] - run.positions[:, 1] * run.velocities[:, 0]
    )
    l_drift = float(np.max(np.abs(angular - angular[0])) / abs(angular[0]))

    # Radial period from perihelion returns (minima of |r|), with parabolic
    # interpolation: plain sample times are quantized to dt, which at this
    # resolution would swamp the integrator's genuine O(dt^2) phase error.
    radius = np.linalg.norm(run.positions, axis=1)
    minima = np.flatnonzero((radius[1:-1] < radius[:-2]) & (radius[1:-1] < radius[2:])) + 1
    t_minima = []
    for i in minima:
        y0, y1, y2 = radius[i - 1], radius[i], radius[i + 1]
        t_minima.append(run.times[i] + dt * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))
    measured_period = float(np.mean(np.diff(t_minima))) if len(t_minima) > 2 else np.nan
    period_err = abs(measured_period - 2 * np.pi) / (2 * np.pi)

    passed = drift < 1e-5 and l_drift < 1e-9 and period_err < 1e-5
    return {
        "level": "3 Kepler two-body (e=0.6)",
        "energy_drift": drift,
        "angular_momentum_drift": l_drift,
        "period_error_vs_2pi": period_err,
        "gate": "E drift<1e-5, L drift<1e-9 (KDK exact for central force), T err<1e-5",
        "passed": passed,
    }


def level4_mars_horizons() -> dict[str, object]:
    """Open data: integrate Sun-Mars two-body, compare to JPL Horizons ephemeris.

    The residual is physical, not numerical: Horizons includes planetary
    perturbations (chiefly Jupiter) that the two-body model omits, so the gate
    is set at the known size of those perturbations over six months.
    """
    gm_sun = 2.9591220828559093e-4  # AU^3 / day^2 (IAU)
    cache = DATA_DIR / "horizons_mars_2025.npz"

    if cache.exists():
        data = np.load(cache)
        r_true, v0, t_days = data["r"], data["v0"], data["t"]
    else:
        from astroquery.jplhorizons import Horizons

        query = Horizons(
            id="499", location="@sun",
            epochs={"start": "2025-01-01", "stop": "2025-07-01", "step": "1d"},
        )
        table = query.vectors(refplane="ecliptic")
        r_true = np.column_stack([table["x"], table["y"], table["z"]]).astype(float)
        v0 = np.array([table["vx"][0], table["vy"][0], table["vz"][0]], dtype=float)
        t_days = np.arange(len(r_true), dtype=float)
        DATA_DIR.mkdir(exist_ok=True)
        np.savez_compressed(cache, r=r_true, v0=v0, t=t_days)

    def force(q: np.ndarray) -> np.ndarray:
        r = float(np.linalg.norm(q))
        return -gm_sun * q / r**3

    substeps = 20  # dt = 0.05 day
    run = leapfrog(force, r_true[0], v0, dt=1.0 / substeps,
                   n_steps=(len(t_days) - 1) * substeps)
    r_model = run.positions[::substeps]

    rel_rms = float(
        np.sqrt(np.mean(np.sum((r_model - r_true) ** 2, axis=1)))
        / np.mean(np.linalg.norm(r_true, axis=1))
    )
    passed = rel_rms < 5e-3
    return {
        "level": "4 Mars vs JPL Horizons (open ephemeris, 182 days)",
        "n_epochs": len(t_days),
        "relative_rms_vs_ephemeris": rel_rms,
        "gate": "rel RMS < 5e-3 (residual = real planetary perturbations)",
        "passed": passed,
    }


def level5_cascade() -> dict[str, object]:
    """Frontier tie-in: the dyadic cascade from the Stage 1 programme."""
    from socrates.dualscale import compare_regularization, enstrophy_bound

    runs = compare_regularization(n_shells=20, alpha_prime=1e-5, t_max=4.0, cfl=0.05)
    tdual = runs["tdual"]
    ceiling = enstrophy_bound(1e-5, tdual.energy[0])
    passed = (
        tdual.terminated == "t_max"
        and runs["classical"].terminated != "t_max"
        and tdual.max_enstrophy <= ceiling
    )
    return {
        "level": "5 dyadic cascade (frontier)",
        "classical_terminated": runs["classical"].terminated,
        "tdual_terminated": tdual.terminated,
        "tdual_max_enstrophy": tdual.max_enstrophy,
        "thm_4_2_ceiling": ceiling,
        "gate": "t-dual completes under bound; classical does not complete",
        "passed": passed,
    }


def eiu_exact_mode_demo() -> dict[str, object]:
    """Exact-rational leapfrog on the oscillator: bit growth vs collapse error.

    Quantifies the honest form of the spec's Memory Boundedness claim: without
    collapse, bit-width grows without bound; with collapse, it is bounded at a
    measured, certified accuracy cost.
    """
    force = lambda q: [-x for x in q]  # noqa: E731
    dt = Fraction(1, 100)

    positions, _ = leapfrog_exact(force, [1], [0], dt=dt, n_steps=60)
    bits = [denominator_bits(p[0]) for p in positions]

    x_exact = positions[-1][0]
    collapsed, error = state_collapse(x_exact, max_denominator=10**6)
    bound = collapse_error_bound(collapsed, 10**6)  # Farey bound; the spec's 1/(q*D_max) is false

    return {
        "level": "EIU exact mode (oscillator, 60 steps)",
        "bits_step10": bits[10],
        "bits_step30": bits[30],
        "bits_step60": bits[60],
        "growth": "unbounded without collapse (linear-in-steps here)",
        "collapse_error": float(abs(error)),
        "certified_error_bound": float(bound),
        "bound_respected": abs(error) < bound,
        "passed": abs(error) < bound and bits[60] > bits[10],
    }


LADDER = [level1_harmonic, level2_pendulum, level3_kepler, level4_mars_horizons, level5_cascade]


def climb(skip_network: bool = False) -> list[dict[str, object]]:
    results = []
    for rung in LADDER:
        if skip_network and rung is level4_mars_horizons:
            results.append({"level": "4 Mars vs JPL Horizons", "passed": None,
                            "note": "skipped (--skip-network)"})
            continue
        try:
            outcome = rung()
        except Exception as exc:  # network failures etc. are reported, not hidden
            outcome = {"level": rung.__name__, "passed": False, "error": repr(exc)}
        results.append(outcome)
        print(f"\n{'PASS' if outcome.get('passed') else 'FAIL'}  {outcome['level']}")
        for key, value in outcome.items():
            if key not in ("level", "passed"):
                print(f"      {key}: {value}")
        if outcome.get("passed") is False:
            print("\nladder halted: fix this rung before climbing further")
            break
    return results


if __name__ == "__main__":
    skip = "--skip-network" in sys.argv
    results = climb(skip_network=skip)
    print("\n" + "=" * 60)
    print("EIU exact-mode demonstration")
    demo = eiu_exact_mode_demo()
    for key, value in demo.items():
        print(f"  {key}: {value}")
    n_passed = sum(1 for r in results if r.get("passed"))
    print(f"\nladder: {n_passed}/{len(results)} rungs passed; "
          f"EIU demo {'passed' if demo['passed'] else 'failed'}")
