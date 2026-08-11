"""T-dual effective scale geometry.

The paper's Section 2, as executable arithmetic. Every theorem is stated here
as a predicate that can be checked exactly on rationals, so the Python side
mirrors the Lean development rather than merely paraphrasing it.

The central object is the effective radius

    R_eff(alpha', R) = max(R, alpha'/R)

whose fixed point at R = sqrt(alpha') is the universal minimal scale. Note the
division-by-zero discipline: R = 0 raises rather than returning a convention
value. That is the exact edge case which made the paper's early Lean
axiomatization inconsistent, and it is worth refusing loudly in code too.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

Rational = Fraction | int


def effective_radius(alpha_prime: float, radius: float) -> float:
    """R_eff = max(R, alpha'/R). Theorem 2.1."""
    if radius <= 0:
        raise ValueError("radius must be strictly positive")
    if alpha_prime <= 0:
        raise ValueError("alpha_prime must be strictly positive")
    return max(radius, alpha_prime / radius)


def effective_radius_exact(alpha_prime: Rational, radius: Rational) -> Fraction:
    """Exact-rational R_eff, for certificate-grade checks."""
    alpha_prime, radius = Fraction(alpha_prime), Fraction(radius)
    if radius <= 0 or alpha_prime <= 0:
        raise ValueError("alpha_prime and radius must be strictly positive")
    return max(radius, alpha_prime / radius)


def effective_wavenumber(alpha_prime: float, k: np.ndarray | float) -> np.ndarray:
    """k_eff = min(k, 1/(alpha' k)), the Fourier-space image of R_eff.

    Section 5.1: modes above 1/sqrt(alpha') act as their duals below it.
    """
    k_arr = np.asarray(k, dtype=float)
    if np.any(k_arr <= 0):
        raise ValueError("wavenumbers must be strictly positive")
    return np.minimum(k_arr, 1.0 / (alpha_prime * k_arr))


def minimal_scale(alpha_prime: float) -> float:
    """The universal floor sqrt(alpha') that R_eff can never go below."""
    return float(np.sqrt(alpha_prime))


def maximal_wavenumber(alpha_prime: float) -> float:
    """The dual ceiling 1/sqrt(alpha') on effective wavenumber."""
    return 1.0 / np.sqrt(alpha_prime)


# -- The four theorems of Section 2, as checkable predicates ----------------


def check_tdual_bound(alpha_prime: Rational, radius: Rational) -> bool:
    """Theorem 2.2: R_eff >= sqrt(alpha'). Checked exactly via R_eff^2 >= alpha'.

    Squaring avoids irrational square roots entirely, so the check is decided
    in exact rational arithmetic rather than approximated in floating point.
    """
    r_eff = effective_radius_exact(alpha_prime, radius)
    return r_eff * r_eff >= Fraction(alpha_prime)


def check_bounce(alpha_prime: Rational, radius: Rational) -> bool:
    """Theorem 2.3: below sqrt(alpha'), R_eff = alpha'/R -- contraction reflects."""
    alpha_prime, radius = Fraction(alpha_prime), Fraction(radius)
    if radius * radius >= alpha_prime:  # not in the sub-cutoff regime
        return True
    return effective_radius_exact(alpha_prime, radius) == alpha_prime / radius


def check_inertial_invisibility(alpha_prime: Rational, radius: Rational) -> bool:
    """Theorem 2.4: above the cutoff, R_eff = R exactly -- no deformation.

    This is the load-bearing one. Without it the framework would not be
    regularizing Navier-Stokes but replacing it with a different equation.
    """
    alpha_prime, radius = Fraction(alpha_prime), Fraction(radius)
    if radius * radius < alpha_prime:
        return True
    return effective_radius_exact(alpha_prime, radius) == radius


def check_tduality(alpha_prime: Rational, radius: Rational) -> bool:
    """Theorem 2.5 (corrected): R_eff(alpha', alpha'/R) == R_eff(alpha', R).

    Effective geometry cannot distinguish a radius from its dual. Note this is
    plain invariance under R <-> alpha'/R; the draft's stated form
    `alpha'/R_eff(alpha', alpha'/R) == R_eff(alpha', R)` carries a spurious
    alpha'/(.) on the left and is false (alpha'=1/4, R=1000 gives 1/4000
    against 1000). The invariance below is what the max-form actually
    satisfies, and it is what the physical claim requires.
    """
    alpha_prime, radius = Fraction(alpha_prime), Fraction(radius)
    return effective_radius_exact(
        alpha_prime, alpha_prime / radius
    ) == effective_radius_exact(alpha_prime, radius)


def certify_geometry(
    alpha_prime: Rational = Fraction(1, 4),
    *,
    n_samples: int = 200,
    seed: int | None = 0,
) -> dict[str, object]:
    """Exact-arithmetic certificate for Theorems 2.2-2.5 over sampled radii.

    Samples are drawn as rationals spanning both sides of the cutoff, so the
    bounce branch and the inertial branch are both genuinely exercised -- a
    sample entirely above sqrt(alpha') would pass vacuously.
    """
    rng = np.random.default_rng(seed)
    alpha_prime = Fraction(alpha_prime)

    radii = [
        Fraction(int(rng.integers(1, 500)), int(rng.integers(1, 500)))
        for _ in range(n_samples)
    ]
    radii += [Fraction(1, 1000), Fraction(1000), alpha_prime, Fraction(1, 2)]

    cutoff_sq = alpha_prime
    below = sum(1 for r in radii if r * r < cutoff_sq)

    return {
        "alpha_prime": alpha_prime,
        "n_samples": len(radii),
        "n_below_cutoff": below,
        "n_above_cutoff": len(radii) - below,
        "thm_2.2_tdual_bound": all(check_tdual_bound(alpha_prime, r) for r in radii),
        "thm_2.3_bounce": all(check_bounce(alpha_prime, r) for r in radii),
        "thm_2.4_inertial_invisibility": all(
            check_inertial_invisibility(alpha_prime, r) for r in radii
        ),
        "thm_2.5_tduality": all(check_tduality(alpha_prime, r) for r in radii),
        "arithmetic": "exact (Fraction)",
    }
