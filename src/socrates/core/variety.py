"""Sampling point clouds from real algebraic varieties.

The bridge from exact algebra to topology: a variety V(I) is defined by
polynomial equations, but persistent homology needs a finite point cloud. We
sample V(I) by seeding random points and projecting them onto the variety with
a damped Gauss-Newton iteration on the residual map F(x) = (f_1(x), ..., f_k(x)).

Points that fail to converge are discarded rather than silently kept, so the
returned cloud consists only of points verified to lie within `tolerance` of
the variety.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .polynomial import Polynomial

_EPS = 1e-12


def _gradient(poly: Polynomial, point: np.ndarray) -> np.ndarray:
    """Analytic gradient of `poly` at `point` via term-wise differentiation."""
    grad = np.zeros(poly.nvars)
    for exp, coeff in poly.terms.items():
        for var in range(poly.nvars):
            power = exp[var]
            if power == 0:
                continue
            term = float(coeff) * power
            for other, other_power in enumerate(exp):
                effective = other_power - 1 if other == var else other_power
                if effective:
                    term *= point[other] ** effective
            grad[var] += term
    return grad


def _residual(polys: Sequence[Polynomial], point: np.ndarray) -> np.ndarray:
    return np.array([p.evaluate(point) for p in polys])


def _jacobian(polys: Sequence[Polynomial], point: np.ndarray) -> np.ndarray:
    return np.vstack([_gradient(p, point) for p in polys])


def project_to_variety(
    polys: Sequence[Polynomial],
    point: np.ndarray,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> np.ndarray | None:
    """Project `point` onto V(polys) by damped Gauss-Newton; None if it diverges.

    Uses the minimum-norm least-squares step (via pinv), which handles both
    under- and over-determined systems and degenerates gracefully at singular
    points of the variety.
    """
    x = np.asarray(point, dtype=float).copy()

    for _ in range(max_iterations):
        residual = _residual(polys, x)
        error = float(np.linalg.norm(residual))
        if error < tolerance:
            return x

        jacobian = _jacobian(polys, x)
        if not np.all(np.isfinite(jacobian)) or np.linalg.norm(jacobian) < _EPS:
            return None

        step = -np.linalg.pinv(jacobian) @ residual

        # Backtracking line search keeps the iteration stable near singularities.
        alpha = 1.0
        for _ in range(30):
            candidate = x + alpha * step
            if np.all(np.isfinite(candidate)) and float(
                np.linalg.norm(_residual(polys, candidate))
            ) < error:
                x = candidate
                break
            alpha *= 0.5
        else:
            return None

    return x if float(np.linalg.norm(_residual(polys, x))) < tolerance else None


def sample_variety(
    polys: Sequence[Polynomial],
    n_points: int,
    *,
    bounds: tuple[float, float] = (-2.0, 2.0),
    tolerance: float = 1e-9,
    seed: int | None = None,
    max_attempts_factor: int = 50,
) -> np.ndarray:
    """Sample up to `n_points` points lying on the real variety V(polys).

    Returns an (m, nvars) array with m <= n_points; m falls short when the real
    locus is small or empty relative to the sampling box, which is itself
    informative (an empty return means no real points were found in `bounds`).
    """
    if not polys:
        raise ValueError("at least one defining polynomial is required")

    nvars = polys[0].nvars
    if any(p.nvars != nvars for p in polys):
        raise ValueError("all polynomials must have the same arity")

    rng = np.random.default_rng(seed)
    lo, hi = bounds
    accepted: list[np.ndarray] = []
    max_attempts = n_points * max_attempts_factor

    for _ in range(max_attempts):
        if len(accepted) >= n_points:
            break
        seed_point = rng.uniform(lo, hi, size=nvars)
        result = project_to_variety(polys, seed_point, tolerance=tolerance)
        # Keep the point only if it stayed inside a generous box; runaway
        # projections land far outside and would distort the topology.
        if result is not None and np.all(np.abs(result) <= 10 * max(abs(lo), abs(hi))):
            accepted.append(result)

    return np.array(accepted) if accepted else np.empty((0, nvars))


def sphere_polynomial(dimension: int, radius: float = 1.0) -> Polynomial:
    """The polynomial x_0^2 + ... + x_n^2 - r^2 whose variety is an n-sphere."""
    from fractions import Fraction

    nvars = dimension + 1
    terms: dict[tuple[int, ...], Fraction] = {}
    for i in range(nvars):
        exp = [0] * nvars
        exp[i] = 2
        terms[tuple(exp)] = Fraction(1)
    terms[(0,) * nvars] = -Fraction(radius).limit_denominator() ** 2
    return Polynomial(terms, nvars)


def torus_polynomial(major: float = 2.0, minor: float = 1.0) -> Polynomial:
    """Implicit torus in R^3: (x^2+y^2+z^2 + R^2 - r^2)^2 - 4R^2(x^2+y^2) = 0."""
    from fractions import Fraction

    big_r = Fraction(major).limit_denominator()
    small_r = Fraction(minor).limit_denominator()

    x2 = Polynomial({(2, 0, 0): Fraction(1)}, 3)
    y2 = Polynomial({(0, 2, 0): Fraction(1)}, 3)
    z2 = Polynomial({(0, 0, 2): Fraction(1)}, 3)
    const = Polynomial.constant(big_r**2 - small_r**2, 3)

    inner = x2 + y2 + z2 + const
    return inner * inner - Polynomial.constant(4 * big_r**2, 3) * (x2 + y2)
