"""Symplectic integration of Hamiltonian systems, in float or exact rationals.

The solver proposed for the HoloAlg programme. Two design commitments:

**Symplectic, not just accurate.** Kick-drift-kick leapfrog (Stormer-Verlet)
preserves the symplectic 2-form exactly, so energy error stays bounded and
oscillatory over arbitrarily long runs instead of drifting secularly. For the
validation ladder -- where the *conserved quantities are the oracle* -- this is
the property that matters, and it is why the same scheme drives the N-body
module.

**Exact mode.** With `exact=True` every state component is a `Fraction` and the
integrator map is applied in exact rational arithmetic: the numerical orbit is
a theorem about the discrete map, not an approximation of it. This is the
EIU/HaloAlg exactness hypothesis made testable -- including its cost, since
bit-width grows with every step unless a lossy state collapse is admitted
(see `socrates.eiu.state_collapse`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

Vector = np.ndarray
Force = Callable[[Vector], Vector]


@dataclass
class SolverResult:
    """Trajectory of a leapfrog integration with conservation diagnostics."""

    times: np.ndarray
    positions: np.ndarray  # (n_steps+1, dim)
    velocities: np.ndarray
    label: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def energy(self, potential: Callable[[Vector], float], mass: float = 1.0) -> np.ndarray:
        kinetic = 0.5 * mass * np.sum(self.velocities**2, axis=1)
        return kinetic + np.array([potential(q) for q in self.positions])

    def energy_drift(self, potential: Callable[[Vector], float], mass: float = 1.0) -> float:
        e = self.energy(potential, mass)
        scale = float(np.max(np.abs(e))) or 1.0
        return float((e.max() - e.min()) / scale)


def leapfrog(
    force: Force,
    q0: Sequence[float],
    v0: Sequence[float],
    *,
    dt: float,
    n_steps: int,
    mass: float = 1.0,
    label: str = "",
) -> SolverResult:
    """Kick-drift-kick leapfrog. Second order, symplectic, time-reversible."""
    q = np.array(q0, dtype=float)
    v = np.array(v0, dtype=float)

    positions = np.empty((n_steps + 1, q.size))
    velocities = np.empty_like(positions)
    positions[0], velocities[0] = q, v

    a = force(q) / mass
    for i in range(1, n_steps + 1):
        v_half = v + 0.5 * dt * a
        q = q + dt * v_half
        a = force(q) / mass
        v = v_half + 0.5 * dt * a
        positions[i], velocities[i] = q, v

    return SolverResult(
        times=np.arange(n_steps + 1) * dt,
        positions=positions,
        velocities=velocities,
        label=label,
        metadata={"dt": dt, "n_steps": n_steps, "scheme": "leapfrog/KDK"},
    )


def leapfrog_exact(
    force: Callable[[list[Fraction]], list[Fraction]],
    q0: Sequence[Fraction | int],
    v0: Sequence[Fraction | int],
    *,
    dt: Fraction,
    n_steps: int,
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """Leapfrog in exact rational arithmetic: the EIU-mode integrator.

    The force must be polynomial/rational in the state (harmonic and power-law
    forces qualify; gravity's 1/r^2 does not, since r involves a square root --
    an honest boundary of the exactness programme worth knowing precisely).
    Returns (positions, velocities) as lists of Fraction vectors.
    """
    dt = Fraction(dt)
    q = [Fraction(x) for x in q0]
    v = [Fraction(x) for x in v0]
    half = Fraction(1, 2)

    positions, velocities = [list(q)], [list(v)]
    a = force(q)
    for _ in range(n_steps):
        v_half = [vi + half * dt * ai for vi, ai in zip(v, a, strict=True)]
        q = [qi + dt * vhi for qi, vhi in zip(q, v_half, strict=True)]
        a = force(q)
        v = [vhi + half * dt * ai for vhi, ai in zip(v_half, a, strict=True)]
        positions.append(list(q))
        velocities.append(list(v))
    return positions, velocities


def convergence_order(
    force: Force,
    q0: Sequence[float],
    v0: Sequence[float],
    reference: Callable[[float], np.ndarray],
    *,
    t_end: float,
    dts: Sequence[float] = (0.02, 0.01, 0.005, 0.0025),
) -> float:
    """Measured order of accuracy against an analytic reference solution.

    Fits log(error) vs log(dt); leapfrog should measure ~= 2.0. A measured
    order far from nominal is the classic symptom of an implementation bug
    that per-run eyeballing never catches.
    """
    errors = []
    for dt in dts:
        n = int(round(t_end / dt))
        result = leapfrog(force, q0, v0, dt=dt, n_steps=n)
        errors.append(float(np.linalg.norm(result.positions[-1] - reference(n * dt))))
    slope = np.polyfit(np.log(np.array(dts)), np.log(np.array(errors)), 1)[0]
    return float(slope)
