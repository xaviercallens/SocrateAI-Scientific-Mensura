"""Gravitational N-body simulation of structure formation.

Generates a synthetic universe whose topology can be compared against the
observed SDSS catalog. Starting from a near-uniform particle distribution with
small perturbations, gravitational instability grows those perturbations into
the clusters, filaments and voids of the cosmic web -- so the simulation is a
physical control for the topological signature measured from real data.

Force evaluation is delegated to the Rust extension (`socrates._numerics`),
which is the performance bottleneck at O(N^2) per step. Integration uses
kick-drift-kick leapfrog, chosen because it is symplectic: energy error stays
bounded over long integrations instead of drifting secularly as it would with
Runge-Kutta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationState:
    """Particle positions, velocities and masses at one instant (G = 1 units)."""

    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    time: float

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=1)))


def _accelerations(positions: np.ndarray, masses: np.ndarray, softening: float) -> np.ndarray:
    """Gravitational accelerations, via Rust when available, else NumPy."""
    try:
        from socrates._numerics import gravity_accelerations

        return np.asarray(
            gravity_accelerations(
                np.ascontiguousarray(positions), masses.tolist(), softening
            )
        )
    except ImportError:
        return _accelerations_numpy(positions, masses, softening)


def _accelerations_numpy(
    positions: np.ndarray, masses: np.ndarray, softening: float
) -> np.ndarray:
    """Reference NumPy implementation -- the correctness oracle for the Rust kernel."""
    delta = positions[None, :, :] - positions[:, None, :]
    r2 = np.sum(delta**2, axis=-1) + softening**2
    np.fill_diagonal(r2, np.inf)  # no self-interaction
    inv_r3 = masses[None, :] / r2**1.5
    return np.sum(delta * inv_r3[:, :, None], axis=1)


def initial_conditions(
    n_particles: int,
    *,
    box_size: float = 100.0,
    perturbation: float = 0.15,
    seed: int | None = None,
) -> SimulationState:
    """A perturbed lattice: the standard starting point for structure formation.

    A pure random cloud already contains Poisson structure at all scales; a
    lattice plus controlled perturbation isolates gravitational growth as the
    origin of the structure that develops.
    """
    rng = np.random.default_rng(seed)

    side = int(np.ceil(n_particles ** (1 / 3)))
    spacing = box_size / side
    grid = np.mgrid[0:side, 0:side, 0:side].reshape(3, -1).T[:n_particles] * spacing
    positions = grid + rng.normal(0.0, perturbation * spacing, grid.shape)

    return SimulationState(
        positions=positions,
        velocities=np.zeros_like(positions),
        masses=np.ones(len(positions)),
        time=0.0,
    )


def step(state: SimulationState, dt: float, *, softening: float = 0.5) -> SimulationState:
    """One kick-drift-kick leapfrog step."""
    acc = _accelerations(state.positions, state.masses, softening)
    half_v = state.velocities + 0.5 * dt * acc
    new_positions = state.positions + dt * half_v
    new_acc = _accelerations(new_positions, state.masses, softening)
    new_velocities = half_v + 0.5 * dt * new_acc

    return SimulationState(
        positions=new_positions,
        velocities=new_velocities,
        masses=state.masses,
        time=state.time + dt,
    )


def simulate(
    n_particles: int = 512,
    *,
    n_steps: int = 100,
    dt: float = 0.05,
    box_size: float = 100.0,
    softening: float = 0.5,
    seed: int | None = None,
) -> SimulationState:
    """Evolve a perturbed lattice under self-gravity and return the final state."""
    state = initial_conditions(n_particles, box_size=box_size, seed=seed)
    for _ in range(n_steps):
        state = step(state, dt, softening=softening)
    return state


def clustering_ratio(initial: SimulationState, final: SimulationState) -> float:
    """Growth in density contrast between two states.

    Ratio of nearest-neighbour distance spread; values above 1 indicate that
    gravitational clustering has amplified the initial density fluctuations.
    """
    from scipy.spatial import cKDTree

    def spread(state: SimulationState) -> float:
        tree = cKDTree(state.positions)
        distances, _ = tree.query(state.positions, k=2)
        return float(np.std(distances[:, 1]))

    initial_spread = spread(initial)
    return spread(final) / initial_spread if initial_spread > 0 else float("nan")
