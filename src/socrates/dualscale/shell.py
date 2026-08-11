"""Dyadic shell models of the energy cascade, classical and T-dual regularized.

Stage 1 of the paper's programme: the disease in vitro. The Katz-Pavlovic /
Desnyansky-Novikov dyadic model is an ODE system that provably blows up in
finite time in the inviscid regime, so it is the sharpest available testbed
for asking whether T-dual regularization actually prevents blow-up.

    du_n/dt = k_{n-1} u_{n-1}^2 - k_n u_n u_{n+1} - nu k_n^2 u_n

Two design decisions carry the scientific weight of this module:

**Adaptive timestepping.** The fastest timescale in the system is
1/(max_n k_n |u_n|). With dyadic k_n = 2^n and 30+ shells, a fixed explicit
step violates stability by many orders of magnitude and the integration
diverges *numerically*, which is indistinguishable by eye from physical
blow-up. Any comparison run at fixed dt would be measuring the integrator, not
the dynamics. We therefore step adaptively and report the reason for
termination.

**Energy conservation as an oracle.** The inviscid model conserves
E = (1/2) sum u_n^2 exactly (the nonlinear terms telescope). Drift in E is a
direct measure of integration error, so it distinguishes a trustworthy run
from a numerically contaminated one without appeal to the result being tested.

References
----------
Katz & Pavlovic (2005), Trans. AMS 357, 695-708.
Cheskidov (2008), Trans. AMS 360, 5101-5120.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import effective_wavenumber


@dataclass
class ShellResult:
    """Trajectory of a shell-model integration, with diagnostics."""

    times: np.ndarray
    enstrophy: np.ndarray
    energy: np.ndarray
    peak_wavenumber: np.ndarray
    final_state: np.ndarray
    wavenumbers: np.ndarray
    terminated: str
    label: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def energy_drift(self) -> float:
        """Relative change in total energy -- the integration-quality oracle."""
        if self.energy.size == 0 or self.energy[0] == 0:
            return float("nan")
        return float(abs(self.energy[-1] - self.energy[0]) / self.energy[0])

    @property
    def max_enstrophy(self) -> float:
        return float(np.max(self.enstrophy)) if self.enstrophy.size else float("nan")

    def summary(self) -> dict[str, object]:
        return {
            "label": self.label,
            "terminated": self.terminated,
            "t_final": float(self.times[-1]) if self.times.size else 0.0,
            "max_enstrophy": self.max_enstrophy,
            "final_enstrophy": float(self.enstrophy[-1]) if self.enstrophy.size else float("nan"),
            "energy_drift": self.energy_drift,
            **self.metadata,
        }


def dyadic_wavenumbers(n_shells: int, *, base: float = 2.0) -> np.ndarray:
    """k_n = base^n, the dyadic ladder of scales."""
    return base ** np.arange(n_shells, dtype=float)


def _rhs(u: np.ndarray, k: np.ndarray, viscosity: float) -> np.ndarray:
    """Katz-Pavlovic right-hand side with boundary shells held closed."""
    forward = np.zeros_like(u)
    forward[1:] = k[:-1] * u[:-1] ** 2  # energy in from below

    backward = np.zeros_like(u)
    backward[:-1] = k[:-1] * u[:-1] * u[1:]  # energy out to above

    du = forward - backward
    if viscosity > 0:
        du -= viscosity * k**2 * u
    return du


def _rk4_step(u: np.ndarray, dt: float, k: np.ndarray, viscosity: float) -> np.ndarray:
    k1 = _rhs(u, k, viscosity)
    k2 = _rhs(u + 0.5 * dt * k1, k, viscosity)
    k3 = _rhs(u + 0.5 * dt * k2, k, viscosity)
    k4 = _rhs(u + dt * k3, k, viscosity)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_shell_model(
    wavenumbers: np.ndarray,
    *,
    t_max: float = 5.0,
    viscosity: float = 0.0,
    initial: np.ndarray | None = None,
    cfl: float = 0.05,
    max_steps: int = 200_000,
    enstrophy_ceiling: float = 1e30,
    min_dt: float = 1e-12,
    n_samples: int = 2000,
    label: str = "",
) -> ShellResult:
    """Integrate the dyadic model with adaptive RK4.

    The step is chosen as cfl / max_n(k_n |u_n|), the inverse of the fastest
    nonlinear rate, so stability tracks the dynamics instead of being fixed in
    advance. Termination is reported explicitly: reaching `t_max`, exceeding
    `enstrophy_ceiling` (candidate blow-up), or hitting `max_steps` (dt
    collapsed -- itself the signature of a genuine finite-time singularity,
    since the fastest rate diverges).
    """
    k = np.asarray(wavenumbers, dtype=float)
    n = k.size

    u = np.zeros(n) if initial is None else np.array(initial, dtype=float)
    if initial is None:
        u[0] = 1.0

    sample_times = np.linspace(0.0, t_max, n_samples)
    next_sample = 0

    times: list[float] = []
    enstrophy: list[float] = []
    energy: list[float] = []
    peak_k: list[float] = []

    def record(t: float, state: np.ndarray) -> None:
        times.append(t)
        enstrophy.append(float(np.sum(k**2 * state**2)))
        energy.append(float(0.5 * np.sum(state**2)))
        weight = k**2 * state**2
        peak_k.append(float(k[int(np.argmax(weight))]) if weight.max() > 0 else 0.0)

    record(0.0, u)
    t = 0.0
    terminated = "t_max"

    for step_index in range(max_steps):
        rate = float(np.max(k * np.abs(u)))
        if viscosity > 0:
            rate = max(rate, viscosity * float(np.max(k**2)))
        if rate <= 0 or not np.isfinite(rate):
            terminated = "degenerate" if rate <= 0 else "non_finite"
            break

        dt = cfl / rate
        # A collapsing step size is the direct signature of finite-time
        # blow-up: the fastest nonlinear rate max_n(k_n |u_n|) is diverging.
        # Detecting it here is both faster and more honest than integrating
        # into overflow, which would confound the physics with float limits.
        if dt < min_dt:
            record(t, u)
            terminated = "dt_collapse"
            break
        if t + dt > t_max:
            dt = t_max - t

        u = _rk4_step(u, dt, k, viscosity)
        t += dt

        if not np.all(np.isfinite(u)):
            terminated = "non_finite"
            break

        while next_sample < n_samples and sample_times[next_sample] <= t:
            record(t, u)
            next_sample += 1

        current_enstrophy = float(np.sum(k**2 * u**2))
        if current_enstrophy > enstrophy_ceiling:
            record(t, u)
            terminated = "enstrophy_ceiling"
            break

        if t >= t_max:
            break

        if step_index == max_steps - 1:
            terminated = "max_steps"

    if times[-1] != t:
        record(t, u)

    return ShellResult(
        times=np.array(times),
        enstrophy=np.array(enstrophy),
        energy=np.array(energy),
        peak_wavenumber=np.array(peak_k),
        final_state=u,
        wavenumbers=k,
        terminated=terminated,
        label=label,
        metadata={
            "n_shells": n,
            "viscosity": viscosity,
            "cfl": cfl,
            "k_max": float(k.max()),
        },
    )


def compare_regularization(
    n_shells: int = 25,
    alpha_prime: float = 1e-6,
    *,
    t_max: float = 5.0,
    viscosity: float = 0.0,
    cfl: float = 0.05,
) -> dict[str, ShellResult]:
    """Run the classical and T-dual regularized cascades under identical settings.

    Both runs use the same integrator, tolerances and initial data; the only
    difference is the wavenumber ladder. That isolates the regularization as
    the sole causal variable.
    """
    k_raw = dyadic_wavenumbers(n_shells)
    k_eff = effective_wavenumber(alpha_prime, k_raw)

    return {
        "classical": simulate_shell_model(
            k_raw, t_max=t_max, viscosity=viscosity, cfl=cfl, label="classical"
        ),
        "tdual": simulate_shell_model(
            k_eff, t_max=t_max, viscosity=viscosity, cfl=cfl, label=f"tdual(a'={alpha_prime:g})"
        ),
    }


def enstrophy_bound(alpha_prime: float, energy: float) -> float:
    """The paper's Theorem 4.2 bound, stated with its energy factor.

    Per shell, k_eff^2 <= 1/alpha'. Summed against the energy distribution,
    total enstrophy is bounded by (1/alpha') * sum u_n^2 = (2/alpha') * E.
    The bare constant 1/alpha' is the *per-mode* ceiling, not the total -- a
    distinction worth keeping explicit when plotting.
    """
    return 2.0 * energy / alpha_prime


def convergence_study(
    n_shells: int = 22,
    *,
    cfl_values: tuple[float, ...] = (0.2, 0.1, 0.05, 0.025),
    t_max: float = 3.0,
) -> list[dict[str, object]]:
    """Refine the timestep and check that the classical result is dt-independent.

    This is the control that separates physical blow-up from numerical
    instability: a genuine finite-time singularity persists (and its blow-up
    time converges) as cfl -> 0, whereas an integrator artifact moves or
    disappears.
    """
    k = dyadic_wavenumbers(n_shells)
    rows = []
    for cfl in cfl_values:
        result = simulate_shell_model(k, t_max=t_max, cfl=cfl, label=f"cfl={cfl}")
        rows.append(
            {
                "cfl": cfl,
                "terminated": result.terminated,
                "t_final": float(result.times[-1]),
                "max_enstrophy": result.max_enstrophy,
                "energy_drift": result.energy_drift,
            }
        )
    return rows
