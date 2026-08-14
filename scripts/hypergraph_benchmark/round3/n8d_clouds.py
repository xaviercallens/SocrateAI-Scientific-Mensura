"""Round-3 (N8d): faithful reconstruction of every round-2 production point cloud.

The round-2 skeptic's largest self-declared coverage gap was that only problems
05 and 06 were re-measured against the pre-N8 baseline. Closing that gap needs
all ten clouds available as data, at their *production* k / max_radius / n_grid,
without paying for each script's solver-verification runs (which do not feed the
cloud) or its compare() call (which is what we are re-deriving).

Every builder below reuses the round-2 module's OWN cloud-construction function
wherever the module is importable (all except 06, whose script does its work at
module scope and so cannot be imported without running the whole benchmark);
06 is reconstructed literally from its module-level code. Clouds are cached to
the scratchpad as pickles so the sweep can be re-run cheaply.
"""

from __future__ import annotations

import importlib.util
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROUND2 = REPO / "scripts" / "hypergraph_benchmark" / "round2"
sys.path.insert(0, str(REPO / "src"))

CACHE_DIR = Path(
    "/tmp/claude-1000/-home-xavkal-xdev/cb774970-2d62-4712-aaa9-94dafb5c745b"
    "/scratchpad/n8d_clouds"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CloudSpec:
    """A round-2 problem's production configuration, read off its script."""

    problem: str
    label: str
    k: int
    max_radius: int
    n_grid: tuple[int, ...]
    true_dimension: float
    tolerance: float
    samples: int = 40


# Read directly from each round-2 script's module constants / compare() call.
SPECS: dict[str, CloudSpec] = {
    "01": CloudSpec("01", "harmonic oscillator", 6, 3,
                    (50, 100, 200, 400, 800, 1600, 3200), 1.0, 0.15),
    "02": CloudSpec("02", "nonlinear pendulum", 6, 6,
                    (32, 64, 128, 256, 512, 1024, 2048, 4096), 1.0, 0.15),
    "03": CloudSpec("03", "Kepler orbit", 6, 6,
                    (100, 200, 400, 800, 1600, 3200, 6400), 1.0, 0.15),
    "04": CloudSpec("04", "Mars (Horizons IC)", 6, 6,
                    (100, 200, 400, 800, 1600, 3200, 6400), 1.0, 0.2),
    "05": CloudSpec("05", "quasiperiodic torus", 10, 6,
                    (100, 200, 400, 800, 1600, 3200, 6400, 12800), 2.0, 0.3),
    "06": CloudSpec("06", "2D Brownian motion", 10, 6,
                    (100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600), 2.0, 0.3),
    "07": CloudSpec("07", "restricted three-body (L1 halo)", 6, 6,
                    (100, 200, 400, 800, 1600, 3200, 6400, 12800), 1.0, 0.2),
    "08": CloudSpec("08", "Lorenz attractor", 10, 6,
                    (100, 200, 400, 800, 1600, 3200, 6400, 12800), 2.05, 0.5),
    "09": CloudSpec("09", "Rossler attractor", 15, 6,
                    (100, 200, 400, 800, 1600, 3200, 6400, 12800), 2.01, 0.5),
    "10": CloudSpec("10", "driven pendulum (period-1)", 6, 6,
                    (32, 64, 128, 256, 512, 1024, 2048, 4096), 1.0, 0.2),
}

_MODULE_FILES = {
    "01": "01_harmonic_oscillator.py",
    "02": "02_nonlinear_pendulum.py",
    "03": "03_kepler_orbit.py",
    "04": "04_mars_horizons.py",
    "05": "05_quasiperiodic_torus.py",
    "07": "07_restricted_three_body.py",
    "08": "08_lorenz_attractor.py",
    "09": "09_rossler_attractor.py",
    "10": "10_driven_pendulum_periodic.py",
}


def _import_round2(problem: str):
    """Import a round-2 script as a module (all of these are main-guarded)."""
    path = ROUND2 / _MODULE_FILES[problem]
    spec = importlib.util.spec_from_file_location(f"round2_p{problem}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_01():
    m = _import_round2("01")
    v = m.verify_solver(2e-4)
    return m.build_low_discrepancy_point_cloud(v["run"], 3200)


def _build_02():
    m = _import_round2("02")
    return m.generate_single_period_cloud()["bitrev_ordered"]


def _build_03():
    m = _import_round2("03")
    run = m.run_kepler(n_periods=1.02, dt=m.DT)
    return m.build_single_period_points(run, m.DT)


def _build_04():
    m = _import_round2("04")
    data = np.load(m.CACHE)
    r_true, v0 = data["r"], data["v0"]
    period = m.orbital_period_days(r_true[0], v0)
    return m.generate_single_period_cloud(r_true[0], v0, period, 6400)


def _build_05():
    m = _import_round2("05")
    n_max = m.N_GRID[-1]
    return m.build_point_cloud(n_max, n_max / m.DENSITY)


def _build_06():
    # 06_brownian_motion.py runs at module scope, so its construction is
    # reproduced literally here (seed 42, 80_000 unit-variance 2D increments,
    # cumulative sum from the origin, stride = len(path)//25_600 = 3).
    rng = np.random.default_rng(42)
    increments = rng.normal(loc=0.0, scale=1.0, size=(80_000, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(increments, axis=0)], axis=0)
    stride = max(1, len(path) // 25_600)
    return [tuple(p) for p in path[::stride]]


def _build_07():
    m = _import_round2("07")
    x_l1 = m.find_l1(m.MU)
    x0 = x_l1 + m.AX_AMPLITUDE
    vy0_guess = m.linear_vy0_guess(x_l1, m.AX_AMPLITUDE)
    vy0_coarse = m.find_periodic_orbit(x0, vy0_guess, dt=1e-3, t_max=2.2, span_frac=0.4)
    vy0_fine = m.find_periodic_orbit(x0, vy0_coarse, dt=1e-4, t_max=2.2, span_frac=0.05)
    t_half, _ = m.half_period_crossing(np.array([x0, 0.0, 0.0, vy0_fine]), dt=1e-5, t_max=2.2)
    period = 2.0 * t_half
    state0 = np.array([x0, 0.0, 0.0, vy0_fine])
    n_steps_cloud = int(round(1.02 * period / m.DT_CLOUD))
    states_cloud = m.integrate(state0, m.DT_CLOUD, n_steps_cloud)
    pos = states_cloud[:-20, :2]
    return [(float(p[0]), float(p[1])) for p in pos]


def _build_08():
    m = _import_round2("08")
    n_steps = int(m.T_TOTAL / m.DT)
    traj = m.rk4_integrate(np.array([1.0, 1.0, 1.0]), m.DT, n_steps)
    post = traj[int(m.T_TRANSIENT / m.DT):]
    return [(float(p[0]), float(p[1]), float(p[2])) for p in post]


def _build_09():
    m = _import_round2("09")
    trajectory = m.integrate()
    post = trajectory[m.TRANSIENT_STEPS:]
    sub = post[:: m.STRIDE][: m.N_MAX]
    return [(float(p[0]), float(p[1]), float(p[2])) for p in sub]


def _build_10():
    m = _import_round2("10")
    v = m.verify_mode_locking()
    assert v["passed"], "problem 10 mode-locking verification failed"
    cloud = m.generate_single_period_cloud(v["theta_final"], v["omega_final"], v["t_final"])
    assert cloud["cloud_closure_passed"]
    return cloud["bitrev_ordered"]


_BUILDERS = {
    "01": _build_01, "02": _build_02, "03": _build_03, "04": _build_04, "05": _build_05,
    "06": _build_06, "07": _build_07, "08": _build_08, "09": _build_09, "10": _build_10,
}


def load_cloud(problem: str, *, rebuild: bool = False) -> list[tuple[float, ...]]:
    """The production point cloud for a round-2 problem (cached on disk)."""
    cache = CACHE_DIR / f"p{problem}.pkl"
    if cache.exists() and not rebuild:
        with cache.open("rb") as fh:
            return pickle.load(fh)
    points = _BUILDERS[problem]()
    points = [tuple(float(c) for c in p) for p in points]
    with cache.open("wb") as fh:
        pickle.dump(points, fh)
    return points


def anchor_cloud() -> list[tuple[float, float]]:
    """The documented R2-F4 anchor cloud: problem 01's own cloud at n=200.

    docs/MENSURA_BENCHMARK.md 9.5 records the anchor shell sequence
    (6,5,5,6,6,6) at n=200, k=6, max_radius=6 -- note max_radius=6, NOT
    problem 01's production max_radius=3. Because the golden-ratio Weyl index
    sequence is a prefix property (fracs = arange(n)*phi % 1), the first 200
    points of the n_max=3200 cloud ARE the n=200 cloud, so this is the same
    construction the round-3 skeptic rebuilt from the solver independently.
    """
    return load_cloud("01")[:200]


ANCHOR_SHELLS = (6, 5, 5, 6, 6, 6)
ANCHOR_VOLUMES = (7, 12, 17, 23, 29, 35)


def perfect_ring_cloud(n: int = 200) -> list[tuple[float, float]]:
    """A perfectly uniform circle -- the exactly-constant (N1) regime, kept as a
    control so the near-constant and exactly-constant branches stay separable."""
    return [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]


if __name__ == "__main__":
    for problem, spec in SPECS.items():
        pts = load_cloud(problem)
        print(f"problem {problem} ({spec.label}): {len(pts)} points, dim={len(pts[0])}, "
              f"k={spec.k}, max_radius={spec.max_radius}, n_grid_max={spec.n_grid[-1]}, "
              f"true={spec.true_dimension}, tol={spec.tolerance}")
        assert len(pts) >= spec.n_grid[-1], f"problem {problem}: cloud too small"
