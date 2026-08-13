"""AR iteration 1: is the Rossler win a property of the data or of the settings?

Finding R2-F3 recorded a win that turned out to be contingent on one
non-default parameter with a wrong stated justification, so the same question
is put to this one BEFORE it is banked. The claim under test is:

    with `theiler_window="auto"`, the shell-growth estimator's
    `poly_algebraic_min_n` on the Rossler attractor falls from 1600 to 400.

Three sweeps, all against the production cloud unless stated:

  * k -- the one shell-growth knob that changes the graph.
  * max_radius -- the one that changes the fit window.
  * dt / sampling density -- the knob that changes how temporally oversampled
    the trajectory is, which is the mechanism the window addresses, so this is
    the sweep most likely to break it.

The honest reading is printed at the end: the n=400 crossing is within
tolerance by 0.463 of 0.5, i.e. it is a knife-edge grid point, so the sweep
reports BOTH the headline min_n and the more conservative "first n from which
the estimate stays within HALF the tolerance", which is the number that does
not depend on a near-boundary crossing.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_clouds import SPECS, load_cloud  # noqa: E402

from socrates.hypergraph.dimension import mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)


def min_n(series, true_d, tol):
    for i, (n, d) in enumerate(series):
        if not math.isfinite(d) or abs(d - true_d) > tol:
            continue
        if all(math.isfinite(x) and abs(x - true_d) <= tol for _, x in series[i:]):
            return n
    return None


def series_for(points, k, max_radius, window, samples=40):
    out = []
    for n in N_GRID:
        if n > len(points) or k >= n:
            break
        try:
            hg = knn_hypergraph(points[:n], k=k, dedupe=True, theiler_window=window)
            out.append((n, mean_dimension(hg, samples=min(n, samples), max_radius=max_radius)))
        except ValueError:
            out.append((n, float("nan")))
    return out


def rossler_cloud(dt: float, density: float, n_max: int = 12800):
    """Same physics as round2/09, at a chosen integration step and sample density."""
    a, b, c = 0.2, 0.2, 5.7

    def deriv(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    def rk4(s, h):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * h * k1)
        k3 = deriv(s + 0.5 * h * k2)
        k4 = deriv(s + h * k3)
        return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    stride = max(1, int(round((1.0 / density) / dt)))
    transient = int(round(100.0 / dt))
    total = transient + n_max * stride
    state = np.array([1.0, 1.0, 1.0])
    out = []
    for step in range(total + 1):
        if step >= transient and (step - transient) % stride == 0:
            out.append(tuple(float(v) for v in state))
        state = rk4(state, dt)
    return out[:n_max]


def report(label, series, true_d, tol):
    strict = min_n(series, true_d, tol)
    conservative = min_n(series, true_d, tol / 2)
    vals = " ".join(f"{d:6.2f}" for _, d in series)
    print(f"  {label:<26} min_n={str(strict):>5}  min_n(half tol)={str(conservative):>5}  {vals}")


def main() -> None:
    spec = SPECS["09"]
    points = load_cloud("09")
    td, tol = spec.true_dimension, spec.tolerance
    print(f"=== Rossler, true={td}, tolerance={tol}; n_grid={N_GRID}\n")

    print("--- baseline for reference (window OFF, production k/max_radius) ---")
    report("window=0", series_for(points, spec.k, spec.max_radius, 0), td, tol)

    print("\n--- sweep 1: k (window='auto', max_radius=6) ---")
    for k in (8, 10, 12, 15, 20, 25):
        report(f"k={k}", series_for(points, k, spec.max_radius, "auto"), td, tol)

    print("\n--- sweep 2: max_radius (window='auto', k=15) ---")
    for mr in (4, 5, 6, 7, 8):
        report(f"max_radius={mr}", series_for(points, spec.k, mr, "auto"), td, tol)

    print("\n--- sweep 3: integration step / sample density (window='auto', k=15, mr=6) ---")
    for dt, density in ((0.01, 20.0), (0.01, 10.0), (0.01, 40.0), (0.005, 20.0), (0.02, 20.0)):
        cloud = rossler_cloud(dt, density)
        report(f"dt={dt}, {density} pts/tu", series_for(cloud, spec.k, spec.max_radius, "auto"),
               td, tol)

    print(
        "\nHOW TO READ min_n(half tol). The n=400 crossing on the production cloud\n"
        "sits 0.463 from the truth against a 0.5 tolerance, so it is a boundary\n"
        "grid point and a single redraw could move it. The half-tolerance column\n"
        "is the same convergence question asked with no boundary case in it; where\n"
        "the two columns agree the win does not rest on the boundary."
    )


if __name__ == "__main__":
    main()
