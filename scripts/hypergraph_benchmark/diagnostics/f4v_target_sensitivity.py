"""F4 verification (e), decisive half: would re-typing the TARGET change 08/09?

The F4 caveat is only consequential if scoring the shell estimator against D_0
instead of D_2 would change what the benchmark recorded.  That is a testable
question, not an interpretive one, and it does not depend on knowing D_0 for
Lorenz exactly -- it only depends on how SENSITIVE the recorded verdict is to
the target.

METHOD.  Reproduce problems 08 and 09's own cloud construction and estimator
settings exactly (round2/08_lorenz_attractor.py, round2/09_rossler_attractor.py),
compute each estimator's dimension ONCE per n on the production n_grid, then
sweep the scoring target over the whole plausible D_0 range and recompute
`poly_algebraic_min_n` / `traditional_min_n` at each target by exactly the
library's own stable-convergence rule.

If min_n is flat over the target range, the F4 category error -- whatever its
merits as a matter of estimator theory -- cannot be what produced the
"poly needs 2-4x more points" verdicts.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.dimension import mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)


def rk4(rhs, s0, dt, n_steps):
    out = np.empty((n_steps + 1, 3))
    out[0] = s0
    s = s0.copy()
    for i in range(n_steps):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = s
    return out


def lorenz_cloud():
    """round2/08: dt=0.005, transient 10, total 100, no stride, k=10."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def rhs(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    dt, t_transient, t_total = 0.005, 10.0, 100.0
    states = rk4(rhs, np.array([1.0, 1.0, 1.0]), dt, int(t_total / dt))
    post = states[int(t_transient / dt):]
    return [tuple(map(float, p)) for p in post], 10


def rossler_cloud():
    """round2/09: dt=0.01, transient 100, strided to 20 pts/time-unit, k=15."""
    a, b, c = 0.2, 0.2, 5.7

    def rhs(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    dt, t_transient = 0.01, 100.0
    density = 20.0
    n_max = 12800
    t_post = n_max / density
    n_steps = int((t_transient + t_post) / dt) + 10
    states = rk4(rhs, np.array([1.0, 1.0, 1.0]), dt, n_steps)
    post = states[int(t_transient / dt):]
    stride = int(round((1.0 / density) / dt))
    sub = post[::stride][:n_max]
    return [tuple(map(float, p)) for p in sub], 15


def curves(points, k, max_radius=6):
    poly, trad = [], []
    for n in N_GRID:
        if n > len(points):
            break
        sub = points[:n]
        try:
            hg = knn_hypergraph(sub, k=k, dedupe=True)
            p = mean_dimension(hg, samples=min(n, 40), max_radius=max_radius)
        except Exception:
            p = float("nan")
        t = correlation_dimension(sub).dimension
        poly.append((n, p))
        trad.append((n, t))
        print(f"    n={n:<6} poly={p:8.4f}  trad={t:8.4f}", flush=True)
    return poly, trad


def min_n(results, target, tol):
    """The library's own stable-convergence rule, verbatim."""
    for i, (n, d) in enumerate(results):
        if not math.isfinite(d):
            continue
        if abs(d - target) > tol:
            continue
        if all(math.isfinite(x) and abs(x - target) <= tol for _, x in results[i:]):
            return n
    return None


def main() -> None:
    for name, builder, recorded_target in (
        ("08 Lorenz", lorenz_cloud, 2.05),
        ("09 Rossler", rossler_cloud, 2.01),
    ):
        print("=" * 100)
        print(f"{name}   (recorded scoring target {recorded_target}, tolerance 0.5)")
        print("=" * 100)
        pts, k = builder()
        print(f"  cloud: {len(pts)} points, k={k}, max_radius=6")
        poly, trad = curves(pts, k)
        print()
        print(f"  {'target':>7} | {'poly min_n':>10} {'trad min_n':>10} | {'savings':>9}  verdict")
        prev = None
        for target in np.arange(1.90, 2.451, 0.01):
            pn = min_n(poly, float(target), 0.5)
            tn = min_n(trad, float(target), 0.5)
            sav = (1.0 - pn / tn) if (pn and tn) else None
            row = (pn, tn)
            mark = "" if row == prev else "  <-- change"
            if row != prev or abs(target - recorded_target) < 5e-3:
                print(f"  {target:7.2f} | {str(pn):>10} {str(tn):>10} | "
                      f"{('%.3f' % sav) if sav is not None else 'n/a':>9}"
                      f"{mark}"
                      f"{'   [RECORDED TARGET]' if abs(target-recorded_target) < 5e-3 else ''}")
            prev = row
        print()


if __name__ == "__main__":
    main()
