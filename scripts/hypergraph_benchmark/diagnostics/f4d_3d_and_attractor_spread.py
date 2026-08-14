"""F4, step D: the 3D arm, and how large the D_0 - D_2 gap actually is on
Lorenz and Rossler.

Two parts, both aimed at the concrete question "what does this imply for the
recorded 08/09 verdicts?".

D1  THE 3D ARM. Repeats the identification on the Sierpinski tetrahedron: an
    exactly-known fractal in R^3 with D_0 = 2.0000 exactly -- i.e. the same
    ambient dimension and almost the same D_0 as the benchmark's Lorenz
    (D_0 ~ 2.06) and Rossler (~2.02) -- carrying measures with D_2 from 2.000
    down to 0.943. If the shell estimator reports ~2 across that whole range,
    it is tracking the support in exactly the geometry the 08/09 verdicts live
    in, not only in the plane.

D2  HOW BIG IS THE GAP ON THE REAL ATTRACTORS? A category error only rewrites
    a verdict if the two categories actually differ there. D_0 - D_2 is
    measured on Lorenz and Rossler clouds by box counting, with the SAME method
    calibrated on the exact targets first -- box counting is known to compress
    D_0 downward at finite n (step A measured exactly that), so the calibration
    reports how much of a true gap this method recovers, and the attractor
    numbers are read only through that calibration. Reported as a bound, not as
    a point estimate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from f4_multifractal_targets import BY_NAME, TARGETS_3D  # noqa: E402

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.dimension import mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

SEED = 20260814
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
RA, RB, RC = 0.2, 0.2, 5.7


def _rk4(deriv, state0, dt, n_steps):
    out = np.empty((n_steps + 1, len(state0)))
    s = np.asarray(state0, float)
    out[0] = s
    for i in range(n_steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = s
    return out


def lorenz_cloud(n: int) -> np.ndarray:
    """Canonical Lorenz (sigma=10, rho=28, beta=8/3), same parameters as
    scripts/hypergraph_benchmark/08_lorenz_attractor.py; long run, transient
    discarded, strided down to `n` points spread over the whole run."""
    dt, t_transient, t_total = 0.005, 10.0, 4000.0
    traj = _rk4(
        lambda s: np.array(
            [SIGMA * (s[1] - s[0]), s[0] * (RHO - s[2]) - s[1], s[0] * s[1] - BETA * s[2]]
        ),
        [1.0, 1.0, 1.0],
        dt,
        int(t_total / dt),
    )
    post = traj[int(t_transient / dt) :]
    return post[:: max(1, len(post) // n)][:n]


def rossler_cloud(n: int) -> np.ndarray:
    """Canonical Rossler (a=b=0.2, c=5.7), same parameters as
    scripts/hypergraph_benchmark/09_rossler_attractor.py."""
    dt, t_transient, t_total = 0.01, 100.0, 12000.0
    traj = _rk4(
        lambda s: np.array([-s[1] - s[2], s[0] + RA * s[1], RB + s[2] * (s[0] - RC)]),
        [1.0, 1.0, 1.0],
        dt,
        int(t_total / dt),
    )
    post = traj[int(t_transient / dt) :]
    return post[:: max(1, len(post) // n)][:n]


def renyi_by_box_counting(points: np.ndarray, *, min_per_box: float = 8.0, n_levels: int = 5):
    """D_0, D_1, D_2 by box counting on a common, automatically chosen window.

    The finest scale is the largest halving level at which the average
    occupancy of an occupied box is still >= `min_per_box`, so no level is
    counting the sample rather than the set; `n_levels` successive halvings
    below that are used. Returns the three slopes and the window actually used,
    so the same window can be quoted alongside the numbers.
    """
    n = len(points)
    lo = points.min(axis=0)
    span = float((points.max(axis=0) - lo).max())
    finest = 1
    for m in range(1, 20):
        idx = np.floor((points - lo) / (span * 2.0**-m)).astype(np.int64)
        occ = len(np.unique(idx, axis=0))
        if n / occ < min_per_box:
            break
        finest = m
    levels = list(range(max(1, finest - n_levels + 1), finest + 1))
    logs, n_occ, ent, corr = [], [], [], []
    for m in levels:
        eps = 2.0**-m
        idx = np.floor((points - lo) / (span * eps)).astype(np.int64)
        _, counts = np.unique(idx, axis=0, return_counts=True)
        mu = counts / n
        logs.append(math.log(1.0 / eps))
        n_occ.append(math.log(len(counts)))
        ent.append(float(-np.sum(mu * np.log(mu))))
        corr.append(-math.log(float(np.sum(mu * mu))))
    x = np.asarray(logs)
    f = lambda y: float(np.polyfit(x, np.asarray(y), 1)[0])  # noqa: E731
    return f(n_occ), f(ent), f(corr), levels


def main() -> None:
    print("=" * 112)
    print("D1  THE 3D ARM -- Sierpinski tetrahedron in R^3, D_0 = 2.0000 exactly")
    print("    (same ambient dimension and nearly the same D_0 as Lorenz/Rossler)")
    print("=" * 112)
    print(
        f"{'target':<12}{'D0':>7}{'D1':>7}{'D2':>7}{'n':>7}{'k':>4}{'max_r':>6} | "
        f"{'shell':>8}{'wf':>6} | {'GP':>8} | {'|s-D0|':>8}{'|s-D1|':>8}{'|s-D2|':>8}  nearest"
    )
    for t in TARGETS_3D:
        for n in (3200, 12800):
            pts = t.sample(n, SEED)
            tup = list(map(tuple, pts))
            gp = correlation_dimension(tup, n_radii=30)
            for k, mr in ((10, 6), (15, 6), (15, 8), (25, 6)):
                hg = knn_hypergraph(tup, k, dedupe=True, duplicate_tolerance=1e-13)
                s = mean_dimension(hg, samples=80, max_radius=mr)
                errs = {"D0": abs(s - t.D0), "D1": abs(s - t.D1), "D2": abs(s - t.D2)}
                near = min(errs, key=errs.get)
                print(
                    f"{t.name:<12}{t.D0:>7.3f}{t.D1:>7.3f}{t.D2:>7.3f}{n:>7}{k:>4}{mr:>6} | "
                    f"{s:>8.4f}{'':>6} | {gp.dimension:>8.4f} | "
                    f"{errs['D0']:>8.4f}{errs['D1']:>8.4f}{errs['D2']:>8.4f}  {near}",
                    flush=True,
                )

    print()
    print("=" * 112)
    print("D2  HOW BIG IS D_0 - D_2 ON THE REAL ATTRACTORS?")
    print("    Box counting, n=200000, identical automatic window for every row.")
    print("    Calibration rows first: 'recovered' = measured gap / true gap.")
    print("=" * 112)
    n_big = 200_000
    print(
        f"{'cloud':<14}{'D0 true':>9}{'D2 true':>9}{'gap true':>10} | "
        f"{'D0 box':>8}{'D1 box':>8}{'D2 box':>8}{'gap box':>9}{'recovered':>11}  levels"
    )
    calib = []
    for name in ("ST-unif", "ST-mild", "ST-matched", "ST-strong", "SQ-strong", "GA-mild"):
        t = BY_NAME[name]
        pts = t.sample(n_big, SEED)
        d0, d1, d2, lv = renyi_by_box_counting(pts)
        gap_true, gap_box = t.D0 - t.D2, d0 - d2
        rec = gap_box / gap_true if gap_true > 1e-9 else float("nan")
        if gap_true > 0.2:
            calib.append(rec)
        print(
            f"{name:<14}{t.D0:>9.4f}{t.D2:>9.4f}{gap_true:>10.4f} | "
            f"{d0:>8.4f}{d1:>8.4f}{d2:>8.4f}{gap_box:>9.4f}{rec:>11.3f}  {lv}",
            flush=True,
        )
    worst = min(calib)
    print(f"\n    worst-case recovery factor over the calibration rows: {worst:.3f}")
    print("    => a measured box gap g implies a true gap of at most g / %.3f\n" % worst)

    for name, cloud in (("Lorenz", lorenz_cloud(n_big)), ("Rossler", rossler_cloud(n_big))):
        d0, d1, d2, lv = renyi_by_box_counting(cloud)
        gp = correlation_dimension(list(map(tuple, cloud[:20000])), n_radii=30)
        print(
            f"{name:<14}{'?':>9}{'?':>9}{'?':>10} | "
            f"{d0:>8.4f}{d1:>8.4f}{d2:>8.4f}{d0 - d2:>9.4f}"
            f"{'':>11}  {lv}   GP(n=20000)={gp.dimension:.4f} R2={gp.r_squared:.4f}"
        )
        print(f"{'':>14}implied bound on the true D_0 - D_2: <= {(d0 - d2) / worst:.3f}")


if __name__ == "__main__":
    main()
