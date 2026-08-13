"""Round-3 skeptic: does N8d move the recorded R2-F8 control table (docs 10.4)?

docs/POLY_ALGEBRAIC_BENCHMARK.md 10.4 records a control table of
`mean_dimension(k=6, max_radius=6)` on clouds of known dimension at small n,
and draws from it the load-bearing claim

    "Poly NEVER returns ~1.0 for a 2- or 3-dimensional cloud at any n >= 32"

which is what makes the problem-02/10 criterion-(a) wins measurements rather
than tautologies. Several cells in that table are `nan` -- and `nan` is exactly
the value the N8d fallback branch (no confident node, near-constant majority)
can turn into a number near 1. If it does that on the 2D square, the 3D cube or
Lorenz, the recorded control is falsified and the claim it supports weakens.

Neither the round-2 nor the round-3 repair checked this table. This does.
Recorded row values (n = 32, 64, 128, 256, 512, 1024):

    circle (1)        nan     1.0000  1.0000  1.0000  1.0000  1.0000
    uniform square(2) 1.3206  nan     1.6420  1.9437  1.8488  1.9977
    uniform disk (2)  1.5730  1.5642  1.5709  1.8835  1.9620  2.0003
    uniform cube (3)  1.6227  nan     1.7839  2.1302  2.3233  2.4705
    Lorenz (2.05)     nan     nan     1.2270  1.7564  1.8991  1.9129

The exact seeds are not recorded, so this sweeps 12 seeds per family and asks
the structural question instead: at these n, does the shipped tree EVER report
a value in [0.75, 1.25] for a cloud whose true dimension is 2 or 3, in a place
where the pre-N8 tree reported nan or a value outside that band?
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.dimension import (  # noqa: E402
    local_dimension,
    mean_dimension,
    near_constant_consensus,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

N_VALUES = (32, 64, 128, 256, 512, 1024)
K = 6
MAX_RADIUS = 6


def circle(seed, n):
    rng = np.random.default_rng(seed)
    th = np.sort(rng.uniform(0, 2 * math.pi, size=n))
    return [(math.cos(t), math.sin(t)) for t in th], 1.0


def square(seed, n):
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(size=(n, 2))], 2.0


def disk(seed, n):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(size=n))
    th = rng.uniform(0, 2 * math.pi, size=n)
    return [(float(r[i] * math.cos(th[i])), float(r[i] * math.sin(th[i]))) for i in range(n)], 2.0


def cube(seed, n):
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(size=(n, 3))], 3.0


def lorenz(seed, n):
    rng = np.random.default_rng(seed)

    def d(s):
        x, y, z = s
        return np.array([10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z])

    s = np.array([1.0, 1.0, 1.0]) + rng.normal(scale=0.1, size=3)
    dt = 0.01
    traj = []
    for i in range(40_000):
        k1 = d(s)
        k2 = d(s + 0.5 * dt * k1)
        k3 = d(s + 0.5 * dt * k2)
        k4 = d(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= 5_000:
            traj.append(tuple(s))
    idx = np.linspace(0, len(traj) - 1, num=n).astype(int)
    return [traj[i] for i in idx], 2.05


FAMILIES = (
    ("circle", circle),
    ("uniform square", square),
    ("uniform disk", disk),
    ("uniform cube", cube),
    ("lorenz whole attractor", lorenz),
)

SEEDS = tuple(range(501, 513))


def main() -> None:
    print("=== R2-F8 CONTROL TABLE under the shipped N8d tree ===")
    print(f"k={K}, max_radius={MAX_RADIUS}, {len(SEEDS)} seeds per family\n")
    offenders = []
    fallback_on_non1d = []
    for fam_name, builder in FAMILIES:
        for n in N_VALUES:
            pre_vals, ship_vals, changed = [], [], 0
            for seed in SEEDS:
                pts, truth = builder(seed, n)
                try:
                    hg = knn_hypergraph(pts, k=K, dedupe=True)
                except Exception:  # noqa: BLE001 - degenerate tiny clouds
                    continue
                ests = [
                    local_dimension(hg, node, max_radius=MAX_RADIUS)
                    for node in sorted(hg.nodes)
                ]
                cons = near_constant_consensus(ests, threshold=0.9)
                pre = [e.dimension for e in ests if e.r_squared >= 0.9]
                pre_mean = sum(pre) / len(pre) if pre else float("nan")
                ship = mean_dimension(hg, max_radius=MAX_RADIUS)
                pre_vals.append(pre_mean)
                ship_vals.append(ship)
                same = (math.isnan(pre_mean) and math.isnan(ship)) or (
                    math.isfinite(pre_mean) and abs(pre_mean - ship) < 1e-12
                )
                if not same:
                    changed += 1
                # A REGRESSION is the shipped tree newly placing a 2D/3D cloud
                # in [0.75, 1.25] where the pre-N8 tree did not (nan counts as
                # "not there"). A row that was already in that band before N8
                # existed is a pre-existing property of small-n k-NN graphs,
                # not something this repair did.
                newly_in_band = math.isfinite(ship) and 0.75 <= ship <= 1.25 and not (
                    math.isfinite(pre_mean) and 0.75 <= pre_mean <= 1.25
                )
                if truth >= 2.0 and newly_in_band and not same:
                    offenders.append((fam_name, n, seed, truth, pre_mean, ship))
                if truth >= 2.0 and math.isnan(pre_mean) and math.isfinite(ship):
                    offenders.append((fam_name, n, seed, truth, pre_mean, ship))
                if truth >= 2.0 and cons and not pre:
                    fallback_on_non1d.append((fam_name, n, seed, truth, ship))
            fin_pre = [v for v in pre_vals if math.isfinite(v)]
            fin_ship = [v for v in ship_vals if math.isfinite(v)]
            print(
                f"  {fam_name:24s} n={n:5d}  "
                f"pre-N8 nan {len(pre_vals)-len(fin_pre)}/{len(pre_vals)}"
                f" mean {np.mean(fin_pre) if fin_pre else float('nan'):7.4f}"
                f" | shipped nan {len(ship_vals)-len(fin_ship)}/{len(ship_vals)}"
                f" mean {np.mean(fin_ship) if fin_ship else float('nan'):7.4f}"
                f" | rows changed {changed}"
            )
    print()
    print(f"NON-1D clouds where the shipped tree newly reports a value in [0.75, 1.25]: "
          f"{len(offenders)}")
    for o in offenders[:20]:
        print(f"   OFFENDER {o}")
    print(f"fallback branch (no confident node) firing on a NON-1D cloud: "
          f"{len(fallback_on_non1d)}")
    for o in fallback_on_non1d[:20]:
        print(f"   FALLBACK {o}")
    assert not offenders, "R2-F8's 'never ~1.0 for a 2D/3D cloud' control is falsified"
    print("\nR2-F8 CONTROL TABLE CHECK COMPLETE")


if __name__ == "__main__":
    main()
