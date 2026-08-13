"""Round-3 (N8d) step 3: does the graph-level rule make the min-fit-length gate redundant?

The task allows the consensus rule to REPLACE the per-node minimum-fit-length
gate or to sit alongside it. This script decides that empirically on the exact
clouds N8b was calibrated against (round 3's n8b_min_fit_length_gate.py: 8
non-1D clouds x k in {6,8,12} x max_radius in {3,4,6}), where the short-window
damage lives -- max_radius 3 and 4 are precisely the regime the length gate
switches the branch off in, and therefore the regime the consensus rule would
have to cover alone.

For each configuration it reports the spurious admissions (nodes accepted that
R^2 alone would reject, on a structure whose true dimension is not ~1) under:
  n8b        node-local + length gate (shipped)
  n8d        length gate AND graph-level consensus (proposed)
  n8d_nolen  graph-level consensus INSTEAD of the length gate
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_policy import R2_THRESHOLD, measure_graph, near_degenerate_ignoring_length  # noqa: E402
from n8d_sweep import consensus_fires  # noqa: E402

TOL = 0.25
FRAC = 0.50


def grid2d(n_side: int):
    return [(float(i), float(j)) for i in range(n_side) for j in range(n_side)]


def cube3d(n_side: int):
    return [
        (float(i), float(j), float(m))
        for i in range(n_side)
        for j in range(n_side)
        for m in range(n_side)
    ]


def disc2d(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * math.pi
    return list(zip((r * np.cos(th)).tolist(), (r * np.sin(th)).tolist(), strict=True))


def brownian2d(n: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    return list(map(tuple, np.cumsum(rng.normal(size=(n, 2)), axis=0).tolist()))


def lorenz(n: int):
    dt = 0.01
    x = np.array([1.0, 1.0, 1.0])
    out = []
    for i in range(n * 5 + 5000):
        dx = np.array(
            [10.0 * (x[1] - x[0]), x[0] * (28.0 - x[2]) - x[1], x[0] * x[1] - (8.0 / 3.0) * x[2]]
        )
        x = x + dt * dx
        if i >= 5000 and (i - 5000) % 5 == 0:
            out.append(tuple(x.tolist()))
    return out[:n]


def problem01_circle(n: int = 200):
    """Problem 01's phase-space ring: TRUE dimension 1 (the regime N8 is for)."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    n_steps = 31416
    idx = [min(int((i * phi) % 1.0 * n_steps), n_steps) for i in range(n)]
    return [
        (math.cos(2 * math.pi * j / n_steps), -math.sin(2 * math.pi * j / n_steps)) for j in idx
    ]


NON_1D_CASES = [
    ("2D grid 20x20", grid2d(20), 2.0),
    ("2D grid 30x30", grid2d(30), 2.0),
    ("2D random disc n=800", disc2d(800), 2.0),
    ("2D random disc n=2000", disc2d(2000), 2.0),
    ("2D Brownian (problem 06) n=1500", brownian2d(1500), 2.0),
    ("3D cube 10x10x10", cube3d(10), 3.0),
    ("3D cube 14x14x14", cube3d(14), 3.0),
    ("Lorenz attractor n=1500", lorenz(1500), 2.06),
]


def score(estimates, *, policy: str):
    """(#admitted beyond R^2, worst |dim - truth| among them, kept mean)."""
    if policy == "n8b":
        fires, is_near = True, lambda e: e.near_degenerate
    elif policy == "n8d":
        is_near = lambda e: e.near_degenerate  # noqa: E731
        fires = consensus_fires(estimates, tol=TOL, frac=FRAC, use_length_gate=True)
    elif policy == "n8d_nolen":
        is_near = near_degenerate_ignoring_length
        fires = consensus_fires(estimates, tol=TOL, frac=FRAC, use_length_gate=False)
    else:
        raise ValueError(policy)
    kept = [e for e in estimates if e.r_squared >= R2_THRESHOLD or (fires and is_near(e))]
    admitted = [e for e in kept if e.r_squared < R2_THRESHOLD]
    mean = sum(e.dimension for e in kept) / len(kept) if kept else float("nan")
    return admitted, mean


def main() -> None:
    print("=" * 108)
    print("N8d STEP 3: is the min-fit-length gate still needed once the graph rule exists?")
    print(f"    consensus: |mean(confident dims) - 1| <= {TOL}, else near_deg_frac >= {FRAC}")
    print("=" * 108)

    totals = {"n8b": 0, "n8d": 0, "n8d_nolen": 0}
    worst = {"n8b": 0.0, "n8d": 0.0, "n8d_nolen": 0.0}
    print(f"\n{'cloud':<34}{'k':>3}{'R':>3} | " + " | ".join(f"{p:>22}" for p in totals))
    for name, points, truth in NON_1D_CASES:
        for k in (6, 8, 12):
            for max_radius in (3, 4, 6):
                m = measure_graph(points, k=k, max_radius=max_radius, samples=200)
                if m is None:
                    continue
                cells = []
                any_admit = False
                for policy in totals:
                    admitted, _mean = score(m.estimates, policy=policy)
                    totals[policy] += len(admitted)
                    if admitted:
                        any_admit = True
                        w = max(abs(e.dimension - truth) for e in admitted)
                        worst[policy] = max(worst[policy], w)
                        cells.append(f"{len(admitted):>3} adm, worst {w:.4f}")
                    else:
                        cells.append(f"{0:>3} adm{'':>16}")
                if any_admit:
                    print(
                        f"{name:<34}{k:>3}{max_radius:>3} | "
                        + " | ".join(f"{c:>22}" for c in cells)
                    )

    print("\n  TOTAL spurious admissions over 72 non-1D configurations:")
    for policy in totals:
        print(f"    {policy:>10}: {totals[policy]:>4}   worst |dim - truth| = {worst[policy]:.4f}")

    print("\n" + "=" * 108)
    print("THE OTHER SIDE: problem 01's genuinely-1D ring, where the branch MUST work")
    print("=" * 108)
    ring = problem01_circle(200)
    print(f"\n{'max_radius':>10} | " + " | ".join(f"{p:>26}" for p in ("pre_n8", *totals)))
    for max_radius in (3, 4, 5, 6):
        m = measure_graph(ring, k=6, max_radius=max_radius, samples=200)
        assert m is not None
        base = [e for e in m.estimates if e.r_squared >= R2_THRESHOLD]
        base_mean = sum(e.dimension for e in base) / len(base) if base else float("nan")
        cells = [f"kept {len(base):>3}/200 mean {base_mean:.4f}"]
        for policy in totals:
            admitted, mean = score(m.estimates, policy=policy)
            cells.append(f"kept {len(base) + len(admitted):>3}/200 mean {mean:.4f}")
        print(f"{max_radius:>10} | " + " | ".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    main()
