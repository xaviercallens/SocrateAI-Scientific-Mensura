"""N8b: choose and validate a MINIMUM FIT LENGTH for the near-constant gate.

Background. N8 added a second, R^2-independent acceptance branch for shell
sequences with almost no dynamic range, justified by `slope_bound`: a rigorous
cap on |dimension - 1|. Independent verification confirmed the bound is real and
tight, but found the justification's blind spot: `slope_bound` bounds
|dimension - 1|, NOT |dimension - truth|. On a structure whose true local
dimension is not ~1, a short near-constant window is admitted with a dimension
near 1 and therefore badly wrong. Measured: 2D Brownian motion (truth 2) at
k=6, max_radius=3 admitted 13/200 nodes, |dimension - truth| up to 1.18, moving
the sampled mean 1.8321 -> 1.7381, i.e. AWAY from the truth.

MECHANISM (why the damage is specifically a SHORT-WINDOW effect, not a
threshold-calibration effect). In a k-NN graph the radius-1 shell is the node's
degree: it is pinned near k by construction and carries no dimensional
information at all. The dimensional signal only appears once the ball has grown
past that degree plateau. A window of 3 radii is therefore roughly one
informative point plus noise, and "nearly constant" is not distinguishable from
"noisy" in ANY structure -- 1D or not. Lengthening the window does not tighten
the bound (the bound is about spread, not length); it removes the regime where
the near-constant hypothesis is unfalsifiable.

This script picks the threshold by measurement rather than by taste:
  1. anchor preservation  -- problem 01's documented node (fit length 6),
  2. spurious admissions  -- across 9 clouds x k in {6,8,12} x max_radius in
                             {3,4,6}, counting only clouds whose true dimension
                             is NOT ~1 (admitting there is what "spurious"
                             means),
  3. 1D coverage cost     -- how much of the genuine near-constant band
                             (problem 01's own cloud) each threshold gives up.

Run: python scripts/hypergraph_benchmark/round3/n8b_min_fit_length_gate.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph import knn_hypergraph, local_dimension  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CV,
    NEAR_CONSTANT_MIN_FIT_LENGTH,
    NEAR_CONSTANT_SLOPE_BOUND,
)

# --------------------------------------------------------------------------
# A re-implementation of the gate, parameterised by the candidate minimum fit
# length. Deliberately NOT importing the shipped predicate, so the sweep below
# can compare candidate thresholds (including the shipped one) on equal terms.
# --------------------------------------------------------------------------


def gate(shells: list[float], min_fit_length: int) -> tuple[float, float, float, float, bool]:
    """Return (dimension, r_squared, shell_cv, slope_bound, near_degenerate)."""
    n = len(shells)
    xs = [float(r) for r in range(1, n + 1)]
    log_x = [math.log(x) for x in xs]
    log_y = [math.log(y) for y in shells]
    mean_x = sum(log_x) / n
    mean_y = sum(log_y) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in log_x)
    if var_x == 0:
        return 1.0, 0.0, 0.0, 0.0, False
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in log_y)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_x, log_y, strict=True))
    scale = max(1.0, sum(y * y for y in log_y))
    degenerate = ss_tot <= 1e-24 * scale
    r2 = 1.0 if degenerate else 1.0 - ss_res / ss_tot
    mean_lin = sum(shells) / n
    cv = math.sqrt(sum((y - mean_lin) ** 2 for y in shells) / n) / mean_lin
    bound = (max(log_y) - min(log_y)) * sum(abs(x - mean_x) for x in log_x) / (2.0 * var_x)
    near = (
        not degenerate
        and n >= min_fit_length
        and cv <= NEAR_CONSTANT_CV
        and bound <= NEAR_CONSTANT_SLOPE_BOUND
    )
    return slope + 1.0, r2, cv, bound, near


def shells_of(est) -> list[float]:
    vols = est.volumes
    out = []
    prev = 1
    for v in vols:
        s = v - prev
        if s <= 0:
            break
        out.append(float(s))
        prev = v
    return out


# --------------------------------------------------------------------------
# Point clouds (identical constructions to n8_collateral_damage.py)
# --------------------------------------------------------------------------


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

CANDIDATES = (2, 3, 4, 5, 6)


def main() -> None:
    print("=" * 100)
    print("N8b: minimum fit length for the near-constant branch")
    print(f"    CV<={NEAR_CONSTANT_CV}  slope_bound<={NEAR_CONSTANT_SLOPE_BOUND}  (unchanged)")
    print("=" * 100)

    # ---- 1. ANCHOR ------------------------------------------------------
    print("\n[1] ANCHOR: problem 01 n=200 k=6 max_radius=6, shells (6,5,5,6,6,6)")
    anchor = [6.0, 5.0, 5.0, 6.0, 6.0, 6.0]
    for m in CANDIDATES:
        d, r2, cv, b, near = gate(anchor, m)
        print(
            f"    min_fit_length={m}: dim={d:.4f} R2={r2:.4f} cv={cv:.4f} "
            f"bound={b:.4f} near_degenerate={near}  kept={near or r2 >= 0.9}"
        )

    # ---- 2. SPURIOUS ADMISSIONS in structures whose truth is NOT 1 ------
    print("\n[2] SPURIOUS ADMISSIONS (truth != 1). counts = near_degenerate nodes admitted")
    header = f"{'case':<32} {'k':>2} {'R':>2}" + "".join(f"{'m=' + str(m):>7}" for m in CANDIDATES)
    print(header)
    totals = dict.fromkeys(CANDIDATES, 0)
    worst_err = dict.fromkeys(CANDIDATES, 0.0)
    means: dict[tuple, dict[int, float]] = {}
    for label, pts, truth in NON_1D_CASES:
        for k in (6, 8, 12):
            for max_radius in (3, 4, 6):
                hg = knn_hypergraph(pts, k=k, dedupe=True)
                nodes = sorted(hg.nodes)
                step = max(1, len(nodes) // 200)
                nodes = nodes[::step][:200]
                ests = [local_dimension(hg, u, max_radius=max_radius) for u in nodes]
                rows = []
                for e in ests:
                    sh = shells_of(e)
                    if len(sh) < 2:
                        continue
                    rows.append((sh, e.r_squared, e.dimension))
                counts = []
                for m in CANDIDATES:
                    adm = []
                    kept = []
                    for sh, r2, _dim in rows:
                        d, _r2, _cv, _b, near = gate(sh, m)
                        if near:
                            adm.append(d)
                        if near or r2 >= 0.9:
                            kept.append(d)
                    counts.append(len(adm))
                    totals[m] += len(adm)
                    if adm:
                        worst_err[m] = max(worst_err[m], max(abs(a - truth) for a in adm))
                    means.setdefault((label, k, max_radius), {})[m] = (
                        sum(kept) / len(kept) if kept else float("nan")
                    )
                if any(counts):
                    print(
                        f"{label:<32} {k:>2} {max_radius:>2}"
                        + "".join(f"{c:>7}" for c in counts)
                        + f"   truth={truth}"
                    )
    print("    " + "-" * 60)
    print(
        f"{'TOTAL spurious admissions':<32} {'':>2} {'':>2}"
        + "".join(f"{totals[m]:>7}" for m in CANDIDATES)
    )
    print(
        f"{'worst |dim - truth| admitted':<32} {'':>2} {'':>2}"
        + "".join(f"{worst_err[m]:>7.2f}" for m in CANDIDATES)
    )

    # ---- 3. THE HEADLINE CASE -------------------------------------------
    print("\n[3] HEADLINE: 2D Brownian n=1500, k=6, max_radius=3 (truth 2.0)")
    hg = knn_hypergraph(brownian2d(1500), k=6, dedupe=True)
    nodes = sorted(hg.nodes)
    nodes = nodes[:: max(1, len(nodes) // 200)][:200]
    ests = [local_dimension(hg, u, max_radius=3) for u in nodes]
    r2only = [e.dimension for e in ests if e.r_squared >= 0.9]
    print(f"    R^2-only baseline (no near-constant branch): mean={sum(r2only) / len(r2only):.4f}")
    for m in CANDIDATES:
        adm, kept = [], []
        for e in ests:
            sh = shells_of(e)
            if len(sh) < 2:
                continue
            d, _r2, _cv, _b, near = gate(sh, m)
            if near:
                adm.append((tuple(int(s) for s in sh), d))
            if near or e.r_squared >= 0.9:
                kept.append(d)
        mean = sum(kept) / len(kept) if kept else float("nan")
        we = max((abs(d - 2.0) for _s, d in adm), default=0.0)
        print(
            f"    min_fit_length={m}: admitted={len(adm):>3}  mean={mean:.4f} "
            f"(truth 2.0, |err|={abs(mean - 2.0):.4f})  worst |dim-truth| admitted={we:.4f}"
        )
        if m == 2 and adm:
            for s, d in adm[:5]:
                print(f"        e.g. shells={s} dim={d:.4f} |dim-truth|={abs(d - 2.0):.4f}")

    # ---- 4. 1D COVERAGE COST -------------------------------------------
    print("\n[4] 1D COVERAGE COST: problem 01's own ring (truth 1.0) -- what do we give up?")
    pts = problem01_circle(200)
    for max_radius in (3, 4, 5, 6):
        hg = knn_hypergraph(pts, k=6, dedupe=True)
        ests = [local_dimension(hg, u, max_radius=max_radius) for u in sorted(hg.nodes)]
        line = f"    max_radius={max_radius}: "
        for m in CANDIDATES:
            kept = []
            for e in ests:
                sh = shells_of(e)
                if len(sh) < 2:
                    continue
                d, _r2, _cv, _b, near = gate(sh, m)
                if near or e.r_squared >= 0.9:
                    kept.append(d)
            mean = sum(kept) / len(kept) if kept else float("nan")
            line += f" m={m}:{len(kept):>3}/{len(ests)} mean={mean:.3f} "
        print(line)

    # ---- 5. THE SHIPPED GATE AGREES WITH THE SWEPT COLUMN ---------------
    # Everything above uses this file's own re-implementation, so it could in
    # principle be measuring a gate the library does not actually have. Check
    # the shipped predicate node-for-node against the swept m=SHIPPED column.
    print(f"\n[5] SHIPPED GATE == swept column m={NEAR_CONSTANT_MIN_FIT_LENGTH}?")
    mismatches = 0
    checked = 0
    for label, pts, _truth in NON_1D_CASES[:5] + [("problem 01 ring", problem01_circle(200), 1.0)]:
        for k in (6, 12):
            for max_radius in (3, 4, 6):
                hg = knn_hypergraph(pts, k=k, dedupe=True)
                nodes = sorted(hg.nodes)
                nodes = nodes[:: max(1, len(nodes) // 100)][:100]
                for u in nodes:
                    e = local_dimension(hg, u, max_radius=max_radius)
                    sh = shells_of(e)
                    if len(sh) < 2:
                        continue
                    _d, _r2, _cv, _b, near = gate(sh, NEAR_CONSTANT_MIN_FIT_LENGTH)
                    checked += 1
                    if near != e.near_degenerate:
                        mismatches += 1
                        if mismatches <= 5:
                            print(f"    MISMATCH {label} k={k} R={max_radius} shells={sh}")
    print(f"    checked {checked} nodes, {mismatches} mismatches")
    assert mismatches == 0, "shipped gate does not match the swept threshold"
    print(f"    OK: shipped NEAR_CONSTANT_MIN_FIT_LENGTH = {NEAR_CONSTANT_MIN_FIT_LENGTH}")


if __name__ == "__main__":
    main()
