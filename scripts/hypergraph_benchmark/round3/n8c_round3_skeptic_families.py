"""Round-3 independent skeptic, part 5: new near-constant families, exhaustive
scans by fit length, and the N1 (exactly-constant) regression check.

Families here are chosen to be ones NEITHER prior round used. Every row is
cross-checked against an independent numpy re-derivation of dimension, R^2,
CV and slope_bound, and against a brute-force search over arrangements inside
the observed log band (the bound must never be exceeded).

Run: python scripts/hypergraph_benchmark/round3/n8c_round3_skeptic_families.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.core import Hypergraph  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CV,
    NEAR_CONSTANT_MIN_FIT_LENGTH,
    NEAR_CONSTANT_SLOPE_BOUND,
    _log_log_fit,
    local_dimension,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402


def independent(shells, min_radius: int = 1):
    """numpy re-derivation of everything the gate uses."""
    radii = list(range(min_radius, min_radius + len(shells)))
    x = np.log(np.asarray(radii, float))
    y = np.log(np.asarray(shells, float))
    slope, intercept = np.polyfit(x, y, 1)
    ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    s = np.asarray(shells, float)
    cv = float(s.std() / s.mean())
    var_x = float(np.sum((x - x.mean()) ** 2))
    bound = float(y.max() - y.min()) * float(np.sum(np.abs(x - x.mean()))) / (2 * var_x)
    near = (
        len(shells) >= NEAR_CONSTANT_MIN_FIT_LENGTH
        and cv <= NEAR_CONSTANT_CV
        and bound <= NEAR_CONSTANT_SLOPE_BOUND
    )
    return float(slope) + 1.0, r2, cv, bound, near


def bound_is_respected(shells, trials: int = 4000, seed: int = 5) -> float:
    """Worst |slope| / slope_bound over random y-arrangements inside the SAME
    log band. Must stay <= 1."""
    rng = np.random.default_rng(seed)
    n = len(shells)
    x = np.log(np.arange(1, n + 1, dtype=float))
    lo, hi = math.log(min(shells)), math.log(max(shells))
    _, _, _, bound, _ = independent(shells)
    if bound == 0:
        return 0.0
    var_x = float(np.sum((x - x.mean()) ** 2))
    worst = 0.0
    ys = rng.uniform(lo, hi, size=(trials, n))
    # include the extremal arrangement explicitly
    extremal = np.where(x - x.mean() >= 0, hi, lo)
    ys = np.vstack([ys, extremal])
    for y in ys:
        sl = float(np.sum((x - x.mean()) * (y - y.mean())) / var_x)
        worst = max(worst, abs(sl) / bound)
    return worst


FAMILIES = [
    # (shells, must_be_near_degenerate, why)
    ((14, 14, 15, 15, 16, 16), True, "NEW: gentle monotone ramp, base 14, len 6"),
    ((9, 9, 10, 9, 9), True, "NEW: length exactly at the threshold (5), base 9"),
    (
        (40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 41, 40),
        True,
        "NEW: very long window (len 12), base 40, single dip + single bump",
    ),
    ((1000, 1005, 995, 1002, 998, 1001), True, "NEW: base 1000, +/-0.5% noise"),
    ((17, 16, 16, 16, 16, 15), True, "NEW: gentle monotone DESCENT, base 17"),
    ((23, 25, 24, 22, 25, 23), True, "NEW: irregular wobble, base 23"),
    ((31, 29, 30, 33, 28, 30, 31), True, "NEW: len 7, base 30, +/-10% irregular"),
    ((9, 9, 13, 9, 9), False, "NEW negative control: len 5 but CV too large"),
    ((9, 9, 10, 9), False, "NEW negative control: near-constant but len 4 < 5"),
    ((14, 14, 15, 15), False, "NEW negative control: the len-6 ramp truncated to len 4"),
    ((7, 14, 21, 28, 35, 42), False, "NEW positive control: exact r^1 (true 2D)"),
    ((13, 17, 13, 13, 17, 15), True, "NEW: the REAL round-2 problem-06 culprit shells"),
    ((2, 20, 2, 20, 2, 20), False, "NEW negative control: 10x oscillation"),
    ((60, 55, 40, 25, 12, 4), False, "NEW negative control: strong monotone decay"),
]


def main() -> int:
    print("=" * 108)
    print("PART 5A: NEAR-CONSTANT FAMILIES NEITHER PRIOR ROUND USED")
    print("=" * 108)
    print(
        f"{'shells':<44} {'len':>3} {'dim':>8} {'R^2':>7} {'CV':>7} {'bound':>7} "
        f"{'near':>5} {'exp':>5} {'|sl|/bnd':>9}"
    )
    failures = []
    for shells, expect, why in FAMILIES:
        xs = [float(r) for r in range(1, len(shells) + 1)]
        fit = _log_log_fit(xs, [float(s) for s in shells])
        dim, r2, cv, bound, near_ind = independent(shells)
        ratio = bound_is_respected(shells)

        assert abs((fit.slope + 1.0) - dim) < 1e-9, shells
        assert fit.degenerate or abs(fit.r_squared - r2) < 1e-9, shells
        assert abs(fit.shell_cv - cv) < 1e-9, shells
        assert abs(fit.slope_bound - bound) < 1e-9, shells
        assert fit.near_degenerate == near_ind, shells
        assert ratio <= 1.0 + 1e-9, f"BOUND VIOLATED on {shells}: ratio {ratio}"

        ok = fit.near_degenerate == expect
        if not ok:
            failures.append((shells, why))
        print(
            f"{str(shells):<44} {len(shells):3d} {fit.slope + 1.0:8.4f} {fit.r_squared:7.4f} "
            f"{fit.shell_cv:7.4f} {fit.slope_bound:7.4f} {str(fit.near_degenerate):>5} "
            f"{str(expect):>5} {ratio:9.6f}  {'' if ok else '<-- MISMATCH'}  {why}"
        )
    assert not failures, f"family expectations violated: {failures}"
    print(
        "\n  all families behave as expected; slope_bound never exceeded "
        "(max ratio over all rows <= 1.0)"
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 108)
    print("PART 5B: EXHAUSTIVE INTEGER SCANS BY FIT LENGTH")
    print("=" * 108)
    print(f"{'len':>3} {'range':>10} {'total':>9} {'admitted':>9} {'rate':>8} {'worst|dim-1|':>13}")
    for length, hi in ((3, 12), (4, 12), (5, 9), (6, 7)):
        total = admitted = 0
        worst = 0.0
        worst_seq = None
        xs = [float(r) for r in range(1, length + 1)]
        for seq in itertools.product(range(1, hi + 1), repeat=length):
            total += 1
            fit = _log_log_fit(xs, [float(s) for s in seq])
            if fit.near_degenerate:
                admitted += 1
                d = abs(fit.slope)
                if d > worst:
                    worst, worst_seq = d, seq
        print(
            f"{length:3d} {f'[1,{hi}]':>10} {total:9d} {admitted:9d} "
            f"{admitted / total:8.4%} {worst:13.4f}  {worst_seq or ''}"
        )
        if length < NEAR_CONSTANT_MIN_FIT_LENGTH:
            assert admitted == 0, f"length {length} must admit nothing, got {admitted}"
        assert worst <= NEAR_CONSTANT_SLOPE_BOUND + 1e-12, "slope bound contract broken"
    print(f"\n  lengths below {NEAR_CONSTANT_MIN_FIT_LENGTH} admit exactly zero sequences.")
    print(
        f"  admitted |dim-1| never exceeds NEAR_CONSTANT_SLOPE_BOUND = {NEAR_CONSTANT_SLOPE_BOUND}."
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 108)
    print("PART 5C: N1 REGRESSION -- the exactly-constant branch must be untouched")
    print("=" * 108)
    bad = 0
    for shell in range(1, 21):
        for n in range(2, 12):
            fit = _log_log_fit([float(r) for r in range(1, n + 1)], [float(shell)] * n)
            if not (fit.degenerate and not fit.near_degenerate and fit.r_squared == 1.0):
                bad += 1
                print(f"    FAIL shell={shell} n={n}: {fit}")
    print(f"  200 (shell, length) exactly-constant combos: {200 - bad} correct, {bad} wrong")
    assert bad == 0

    # 400-point circle: the case N1 exists for.
    pts = [(math.cos(2 * math.pi * i / 400), math.sin(2 * math.pi * i / 400)) for i in range(400)]
    hg = knn_hypergraph(pts, k=6)
    e = local_dimension(hg, 0, max_radius=6)
    print(
        f"  400-pt circle k=6: dim={e.dimension:.6f} R^2={e.r_squared:.6f} "
        f"degenerate={e.degenerate} near_degenerate={e.near_degenerate} "
        f"genuinely_well_fit={e.is_genuinely_well_fit(0.9)}"
    )
    assert e.degenerate is True and e.near_degenerate is False and e.r_squared == 1.0

    # 30x30 grid: a genuine 2D measurement must NOT be excused by either flag.
    grid = [(float(i), float(j)) for i in range(30) for j in range(30)]
    hgg = knn_hypergraph(grid, k=8)
    centre = 15 * 30 + 15
    g = local_dimension(hgg, centre, max_radius=6)
    print(
        f"  30x30 grid k=8 centre: dim={g.dimension:.4f} R^2={g.r_squared:.4f} "
        f"degenerate={g.degenerate} near_degenerate={g.near_degenerate} "
        f"genuinely_well_fit={g.is_genuinely_well_fit(0.9)}"
    )
    assert g.near_degenerate is False and g.is_genuinely_well_fit(0.9) is True
    assert abs(g.dimension - 2.0) < 0.05

    # An empty-ish graph / short fit path must still work.
    tiny = Hypergraph(((0, 1),))
    t = local_dimension(tiny, 0, max_radius=6)
    print(f"  2-node graph: dim={t.dimension} R^2={t.r_squared} near={t.near_degenerate}")
    assert t.near_degenerate is False

    print("\n  PART 5 RESULT: families, scans and N1 all behave correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
