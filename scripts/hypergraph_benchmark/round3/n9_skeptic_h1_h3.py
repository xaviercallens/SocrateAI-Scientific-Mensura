"""Independent skeptic verification of H1 (Theiler window) and H3 (criterion c).

Written from the DEFINITIONS, not from the implementations. Nothing here imports
a helper out of baseline.py other than the public entry points under test.

Pre-H1 reference: the module as it exists at git HEAD (the H1 edits are
uncommitted), extracted to a scratch file and imported as a separate module, so
"reproduces prior results" is checked against the actual prior code and not
against the shipped function calling itself.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from socrates.hypergraph.baseline import (  # noqa: E402
    correlation_dimension,
    minimum_points_for_target_accuracy,
    theiler_window_from_autocorrelation,
)


def _load_pre_h1(path: Path):
    spec = importlib.util.spec_from_file_location("baseline_pre_h1", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["baseline_pre_h1"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Clouds
# --------------------------------------------------------------------------
def lorenz(n: int, dt: float = 0.005, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    xyz = np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)
    out = []
    for step in range(n + 5000):
        x, y, z = xyz
        d = np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        xyz = xyz + dt * d
        if step >= 5000:
            out.append(xyz.copy())
    return np.array(out)


def rossler(n: int, dt: float = 0.01, stride: int = 5) -> np.ndarray:
    a, b, c = 0.2, 0.2, 5.7
    xyz = np.array([1.0, 1.0, 1.0])
    out = []
    for step in range(n * stride + 5000):
        x, y, z = xyz
        d = np.array([-y - z, x + a * y, b + z * (x - c)])
        xyz = xyz + dt * d
        if step >= 5000 and (step - 5000) % stride == 0:
            out.append(xyz.copy())
    return np.array(out[:n])


def iid_cube(n: int, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 3))


def bit_reversal_permutation(n: int) -> np.ndarray:
    """The kind of reordering problems 02/10/03/04 apply: index bit-reversal."""
    bits = max(1, int(math.ceil(math.log2(max(n, 2)))))
    perm = []
    for i in range(1 << bits):
        r = int(format(i, f"0{bits}b")[::-1], 2)
        if r < n:
            perm.append(r)
    return np.array(perm[:n], dtype=np.int64)


# --------------------------------------------------------------------------
# Brute-force reference correlation sum, written from the definition
# --------------------------------------------------------------------------
def brute_correlation_sums(pts: np.ndarray, times: np.ndarray, window: int, radii: np.ndarray):
    n = len(pts)
    kept_num = np.zeros(len(radii))
    kept_den = 0
    for i, j in itertools.combinations(range(n), 2):
        if abs(int(times[i]) - int(times[j])) < window:
            continue
        kept_den += 1
        d = float(np.linalg.norm(pts[i] - pts[j]))
        kept_num += (radii >= d).astype(float)
    return kept_num / kept_den, kept_den


def radii_of(pts: np.ndarray, n_radii=20, r_min_frac=0.01, r_max_frac=0.2) -> np.ndarray:
    arr = np.asarray(pts, dtype=float)
    diag = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
    return np.logspace(math.log10(r_min_frac * diag), math.log10(r_max_frac * diag), n_radii)


def slope_from(radii, sums):
    lr, lc = [], []
    for r, c in zip(radii, sums, strict=True):
        if 0.0 < c < 1.0:
            lr.append(math.log(r))
            lc.append(math.log(c))
    if len(lr) < 2:
        return float("nan")
    return float(np.polyfit(lr, lc, 1)[0])


FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


# ==========================================================================
def gate_bitforbit(pre) -> None:
    print("\n=== H1 GATE A: default / W=0 / W=1 reproduce pre-H1 BIT-FOR-BIT ===")
    cases = {
        "lorenz dt=0.005 n=6400": lorenz(6400),
        "rossler stride=5 n=6400": rossler(6400),
        "iid cube n=3200": iid_cube(3200),
        "lorenz n=400 (small-n prefix)": lorenz(400),
    }
    for label, pts in cases.items():
        ref = pre.correlation_dimension(pts)
        got_default = correlation_dimension(pts)
        got_w0 = correlation_dimension(pts, theiler_window=0)
        got_w1 = correlation_dimension(pts, theiler_window=1)
        same = all(
            tuple(g.correlation_sums) == tuple(ref.correlation_sums)
            and g.dimension == ref.dimension
            and g.r_squared == ref.r_squared
            and g.radii == ref.radii
            for g in (got_default, got_w0, got_w1)
        )
        check(
            f"bit-for-bit {label}",
            same,
            f"pre-H1 dim={ref.dimension!r} default={got_default.dimension!r}",
        )
    # also through minimum_points_for_target_accuracy
    pts = lorenz(3200)
    a = pre.minimum_points_for_target_accuracy(pts, 2.05, 0.5)
    b = minimum_points_for_target_accuracy(pts, 2.05, 0.5)
    check("min_points default identical", a == b, f"pre={a} post={b}")


def gate_time_index(pre) -> None:
    print("\n=== H1 GATE B: exclusion keys on TRAJECTORY TIME, not array position ===")
    n = 500
    pts = lorenz(n)
    win = 13
    radii = radii_of(pts)

    # 1. brute force, natural order
    ref_sums, ref_den = brute_correlation_sums(pts, np.arange(n), win, radii)
    est = correlation_dimension(pts, theiler_window=win)
    d = float(np.max(np.abs(np.array(est.correlation_sums) - ref_sums)))
    check("natural order matches brute force exactly", d == 0.0, f"max|dC|={d:.3e}")
    check(
        "excluded pair count matches brute force",
        est.n_excluded_pairs == n * (n - 1) // 2 - ref_den,
        f"impl={est.n_excluded_pairs} brute={n * (n - 1) // 2 - ref_den}",
    )

    # 2. BIT-REVERSAL reordering (the 02/10/03/04 case) with explicit time_indices
    perm = bit_reversal_permutation(n)
    pts_perm = pts[perm]
    est_perm = correlation_dimension(pts_perm, theiler_window=win, time_indices=perm)
    same = tuple(est_perm.correlation_sums) == tuple(est.correlation_sums)
    check(
        "bit-reversal reorder + time_indices == natural order (bit-for-bit)",
        same,
        f"dim {est.dimension!r} vs {est_perm.dimension!r}",
    )
    brute_perm, _ = brute_correlation_sums(pts_perm, perm, win, radii_of(pts_perm))
    dd = float(np.max(np.abs(np.array(est_perm.correlation_sums) - brute_perm)))
    check("bit-reversal case matches brute force exactly", dd == 0.0, f"max|dC|={dd:.3e}")

    # 3. the bug the parameter prevents: omit time_indices on a reordered cloud
    est_bug = correlation_dimension(pts_perm, theiler_window=win)
    check(
        "omitting time_indices on a reordered cloud DOES change the answer",
        tuple(est_bug.correlation_sums) != tuple(est.correlation_sums),
        f"correct={est.dimension:.6f} position-keyed={est_bug.dimension:.6f}",
    )

    # 4. random permutation, unsorted / gapped / negative / duplicate time indices
    rng = np.random.default_rng(7)
    rp = rng.permutation(n)
    e4 = correlation_dimension(pts[rp], theiler_window=win, time_indices=rp)
    check(
        "random permutation + time_indices invariant",
        tuple(e4.correlation_sums) == tuple(est.correlation_sums),
    )

    gap_t = np.arange(n) * 3
    e5 = correlation_dimension(pts, theiler_window=7, time_indices=gap_t)
    b5, _ = brute_correlation_sums(pts, gap_t, 7, radii)
    check("gapped times t=3i, W=7 matches brute force", np.array_equal(e5.correlation_sums, b5))

    neg_t = np.arange(n) - 250
    e6 = correlation_dimension(pts, theiler_window=win, time_indices=neg_t)
    check(
        "negative time indices == shifted contiguous",
        tuple(e6.correlation_sums) == tuple(est.correlation_sums),
    )

    dup_t = np.arange(n) // 2  # every time index appears twice
    e7 = correlation_dimension(pts, theiler_window=5, time_indices=dup_t)
    b7, _ = brute_correlation_sums(pts, dup_t, 5, radii)
    check("duplicate time indices match brute force", np.array_equal(e7.correlation_sums, b7))

    # 5. non-monotone but valid trajectory indices (subsampled + shuffled)
    sub_t = rng.choice(4 * n, size=n, replace=False)
    e8 = correlation_dimension(pts, theiler_window=11, time_indices=sub_t)
    b8, _ = brute_correlation_sums(pts, sub_t, 11, radii)
    check("random sparse time indices match brute force", np.array_equal(e8.correlation_sums, b8))


def gate_control() -> None:
    print("\n=== H1 GATE C: the window must not invent dimension on i.i.d. data ===")
    pts = iid_cube(3200, seed=11)
    w = theiler_window_from_autocorrelation(pts)
    d0 = correlation_dimension(pts, theiler_window=0).dimension
    da = correlation_dimension(pts, theiler_window="auto").dimension
    check("auto window is 1 on i.i.d. data", w == 1, f"W={w}")
    check("auto == W=0 on i.i.d. data", d0 == da, f"{d0:.6f} vs {da:.6f}")
    sweep = {
        w2: correlation_dimension(pts, theiler_window=w2).dimension for w2 in (5, 20, 100, 300)
    }
    spread = max(sweep.values()) - min(sweep.values())
    check(
        "fixed windows do not inflate i.i.d. estimate",
        spread < 0.05,
        f"{ {k: round(v, 4) for k, v in sweep.items()} } spread={spread:.4f}",
    )


def gate_chunking() -> None:
    print("\n=== H1 GATE D: chunked pair enumeration is exact (stress the chunk boundary) ===")
    import socrates.hypergraph.baseline as bl

    old = bl._PAIR_CHUNK
    pts = lorenz(300)
    radii = radii_of(pts)
    ref, _ = brute_correlation_sums(pts, np.arange(300), 25, radii)
    ok = True
    detail = []
    for chunk in (1, 2, 7, 31, 999, 10**9):
        bl._PAIR_CHUNK = chunk
        got = correlation_dimension(pts, theiler_window=25)
        same = np.array_equal(np.array(got.correlation_sums), ref)
        ok = ok and same
        detail.append(f"chunk={chunk}:{'ok' if same else 'MISMATCH'}")
    bl._PAIR_CHUNK = old
    check("chunk size does not change the answer", ok, " ".join(detail))


def gate_edge() -> None:
    print("\n=== H1 GATE E: adversarial / edge inputs ===")
    pts = lorenz(200)
    try:
        correlation_dimension(pts, theiler_window=10_000)
        check("window excluding all pairs raises", False, "no exception")
    except ValueError as e:
        check("window excluding all pairs raises ValueError", True, str(e)[:70])
    for bad in (-1, 2.5, True, "sometimes"):
        try:
            correlation_dimension(pts, theiler_window=bad)
            check(f"rejects theiler_window={bad!r}", False, "accepted")
        except ValueError:
            check(f"rejects theiler_window={bad!r}", True)
    try:
        correlation_dimension(pts, theiler_window=5, time_indices=np.arange(199))
        check("rejects wrong-length time_indices", False, "accepted")
    except ValueError:
        check("rejects wrong-length time_indices", True)
    # monotonic non-negativity of correlation sums under a window
    e = correlation_dimension(pts, theiler_window=17)
    cs = np.array(e.correlation_sums)
    check(
        "correlation sums non-negative and non-decreasing",
        bool((cs >= 0).all() and (np.diff(cs) >= -1e-15).all()),
    )
    check("correlation sums <= 1", bool((cs <= 1.0 + 1e-12).all()), f"max={cs.max():.6f}")


def gate_h3() -> None:
    print("\n=== H3: criterion (c) known-answer behaviour, re-derived here ===")
    from socrates.hypergraph.comparison import compare_accuracy_at_max_n

    rng = np.random.default_rng(42)
    # WIN case: planar Brownian motion, true D2 = 2 (Taylor 1953)
    steps = rng.standard_normal((1600, 2))
    brownian = [tuple(p) for p in np.cumsum(steps, axis=0)]
    r = compare_accuracy_at_max_n(brownian, 2.0, 1600, k=10, tolerance=0.15)
    check(
        "WIN: brownian poly wins criterion (c)",
        r.poly_algebraic_accuracy_win and not r.traditional_accuracy_win,
        f"poly={r.poly_algebraic_estimate:.4f} trad={r.traditional_estimate:.4f} "
        f"sentinel={r.poly_algebraic_sentinel_fraction:.3f}",
    )
    # LOSS case: uniform 3-cube
    cube = [tuple(p) for p in rng.random((400, 3))]
    r2 = compare_accuracy_at_max_n(cube, 3.0, 400, k=12, tolerance=0.15)
    # The load-bearing assertion is that poly does NOT win here (20/20 draws in
    # n9_skeptic_h3_seeds.py). The mirror flag `traditional_accuracy_win` is only
    # 11/20 across seeds -- on the other 9 the baseline's own error also exceeds
    # the 0.15 tolerance, so (c4) mirrored refuses and nobody wins. That is the
    # criterion behaving correctly, but it means the shipped test's
    # `traditional_accuracy_win is True` assertion is a property of seed 1, not
    # of the 3-cube. Reported, not asserted.
    check(
        "LOSS: poly does not win on a uniform 3-cube",
        not r2.poly_algebraic_accuracy_win and r2.more_accurate == "traditional",
        f"poly={r2.poly_algebraic_estimate:.4f} trad={r2.traditional_estimate:.4f} "
        f"trad_win={r2.traditional_accuracy_win} (seed-dependent, informational)",
    )
    # TIE case: uniform square
    sq = [tuple(p) for p in rng.random((800, 2))]
    r3 = compare_accuracy_at_max_n(sq, 2.0, 800, k=10, tolerance=0.15, margin=0.2)
    check(
        "TIE: uniform square is a tie, neither wins",
        r3.more_accurate == "tie"
        and not r3.poly_algebraic_accuracy_win
        and not r3.traditional_accuracy_win,
        f"poly={r3.poly_algebraic_estimate:.4f} trad={r3.traditional_estimate:.4f} "
        f"more_accurate={r3.more_accurate}",
    )
    # DEGENERATE case: exact circle -> circulant ring lattice, poly pinned at 1.0
    th = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    circle = [(float(math.cos(t)), float(math.sin(t))) for t in th]
    r4 = compare_accuracy_at_max_n(circle, 1.0, 400, k=6, tolerance=0.05, margin=0.05)
    conditions_else_pass = (
        math.isfinite(r4.poly_algebraic_abs_error)
        and math.isfinite(r4.traditional_abs_error)
        and r4.poly_algebraic_abs_error <= r4.tolerance
        and r4.traditional_abs_error > r4.tolerance
        and (r4.traditional_abs_error - r4.poly_algebraic_abs_error) >= r4.margin
    )
    check(
        "DEGENERATE: circle is refused, and ONLY (c2) refuses it",
        (not r4.poly_algebraic_accuracy_win)
        and conditions_else_pass
        and r4.poly_algebraic_degenerate_fraction > 0.5,
        f"poly={r4.poly_algebraic_estimate:.4f} err={r4.poly_algebraic_abs_error:.4f} "
        f"trad_err={r4.traditional_abs_error:.4f} "
        f"degen={r4.poly_algebraic_degenerate_fraction:.3f} "
        f"c1/c3/c4/c5_all_pass={conditions_else_pass} reason={r4.verdict_reason[:60]}",
    )
    check(
        "DEGENERATE: more_accurate still ranks poly ahead (guard is separate)",
        r4.more_accurate == "poly_algebraic",
        f"more_accurate={r4.more_accurate}",
    )
    # anti-gaming: no tolerance manufactures a win on a sub-margin gap
    wins = [
        t / 100
        for t in range(0, 101)
        if compare_accuracy_at_max_n(
            brownian, 2.0, 1600, k=10, tolerance=t / 100
        ).poly_algebraic_accuracy_win
    ]
    lo, hi = r.poly_algebraic_abs_error, r.traditional_abs_error
    check(
        "anti-gaming: winning tolerances are exactly [poly_err, trad_err)",
        len(wins) > 0 and min(wins) >= lo - 0.011 and max(wins) < hi,
        f"tolerance win-set [{min(wins):.2f},{max(wins):.2f}] vs errors [{lo:.4f},{hi:.4f})",
    )
    nowin = [
        t / 100
        for t in range(0, 101)
        if compare_accuracy_at_max_n(
            brownian, 2.0, 1600, k=10, tolerance=t / 100, margin=hi - lo + 0.01
        ).poly_algebraic_accuracy_win
    ]
    check("anti-gaming: margin above the gap kills every tolerance", nowin == [], f"{nowin}")


def main() -> int:
    pre_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if pre_path is None:
        raise SystemExit("usage: n9_skeptic_h1_h3.py <path-to-pre-H1-baseline.py>")
    pre = _load_pre_h1(pre_path)
    gate_bitforbit(pre)
    gate_time_index(pre)
    gate_control()
    gate_chunking()
    gate_edge()
    gate_h3()
    print("\n=== SUMMARY ===")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all skeptic gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
