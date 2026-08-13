"""Round-3 independent skeptic, part 2: collateral damage on structures whose
true dimension is NOT ~1, with point clouds built from scratch here.

Deliberately does NOT reuse scripts/hypergraph_benchmark/round3/
n8_collateral_damage.py. Differences that matter:

  * 2D Brownian: 10 independent seeds (the round-1 skeptic used one), plus the
    ACTUAL round-2 problem-06 production cloud (seed 42, 80k steps, strided to
    25,600, evaluated on the same prefixes compare() uses, at that script's own
    K=10 and max_radius=6).
  * 2D torus: the round-1 skeptic's Lissajous sampling was degenerate (dt=0.01
    over n=1500 traces a CURVE, so his own baseline already read ~1.0 and the
    row carried no information). Three genuinely-2D torus clouds are built
    here instead, and each one is REQUIRED to pass a filling check -- the
    R^2-only baseline must itself read >= 1.6 -- before its damage row is
    allowed to count.
  * max_radius sweep extended to 5,6,7,8. Prior rounds swept {3,4,6}. Fit
    windows of length >= 5 are the ONLY place N8b's gate can still fire, so
    that is where an attack has to look.

Run: python scripts/hypergraph_benchmark/round3/n8c_round3_skeptic_collateral.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.dimension import local_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

SAMPLES = 200

# ---------------------------------------------------------------------------
# Point clouds, all built here
# ---------------------------------------------------------------------------


def brownian2d(n: int, seed: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n - 1, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    return [tuple(map(float, p)) for p in path]


def brownian_round2_cloud(target: int = 25_600) -> list[tuple[float, float]]:
    """The literal round-2 problem-06 production cloud."""
    rng = np.random.default_rng(42)
    increments = rng.normal(loc=0.0, scale=1.0, size=(80_000, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(increments, axis=0)], axis=0)
    stride = max(1, len(path) // target)
    return [tuple(map(float, p)) for p in path[::stride]]


def torus_uniform_4d(n: int, seed: int = 11) -> list[tuple[float, ...]]:
    """Genuinely 2-dimensional: a uniform sample of the flat 2-torus embedded
    in R^4 as (cos a, sin a, cos b, sin b). No sampling artifact possible --
    the two angles are drawn independently."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 2 * math.pi, n)
    b = rng.uniform(0.0, 2 * math.pi, n)
    return [(math.cos(a[i]), math.sin(a[i]), math.cos(b[i]), math.sin(b[i])) for i in range(n)]


def torus_quasiperiodic_4d(n: int, t_max: float = 200_000.0) -> list[tuple[float, ...]]:
    """Problem 05's ACTUAL dynamical object: the full phase-space orbit
    (x, vx, y, vy) = (cos t, -sin t, cos(w t), -w sin(w t)) with w = sqrt(2).
    This is a genuine quasi-periodic 2-torus. t_max is large enough (~31,800
    periods) that a Weyl-sampled set of n times is a near-uniform sample of
    the invariant measure rather than a traced curve -- which is exactly the
    failure the round-1 skeptic flagged in his own row."""
    w = math.sqrt(2.0)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    fr = (np.arange(n) * phi) % 1.0
    ts = np.sort(fr) * t_max
    return [(math.cos(t), -math.sin(t), math.cos(w * t), -w * math.sin(w * t)) for t in ts]


def torus_lissajous_2d(n: int, t_max: float = 200_000.0) -> list[tuple[float, float]]:
    """Problem 05 as the benchmark script actually samples it -- the (x, y)
    Lissajous projection -- but over a t_max long enough that the square is
    genuinely filled (the benchmark uses t_max=1800; longer only helps)."""
    w = math.sqrt(2.0)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    fr = (np.arange(n) * phi) % 1.0
    ts = np.sort(fr) * t_max
    return [(math.cos(t), math.cos(w * t)) for t in ts]


def torus_problem05_asbuilt(n: int = 500, t_max: float = 1800.0) -> list[tuple[float, float]]:
    """Byte-for-byte problem 05's own build_point_cloud."""
    w = math.sqrt(2.0)
    ts = [t_max * i / n for i in range(n)]
    return [(math.cos(t), math.cos(w * t)) for t in ts]


# ---------------------------------------------------------------------------
# Damage measurement
# ---------------------------------------------------------------------------


def measure(points, k: int, max_radius: int) -> dict:
    hg = knn_hypergraph(points, k=k, dedupe=True)
    nodes = sorted(hg.nodes)
    step = max(1, len(nodes) // SAMPLES)
    nodes = nodes[::step][:SAMPLES]
    ests = [local_dimension(hg, u, max_radius=max_radius) for u in nodes]

    baseline = [e for e in ests if e.r_squared >= 0.9]  # pre-N8 behaviour
    gated = [e for e in ests if e.is_well_fit(0.9)]  # shipped behaviour
    admitted = [e for e in ests if e.near_degenerate and e.r_squared < 0.9]

    mean_base = sum(e.dimension for e in baseline) / len(baseline) if baseline else float("nan")
    mean_gate = sum(e.dimension for e in gated) / len(gated) if gated else float("nan")
    return {
        "n_sampled": len(ests),
        "admitted": admitted,
        "n_admitted": len(admitted),
        "mean_baseline": mean_base,
        "mean_gated": mean_gate,
        "n_baseline": len(baseline),
        "n_gated": len(gated),
    }


def run_family(label: str, points, truth: float, ks, radii, require_filling: bool) -> list[dict]:
    rows = []
    print(f"\n--- {label}  (n={len(points)}, true dimension {truth}) ---")
    print(
        f"{'k':>3} {'R':>2} {'base_n':>7} {'base_mean':>10} {'gate_n':>7} "
        f"{'gate_mean':>10} {'admit':>6} {'worst|d-truth|':>15} {'delta_mean':>11}"
    )
    for k in ks:
        for mr in radii:
            m = measure(points, k, mr)
            worst = (
                max(abs(e.dimension - truth) for e in m["admitted"])
                if m["admitted"]
                else float("nan")
            )
            delta = (
                abs(m["mean_gated"] - m["mean_baseline"])
                if not (math.isnan(m["mean_gated"]) or math.isnan(m["mean_baseline"]))
                else float("nan")
            )
            flag = ""
            if require_filling and not (m["mean_baseline"] >= 1.6):
                flag = "  <-- NOT FILLING, row does not count"
            print(
                f"{k:3d} {mr:2d} {m['n_baseline']:7d} {m['mean_baseline']:10.4f} "
                f"{m['n_gated']:7d} {m['mean_gated']:10.4f} {m['n_admitted']:6d} "
                f"{worst:15.4f} {delta:11.4f}{flag}"
            )
            rows.append(
                {
                    "label": label,
                    "k": k,
                    "max_radius": mr,
                    "truth": truth,
                    "counts_as_evidence": (not require_filling) or m["mean_baseline"] >= 1.6,
                    **m,
                    "worst": worst,
                    "delta": delta,
                }
            )
    return rows


def main() -> int:
    all_rows: list[dict] = []
    ks = (6, 8, 10, 12)
    radii = (3, 4, 5, 6, 7, 8)

    print("=" * 100)
    print("PART 2: COLLATERAL DAMAGE, clouds built from scratch in this file")
    print("=" * 100)

    # ---- the mandated Brownian case, plus 9 more seeds -------------------
    print("\n### 2D BROWNIAN MOTION (true dimension 2) ###")
    all_rows += run_family("Brownian seed=101 n=1500", brownian2d(1500, 101), 2.0, ks, radii, False)
    print("\n  the specifically mandated configuration, across 10 fresh seeds, k=6 R=3:")
    for seed in range(101, 111):
        m = measure(brownian2d(1500, seed), 6, 3)
        print(
            f"    seed={seed:3d}  admitted={m['n_admitted']:2d}  "
            f"base_mean={m['mean_baseline']:.4f}  gate_mean={m['mean_gated']:.4f}"
        )
        all_rows.append(
            {
                "label": f"Brownian k=6 R=3 seed={seed}",
                "k": 6,
                "max_radius": 3,
                "truth": 2.0,
                "counts_as_evidence": True,
                **m,
                "worst": float("nan"),
                "delta": abs(m["mean_gated"] - m["mean_baseline"]),
            }
        )

    print("\n  the same 10 seeds at the module DEFAULT max_radius=6 (length-6 windows,")
    print("  i.e. the only place the gate can still fire):")
    for seed in range(101, 111):
        m = measure(brownian2d(1500, seed), 6, 6)
        worst = (
            max(abs(e.dimension - 2.0) for e in m["admitted"]) if m["admitted"] else float("nan")
        )
        print(
            f"    seed={seed:3d}  admitted={m['n_admitted']:2d}  worst|d-2|={worst:7.4f}  "
            f"base_mean={m['mean_baseline']:.4f}  gate_mean={m['mean_gated']:.4f}"
        )
        all_rows.append(
            {
                "label": f"Brownian k=6 R=6 seed={seed}",
                "k": 6,
                "max_radius": 6,
                "truth": 2.0,
                "counts_as_evidence": True,
                **m,
                "worst": worst,
                "delta": abs(m["mean_gated"] - m["mean_baseline"]),
            }
        )

    # ---- the actual round-2 problem-06 production cloud -------------------
    print("\n### ROUND-2 PROBLEM 06 PRODUCTION CLOUD (seed 42, 80k steps, K=10, R=6) ###")
    print("  compare() defaults to max_radius=6, so the gate IS live here.")
    r2cloud = brownian_round2_cloud()
    print(f"  cloud: {len(r2cloud)} points")
    for n in (100, 200, 400, 800, 1600, 3200, 6400, 12_800, 25_600):
        if n > len(r2cloud):
            break
        m = measure(r2cloud[:n], 10, 6)
        print(
            f"    n={n:6d}  admitted={m['n_admitted']:2d}  "
            f"base_mean={m['mean_baseline']:.4f}  gate_mean={m['mean_gated']:.4f}  "
            f"delta={abs(m['mean_gated'] - m['mean_baseline']):.4f}"
        )
        all_rows.append(
            {
                "label": f"round2-06 production n={n}",
                "k": 10,
                "max_radius": 6,
                "truth": 2.0,
                "counts_as_evidence": True,
                **m,
                "worst": (
                    max(abs(e.dimension - 2.0) for e in m["admitted"])
                    if m["admitted"]
                    else float("nan")
                ),
                "delta": abs(m["mean_gated"] - m["mean_baseline"]),
            }
        )

    # ---- problem 05: quasiperiodic torus ---------------------------------
    print("\n### PROBLEM 05 QUASIPERIODIC TORUS (true dimension 2) ###")
    print("  The round-1 skeptic could not get a clean read here; his own Lissajous")
    print("  sampling traced a curve. Each cloud below must pass a filling check")
    print("  (R^2-only baseline >= 1.6) before its row is allowed to count.")
    all_rows += run_family(
        "torus: uniform flat 2-torus in R^4, n=2000",
        torus_uniform_4d(2000),
        2.0,
        ks,
        radii,
        True,
    )
    all_rows += run_family(
        "torus: quasiperiodic orbit (x,vx,y,vy), n=2000, t_max=2e5",
        torus_quasiperiodic_4d(2000),
        2.0,
        ks,
        radii,
        True,
    )
    all_rows += run_family(
        "torus: Lissajous (x,y) projection, n=2000, t_max=2e5",
        torus_lissajous_2d(2000),
        2.0,
        ks,
        radii,
        True,
    )
    all_rows += run_family(
        "torus: problem 05 AS BUILT (n=500, t_max=1800)",
        torus_problem05_asbuilt(),
        2.0,
        (6, 8, 10, 12),
        (3, 4, 5, 6),
        True,
    )

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    counted = [r for r in all_rows if r["counts_as_evidence"]]
    ignored = [r for r in all_rows if not r["counts_as_evidence"]]
    print(
        f"  rows measured: {len(all_rows)}  counted: {len(counted)}  "
        f"ignored (cloud not filling): {len(ignored)}"
    )

    mandated = [
        r
        for r in counted
        if r["label"].startswith("Brownian") and r["k"] == 6 and r["max_radius"] == 3
    ]
    print(f"\n  MANDATED CASE -- 2D Brownian, k=6, max_radius=3 ({len(mandated)} configs):")
    print(f"    total near_degenerate admissions: {sum(r['n_admitted'] for r in mandated)}")

    torus3 = [r for r in counted if r["label"].startswith("torus") and r["max_radius"] == 3]
    print(f"\n  MANDATED CASE -- torus, max_radius=3 ({len(torus3)} counted configs):")
    print(f"    total near_degenerate admissions: {sum(r['n_admitted'] for r in torus3)}")

    with_adm = [r for r in counted if r["n_admitted"] > 0]
    print(f"\n  ALL counted rows with ANY admission: {len(with_adm)}")
    for r in sorted(with_adm, key=lambda r: -r["n_admitted"]):
        print(
            f"    {r['label']:<52} k={r['k']:2d} R={r['max_radius']} "
            f"admitted={r['n_admitted']:3d}/{r['n_sampled']} worst|d-truth|={r['worst']:.4f} "
            f"delta_mean={r['delta']:.4f}"
        )
    if not with_adm:
        print("    (none)")

    deltas = [r["delta"] for r in counted if not math.isnan(r["delta"])]
    print(f"\n  worst |mean_gated - mean_baseline| over all counted rows: {max(deltas):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
