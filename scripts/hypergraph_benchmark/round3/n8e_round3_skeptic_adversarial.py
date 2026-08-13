"""Round-3 skeptic: adversarial attack on the N8d graph-level consensus rule.

Three things this tries to break, on point clouds the repair agent never used:

  A. THE STRUCTURAL CLAIM. If the confident branch decides the verdict, every
     pooled dimension is a convex combination of values in [1 - T, 1 + T] with
     T = NEAR_CONSTANT_CONSENSUS_TOLERANCE = 0.25 (the confident MEAN is in that
     band by the rule itself; every near-constant node is in it by
     `slope_bound <= NEAR_CONSTANT_SLOPE_BOUND = 0.25`). So both the pre-N8 and
     the shipped mean land in [0.75, 1.25], and no accept/reject verdict whose
     tolerance band misses that interval can move. If that reasoning is wrong,
     some cloud here should produce a shipped mean outside [0.75, 1.25] while
     differing from the pre-N8 mean.

  B. THE RESIDUE CLASS. The round-2 skeptic found 14 of 141 configurations where
     the node-local gate admitted a spurious ~1D node into a 2D structure. This
     re-runs that hunt on FRESH clouds (new Brownian seeds, filled square/disc/
     cube, Lorenz, Rossler, flat torus, and lattices) over a wider k x max_radius
     grid, and asks whether the graph verdict actually refuses them.

  C. THE FALLBACK BRANCH. The dangerous hole would be a genuinely NON-1D graph
     with no confident node and a near-constant MAJORITY, which the fallback
     would then endorse. This searches for one.

Run: python n8e_round3_skeptic_adversarial.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CONSENSUS_TOLERANCE,
    NEAR_CONSTANT_SLOPE_BOUND,
    local_dimension,
    mean_dimension,
    near_constant_consensus,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

BAND_LO = 1.0 - NEAR_CONSTANT_CONSENSUS_TOLERANCE
BAND_HI = 1.0 + NEAR_CONSTANT_CONSENSUS_TOLERANCE


# --------------------------------------------------------------------------
# Clouds -- none of these are the repair agent's.
# --------------------------------------------------------------------------


def brownian(seed, n=1200, dim=2):
    rng = np.random.default_rng(seed)
    path = np.cumsum(rng.normal(size=(n, dim)), axis=0)
    return [tuple(p) for p in path], float(dim)


def uniform_square(seed, n=1200):
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(size=(n, 2))], 2.0


def uniform_disc(seed, n=1200):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(size=n))
    th = rng.uniform(0, 2 * math.pi, size=n)
    return [(float(r[i] * math.cos(th[i])), float(r[i] * math.sin(th[i]))) for i in range(n)], 2.0


def uniform_cube(seed, n=1200):
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(size=(n, 3))], 3.0


def flat_torus(seed, n=1200):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * math.pi, size=n)
    b = rng.uniform(0, 2 * math.pi, size=n)
    return [
        (math.cos(a[i]), math.sin(a[i]), math.cos(b[i]), math.sin(b[i])) for i in range(n)
    ], 2.0


def square_lattice(side=32):
    return [(float(i), float(j)) for i in range(side) for j in range(side)], 2.0


def jittered_lattice(seed, side=32, amp=0.15):
    rng = np.random.default_rng(seed)
    return [
        (i + float(rng.normal(0, amp)), j + float(rng.normal(0, amp)))
        for i in range(side)
        for j in range(side)
    ], 2.0


def _rk4(deriv, s, dt, n):
    out = np.empty((n, len(s)))
    s = np.asarray(s, dtype=float)
    for i in range(n):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = s
    return out


def lorenz(n=1500):
    def d(s):
        x, y, z = s
        return np.array([10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z])

    traj = _rk4(d, [1.0, 1.0, 1.0], 0.01, 20_000 + n * 3)
    traj = traj[20_000 :: 3][:n]
    return [tuple(p) for p in traj], 2.05


def rossler(n=1500):
    def d(s):
        x, y, z = s
        return np.array([-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)])

    traj = _rk4(d, [1.0, 1.0, 1.0], 0.02, 20_000 + n * 3)
    traj = traj[20_000 :: 3][:n]
    return [tuple(p) for p in traj], 2.01


CLOUDS = []
for s in range(201, 211):
    CLOUDS.append((f"brownian2d s={s}", *brownian(s)))
CLOUDS += [
    ("brownian3d s=301", *brownian(301, n=1200, dim=3)),
    ("uniform square s=401", *uniform_square(401)),
    ("uniform disc s=402", *uniform_disc(402)),
    ("uniform cube s=403", *uniform_cube(403)),
    ("flat torus s=404", *flat_torus(404)),
    ("square lattice 40x40", *square_lattice()),
    ("jittered lattice s=405", *jittered_lattice(405)),
    ("lorenz", *lorenz()),
    ("rossler", *rossler()),
]

K_VALUES = (6, 8, 10, 12)
RADII = (3, 4, 5, 6, 7, 8)


def analyse(points, k, max_radius, samples=120):
    hg = knn_hypergraph(points, k=k, dedupe=True)
    nodes = sorted(hg.nodes)
    if samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    ests = [local_dimension(hg, n, max_radius=max_radius) for n in nodes]
    cons = near_constant_consensus(ests, threshold=0.9)
    pre = [e.dimension for e in ests if e.r_squared >= 0.9]
    nl = [e.dimension for e in ests if e.r_squared >= 0.9 or e.near_degenerate]
    kept = nl if cons else pre
    admitted = [e for e in ests if e.near_degenerate] if cons else []
    return {
        "ests": ests,
        "consensus": cons,
        "n_conf": len(pre),
        "conf_mean": (sum(pre) / len(pre)) if pre else float("nan"),
        "nd_frac": sum(1 for e in ests if e.near_degenerate) / len(ests),
        "pre_mean": (sum(pre) / len(pre)) if pre else float("nan"),
        "shipped_mean": (sum(kept) / len(kept)) if kept else float("nan"),
        "nl_mean": (sum(nl) / len(nl)) if nl else float("nan"),
        "admitted": admitted,
        "hg": hg,
    }


def main() -> None:
    print("=== ADVERSARIAL SWEEP: non-1D clouds the repair agent never used ===")
    print(f"clouds={len(CLOUDS)}  k in {K_VALUES}  max_radius in {RADII}  "
          f"=> {len(CLOUDS) * len(K_VALUES) * len(RADII)} configurations, 120 nodes each\n")

    rows = 0
    cand_rows = 0
    cand_nodes = 0
    admitted_rows = 0
    admitted_nodes = 0
    worst_admitted_err = 0.0
    worst_bound_violation = 0.0
    band_violations = []
    fallback_fires = []
    changed_rows = []
    prod_mismatch = 0

    for name, pts, truth in CLOUDS:
        for k in K_VALUES:
            for radius in RADII:
                a = analyse(pts, k, radius)
                rows += 1
                nd = [e for e in a["ests"] if e.near_degenerate]
                if nd:
                    cand_rows += 1
                    cand_nodes += len(nd)
                # Invariant the whole branch rests on: |dim - 1| <= slope_bound
                # <= NEAR_CONSTANT_SLOPE_BOUND for every near-constant node.
                for e in nd:
                    worst_bound_violation = max(
                        worst_bound_violation, abs(e.dimension - 1.0) - e.slope_bound
                    )
                    assert e.slope_bound <= NEAR_CONSTANT_SLOPE_BOUND + 1e-12
                if a["admitted"]:
                    admitted_rows += 1
                    admitted_nodes += len(a["admitted"])
                    worst_admitted_err = max(
                        worst_admitted_err, max(abs(e.dimension - truth) for e in a["admitted"])
                    )
                # Claim A: whenever the shipped verdict differs from pre-N8,
                # both means must sit inside [0.75, 1.25].
                if not (
                    math.isnan(a["pre_mean"]) and math.isnan(a["shipped_mean"])
                ) and not math.isclose(
                    a["pre_mean"], a["shipped_mean"], rel_tol=0, abs_tol=1e-12
                ):
                    changed_rows.append(
                        (name, k, radius, a["pre_mean"], a["shipped_mean"], truth)
                    )
                    if math.isfinite(a["shipped_mean"]) and not (
                        BAND_LO - 1e-9 <= a["shipped_mean"] <= BAND_HI + 1e-9
                    ):
                        band_violations.append((name, k, radius, a["shipped_mean"]))
                # Claim C: fallback branch firing on a non-1D structure.
                if a["consensus"] and a["n_conf"] == 0:
                    fallback_fires.append(
                        (name, k, radius, a["nd_frac"], truth, a["shipped_mean"])
                    )
                # production mean_dimension must agree with my recomputation
                # (spot-checked on every 6th configuration -- it is a second
                # full pass over the graph and doubles the sweep's cost)
                if rows % 6:
                    continue
                md = mean_dimension(a["hg"], samples=120, max_radius=radius)
                if not (
                    (math.isnan(md) and math.isnan(a["shipped_mean"]))
                    or abs(md - a["shipped_mean"]) < 1e-12
                ):
                    prod_mismatch += 1
                    print(f"  !! production mismatch {name} k={k} R={radius}: "
                          f"{md} vs {a['shipped_mean']}")

    print(f"configurations measured                       : {rows}")
    print(f"rows with a node-local near-constant CANDIDATE : {cand_rows}  ({cand_nodes} nodes)")
    print(f"rows where the graph ADMITTED one              : {admitted_rows}  "
          f"({admitted_nodes} nodes)")
    print(f"worst |dim - truth| among admitted             : {worst_admitted_err:.4f}")
    print(f"worst violation of |dim-1| <= slope_bound      : {worst_bound_violation:.3e} "
          f"(must be <= 0)")
    print(f"production mean_dimension mismatches           : {prod_mismatch}")
    print()
    print(f"[A] rows where shipped mean != pre-N8 mean     : {len(changed_rows)}")
    print(f"    of those, shipped mean outside [{BAND_LO}, {BAND_HI}] : {len(band_violations)}")
    for r in band_violations[:10]:
        print(f"      VIOLATION {r}")
    for r in changed_rows[:12]:
        print(f"      changed: {r[0]} k={r[1]} R={r[2]} pre={r[3]:.4f} "
              f"shipped={r[4]:.4f} truth={r[5]}")
    print()
    print(f"[C] fallback-branch (no confident node) firings on these non-1D clouds: "
          f"{len(fallback_fires)}")
    for r in fallback_fires[:10]:
        print(f"      {r[0]} k={r[1]} R={r[2]} nd_frac={r[3]:.3f} truth={r[4]} shipped={r[5]:.4f}")

    assert worst_bound_violation <= 0.0, "slope_bound is NOT a bound on |dimension - 1|"
    assert not band_violations, "structural claim A is FALSE"
    assert prod_mismatch == 0
    print("\nADVERSARIAL SWEEP COMPLETE")


if __name__ == "__main__":
    main()
