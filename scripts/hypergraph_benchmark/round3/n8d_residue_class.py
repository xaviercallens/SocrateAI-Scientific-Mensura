"""Round-3 (N8d) step 5: does the graph rule close the round-2 SKEPTIC's residue class?

The round-2 skeptic's headline finding was that the node-local gate's residue was
not one node but a CLASS: 14 of 141 configurations across fresh Brownian seeds,
three torus constructions and problem 05's as-built cloud, at k in {6,8,10,12}
and max_radius in {3..8}, each admitting a near-constant node up to 1.17 away
from the truth.

Those clouds are rebuilt here with that script's own constructions and rerun
under the shipped consensus rule. `admitted` means "accepted by the production
acceptance rule despite failing R^2" -- which is the quantity that matters;
n8c_round3_skeptic_collateral.py counts near-constant CANDIDATES (the node-local
flag), a number the consensus rule deliberately leaves untouched.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from n8d_policy import R2_THRESHOLD, measure_graph  # noqa: E402

from socrates.hypergraph.dimension import near_constant_consensus  # noqa: E402

OMEGA2 = math.sqrt(2.0)


def brownian2d(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return list(map(tuple, np.cumsum(rng.normal(size=(n, 2)), axis=0).tolist()))


def brownian_round2_cloud(target: int = 25_600):
    rng = np.random.default_rng(42)
    path = np.concatenate(
        [np.zeros((1, 2)), np.cumsum(rng.normal(size=(80_000, 2)), axis=0)], axis=0
    )
    return [tuple(map(float, p)) for p in path[:: max(1, len(path) // target)]]


def torus_uniform_4d(n: int, seed: int = 11):
    rng = np.random.default_rng(seed)
    a = rng.random(n) * 2 * math.pi
    b = rng.random(n) * 2 * math.pi
    return list(
        zip(
            np.cos(a).tolist(), np.sin(a).tolist(),
            np.cos(b).tolist(), np.sin(b).tolist(), strict=True,
        )
    )


def torus_quasiperiodic_4d(n: int, t_max: float = 200_000.0):
    t = np.linspace(0.0, t_max, n, endpoint=False)
    return list(
        zip(
            np.cos(t).tolist(), (-np.sin(t)).tolist(),
            np.cos(OMEGA2 * t).tolist(), (-OMEGA2 * np.sin(OMEGA2 * t)).tolist(), strict=True,
        )
    )


def torus_lissajous_2d(n: int, t_max: float = 200_000.0):
    t = np.linspace(0.0, t_max, n, endpoint=False)
    return list(zip(np.cos(t).tolist(), np.cos(OMEGA2 * t).tolist(), strict=True))


def torus_problem05_asbuilt(n: int = 500, t_max: float = 1800.0):
    return [
        (math.cos(t_max * i / n), math.cos(OMEGA2 * t_max * i / n)) for i in range(n)
    ]


FAMILIES = [
    *[(f"Brownian seed={s} n=1500", brownian2d(1500, s), 2.0) for s in range(101, 111)],
    ("Brownian round-2 production n=6400", brownian_round2_cloud()[:6400], 2.0),
    ("torus uniform flat 4D n=2000", torus_uniform_4d(2000), 2.0),
    ("torus quasiperiodic 4D n=2000", torus_quasiperiodic_4d(2000), 2.0),
    ("torus Lissajous 2D n=2000", torus_lissajous_2d(2000), 2.0),
    ("torus problem05 as built n=500", torus_problem05_asbuilt(), 2.0),
]


def main() -> int:
    print("=" * 104)
    print("N8d STEP 5: the round-2 skeptic's residue CLASS, rerun under the shipped rule")
    print("=" * 104)

    rows = 0
    cand_rows = 0
    adm_rows = 0
    total_candidates = 0
    total_admitted = 0
    worst_admitted = 0.0
    print(f"\n{'family':<36}{'k':>3}{'R':>3} {'cand':>5} {'adm':>5} {'consensus':>10} "
          f"{'worst|d-truth|':>14}")
    for label, points, truth in FAMILIES:
        for k in (6, 8, 10, 12):
            for max_radius in (3, 4, 5, 6, 7, 8):
                m = measure_graph(points, k=k, max_radius=max_radius, samples=200)
                if m is None:
                    continue
                rows += 1
                candidates = [e for e in m.estimates if e.near_degenerate]
                fires = near_constant_consensus(list(m.estimates), threshold=R2_THRESHOLD)
                admitted = [
                    e
                    for e in m.estimates
                    if e.r_squared < R2_THRESHOLD
                    and e.is_well_fit(R2_THRESHOLD, near_constant_consensus=fires)
                ]
                total_candidates += len(candidates)
                total_admitted += len(admitted)
                if candidates:
                    cand_rows += 1
                if admitted:
                    adm_rows += 1
                    w = max(abs(e.dimension - truth) for e in admitted)
                    worst_admitted = max(worst_admitted, w)
                if candidates or admitted:
                    w = (
                        max(abs(e.dimension - truth) for e in admitted) if admitted else 0.0
                    )
                    print(f"{label:<36}{k:>3}{max_radius:>3} {len(candidates):>5} "
                          f"{len(admitted):>5} {str(fires):>10} {w:>14.4f}")

    print(f"\n  configurations measured                 : {rows}")
    print(f"  rows with a near-constant CANDIDATE     : {cand_rows}  "
          f"({total_candidates} candidate nodes)")
    print(f"  rows where one was actually ADMITTED    : {adm_rows}  "
          f"({total_admitted} admitted nodes)")
    print(f"  worst |dim - truth| among admitted      : {worst_admitted:.4f}")
    print("\n  (round 2 reported 14 rows / up to 1.17 away from truth, counting candidates;")
    print("   the consensus rule leaves the candidate count alone and gates the ACCEPTANCE.)")
    return 0 if total_admitted == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
