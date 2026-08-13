"""Round-3 (N8d) step 2: score candidate graph-level consensus rules on min_n.

Evaluates every candidate against the number that is actually recorded in the
benchmark -- `poly_algebraic_min_n` -- for ALL TEN round-2 problems at their
production k / max_radius / n_grid / tolerance, against the pre-N8 baseline
(R^2 alone) and against the currently shipped node-local gate.

Candidate rules (all evaluated on identical per-node estimates):

  pre_n8        r_squared >= 0.9 only. The baseline every recorded round-2
                number was produced under.
  n8b           shipped: r_squared >= 0.9 OR near_degenerate (node-local, with
                the min-fit-length gate).
  n8d(tol,frac) proposed: the near-constant branch may contribute only if the
                GRAPH agrees it is ~1D --
                  * if any node has r_squared >= 0.9, require the mean of those
                    confident dimensions to be within `tol` of 1;
                  * if none does, the branch is the only evidence there is, so
                    require near_degenerate_fraction >= `frac`.
  n8d_nolen     the same rule with the per-node min-fit-length gate REMOVED, to
                test whether the graph-level decision can replace it outright.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_clouds import SPECS, anchor_cloud, load_cloud  # noqa: E402
from n8d_policy import (  # noqa: E402
    R2_THRESHOLD,
    measure_graph,
    measure_grid,
    near_degenerate_ignoring_length,
)

ONE_D = {"01", "02", "03", "04", "07", "10"}


def consensus_fires(estimates, *, tol: float, frac: float, use_length_gate: bool) -> bool:
    """The proposed graph-level decision, made once per graph."""
    def is_near(e):
        return e.near_degenerate if use_length_gate else near_degenerate_ignoring_length(e)

    confident = [e for e in estimates if e.r_squared >= R2_THRESHOLD]
    if confident:
        consensus_dim = sum(e.dimension for e in confident) / len(confident)
        return abs(consensus_dim - 1.0) <= tol
    if not estimates:
        return False
    near_frac = sum(1 for e in estimates if is_near(e)) / len(estimates)
    return near_frac >= frac


def graph_mean(estimates, *, policy: str, tol: float = 0.25, frac: float = 0.1) -> float:
    def is_near(e):
        return e.near_degenerate

    if policy == "pre_n8":
        kept = [e for e in estimates if e.r_squared >= R2_THRESHOLD]
    elif policy == "n8b":
        kept = [e for e in estimates if e.r_squared >= R2_THRESHOLD or e.near_degenerate]
    elif policy in ("n8d", "n8d_nolen"):
        use_len = policy == "n8d"
        if not use_len:
            def is_near(e):  # noqa: F811
                return near_degenerate_ignoring_length(e)
        fires = consensus_fires(estimates, tol=tol, frac=frac, use_length_gate=use_len)
        kept = [
            e for e in estimates if e.r_squared >= R2_THRESHOLD or (fires and is_near(e))
        ]
    else:
        raise ValueError(policy)
    if not kept:
        return float("nan")
    return sum(e.dimension for e in kept) / len(kept)


def min_n_for(measurements, true_dim, tolerance, **kw) -> int | None:
    dims = [(m.n, graph_mean(m.estimates, **kw)) for m in measurements]
    for i, (n, dim) in enumerate(dims):
        if not math.isfinite(dim) or abs(dim - true_dim) > tolerance:
            continue
        if all(math.isfinite(d) and abs(d - true_dim) <= tolerance for _, d in dims[i:]):
            return n
    return None


def main() -> None:
    candidates = [
        ("pre_n8", {"policy": "pre_n8"}),
        ("n8b(shipped)", {"policy": "n8b"}),
        ("n8d t.25 f.10", {"policy": "n8d", "tol": 0.25, "frac": 0.10}),
        ("n8d t.25 f.50", {"policy": "n8d", "tol": 0.25, "frac": 0.50}),
        ("n8d t.15 f.10", {"policy": "n8d", "tol": 0.15, "frac": 0.10}),
        ("n8d t.50 f.10", {"policy": "n8d", "tol": 0.50, "frac": 0.10}),
        ("nolen t.25 f.10", {"policy": "n8d_nolen", "tol": 0.25, "frac": 0.10}),
    ]

    print("=" * 118)
    print("N8d STEP 2: poly_algebraic_min_n under each candidate, ALL TEN round-2 problems")
    print("=" * 118)
    header = " ".join(f"{name:>15}" for name, _ in candidates)
    print(f"\n{'prob':>4} {'true':>5} {'tol':>5} " + header)

    all_rows = {}
    for problem, spec in SPECS.items():
        points = load_cloud(problem)
        rows = measure_grid(points, spec)
        all_rows[problem] = rows
        cells = []
        for _name, kw in candidates:
            v = min_n_for(rows, spec.true_dimension, spec.tolerance, **kw)
            cells.append(f"{str(v):>15}")
        print(f"{problem:>4} {spec.true_dimension:>5} {spec.tolerance:>5} " + " ".join(cells))

    print("\n" + "=" * 118)
    print("PER-n MEANS where any candidate differs from pre_n8")
    print("=" * 118)
    for problem, spec in SPECS.items():
        for m in all_rows[problem]:
            vals = {name: graph_mean(m.estimates, **kw) for name, kw in candidates}
            base = vals["pre_n8"]
            differs = any(
                not (math.isnan(v) and math.isnan(base))
                and abs(
                    (v if not math.isnan(v) else -99)
                    - (base if not math.isnan(base) else -99)
                )
                > 1e-12
                for v in vals.values()
            )
            if differs:
                cells = " ".join(f"{name}={v:.4f}" for name, v in vals.items())
                print(f"  p{problem} n={m.n:>6} truth={spec.true_dimension}: {cells}")

    print("\n" + "=" * 118)
    print("ANCHOR (problem 01 cloud, n=200, k=6, max_radius=6) under each candidate")
    print("=" * 118)
    anchor = measure_graph(anchor_cloud(), k=6, max_radius=6, samples=200)
    assert anchor is not None
    for name, kw in candidates:
        kept = [
            e
            for e in anchor.estimates
            if e.r_squared >= R2_THRESHOLD
            or (
                kw["policy"] == "n8b" and e.near_degenerate
            )
            or (
                kw["policy"] in ("n8d", "n8d_nolen")
                and consensus_fires(
                    anchor.estimates,
                    tol=kw["tol"],
                    frac=kw["frac"],
                    use_length_gate=kw["policy"] == "n8d",
                )
                and (
                    e.near_degenerate
                    if kw["policy"] == "n8d"
                    else near_degenerate_ignoring_length(e)
                )
            )
        ]
        mean = sum(e.dimension for e in kept) / len(kept) if kept else float("nan")
        print(f"  {name:>15}: kept {len(kept):>3}/200  mean={mean:.6f}")


if __name__ == "__main__":
    main()
