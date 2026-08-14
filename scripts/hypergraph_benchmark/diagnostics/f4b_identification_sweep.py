"""F4, step B: which theoretical quantity does each estimator actually track?

Runs BOTH estimators on the SAME i.i.d. point clouds drawn from exactly-known
multifractal measures, over a grid of configurations, and scores each result
against the three candidate answers D_0 (box/support), D_1 (information) and
D_2 (correlation).

Two extra reference estimators are computed on the identical clouds so the
identification does not rest on the two implementations alone. Both count
points in EUCLIDEAN balls of radius eps around each sample point; they differ
only in how they average, and that difference is exactly the D_1 / D_2
distinction:

    D_1 reference:  slope of  mean_i log #B(x_i, eps)  vs log eps
    D_2 reference:  slope of  log mean_i #B(x_i, eps)  vs log eps

(the second is the correlation integral, since mean_i mu(B(x_i,eps)) = C(eps)).
Averaging the LOG of a ball count gives the information dimension; taking the
LOG of the AVERAGE gives the correlation dimension. Both are measured here, so
"the shell estimator averages per-node log-slopes, therefore it must be D_1" is
tested rather than asserted.

Configuration grid: k in {8, 10, 15, 25} x max_radius in {4, 6, 8} x
n in {1600, 6400, 12800}. No single configuration carries any conclusion.

AGGREGATION. `mean_dimension` is not called directly, because it re-runs the
whole BFS for each of the three aggregates the benchmark reports and that made
the sweep four times more expensive than it needs to be. Instead
`local_dimension` is called once per (node, max_radius) and the library's OWN
`near_constant_consensus` / `is_well_fit` are applied to the resulting
estimates -- the same code path `mean_dimension` uses, at the same thresholds
(0.9). `--selfcheck` asserts this reproduces `mean_dimension` exactly on a
spread of configurations before any sweep is trusted.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from f4_multifractal_targets import ALL_TARGETS, BY_NAME  # noqa: E402

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    local_dimension,
    mean_dimension,
    near_constant_consensus,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

SEED = 20260814
N_GRID = (1600, 6400, 12800)
K_GRID = (8, 10, 15, 25)
RADIUS_GRID = (4, 6, 8)
SAMPLES = 80
OUT = Path(__file__).resolve().parent / "f4b_results.json"


def sampled_nodes(hg, samples: int) -> list[int]:
    """Byte-for-byte the node subset `dimension._sampled_nodes` picks."""
    nodes = sorted(hg.nodes)
    if samples is not None and samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    return nodes


def aggregate(estimates) -> dict:
    """`mean_dimension`'s aggregation, applied to estimates already computed."""
    consensus = near_constant_consensus(estimates, threshold=0.9)
    well_fit = [
        e.dimension
        for e in estimates
        if e.is_well_fit(threshold=0.9, near_constant_consensus=consensus)
    ]
    return {
        "shell": (sum(well_fit) / len(well_fit)) if well_fit else float("nan"),
        "shell_unfiltered": float(np.mean([e.dimension for e in estimates])),
        "n_well_fit": len(well_fit),
        "n_sampled": len(estimates),
        "degen_frac": sum(1 for e in estimates if e.degenerate) / len(estimates),
        "near_degen_frac": sum(1 for e in estimates if e.near_degenerate) / len(estimates),
    }


def euclidean_ball_dimensions(pts: np.ndarray, *, n_radii: int = 24) -> tuple[float, float]:
    """(D_1 reference, D_2 reference) by counting points in Euclidean balls."""
    tree = cKDTree(pts)
    diag = float(np.sqrt(np.sum((pts.max(axis=0) - pts.min(axis=0)) ** 2)))
    radii = np.logspace(math.log10(0.01 * diag), math.log10(0.2 * diag), n_radii)
    counts = np.array([tree.query_ball_point(pts, r, return_length=True) for r in radii])
    counts = counts - 1  # drop the self-match
    log_r = np.log(radii)
    ok = counts.min(axis=1) > 0
    d1 = float(np.polyfit(log_r[ok], np.mean(np.log(counts[ok]), axis=1), 1)[0])
    positive = counts.mean(axis=1) > 0
    d2 = float(np.polyfit(log_r[positive], np.log(counts[positive].mean(axis=1)), 1)[0])
    return d1, d2


def selfcheck() -> None:
    """`aggregate` must reproduce `mean_dimension` exactly, or the sweep is void."""
    checks = [
        ("SQ-unif", 1600, 10, 6),
        ("SQ-strong", 1600, 15, 4),
        ("GA-mild", 3200, 8, 8),
        ("GA-strong", 3200, 25, 6),
        ("CD-unif", 1600, 15, 6),
        ("ST-strong", 3200, 15, 6),
    ]
    for name, n, k, mr in checks:
        t = BY_NAME[name]
        hg = knn_hypergraph(
            list(map(tuple, t.sample(n, SEED))), k, dedupe=True, duplicate_tolerance=1e-13
        )
        ests = [local_dimension(hg, s, max_radius=mr) for s in sampled_nodes(hg, SAMPLES)]
        mine = aggregate(ests)["shell"]
        theirs = mean_dimension(hg, samples=SAMPLES, max_radius=mr)
        same = (math.isnan(mine) and math.isnan(theirs)) or mine == theirs
        print(f"  selfcheck {name:<11} n={n:<6} k={k:<3} r={mr}: {mine!r} vs {theirs!r}  {same}")
        assert same, f"aggregation diverges from mean_dimension on {name}"
    print("  aggregation reproduces mean_dimension exactly on all checks\n")


def main() -> None:
    print("Self-check of the aggregation against the library's own mean_dimension:")
    selfcheck()
    rows = []
    for t in ALL_TARGETS:
        for n in N_GRID:
            pts = t.sample(n, SEED)
            tup = list(map(tuple, pts))
            gp = correlation_dimension(tup, n_radii=30)
            d1_ref, d2_ref = euclidean_ball_dimensions(pts)
            for k in K_GRID:
                hg = knn_hypergraph(tup, k, dedupe=True, duplicate_tolerance=1e-13)
                picked = sampled_nodes(hg, SAMPLES)
                for mr in RADIUS_GRID:
                    agg = aggregate([local_dimension(hg, s, max_radius=mr) for s in picked])
                    rows.append(
                        {
                            "target": t.name,
                            "support": t.support,
                            "D0": t.D0,
                            "D1": t.D1,
                            "D2": t.D2,
                            "n": n,
                            "k": k,
                            "max_radius": mr,
                            "gp": gp.dimension,
                            "gp_r2": gp.r_squared,
                            "d1_ref": d1_ref,
                            "d2_ref": d2_ref,
                            **agg,
                        }
                    )
            print(f"  done {t.name} n={n}", flush=True)
            OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
