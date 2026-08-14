"""F4, step C: WHY the shell estimator lands where it lands.

The identification sweep (f4b) says *which* number the shell estimator returns.
This script decomposes that number, so the answer is a mechanism rather than a
coincidence.

Around each sampled source node, for graph-hop radius h = 1..H, record the ball
size V(h) and the ball's Euclidean extent R(h). Then per node

    a = d log V / d log h      what a ball-growth estimator reads off the graph
    b = d log V / d log R      the AMBIENT-metric ball-count exponent
    c = d log R / d log h      the graph-metric distortion, dimensionless

The two factors have completely different status:

  * b is a real dimension-type quantity. For a self-similar measure the
    pointwise dimension alpha(x) = lim log mu(B(x,eps)) / log eps equals D_1 for
    mu-almost every x, so averaging b over nodes drawn FROM the measure targets
    D_1 -- the information dimension. (f4b measures the same thing far more
    stably with a KD-tree at fixed Euclidean radii; this graph-side version is
    here only to complete the factorisation.)
  * c is not a dimension at all. It measures how far one graph hop travels in
    space, which on a k-NN graph is set by the local SAMPLE DENSITY: hops are
    short where points are dense and long where they are sparse. For a uniform
    sample of a d-manifold c -> 1 and the distortion vanishes, which is why
    every integer-dimension calibration case in the repo (circle, filled
    square, filled cube) hides it.

If c is measurably > 1 on a multifractal then the shell estimate is not D_1, is
not D_2, and cannot be made into either by more sampling: it is a graph
statistic multiplied by a density-dependent factor.

NUMERICAL GUARD (a real defect in this script's first version). Where the
measure is strongly concentrated, a ball can spend all H hops inside one dense
cluster, so log R barely moves and the per-node fit of log V against log R is
ill-conditioned -- the first version reported b = 6490 on GA-strong from
exactly this. Nodes whose log R range is below MIN_LOG_R_RANGE are therefore
excluded from b and c and counted separately as "decoupled": on those nodes the
graph metric has come loose from the ambient metric altogether, which is
itself part of the finding and is reported as a fraction rather than swept into
a mean. The aggregate (geometric-mean-over-nodes) fit is reported alongside as
an independent, non-per-node route to the same two factors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from f4_multifractal_targets import BY_NAME  # noqa: E402

from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

SEED = 20260814
H = 6
MIN_LOG_R_RANGE = 0.5  # ~1.65x growth in ball extent over the whole hop window

NAMES = (
    "SQ-unif",
    "SQ-mild",
    "SQ-matched",
    "SQ-strong",
    "GA-unif",
    "GA-mild",
    "GA-strong",
    "CD-unif",
    "CD-matched",
    "CD-strong",
)


def ball_profiles(pts: np.ndarray, hg, sources: list[int], h_max: int):
    """(V(h), R(h)) per source: ball size and ball Euclidean extent (max radius)."""
    adj = hg.adjacency()
    out = []
    for s in sources:
        visited = {s}
        frontier = {s}
        vs, rs = [], []
        for _ in range(h_max):
            nxt: set[int] = set()
            for u in frontier:
                nxt |= adj.get(u, set()) - visited
            if not nxt:
                break
            visited |= nxt
            frontier = nxt
            idx = np.fromiter(visited, dtype=np.int64, count=len(visited))
            vs.append(len(visited))
            rs.append(float(np.max(np.linalg.norm(pts[idx] - pts[s], axis=1))))
        if len(vs) == h_max and min(rs) > 0:
            out.append((np.asarray(vs, float), np.asarray(rs, float)))
    return out


def decompose(profiles):
    lh = np.log(np.arange(1, H + 1, dtype=float))
    a, b, c = [], [], []
    decoupled = 0
    lv_stack, lr_stack = [], []
    for vs, rs in profiles:
        lv, lr = np.log(vs), np.log(rs)
        lv_stack.append(lv)
        lr_stack.append(lr)
        a.append(np.polyfit(lh, lv, 1)[0])
        if lr.max() - lr.min() < MIN_LOG_R_RANGE:
            decoupled += 1
            continue
        b.append(np.polyfit(lr, lv, 1)[0])
        c.append(np.polyfit(lh, lr, 1)[0])
    LV = np.mean(np.vstack(lv_stack), axis=0)
    LR = np.mean(np.vstack(lr_stack), axis=0)
    return {
        "a": float(np.mean(a)),
        "b": float(np.mean(b)) if b else float("nan"),
        "c": float(np.mean(c)) if c else float("nan"),
        "decoupled": decoupled / len(profiles),
        "n_nodes": len(profiles),
        "a_agg": float(np.polyfit(lh, LV, 1)[0]),
        "b_agg": float(np.polyfit(LR, LV, 1)[0]),
        "c_agg": float(np.polyfit(lh, LR, 1)[0]),
    }


def main() -> None:
    print("=" * 118)
    print("C1  DECOMPOSITION  a ~ b * c    n=12800, k=15, H=6, 120 sampled sources")
    print("    a: ball growth per graph hop | b: ambient ball-count exponent (targets D_1)")
    print("    c: hops-to-space distortion (1.0 iff graph hops are proportional to distance)")
    print("    'decoup': fraction of sources whose ball never leaves one dense cluster")
    print("=" * 118)
    print(
        f"{'target':<12}{'D0':>7}{'D1':>7}{'D2':>7} | {'a':>7}{'b':>7}{'c':>7}{'decoup':>8} | "
        f"{'a_agg':>7}{'b_agg':>7}{'c_agg':>7} | {'b-D1':>7}{'b-D2':>7}"
    )
    for name in NAMES:
        t = BY_NAME[name]
        pts = t.sample(12800, SEED)
        hg = knn_hypergraph(list(map(tuple, pts)), 15, dedupe=True, duplicate_tolerance=1e-13)
        nodes = sorted(hg.nodes)
        sources = nodes[:: max(1, len(nodes) // 120)][:120]
        d = decompose(ball_profiles(pts, hg, sources, H))
        print(
            f"{t.name:<12}{t.D0:>7.3f}{t.D1:>7.3f}{t.D2:>7.3f} | "
            f"{d['a']:>7.3f}{d['b']:>7.3f}{d['c']:>7.3f}{d['decoupled']:>8.2f} | "
            f"{d['a_agg']:>7.3f}{d['b_agg']:>7.3f}{d['c_agg']:>7.3f} | "
            f"{d['b'] - t.D1:>+7.3f}{d['b'] - t.D2:>+7.3f}"
        )

    print()
    print("=" * 118)
    print("C2  Why the Cantor-dust support breaks the shell estimator: shell sequences")
    print("=" * 118)
    for name in ("CD-unif", "GA-unif"):
        t = BY_NAME[name]
        pts = t.sample(6400, SEED)
        hg = knn_hypergraph(list(map(tuple, pts)), 15, dedupe=True, duplicate_tolerance=1e-13)
        adj = hg.adjacency()
        nodes = sorted(hg.nodes)
        print(f"  {name} (k=15, n=6400): 5 sampled shell sequences")
        for s in nodes[:: len(nodes) // 5][:5]:
            visited, frontier, shells = {s}, {s}, []
            for _ in range(8):
                nxt: set[int] = set()
                for u in frontier:
                    nxt |= adj.get(u, set()) - visited
                if not nxt:
                    break
                visited |= nxt
                frontier = nxt
                shells.append(len(nxt))
            print(f"    {shells}")


if __name__ == "__main__":
    main()
