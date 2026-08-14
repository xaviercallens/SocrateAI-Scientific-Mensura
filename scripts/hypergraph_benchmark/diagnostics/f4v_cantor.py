"""F4 verification (d): WHY does the shell estimator break on Cantor dust?

The orchestrator's reading is "failure on a totally disconnected support".
That is a TOPOLOGICAL, binary property, and it makes a sharp prediction that
can be falsified.

THE EXPERIMENT.  Take the same 4 corner maps of the unit square and sweep the
contraction ratio r:

    r  = 1/2   images tile the square  -> attractor = the FULL SQUARE,
               connected, D0 = 2 exactly
    r  < 1/2   images are disjoint     -> attractor = C_r x C_r, TOTALLY
               DISCONNECTED for every r < 1/2, D0 = log 4 / log(1/r)

So connectivity flips discontinuously at r = 1/2 while D0 varies continuously.
Probabilities are held UNIFORM throughout, so D0 = D1 = D2 exactly and there is
no measure confound at all: any deviation of an estimator from D0 is a pure
support-geometry effect.

  * If "disconnected support breaks it" is the mechanism, the shell estimator
    must break at r = 0.49 (gaps of width 0.02) just as it does at r = 1/3.
  * If the mechanism is instead "gap width vs the k-NN neighbour scale", the
    estimator must degrade SMOOTHLY as r falls, and be fine at r = 0.49.

Also reported per configuration: k-NN graph component structure, raw shell
sequences, ball-fraction growth, and the fit-window length -- so a
max_radius-saturation artifact can be distinguished from a fit-quality collapse.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.dimension import (  # noqa: E402
    local_dimension,
    near_constant_consensus,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

SEED = 20260814
V4 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def sample(v, r, p, n, seed, depth=None):
    if depth is None:
        depth = int(math.ceil(60.0 / math.log2(1.0 / r)))
    rng = np.random.default_rng(seed)
    x = np.repeat(v[0][None, :], n, axis=0).astype(float)
    for _ in range(depth):
        idx = rng.choice(len(v), size=n, p=np.asarray(p))
        x = r * x + (1.0 - r) * v[idx]
    return x


def components(hg):
    adj = hg.adjacency()
    seen, sizes = set(), []
    for s in hg.nodes:
        if s in seen:
            continue
        q, size = deque([s]), 0
        seen.add(s)
        while q:
            u = q.popleft()
            size += 1
            for w in adj.get(u, ()):
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def sampled_nodes(hg, samples):
    nodes = sorted(hg.nodes)
    step = max(1, len(nodes) // samples)
    return nodes[::step][:samples]


def nn_scale(pts, k):
    from scipy.spatial import cKDTree

    d, _ = cKDTree(pts).query(pts, k=k + 1)
    return float(np.median(d[:, k]))


def run(n=6400, ks=(8, 15, 25), mrs=(4, 6, 8)):
    ratios = [0.5, 0.49, 0.47, 0.45, 0.40, 1.0 / 3.0]
    print(f"n = {n}, uniform p (so D0 = D1 = D2 exactly), 4 corner maps of the unit square\n")
    for r in ratios:
        d0 = math.log(4) / math.log(1.0 / r)
        pts = sample(V4, r, (0.25,) * 4, n, SEED)
        tup = list(map(tuple, pts))
        gap = max(0.0, 1.0 - 2.0 * r)  # level-1 gap width, in units of the unit square
        print("=" * 112)
        print(f"r = {r:.4f}   D0 = D1 = D2 = {d0:.4f}   level-1 gap width = {gap:.4f}   "
              f"{'CONNECTED (full square)' if r >= 0.5 else 'TOTALLY DISCONNECTED'}")
        for k in ks:
            hg = knn_hypergraph(tup, k, dedupe=True, duplicate_tolerance=1e-13)
            comps = components(hg)
            s = nn_scale(pts, k)
            print(f"  k={k:<3} median k-th-NN distance = {s:.5f}  (gap/scale = "
                  f"{gap / s if s else float('inf'):8.2f})   graph: {len(comps)} components, "
                  f"top {comps[:5]}")
            picked = sampled_nodes(hg, 80)
            for mr in mrs:
                ests = [local_dimension(hg, x, max_radius=mr) for x in picked]
                cons = near_constant_consensus(ests, threshold=0.9)
                wf = [e.dimension for e in ests
                      if e.is_well_fit(threshold=0.9, near_constant_consensus=cons)]
                shell = (sum(wf) / len(wf)) if wf else float("nan")
                unf = float(np.mean([e.dimension for e in ests]))
                # fit-window length actually used, and how fast the ball fills
                zero_dim = sum(1 for e in ests if e.dimension == 0.0)
                r2 = float(np.median([e.r_squared for e in ests]))
                ballfrac = float(np.median([e.volumes[-1] / len(hg.nodes) for e in ests]))
                print(f"    mr={mr}: shell={shell:+7.4f} (err vs D0 {shell - d0:+6.3f})  "
                      f"unfilt={unf:+7.4f}  wf={len(wf):>2}/80  medR2={r2:5.2f}  "
                      f"zero-dim nodes={zero_dim:>2}/80  median ball({mr})/N={ballfrac:.3f}")
            # shell sequences for three nodes, at the largest max_radius
            for x in picked[:3]:
                e = local_dimension(hg, x, max_radius=max(mrs))
                vols = list(e.volumes)
                shells = [vols[0]] + [vols[i] - vols[i - 1] for i in range(1, len(vols))]
                print(f"      node {x:<6} shells={shells}  dim={e.dimension:+.4f} "
                      f"R2={e.r_squared:.3f}")
        print()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6400
    run(n=n)
