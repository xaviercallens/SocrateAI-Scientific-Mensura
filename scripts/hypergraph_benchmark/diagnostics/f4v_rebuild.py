"""F4 verification: independent rebuild of the multifractal targets + re-run.

Deliberately does NOT import f4_multifractal_targets. Everything here is
re-derived and re-implemented:

  * the IFS maps are applied as EXPLICIT COMPOSITIONS x <- r*x + (1-r)*v_i,
    iterated per point from a starting vertex (a "one independent chain per
    point" chaos game), rather than by the closed-form weighted-sum address
    expansion the original module uses.  Different algorithm, same measure.
  * D0/D1/D2 are re-derived from the multifractal formalism (see f4v_skeptic_analyse).
  * D0/D1/D2 are ALSO measured empirically by box counting off the cloud, which
    is a known-answer test of the sampler and the algebra simultaneously.
  * the estimators are the real library ones, called through the public API.

Usage:  python f4v_rebuild.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    local_dimension,
    mean_dimension,
    near_constant_consensus,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

SEED = 20260814

SQ_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
GA_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
ST_V = 0.5 * np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
)


def theory(n_maps: int, r: float, p) -> tuple[float, float, float]:
    L = math.log(1.0 / r)
    return (
        math.log(n_maps) / L,
        -sum(pi * math.log(pi) for pi in p if pi > 0) / L,
        -math.log(sum(pi * pi for pi in p)) / L,
    )


def matched_probs(n: int, r: float, want_d2: float):
    s = math.exp(-want_d2 * math.log(1.0 / r))
    A = (n - 1) ** 2 + (n - 1)
    B = -2.0 * (n - 1)
    C = 1.0 - s
    b = (-B - math.sqrt(B * B - 4 * A * C)) / (2 * A)
    return (1.0 - (n - 1) * b,) + (b,) * (n - 1)


MATCHED_D2 = theory(3, 0.5, (0.6, 0.2, 0.2))[2]


def sample_ifs(v: np.ndarray, r: float, p, n: int, seed: int, depth: int = 64) -> np.ndarray:
    """n i.i.d. draws from the self-similar measure, by explicit map iteration.

    Independent chain per point: x_0 = v[0] (a point of the attractor), then
    x_{m+1} = r*x_m + (1-r)*v[i_{m+1}] with i drawn i.i.d. from p.  After
    `depth` steps the memory of x_0 is r^depth, i.e. below double precision, and
    the law of x_depth is exactly mu up to that.
    """
    rng = np.random.default_rng(seed)
    x = np.repeat(v[0][None, :], n, axis=0).astype(float)
    for _ in range(depth):
        idx = rng.choice(len(v), size=n, p=np.asarray(p))
        x = r * x + (1.0 - r) * v[idx]
    return x


def boxcount_dq(pts: np.ndarray, r: float, levels, *, occ_frac: float = 0.10) -> dict:
    """Empirical D0/D1/D2 by box counting at eps = r^m.  Independent of everything.

    D_0 saturates first at finite n: once boxes hold ~1 point, counting occupied
    boxes counts the SAMPLE, not the set.  So the D_0 fit is restricted to the
    unsaturated window (occupied <= occ_frac * n); D_1/D_2 are fit over the full
    range, which is safe because they are mass-weighted and saturate much later.
    Per-level local slopes are reported so the saturation is visible rather than
    hidden inside a single regression.
    """
    lo = pts.min(axis=0)
    span = float((pts.max(axis=0) - lo).max())
    n = len(pts)
    x, y0, y1, y2, occ = [], [], [], [], []
    for m in levels:
        eps = r**m
        idx = np.floor((pts - lo) / (span * eps + 1e-300)).astype(np.int64)
        _, counts = np.unique(idx, axis=0, return_counts=True)
        mu = counts / n
        x.append(math.log(1.0 / eps))
        y0.append(math.log(len(counts)))
        y1.append(float(-np.sum(mu * np.log(mu))))
        y2.append(-math.log(float(np.sum(mu * mu))))
        occ.append(len(counts))
    x = np.asarray(x)
    f = lambda xx, y: float(np.polyfit(np.asarray(xx), np.asarray(y), 1)[0])  # noqa: E731
    keep = [i for i, o in enumerate(occ) if o <= occ_frac * n]
    if len(keep) < 2:
        keep = list(range(min(3, len(occ))))
    local0 = [round((y0[i + 1] - y0[i]) / (x[i + 1] - x[i]), 3) for i in range(len(x) - 1)]
    local2 = [round((y2[i + 1] - y2[i]) / (x[i + 1] - x[i]), 3) for i in range(len(x) - 1)]
    return {
        "D0": f([x[i] for i in keep], [y0[i] for i in keep]),
        "D0_allrange": f(x, y0),
        "D1": f(x, y1),
        "D2": f(x, y2),
        "occupied": occ,
        "levels": list(levels),
        "D0_fit_levels": [list(levels)[i] for i in keep],
        "local_slope_D0": local0,
        "local_slope_D2": local2,
    }


def measure_fingerprint(pts: np.ndarray, r: float, m: int) -> dict:
    """Is this cloud actually drawn from the SKEWED measure or from a uniform one?

    At level m the cylinders are r^m boxes.  For probability vector p the
    LARGEST cylinder mass must be max(p)^m and the box-mass histogram must be
    the multinomial spectrum {prod p_{i_j}}.  A uniform-on-support cloud would
    instead have every occupied cylinder at mass ~ 1/N^m.
    """
    lo = pts.min(axis=0)
    span = float((pts.max(axis=0) - lo).max())
    eps = r**m
    idx = np.floor((pts - lo) / (span * eps + 1e-300)).astype(np.int64)
    _, counts = np.unique(idx, axis=0, return_counts=True)
    mu = np.sort(counts / len(pts))[::-1]
    return {
        "level": m,
        "n_occupied": int(len(mu)),
        "max_box_mass": float(mu[0]),
        "min_box_mass": float(mu[-1]),
        "mass_ratio_max_min": float(mu[0] / mu[-1]),
        "top5": [round(float(v), 6) for v in mu[:5]],
    }


def graph_components(hg) -> list[int]:
    adj = hg.adjacency()
    nodes = hg.nodes
    seen: set = set()
    sizes = []
    for s in nodes:
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        size = 0
        while q:
            u = q.popleft()
            size += 1
            for w in adj.get(u, ()):  # noqa
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def aggregate(ests) -> dict:
    consensus = near_constant_consensus(ests, threshold=0.9)
    wf = [e.dimension for e in ests if e.is_well_fit(threshold=0.9, near_constant_consensus=consensus)]
    return {
        "shell": (sum(wf) / len(wf)) if wf else float("nan"),
        "shell_unfiltered": float(np.mean([e.dimension for e in ests])),
        "shell_median_unfiltered": float(np.median([e.dimension for e in ests])),
        "n_well_fit": len(wf),
        "n_sampled": len(ests),
        "degen_frac": sum(1 for e in ests if e.degenerate) / len(ests),
        "near_degen_frac": sum(1 for e in ests if e.near_degenerate) / len(ests),
        "consensus": bool(consensus),
    }


def sampled_nodes(hg, samples: int):
    nodes = sorted(hg.nodes)
    if samples is not None and samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    return nodes


TARGETS = {
    # name: (vertices, ratio, probs)
    "SQ-matched": (SQ_V, 0.5, matched_probs(4, 0.5, MATCHED_D2)),
    "SQ-strong": (SQ_V, 0.5, (0.7, 0.1, 0.1, 0.1)),
    "SQ-unif": (SQ_V, 0.5, (0.25,) * 4),
    "ST-matched": (ST_V, 0.5, matched_probs(4, 0.5, MATCHED_D2)),
    "GA-strong": (GA_V, 0.5, (0.8, 0.1, 0.1)),
    "GA-unif": (GA_V, 0.5, (1 / 3,) * 3),
    "CD-unif": (SQ_V, 1 / 3, (0.25,) * 4),
    "CD-matched": (SQ_V, 1 / 3, matched_probs(4, 1 / 3, MATCHED_D2)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--targets", default="SQ-matched,ST-matched,SQ-strong,GA-strong,CD-unif,CD-matched,SQ-unif,GA-unif")
    ap.add_argument("--n", type=int, default=12800)
    args = ap.parse_args()

    ks = (8, 15) if args.quick else (8, 15, 25)
    mrs = (6,) if args.quick else (4, 6, 8)
    n = args.n
    out = []

    for name in args.targets.split(","):
        v, r, p = TARGETS[name]
        d0, d1, d2 = theory(len(v), r, p)
        t0 = time.time()
        pts = sample_ifs(v, r, p, n, SEED)
        # --- known-answer validation of sampler + algebra, before any estimator
        lv = range(2, 8) if r == 0.5 else range(2, 7)
        bc = boxcount_dq(pts, r, lv)
        fp = measure_fingerprint(pts, r, 4 if r == 0.5 else 3)
        exp_max = max(p) ** fp["level"]
        print("=" * 100)
        print(f"{name}   p = {tuple(round(x,6) for x in p)}   r = {r:.5f}   ambient dim {v.shape[1]}")
        print(f"  THEORY      D0={d0:.5f}  D1={d1:.5f}  D2={d2:.5f}")
        print(f"  BOXCOUNT    D0={bc['D0']:.5f} (unsaturated levels {bc['D0_fit_levels']};"
              f" all-range {bc['D0_allrange']:.5f})  D1={bc['D1']:.5f}  D2={bc['D2']:.5f}")
        print(f"     occupied per level {list(lv)} = {bc['occupied']}   (n={n})")
        print(f"     local slope D0 {bc['local_slope_D0']}   local slope D2 {bc['local_slope_D2']}")
        print(f"  MEASURE FINGERPRINT at level {fp['level']}: occupied={fp['n_occupied']} "
              f"(theory N^m = {len(v)**fp['level']})")
        print(f"     max box mass {fp['max_box_mass']:.6f}  vs theory max(p)^m = {exp_max:.6f}"
              f"   |  uniform-on-support would give {1.0/len(v)**fp['level']:.6f}")
        print(f"     top5 masses {fp['top5']}   max/min ratio {fp['mass_ratio_max_min']:.1f}")

        tup = list(map(tuple, pts))
        gp = correlation_dimension(tup, n_radii=30)
        print(f"  GP (library, n_radii=30): dim={gp.dimension:.5f} r2={gp.r_squared:.4f}"
              f"   |  gp - D2 = {gp.dimension - d2:+.4f}   gp - D0 = {gp.dimension - d0:+.4f}")

        for k in ks:
            hg = knn_hypergraph(tup, k, dedupe=True, duplicate_tolerance=1e-13)
            comps = graph_components(hg)
            picked = sampled_nodes(hg, 80)
            for mr in mrs:
                ests = [local_dimension(hg, s, max_radius=mr) for s in picked]
                agg = aggregate(ests)
                md = mean_dimension(hg, samples=80, max_radius=mr)
                same = (math.isnan(md) and math.isnan(agg["shell"])) or abs(md - agg["shell"]) < 1e-12
                print(f"    k={k:<3} mr={mr}: shell={agg['shell']:+.4f} (mean_dimension={md:+.4f}"
                      f" match={same})  unfilt={agg['shell_unfiltered']:+.4f}"
                      f"  medunfilt={agg['shell_median_unfiltered']:+.4f}"
                      f"  wf={agg['n_well_fit']:>2}/80 degen={agg['degen_frac']:.2f}"
                      f" neardeg={agg['near_degen_frac']:.2f} cons={agg['consensus']}"
                      f"  | sh-D0={agg['shell']-d0:+.3f} sh-D1={agg['shell']-d1:+.3f}"
                      f" sh-D2={agg['shell']-d2:+.3f}")
                out.append({"target": name, "n": n, "k": k, "max_radius": mr,
                            "D0": d0, "D1": d1, "D2": d2, "gp": gp.dimension,
                            "gp_r2": gp.r_squared, "bc": bc["D0"], "bc1": bc["D1"],
                            "bc2": bc["D2"], **agg,
                            "n_components": len(comps), "largest_component": comps[0],
                            "component_sizes_top10": comps[:10],
                            "n_nodes": len(hg.nodes)})
            print(f"    k={k:<3} graph: {len(hg.nodes)} nodes, {len(comps)} components, "
                  f"top sizes {comps[:6]}")
        print(f"  [{time.time()-t0:.1f}s]\n")

    Path(__file__).resolve().parent.joinpath("f4v_rebuild_results.json").write_text(
        json.dumps(out, indent=1)
    )
    print(f"wrote {len(out)} rows")


if __name__ == "__main__":
    main()
