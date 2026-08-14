"""F4, step A: validate the targets, the sampler and BOTH estimators on cases
whose answers are known, BEFORE using any of them to identify anything.

Standing repository rule: validate any estimator you write against known-answer
cases before trusting it on the real target. Three separate validations here,
each of which can fail independently:

  A1  SAMPLER + ALGEBRA. Box-count D_0, D_1, D_2 directly off each sampled
      cloud and compare with the closed forms in f4_multifractal_targets.py.
      This checks that the points really are drawn from the invariant measure
      and that the Renyi algebra is right. Nothing downstream means anything if
      this fails.

  A2  BASELINE. The reviewer's finding ASSUMES Grassberger-Procaccia estimates
      D_2. That is true by construction in the asymptotic limit, but it is an
      assumption about THIS implementation at THIS n, so it is measured: run
      `baseline.correlation_dimension` on every target and check it lands on
      D_2 rather than on D_0 or D_1. On the strongly-skewed targets those three
      differ by ~1, so the check has real power.

  A3  SHELL ESTIMATOR, monofractal anchors. On the uniform-p targets
      D_0 = D_1 = D_2, so any correct dimension estimator must return that one
      number. This is the shell estimator's own known-answer case (the repo's
      own tests use the filled square -> 2). It cannot identify anything, but a
      failure here would mean the identification runs below are measuring a
      broken pipeline rather than a dimension type.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from f4_multifractal_targets import TARGETS, box_count_dimensions  # noqa: E402

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.dimension import mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

N_VALIDATE = 20000
SEED = 20260814


def main() -> None:
    print("=" * 100)
    print("A1  SAMPLER + ALGEBRA: box-counting the actual clouds vs the closed forms")
    print(f"    n = {N_VALIDATE} i.i.d. points from the invariant measure, seed {SEED}")
    print("=" * 100)
    print(
        f"{'target':<12}{'D0 exact':>10}{'D0 box':>9}{'D1 exact':>10}{'D1 box':>9}"
        f"{'D2 exact':>10}{'D2 box':>9}   occupied boxes by level"
    )
    clouds = {}
    for t in TARGETS:
        pts = t.sample(N_VALIDATE, SEED)
        clouds[t.name] = pts
        # Levels chosen so the finest box still holds many points on average:
        # for r=1/2, levels 2..6 (eps 1/4..1/64); for r=1/3, levels 1..4.
        levels = range(2, 7) if t.ratio > 0.4 else range(1, 5)
        bc = box_count_dimensions(pts, ratio=t.ratio, levels=levels)
        print(
            f"{t.name:<12}{t.D0:>10.4f}{bc['D0_boxcount']:>9.4f}"
            f"{t.D1:>10.4f}{bc['D1_boxcount']:>9.4f}"
            f"{t.D2:>10.4f}{bc['D2_boxcount']:>9.4f}   {bc['occupied']}"
        )

    print()
    print("=" * 100)
    print("A2  BASELINE: does Grassberger-Procaccia land on D_2 (not D_0, not D_1)?")
    print("=" * 100)
    print(
        f"{'target':<12}{'D0':>8}{'D1':>8}{'D2':>8} | {'GP':>8}{'R^2':>8} | "
        f"{'|GP-D0|':>8}{'|GP-D1|':>8}{'|GP-D2|':>8}  nearest"
    )
    for t in TARGETS:
        est = correlation_dimension(list(map(tuple, clouds[t.name])), n_radii=30)
        d = est.dimension
        errs = {"D0": abs(d - t.D0), "D1": abs(d - t.D1), "D2": abs(d - t.D2)}
        nearest = min(errs, key=errs.get)
        print(
            f"{t.name:<12}{t.D0:>8.4f}{t.D1:>8.4f}{t.D2:>8.4f} | {d:>8.4f}{est.r_squared:>8.4f} | "
            f"{errs['D0']:>8.4f}{errs['D1']:>8.4f}{errs['D2']:>8.4f}  {nearest}"
        )

    print()
    print("=" * 100)
    print("A3  SHELL ESTIMATOR on the monofractal anchors (D_0 = D_1 = D_2)")
    print("=" * 100)
    print(f"{'target':<12}{'truth':>8}{'n':>7}{'k':>4}{'max_r':>7}{'shell':>9}{'error':>9}")
    for t in TARGETS:
        if len(set(t.probabilities)) != 1:
            continue
        for n in (1600, 6400):
            pts = list(map(tuple, t.sample(n, SEED)))
            for k, mr in ((10, 6), (15, 6), (15, 8)):
                hg = knn_hypergraph(pts, k, dedupe=True, duplicate_tolerance=1e-13)
                val = mean_dimension(hg, samples=120, max_radius=mr)
                print(
                    f"{t.name:<12}{t.D0:>8.4f}{n:>7}{k:>4}{mr:>7}{val:>9.4f}{val - t.D0:>+9.4f}"
                )


if __name__ == "__main__":
    main()
