"""F4, step F: the control that decides what F4 implies for problems 08/09.

THE QUESTION. The ledger (F4/F4b, sections 5 and 11) records that on Lorenz and
Rossler the shell estimate rises monotonically with n, passes THROUGH the
literature D_2, and is still climbing at the largest n affordable. The reviewer
proposes that this is not slow convergence but a category error: the estimator
is converging correctly to a different and larger number (D_0).

Step B/D established the first half of that -- the estimator does track D_0.
This step tests the second half, which is a separate claim and does not follow
from the first: is the observed monotone rise EXPLAINED by the D_0/D_2 gap?

THE CONTROL. Run the identical n-sweep on ST-unif: the Sierpinski tetrahedron
with uniform probabilities. It is in R^3 like Lorenz, its dimension is 2.0000,
and -- the point -- it is exactly MONOFRACTAL: D_0 = D_1 = D_2 = 2.0000 by
construction. On this target a dimension-type mismatch is arithmetically
impossible, because all three candidate answers are the same number. If the
shell estimator STILL shows the ledger's rise-and-overshoot pattern here, then
that pattern is finite-size behaviour and needs no category error to explain
it.

ST-strong (same support, same D_0 = 2.0000, but D_2 = 0.9434) is swept
alongside as the contrast: if the two sweeps look the same, the pattern is
independent of the measure's multifractality altogether.

Lorenz itself is swept in the same table with the same estimator settings, so
the shapes can be compared directly rather than against the ledger's prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from f4_multifractal_targets import BY_NAME  # noqa: E402
from f4d_3d_and_attractor_spread import lorenz_cloud, rossler_cloud  # noqa: E402

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.dimension import mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

SEED = 20260814
N_GRID = (200, 400, 800, 1600, 3200, 6400, 12800)
# (k, max_radius, samples): the first row mirrors the ledger's own F4 sweep on
# problem 08 (k=15, max_radius=5, 60-node sample); the others check the pattern
# is not a property of that one setting.
CONFIGS = ((15, 5, 60), (10, 6, 60), (25, 6, 60))


def sweep(label: str, cloud_fn, truth: str) -> None:
    print(f"\n  {label}   (truth: {truth})")
    for k, mr, samples in CONFIGS:
        vals = []
        for n in N_GRID:
            pts = list(map(tuple, cloud_fn(n)))
            try:
                hg = knn_hypergraph(pts, k, dedupe=True, duplicate_tolerance=1e-13)
                vals.append(mean_dimension(hg, samples=samples, max_radius=mr))
            except Exception as exc:  # noqa: BLE001
                vals.append(float("nan"))
                print(f"      (n={n}: {type(exc).__name__})")
        cells = "".join(f"{v:>9.3f}" if np.isfinite(v) else f"{'nan':>9}" for v in vals)
        finite = [v for v in vals if np.isfinite(v)]
        rise = (max(finite) - min(finite)) if finite else float("nan")
        mono = all(b >= a - 0.02 for a, b in zip(finite, finite[1:], strict=False))
        print(
            f"    k={k:<3} max_r={mr}  {cells}   | total rise {rise:.3f}"
            f" | monotone(+/-0.02): {mono}"
        )


def main() -> None:
    print("=" * 104)
    print("F1  THE n-SWEEP, same estimator settings on every cloud")
    print(f"    n = {N_GRID}")
    print("=" * 104)
    print(f"    {'':<14}" + "".join(f"{n:>9}" for n in N_GRID))

    for name in ("ST-unif", "ST-strong"):
        t = BY_NAME[name]
        sweep(
            name,
            lambda n, t=t: t.sample(n, SEED),
            f"D_0={t.D0:.4f} D_1={t.D1:.4f} D_2={t.D2:.4f}",
        )

    lz = lorenz_cloud(N_GRID[-1])
    rs = rossler_cloud(N_GRID[-1])
    sweep("Lorenz", lambda n: lz[:: max(1, len(lz) // n)][:n], "literature D_2 = 2.05")
    sweep("Rossler", lambda n: rs[:: max(1, len(rs) // n)][:n], "literature D_2 = 2.01")

    print()
    print("=" * 104)
    print("F2  The same sweep for Grassberger-Procaccia, for reference")
    print("=" * 104)
    print(f"    {'':<14}" + "".join(f"{n:>9}" for n in N_GRID))
    for label, fn in (
        ("ST-unif", lambda n: BY_NAME["ST-unif"].sample(n, SEED)),
        ("ST-strong", lambda n: BY_NAME["ST-strong"].sample(n, SEED)),
        ("Lorenz", lambda n: lz[:: max(1, len(lz) // n)][:n]),
        ("Rossler", lambda n: rs[:: max(1, len(rs) // n)][:n]),
    ):
        vals = [correlation_dimension(list(map(tuple, fn(n))), n_radii=30).dimension for n in N_GRID]
        print(f"    {label:<14}" + "".join(f"{v:>9.3f}" for v in vals))


if __name__ == "__main__":
    main()
