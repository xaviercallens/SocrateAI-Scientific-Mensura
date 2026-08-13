"""H3 / criterion (c) adversarial check: is the accuracy win a real property of
the point cloud, or an artifact of the (k, max_radius, seed) chosen to show it?

Motivated by finding R2-F3, where a recorded win turned out to be contingent on
a non-default `max_radius` whose stated justification was wrong. A new win
criterion must be checked against that failure mode before anything is banked
under it, not after.

Three sweeps on planar Brownian motion (true dimension 2, the criterion's own
headline case, at the n used by the unit test):

  1. SEED     -- 20 independent Brownian paths at fixed k and max_radius.
  2. K        -- k in {6, 8, 10, 12, 14} at fixed seed and max_radius.
  3. MAX_RADIUS -- 3..8 at fixed seed and k.

For each configuration we report the criterion (c) verdict and, when it is a
no-win, which condition refused it. A criterion that fires on 1 configuration
out of 30 is a tuned result; one that fires on most of them is a property of
the data. This script states which, rather than asserting either.

Exit code is 0 always -- this is a measurement, not a gate.
"""

from __future__ import annotations

import sys

import numpy as np

from socrates.hypergraph.comparison import compare_accuracy_at_max_n

N = 1600
TRUE_DIMENSION = 2.0
TOLERANCE = 0.15
BASE_SEED = 42
BASE_K = 10
BASE_MAX_RADIUS = 6  # comparison's own default


def brownian_2d(n: int, seed: int) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, 1.0, size=(n * 4, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(increments, axis=0)])
    return [tuple(p) for p in path[::4][:n]]


def refusal(reason: str) -> str:
    """One-word summary of which condition refused a no-win."""
    if reason.startswith("win"):
        return "-"
    if "sentinel" in reason:
        return "c2 sentinel"
    if "poly's own error" in reason:
        return "c3 poly inaccurate"
    if "also within the tolerance" in reason:
        return "c4 trad also ok"
    if "below the margin" in reason:
        return "c5 gap < margin"
    return "c1 no estimate"


def sweep(label: str, configs: list[tuple[str, int, int, int]]) -> tuple[int, int]:
    print(f"\n{label}")
    print(
        f"{'config':>16s}  {'poly':>7s} {'err':>7s}  {'trad':>7s} {'err':>7s}  "
        f"{'gap':>7s}  win  refused by"
    )
    wins = 0
    for name, seed, k, max_radius in configs:
        points = brownian_2d(N, seed)
        r = compare_accuracy_at_max_n(
            points,
            TRUE_DIMENSION,
            N,
            k=k,
            tolerance=TOLERANCE,
            max_radius=max_radius,
        )
        gap = r.traditional_abs_error - r.poly_algebraic_abs_error
        wins += bool(r.poly_algebraic_accuracy_win)
        print(
            f"{name:>16s}  {r.poly_algebraic_estimate:7.4f} {r.poly_algebraic_abs_error:7.4f}  "
            f"{r.traditional_estimate:7.4f} {r.traditional_abs_error:7.4f}  {gap:7.4f}  "
            f"{'YES' if r.poly_algebraic_accuracy_win else ' no':>3s}  {refusal(r.verdict_reason)}"
        )
    print(f"  -> {wins}/{len(configs)} configurations score a criterion (c) win")
    return wins, len(configs)


def main() -> int:
    total_wins = total = 0
    for label, configs in (
        (
            f"SWEEP 1 -- seed (k={BASE_K}, max_radius={BASE_MAX_RADIUS})",
            [(f"seed={s}", s, BASE_K, BASE_MAX_RADIUS) for s in range(20)],
        ),
        (
            f"SWEEP 2 -- k (seed={BASE_SEED}, max_radius={BASE_MAX_RADIUS})",
            [(f"k={k}", BASE_SEED, k, BASE_MAX_RADIUS) for k in (6, 8, 10, 12, 14)],
        ),
        (
            f"SWEEP 3 -- max_radius (seed={BASE_SEED}, k={BASE_K})",
            [(f"max_radius={m}", BASE_SEED, BASE_K, m) for m in range(3, 9)],
        ),
    ):
        w, t = sweep(label, configs)
        total_wins += w
        total += t

    print(f"\nOVERALL: {total_wins}/{total} configurations score a criterion (c) win")
    print(
        "Read this as: a win rate near 1.0 means the criterion is reporting a property\n"
        "of planar Brownian motion; a win rate near 1/total means it is reporting the\n"
        "configuration that was chosen to show it (the R2-F3 failure mode)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
