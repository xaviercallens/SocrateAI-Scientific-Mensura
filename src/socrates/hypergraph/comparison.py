"""Apples-to-apples comparison: Poly-Algebraic Calculus vs the traditional baseline.

Tier B. Two *separate* win criteria live here, and they answer different
questions. Keeping them separate is the point of this module; conflating them
is what docs/POLY_ALGEBRAIC_BENCHMARK.md finding R2-F9 identified as a defect
in the benchmark's own scoreboard.

**Criterion (a), sample efficiency** -- `compare()` / `ComparisonResult`.
At how many points does each method's dimension estimate first become, and
then STAY, within `tolerance` of the known dimension? Fewer points needed is a
genuine compute-cost claim (both methods build a k-NN or pair-count structure
in O(n log n) via the same `scipy.spatial.cKDTree`, so this isolates sample
efficiency, not implementation quality). Point-count parity is enforced: both
methods are evaluated on identical prefixes of the same point cloud.

**Criterion (c), asymptotic accuracy** -- `compare_accuracy_at_max_n()` /
`AccuracyComparisonResult`. Which method is *closer to the truth* at one fixed,
large point count? This criterion does not ask, and does not care, whether
either method converged in the criterion-(a) sense.

Criterion (c) exists because criterion (a) is structurally blind to the one
result the round-2 benchmark actually produced in poly's favour on
non-degenerate targets (finding R2-F9, docs/POLY_ALGEBRAIC_BENCHMARK.md
Sec. 10.5). On the quasiperiodic torus and on planar Brownian motion -- the two
2-dimensional, non-degenerate targets in the suite -- the shell-growth
estimator lands within 0.03-0.05 of the truth with a degenerate fraction of
0.00, while Grassberger-Procaccia is wrong by 0.36-0.61 with R^2 > 0.998, i.e.
confidently, converged-looking, and wrong. Criterion (a) scores both as
non-wins and *must* do so: the baseline never converges there, so there is no
"fewer points" comparison to make, and scoring a baseline's non-convergence as
a poly win would be exactly the self-deception standing rule 2 forbids.

Criterion (c) is therefore not a loosening of rule 2. It measures a different
quantity (distance from truth at fixed n, not points-to-converge), it is
computed by its own code path -- it shares no convergence logic with
`compare()` and calls neither `poly_algebraic_minimum_points` nor
`minimum_points_for_target_accuracy` -- and it carries its own guard against
the degeneracy that made six of the round-1 "passes" vacuous (finding F1). See
`AccuracyComparisonResult.poly_algebraic_accuracy_win` for the exact,
checkable definition and for why it cannot be tuned into existence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .baseline import correlation_dimension, minimum_points_for_target_accuracy
from .dimension import degenerate_fraction, mean_dimension, near_degenerate_fraction
from .pointcloud import DuplicatePointsError, knn_hypergraph


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing the two methods on one point cloud."""

    true_dimension: float
    tolerance: float
    poly_algebraic_min_n: int | None
    traditional_min_n: int | None
    poly_algebraic_estimate_at_max_n: float
    traditional_estimate_at_max_n: float
    max_n_tested: int

    @property
    def poly_algebraic_wins(self) -> bool:
        """Strictly fewer points needed, with BOTH methods actually converging.

        A method that never stably converges in the tested grid cannot be
        said to "win" on efficiency -- that would reward failure to
        converge as if it were failure to need many points. See
        `compute_savings_fraction` for the case both converge.
        """
        if self.poly_algebraic_min_n is None or self.traditional_min_n is None:
            return False
        return self.poly_algebraic_min_n < self.traditional_min_n

    @property
    def compute_savings_fraction(self) -> float | None:
        """Fraction of points saved by the poly-algebraic method, if both converged.

        None if either method never converged in the tested grid (a savings
        percentage is not meaningful when the denominator is undefined).
        """
        if self.poly_algebraic_min_n is None or self.traditional_min_n is None:
            return None
        return 1.0 - (self.poly_algebraic_min_n / self.traditional_min_n)


def poly_algebraic_minimum_points(
    points: list[tuple[float, ...]],
    true_dimension: float,
    tolerance: float,
    *,
    k: int,
    n_grid: tuple[int, ...] = (100, 200, 400, 800, 1600, 3200),
    max_radius: int = 6,
    samples: int = 40,
    theiler_window: int | str = 0,
    time_indices: Sequence[int] | None = None,
    max_ball_fraction: float = 1.0,
) -> int | None:
    """Analogous to `baseline.minimum_points_for_target_accuracy`, for the shell-growth method.

    Requires stable convergence (within tolerance at every larger n tested
    in the grid, not just a single lucky crossing), applying the same
    discipline docs/POLY_ALGEBRAIC_BENCHMARK.md findings F4/F4b demanded.
    Duplicate point clusters are auto-deduplicated (`dedupe=True`) rather
    than raising, since this function is meant to run unattended over many
    (n, k) combinations.

    `theiler_window` / `time_indices` are forwarded to `knn_hypergraph`
    unchanged; `theiler_window=0` (the default) reproduces the pre-AR1 result
    exactly. Each grid point uses the PREFIX `points[:n]`, so `"auto"` is
    re-evaluated per prefix -- intended, since the decorrelation time is a
    property of the sample actually being measured. A window that starves some
    point of neighbours at a small n is recorded as "no estimate at this n"
    rather than aborting the sweep, matching how a duplicate cluster is handled.

    `max_ball_fraction` is forwarded to `dimension.mean_dimension` unchanged
    (finite-size saturation guard; 1.0 = off = pre-AR2 behaviour). It applies to
    the shell-growth side ONLY, and unlike `theiler_window` that is not a fair-
    comparison problem: it is not a correction to the data both estimators read,
    it is a fit-window rule internal to one estimator, and the correlation sum
    has its own scaling-region selection built into `baseline`'s radius grid.
    """
    results = []
    for n in n_grid:
        if n > len(points) or k >= n:
            break
        subset = points[:n]
        try:
            hg = knn_hypergraph(
                subset,
                k=k,
                dedupe=True,
                theiler_window=theiler_window,
                time_indices=None if time_indices is None else time_indices[:n],
            )
        except DuplicatePointsError:
            continue
        except ValueError:
            results.append((n, float("nan")))
            continue
        dim = mean_dimension(
            hg,
            samples=min(n, samples),
            max_radius=max_radius,
            max_ball_fraction=max_ball_fraction,
        )
        results.append((n, dim))

    for i, (_, dim) in enumerate(results):
        if not math.isfinite(dim):
            continue
        if abs(dim - true_dimension) > tolerance:
            continue
        if all(math.isfinite(d) and abs(d - true_dimension) <= tolerance for _, d in results[i:]):
            return results[i][0]
    return None


def compare(
    points: list[tuple[float, ...]],
    true_dimension: float,
    tolerance: float,
    *,
    k: int,
    n_grid: tuple[int, ...] = (100, 200, 400, 800, 1600, 3200),
    max_radius: int = 6,
    theiler_window: int | str = 0,
    time_indices: Sequence[int] | None = None,
    max_ball_fraction: float = 1.0,
) -> ComparisonResult:
    """Run both methods on the same point cloud and report the comparison.

    `theiler_window` is deliberately ONE knob applied to BOTH estimators --
    the k-NN graph the shell-growth method is built on, and the correlation
    sum -- rather than a separate setting per side. Theiler's correction is
    not an advantage granted to one method; it is a statement about what
    trajectory data means, and a comparison in which only one side receives it
    would be rigged in whichever direction the knob was set. `0` (the default)
    leaves both sides on their pre-AR1 code paths, so every recorded number
    reproduces exactly.

    Note that the correction generally makes the TRADITIONAL side harder to
    beat asymptotically while costing it small-n stability (see
    `baseline.correlation_dimension`'s measured caveat), so a poly-algebraic
    win recorded with the window on is a win over a differently-corrected, not
    a weakened, baseline. Both `poly_algebraic_min_n` and `traditional_min_n`
    must be re-derived under the same setting; neither may be quoted from a run
    with a different one.

    `max_ball_fraction` (finite-size saturation guard, 1.0 = off) is by contrast
    a one-sided setting, and deliberately so -- see
    `poly_algebraic_minimum_points` for why it is not the same kind of knob as
    `theiler_window`. It is still a setting a recorded number must be quoted
    with.
    """
    max_n = max(n for n in n_grid if n <= len(points))

    poly_n = poly_algebraic_minimum_points(
        points,
        true_dimension,
        tolerance,
        k=k,
        n_grid=n_grid,
        max_radius=max_radius,
        theiler_window=theiler_window,
        time_indices=time_indices,
        max_ball_fraction=max_ball_fraction,
    )
    trad_n = minimum_points_for_target_accuracy(
        points,
        true_dimension,
        tolerance,
        n_grid=n_grid,
        theiler_window=theiler_window,
        time_indices=time_indices,
    )

    try:
        hg_final = knn_hypergraph(
            points[:max_n],
            k=k,
            dedupe=True,
            theiler_window=theiler_window,
            time_indices=None if time_indices is None else time_indices[:max_n],
        )
        poly_final = mean_dimension(
            hg_final, samples=40, max_radius=max_radius, max_ball_fraction=max_ball_fraction
        )
    except (DuplicatePointsError, ValueError):
        poly_final = float("nan")

    trad_final = correlation_dimension(
        points[:max_n],
        theiler_window=theiler_window,
        time_indices=None if time_indices is None else time_indices[:max_n],
    ).dimension

    return ComparisonResult(
        true_dimension=true_dimension,
        tolerance=tolerance,
        poly_algebraic_min_n=poly_n,
        traditional_min_n=trad_n,
        poly_algebraic_estimate_at_max_n=poly_final,
        traditional_estimate_at_max_n=trad_final,
        max_n_tested=max_n,
    )


# =============================================================================
# Criterion (c): asymptotic accuracy at fixed n
# =============================================================================
#
# Everything below is independent of the criterion-(a) machinery above. It
# calls neither `poly_algebraic_minimum_points` nor
# `minimum_points_for_target_accuracy`, uses no n_grid, and never asks whether
# a method "converged". That independence is deliberate and is the whole point
# of finding R2-F9 (docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 10.5): the two
# non-degenerate 2D targets in the benchmark cannot be scored by criterion (a)
# at all, because the baseline never converges there, so a criterion that is
# merely a re-skin of the convergence search would reproduce the same blindness.
#
# --- DEFAULT_ACCURACY_MARGIN: how much better is "meaningfully better"? -------
#
# The margin is the minimum accuracy gap (traditional |error| minus poly
# |error|) that counts. It exists so that a win cannot be a coin flip: the gap
# has its own run-to-run noise, and a threshold below that noise would score
# resampling luck.
#
# Calibrated, not assumed
# (scripts/hypergraph_benchmark/round3/h3_margin_calibration.py). Ten
# independent redraws (reseeded Brownian paths, reseeded uniform square and
# cube samples, re-phased quasiperiodic torus) at fixed n, k and max_radius,
# measuring the standard deviation of each method's estimate:
#
#     cloud              n      poly sd    trad sd    gap sd (rss)
#     Brownian 2D      1600     0.0652     0.0763        0.100
#     uniform square    800     0.0373     0.0207        0.043
#     quasiperiodic     800     0.0243     0.0619        0.067
#     uniform cube 3D   400     0.0308     0.1045        0.109
#
# The worst measured gap noise is ~0.11 (one sd), so 0.25 is ~2.3 sd above the
# noisiest configuration measured -- a gap this large is not a redraw artifact.
#
# The other side of the calibration, from Sec. 10.5's own table (errors at each
# problem's max n): the two cases this criterion exists to credit have gaps of
# 0.314 (quasiperiodic torus) and 0.579 (Brownian), both comfortably above
# 0.25; Lorenz has a gap of 0.063 and is correctly refused.
#
# HONEST BOUNDARY CASE, stated rather than smoothed over: Rossler's gap is
# 0.276 using the round-2 agent's poly estimate (2.04) but 0.198 using the
# auditor's independent re-measurement (1.90) of the same system, so 0.25
# splits those two readings. No margin resolves that, because the disagreement
# (0.14) is between two measurements of the same quantity, not between two
# thresholds. The conservative reading -- the auditor's own number, which is
# the one Sec. 10.5 reports -- gives no win, and a case sitting this close to
# the line should not be banked in either direction on one measurement.
DEFAULT_ACCURACY_MARGIN = 0.25

# --- DEFAULT_MAX_SENTINEL_FRACTION: the F1 degeneracy guard -------------------
#
# Fraction of sampled nodes allowed to be in a sentinel regime -- `degenerate`
# (exactly constant shell sequence, r_squared overwritten with the sentinel
# 1.0) or `near_degenerate` (near-constant, r_squared uninformative). Both
# regimes pin the reported dimension at or near exactly 1.0 by construction, so
# on a 1-dimensional target a "perfect" poly answer there is not a measurement
# at all -- it is the ring-lattice degeneracy that made six of the ten round-1
# passes vacuous (finding F1, Sec. 4). Without this gate, criterion (c) would
# re-import exactly that failure: a circle scores poly error 0.0000 against
# traditional 0.1162, which is a "win" on the numbers and evidence of nothing.
#
# Measured separation over the same 10 redraws per cloud, 40 sampled nodes each
# (h3_margin_calibration.py, TABLE 2; min/max over redraws):
#   must accept -- uniform square, uniform cube, quasiperiodic torus: 0.000
#                  flat, every redraw. Brownian 2D: 0.000 in 9 of 10 redraws
#                  and 0.025 in one, i.e. a single sampled node of 40. That is
#                  not noise in the threshold; it is the known sporadic
#                  residue -- a length>=5 window that happens to look flat in a
#                  genuinely 2D structure -- that dimension.py's NEAR_CONSTANT_*
#                  commentary documents and states no node-local rule removes.
#   must reject -- circle (exact circulant ring lattice): 1.000;
#                  straight line: 0.925.
# 0.05 is 2 of 40 sampled nodes: twice the worst must-accept value measured
# (0.025) and 18x below the ring lattice it must reject. The gap between the
# two sides is a factor of 37, so the threshold is bracketed by data rather
# than perched between neighbouring measurements -- unlike the bare
# near_degenerate_fraction, whose must-accept and must-reject sides
# dimension.py measured to overlap outright.
#
# Note the asymmetry with the traditional side, which gets no analogous gate:
# it is not an oversight. The gate exists because the shell-growth estimator
# has a documented regime in which the answer 1.0 is *structurally forced*
# regardless of the data. Grassberger-Procaccia has no such forced-answer mode
# -- Sec. 10.5 shows it failing with R^2 > 0.998, i.e. its fit-quality number
# does not flag its errors either, but it is not pinned to a constant. A
# fit-quality gate on the baseline would therefore reject nothing it should.
DEFAULT_MAX_SENTINEL_FRACTION = 0.05

# --- Is a criterion (c) win a property of the data or of the settings? --------
#
# Finding R2-F3 recorded a win that turned out to be contingent on a non-default
# `max_radius` with a wrong stated justification, so the same question was put
# to this criterion before anything was banked under it
# (scripts/hypergraph_benchmark/round3/h3_win_robustness.py; planar Brownian
# motion, n=1600, tolerance 0.15, defaults elsewhere):
#
#     sweep                                       criterion (c) wins
#     k in {6, 8, 10, 12, 14}                            5 / 5
#     max_radius in 3..8                                 6 / 6
#     20 independent seeds                              16 / 20
#     total                                             27 / 31
#
# The win is completely insensitive to both tuning knobs -- the poly estimate
# moves by at most 0.065 across all 11 (k, max_radius) settings and the verdict
# never changes -- so it is not the R2-F3 failure mode. It is NOT insensitive to
# the draw: 4 of 20 seeds refuse, 3 because the shell-growth estimator itself
# misses the 0.15 tolerance (condition c3) and 1 because the gap falls to 0.243,
# just under the margin (condition c5). Both refusals are the criterion working.
# The honest statement is therefore "on ~80% of Brownian draws at this n", not
# "on planar Brownian motion", and a single-cloud win should be reported with
# its seed.


@dataclass(frozen=True)
class AccuracyComparisonResult:
    """Outcome of comparing the two methods' ACCURACY at one fixed point count.

    Every field is a raw measurement; every verdict is a property derived from
    those measurements plus the three stated criterion parameters (`tolerance`,
    `margin`, `max_sentinel_fraction`). Nothing is decided inside the
    measurement function, so a verdict can be re-derived, argued with, or
    recomputed under different parameters from a stored result.
    """

    true_dimension: float
    tolerance: float
    margin: float
    max_sentinel_fraction: float
    n_requested: int
    n_evaluated: int
    poly_algebraic_n_nodes: int
    poly_algebraic_estimate: float
    traditional_estimate: float
    poly_algebraic_degenerate_fraction: float
    poly_algebraic_near_degenerate_fraction: float
    traditional_r_squared: float

    @property
    def poly_algebraic_abs_error(self) -> float:
        """|poly estimate - true dimension|; nan if the estimate is nan."""
        return abs(self.poly_algebraic_estimate - self.true_dimension)

    @property
    def traditional_abs_error(self) -> float:
        """|traditional estimate - true dimension|; nan if the estimate is nan."""
        return abs(self.traditional_estimate - self.true_dimension)

    @property
    def poly_algebraic_sentinel_fraction(self) -> float:
        """Fraction of sampled nodes whose r_squared is not a fit-quality number.

        `degenerate` and `near_degenerate` are mutually exclusive per node (see
        `dimension._log_log_fit`), so this sum is a genuine fraction in [0, 1].
        """
        return (
            self.poly_algebraic_degenerate_fraction + self.poly_algebraic_near_degenerate_fraction
        )

    @property
    def more_accurate(self) -> str:
        """Which method is closer to the truth: 'poly_algebraic', 'traditional',
        'tie', or 'undetermined'.

        A ranking, NOT a verdict -- it applies `margin` but applies neither the
        tolerance conditions nor the degeneracy guard, so
        `more_accurate == 'poly_algebraic'` does not imply
        `poly_algebraic_accuracy_win`. The circle case is exactly this: poly is
        genuinely closer to 1.0, and it is closer for a reason that is not a
        measurement.

        'undetermined' means neither method produced a finite estimate.
        """
        poly_err, trad_err = self.poly_algebraic_abs_error, self.traditional_abs_error
        poly_ok, trad_ok = math.isfinite(poly_err), math.isfinite(trad_err)
        if not poly_ok and not trad_ok:
            return "undetermined"
        if not poly_ok:
            return "traditional"
        if not trad_ok:
            return "poly_algebraic"
        # The strict `<` is deliberate and matches condition (c5)'s `>= margin`
        # exactly, so a gap sitting precisely on the margin is never reported as
        # a tie by one and a win by the other. The equality case is called out
        # separately only because at margin=0 a strict `<` would otherwise rank
        # two identical errors.
        if poly_err == trad_err or abs(poly_err - trad_err) < self.margin:
            return "tie"
        return "poly_algebraic" if poly_err < trad_err else "traditional"

    def _verdict(self) -> tuple[bool, str]:
        """The criterion (c) decision and the reason for it, in one place."""
        poly_err, trad_err = self.poly_algebraic_abs_error, self.traditional_abs_error
        sentinel = self.poly_algebraic_sentinel_fraction

        if not math.isfinite(poly_err):
            return False, (
                f"no win: the shell-growth estimator returned no finite estimate at "
                f"n={self.n_evaluated}, so there is no accuracy to compare"
            )
        if not math.isfinite(trad_err):
            return False, (
                f"no win: the traditional estimator returned no finite estimate at "
                f"n={self.n_evaluated}. A baseline that produces no number is a "
                f"baseline failure, not a poly-algebraic accuracy win (standing rule 2)"
            )
        if not (sentinel <= self.max_sentinel_fraction):
            return False, (
                f"no win: sentinel fraction {sentinel:.3f} exceeds "
                f"{self.max_sentinel_fraction:.3f} (degenerate "
                f"{self.poly_algebraic_degenerate_fraction:.3f} + near-degenerate "
                f"{self.poly_algebraic_near_degenerate_fraction:.3f}); the poly estimate is "
                f"pinned near 1.0 by the shell sequence rather than measured from it, "
                f"so its accuracy is not evidence (finding F1)"
            )
        if poly_err > self.tolerance:
            return False, (
                f"no win: poly's own error {poly_err:.4f} exceeds the tolerance "
                f"{self.tolerance:.4f}; being less wrong than the baseline is not accuracy"
            )
        if trad_err <= self.tolerance:
            return False, (
                f"no win: the traditional error {trad_err:.4f} is also within the tolerance "
                f"{self.tolerance:.4f}; both methods met the stated bar, so the difference "
                f"is not decision-relevant"
            )
        gap = trad_err - poly_err
        if gap < self.margin:
            return False, (
                f"no win: accuracy gap {gap:.4f} is below the margin {self.margin:.4f}, "
                f"i.e. within the measured run-to-run spread of the two estimators"
            )
        return True, (
            f"win: poly error {poly_err:.4f} <= tolerance {self.tolerance:.4f} < traditional "
            f"error {trad_err:.4f} (gap {gap:.4f} >= margin {self.margin:.4f}), with "
            f"sentinel fraction {sentinel:.3f} <= {self.max_sentinel_fraction:.3f}"
        )

    @property
    def poly_algebraic_accuracy_win(self) -> bool:
        """Criterion (c). Does this count as a poly-algebraic win on accuracy?

        True iff ALL FIVE of the following hold at the single fixed point count
        `n_evaluated`. Convergence is deliberately not among them.

          (c1) Both methods produced a finite estimate. If the baseline
               produced none, that is the baseline failing, and standing rule 2
               forbids banking a baseline failure as a poly win -- the same
               reason `ComparisonResult.poly_algebraic_wins` returns False when
               `traditional_min_n is None`.
          (c2) `poly_algebraic_sentinel_fraction <= max_sentinel_fraction`.
               Poly is not in the degenerate / near-constant regime where its
               answer is structurally pinned near 1.0 (finding F1). This is
               what stops a ring lattice from being re-scored as an accuracy
               win.
          (c3) `poly_algebraic_abs_error <= tolerance`. Poly is accurate in its
               own right, at the same bar the rest of the benchmark uses. Being
               merely less wrong than a badly wrong baseline is not a win.
          (c4) `traditional_abs_error > tolerance`. The baseline fails that
               same bar. The separation therefore changes what a user would
               conclude, rather than ranking two acceptable answers.
          (c5) `traditional_abs_error - poly_algebraic_abs_error >= margin`.
               The gap exceeds the estimators' measured run-to-run spread, so
               it is not a redraw artifact.

        WHY THIS CANNOT BE TUNED INTO EXISTENCE. (c3) and (c4) pull in opposite
        directions in `tolerance`: raising it makes (c3) easier and (c4)
        harder, lowering it does the reverse. A win therefore requires
        `poly_err <= tolerance < trad_err`, so the tolerance must land inside an
        interval that the DATA defines and whose width is exactly the accuracy
        gap -- and (c5) requires that width to be at least `margin`. There is no
        direction to move `tolerance` in that manufactures a win; if the two
        errors do not straddle by at least `margin`, no tolerance works. This is
        the specific defect finding R2-F2 identified in criterion (b) (a
        criterion satisfiable by choosing its own parameters), and it is why
        criterion (c) is stated this way rather than as a bare "poly closer to
        truth" comparison.

        WHAT IT DOES NOT CLAIM. It is a statement about ONE point cloud at ONE
        point count with ONE parameter set. It says nothing about sample
        efficiency (that is criterion (a)), nothing about either method's
        asymptotic behaviour beyond `n_evaluated`, and nothing about whether
        the poly estimate would survive reseeding -- `margin` bounds that risk
        from measured spread, it does not eliminate it, and h3_win_robustness.py
        measures 16 of 20 Brownian draws winning at n=1600, not 20 of 20.
        Independent redraws remain the caller's job. What the same script does
        rule out is the R2-F3 failure mode: across k in {6,...,14} and
        max_radius in 3..8 the verdict never changes.
        """
        return self._verdict()[0]

    @property
    def traditional_accuracy_win(self) -> bool:
        """The mirror of criterion (c) for the baseline: (c1), (c3) and (c4)
        with the roles swapped, plus (c5).

        Present so the criterion is visibly two-sided and can be seen to fire
        against the poly-algebraic method, which it does on a uniform 3-cube.
        There is no mirror of (c2): the degeneracy guard answers a failure mode
        specific to the shell-growth estimator (an answer of 1.0 forced by a
        constant shell sequence), and the correlation sum has no analogue --
        see DEFAULT_MAX_SENTINEL_FRACTION.
        """
        poly_err, trad_err = self.poly_algebraic_abs_error, self.traditional_abs_error
        if not math.isfinite(poly_err) or not math.isfinite(trad_err):
            return False
        return (
            trad_err <= self.tolerance
            and poly_err > self.tolerance
            and (poly_err - trad_err) >= self.margin
        )

    @property
    def verdict_reason(self) -> str:
        """Human-readable statement of which criterion (c) condition decided it."""
        return self._verdict()[1]


def compare_accuracy_at_max_n(
    points: list[tuple[float, ...]],
    true_dimension: float,
    max_n: int,
    *,
    k: int,
    tolerance: float,
    margin: float = DEFAULT_ACCURACY_MARGIN,
    max_sentinel_fraction: float = DEFAULT_MAX_SENTINEL_FRACTION,
    max_radius: int = 6,
    samples: int = 40,
    n_radii: int = 20,
) -> AccuracyComparisonResult:
    """Criterion (c): run BOTH estimators once at a fixed point count and compare
    their accuracy against a known `true_dimension`.

    Neither method is required to have converged in the criterion-(a) sense,
    and no convergence search is performed -- see the module docstring and
    `AccuracyComparisonResult.poly_algebraic_accuracy_win` for why this is a
    separate criterion rather than a variant of `compare()`.

    Both methods see the same prefix `points[:n_evaluated]`, where
    `n_evaluated = min(max_n, len(points))`. Exact parity of what each method
    ultimately consumes is reported rather than assumed: `poly_algebraic_n_nodes`
    is the node count of the k-NN graph actually built, which can be smaller
    than `n_evaluated` because near-duplicate points are dropped (`dedupe=True`,
    finding F3) and because `knn_hypergraph` derives its nodes from its edges.
    The correlation sum sees the prefix as given.

    A non-finite estimate from either side is recorded as nan rather than
    raised: this function is meant to run unattended across a benchmark suite,
    and "this method produced no answer here" is a result worth recording. It
    is never a win -- see (c1).

    Raises ValueError only for inputs on which the question is not defined
    (fewer than 3 usable points, non-positive k, negative tolerance/margin, or
    a `max_sentinel_fraction` outside [0, 1]).
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if not 0.0 <= max_sentinel_fraction <= 1.0:
        raise ValueError("max_sentinel_fraction must lie in [0, 1]")
    if max_n < 3:
        raise ValueError("need at least 3 points for an accuracy comparison")

    n_evaluated = min(max_n, len(points))
    if n_evaluated < 3:
        raise ValueError(
            f"need at least 3 points for an accuracy comparison; got {n_evaluated} "
            f"(max_n={max_n}, len(points)={len(points)})"
        )
    prefix = points[:n_evaluated]

    # --- poly-algebraic side: estimate and its own degeneracy diagnostics ---
    # `mean_dimension`, `degenerate_fraction` and `near_degenerate_fraction`
    # sample nodes identically given the same `samples` and `max_radius`, so
    # the fractions describe exactly the node set the estimate was pooled over.
    # They are called separately rather than fused into one pass so that this
    # module cannot drift out of step with dimension.py's sampling rule; the
    # extra work is three BFS passes over `samples` nodes on a cached adjacency,
    # negligible against building the k-NN graph itself.
    try:
        hg = knn_hypergraph(prefix, k=k, dedupe=True)
    except (DuplicatePointsError, ValueError):
        poly_estimate = float("nan")
        poly_degenerate = float("nan")
        poly_near_degenerate = float("nan")
        poly_n_nodes = 0
    else:
        poly_estimate = mean_dimension(hg, samples=samples, max_radius=max_radius)
        poly_degenerate = degenerate_fraction(hg, samples=samples, max_radius=max_radius)
        poly_near_degenerate = near_degenerate_fraction(hg, samples=samples, max_radius=max_radius)
        poly_n_nodes = len(hg.nodes)

    # --- traditional side ---
    try:
        trad = correlation_dimension(prefix, n_radii=n_radii)
        trad_estimate, trad_r_squared = trad.dimension, trad.r_squared
    except ValueError:
        trad_estimate, trad_r_squared = float("nan"), 0.0

    return AccuracyComparisonResult(
        true_dimension=true_dimension,
        tolerance=tolerance,
        margin=margin,
        max_sentinel_fraction=max_sentinel_fraction,
        n_requested=max_n,
        n_evaluated=n_evaluated,
        poly_algebraic_n_nodes=poly_n_nodes,
        poly_algebraic_estimate=poly_estimate,
        traditional_estimate=trad_estimate,
        poly_algebraic_degenerate_fraction=poly_degenerate,
        poly_algebraic_near_degenerate_fraction=poly_near_degenerate,
        traditional_r_squared=trad_r_squared,
    )
