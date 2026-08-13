"""Emergent dimension estimation via ball-volume growth (Pre-Geometric Dimension Variable).

Tier B: this is a well-defined statistic of a graph, not a new physical
claim. If a hypergraph's adjacency graph behaves like a d-dimensional
lattice at some scale, the number of nodes within radius r of a source grows
as |ball(r)| ~ r^d, so d is recoverable from the log-log slope. This is the
standard volume-growth dimension used in causal set theory and in Wolfram's
own work; nothing here is novel except the packaging.

The fit is done on *shell sizes* (dV(r) = |ball(r)| - |ball(r-1)|), not on
cumulative volume directly. This matters: cumulative volume on a lattice is
affine rather than homogeneous (a 1D path has |ball(r)| = 2r+1 from an
interior point, not r^1 -- the "+1" is a real offset, not noise), which
biases a naive log-log fit on volume itself well below the true dimension
at any radius small enough to be computationally tractable. The shell size
of a d-dimensional lattice scales as r^(d-1) with *no* additive offset (a
path's shell is the exact constant 2; a 2D grid's shell is exactly 4r), so
`dimension = shell_slope + 1` converges immediately rather than
asymptotically. Verified against both cases directly, not assumed.

The "dynamic"/"fluid" dimension idea (concept #4 in the brief) is realized
directly: `local_dimension` computes this per-source, so a hypergraph can
have different estimated dimension at different regions, and
`dimension_profile` tracks it over an evolution history.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .core import Hypergraph, Node

# --- Near-constant ("low dynamic range") shell-sequence gate -------------------
#
# See docs/POLY_ALGEBRAIC_BENCHMARK.md findings R2-F4 (§9.5), §10.3 and next
# step N8. R^2 is a *fraction of variance explained*. A shell sequence that is
# nearly -- but not exactly -- constant has almost no variance to explain, so
# R^2 collapses on its own merits even when the fit is excellent and the
# dimension estimate is accurate. N1's fix only caught the *exactly* constant
# case (ss_tot within float noise of zero); the near-constant band sat
# unguarded immediately next to it. Gating on R^2 alone therefore discards
# genuinely good nodes, and -- worse -- cannot rank them: the benchmark found a
# near-perfect (6,5,6,6,6,6) at R^2=0.0889 scoring *below* a wildly oscillating
# garbage sequence (4,8,4,8,4,8) at R^2=0.1027.
#
# The fix is a second, R^2-independent gate on the shell sequence's *relative
# spread* and its *length*. Three conditions must all hold, and they do
# different jobs:
#
#   0. NEAR_CONSTANT_MIN_FIT_LENGTH -- the fit window must span at least this
#      many radii. This is NOT a statistical nicety; it is the condition that
#      makes conditions 1 and 2 mean anything, and leaving it out was a real,
#      measured defect in this gate's first version (see the limitation
#      section below). In a k-NN graph the radius-1 shell IS the node's
#      degree: pinned near k by construction, carrying no dimensional
#      information whatsoever. The dimensional signal only appears once the
#      ball has grown past that degree plateau. Over a 3-radius window the fit
#      is therefore roughly one informative point plus noise, and
#      "near-constant" is indistinguishable from "noisy" in ANY structure,
#      1-dimensional or not -- so conditions 1 and 2 stop being evidence about
#      the geometry and become evidence about the sample size. There is also a
#      plain degrees-of-freedom statement of the same thing: a 2-parameter fit
#      over n points has n-2 residual degrees of freedom, so at n=3 there is
#      exactly one and the residual pattern is essentially unconstrained.
#
#      5 is calibrated, not assumed
#      (scripts/hypergraph_benchmark/round3/n8b_min_fit_length_gate.py sweeps
#      2..6 over 8 point clouds whose true dimension is NOT 1, x k in
#      {6,8,12} x max_radius in {3,4,6}). Measured spurious admissions in
#      non-1D structures: length>=2 (the unguarded original) 44, >=3 44,
#      >=4 6, >=5 1, >=6 1. Measured cost on problem 01's own genuinely-1D
#      ring, the case this whole gate exists for: at max_radius 5 and 6
#      (including the module default) length>=5 keeps *exactly* the same nodes
#      as the unguarded version -- 196/200 at max_radius=6 -- so the anchor
#      case and the default configuration pay nothing at all. 6 buys no
#      further reduction and costs real 1D coverage (32/200 instead of 130/200
#      at max_radius=5), so 5 strictly dominates it. The single admission that
#      survives at 5 is one node of 200 on 2D Brownian at k=12, max_radius=6;
#      no length threshold that preserves the length-6 anchor can remove it.
#
#   1. NEAR_CONSTANT_CV -- the coefficient of variation (population stdev /
#      mean) of the raw shell counts. This identifies the regime: below this,
#      the sequence carries so little dynamic range that a variance-explained
#      ratio has essentially no variance to work with and its value is noise.
#      0.15 is chosen with margin on both sides of the measured data: the
#      worst near-constant case in §9.5's table is (7,5,5,6,6,6) at CV=0.1178,
#      and the oscillating garbage row (4,8,4,8,4,8) sits at CV=0.3333, so the
#      threshold is 1.27x above everything it must accept and 2.2x below the
#      row it must keep rejecting.
#
#   2. NEAR_CONSTANT_SLOPE_BOUND -- a *rigorous* bound on how wrong the
#      accepted dimension can be, which turns condition 1 from a heuristic
#      into a guarantee. If every log(shell) lies in a band of width
#      W = log(max shell / min shell), then for ANY arrangement of the values
#      within that band the least-squares slope obeys
#          |slope| <= W * sum|log x_i - mean log x| / (2 * var(log x)),
#      because the maximizing arrangement puts each y_i at whichever end of
#      the band matches sign(x_i - mean_x). So a near-constant sequence cannot
#      produce a slope far from 0, i.e. cannot produce a dimension far from 1,
#      no matter how its residuals are arranged. Independently re-derived and
#      brute-forced over 20,000 random sequences: the bound was never exceeded
#      and is saturated exactly (ratio 1.000000) by the extremal arrangement
#      at every length 4..7, so it is genuine and tight, not slack.
#
# WHAT THE BOUND DOES AND DOES NOT GUARANTEE -- read this precisely, because
# the first version of this comment got it wrong and the error was not
# academic.
#
#   `slope_bound` bounds |dimension - 1|. It does NOT bound
#   |dimension - true dimension|. It is an *error* bound only where the true
#   local dimension really is ~1.
#
# Accepting only fits with bound <= 0.25 therefore caps |dimension - 1| at
# 0.25 on this branch, and nothing more. Two consequences follow, both
# measured rather than hypothesised:
#
#   (a) Even as a distance-from-1 cap, 0.25 is looser than some callers'
#       tolerances. An exhaustive scan of integer shell sequences finds the
#       gate will admit |dimension - 1| up to 0.2377 (on (3,3,4,4)), which
#       exceeds problem 01's own +/-0.15 acceptance tolerance. On problem 01
#       the realised worst is 0.077, so nothing breaks there -- but this
#       branch's guarantee must not be cited as if it were tighter than the
#       benchmark criterion it feeds.
#
#   (b) Where a structure whose true dimension is NOT ~1 throws up a
#       near-constant window by chance, this branch would admit a node whose
#       dimension is near 1 and therefore badly wrong, and `slope_bound` says
#       nothing about that. This was not hypothetical. Before condition 0
#       existed, 2D Brownian motion (true dimension 2) at k=6, max_radius=3
#       admitted 13 of 200 sampled nodes -- every one respecting slope_bound,
#       with |dimension - 1| <= 0.243, while sitting up to 1.18 away from the
#       truth -- and dragged the sampled mean from 1.832 to 1.738, i.e. AWAY
#       from 2. Across 8 non-1D clouds x k in {6,8,12} x max_radius in
#       {3,4,6} there were 44 such admissions.
#
#       Condition 0 (minimum fit length 5) is the fix, because the damage was
#       entirely a short-window effect rather than a threshold-calibration
#       one: it removes 43 of those 44 admissions, takes the Brownian
#       max_radius=3 case from 13 to 0 and its sampled mean from 1.7381 back
#       to 1.8321, and changes nothing at all about the anchor case or the
#       module default.
#
#       WHAT CONDITION 0 DOES NOT FIX, MEASURED. The residue is NOT a single
#       node. An independent round-3 sweep (fresh point clouds, 10 Brownian
#       seeds, three genuinely torus-filling clouds, k in {6,8,10,12} x
#       max_radius in {3..8}) finds sporadic admissions wherever a length>=5
#       window happens to look flat in a 2D structure: 14 of 141 configs, at
#       k=6/8/10/12 and at max_radius 5 AND 6, on Brownian, on the Lissajous
#       torus projection, and on problem 05's own as-built cloud. Admitted
#       nodes sit up to 1.17 from the truth. Density is low (1-2 nodes per
#       200 sampled) and the usual effect on a sampled mean is <= 0.03.
#
#       THAT IS NOT ALWAYS HARMLESS, because a spurious node's LEVERAGE on a
#       mean is set by how many nodes passed R^2, which at small n is few.
#       Measured consequence on a recorded benchmark number: round-2 problem
#       06 (2D Brownian, k=10, compare()'s default max_radius=6) at n=400 has
#       only 4 of 40 sampled nodes passing R^2. One admitted node with shells
#       (13,17,13,13,17,15), dimension 1.0587 against a truth of 2, moves the
#       sampled mean 1.8434 -> 1.6864, i.e. from inside the +/-0.3 tolerance to
#       outside it, and moves `poly_algebraic_min_n` from 400 to 800 -- the
#       value docs/POLY_ALGEBRAIC_BENCHMARK.md records for that problem. Its
#       window is length 6, the same length as the anchor this gate exists to
#       keep, so no length threshold can remove it without removing the thing
#       the gate is for. Round-2 problem 05 was checked the same way and is
#       unchanged (400 both ways).
#
#       A node-local spread test cannot close this -- the culprit's CV
#       (0.1224) is only 4% above the worst CV the gate is required to accept
#       ((7,5,5,6,6,6) at 0.1178), so tightening NEAR_CONSTANT_CV would be
#       knife-edge. Condition 3 below is the fix.
#
#   3. GRAPH-LEVEL CONSENSUS (`near_constant_consensus`) -- the condition that
#      actually closes (b), and the only one here that is not node-local.
#
#      A near-constant window is evidence for "~1D" only under the premise
#      that the *structure* is a low-dynamic-range ~1D object; on a 2D
#      structure the identical window is a coincidence. That premise is a
#      statement about the whole graph, so it is tested once per graph rather
#      than once per node: the branch may contribute to `mean_dimension` only
#      if the graph's own independent evidence agrees.
#
#        * If any sampled node is CONFIDENT (r_squared >= the caller's
#          threshold -- this includes the exactly-constant/degenerate nodes,
#          whose sentinel r_squared=1.0 and dimension 1.0 are genuine 1D
#          evidence), then the mean of those confident dimensions must lie
#          within NEAR_CONSTANT_CONSENSUS_TOLERANCE of 1.
#        * If NO node is confident, the near-constant branch is the only
#          evidence the graph has. Trusting it then requires it to be the
#          DOMINANT description rather than a handful of flat windows:
#          `near_degenerate_fraction` must reach
#          NEAR_CONSTANT_CONSENSUS_FRACTION.
#
#      MEASURED CALIBRATION (scripts/hypergraph_benchmark/round3/n8d_*.py).
#      The round-2 skeptic proposed this test but calibrated it on a single
#      must-fire example (problem 01's ring at 0.98) against damaged graphs at
#      <= 0.01, and warned that one positive was too narrow a base. Broadening
#      it to all ten round-2 production clouds at their production
#      k/max_radius, at every n in their production n_grid, showed that warning
#      was justified and that a bare fraction threshold does NOT work:
#      1D problems 03/04/07 sit at near_degenerate_fraction 0.025-0.05, which
#      is at or BELOW round-2 problem 06's 0.025 at the very n where it does
#      its damage, while 2D Lorenz reaches 0.60 and 2D Rossler 0.425. The
#      must-fire and must-not-fire sides overlap outright.
#
#      The confident-consensus statistic separates where the fraction cannot,
#      because it asks what the graph's *dimension evidence* says rather than
#      how many flat windows it has:
#        must accept  -- every 1D production graph with confident nodes
#                        (problems 01/02/03/04/07/10, all n): |mean - 1| =
#                        0.0000 exactly, at every single n. On a genuinely 1D
#                        k-NN graph the confident nodes ARE the constant-shell
#                        nodes, which report exactly 1.0.
#        must reject  -- round-2 problem 06 at n=400, the recorded regression:
#                        |mean - 1| = 0.8434.
#      0.25 sits 3.4x below the case it must reject, with the entire must-
#      accept side pinned at 0. It is deliberately the same number as
#      NEAR_CONSTANT_SLOPE_BOUND, and for the same reason rather than by
#      coincidence: 0.25 is the widest deviation from 1 this branch ever
#      claims, so requiring the graph's independent evidence to fall inside
#      the same band is the branch's own guarantee restated at graph scale.
#
#      The fallback fraction is the one place a majority is required, and 0.5
#      is the only regime with margin on both sides:
#        must accept  -- the documented anchor graph (problem 01, n=200, k=6,
#                        max_radius=6): 0.98, no confident node anywhere.
#        must reject  -- 2D Rossler at n=100, also with no confident node:
#                        0.325.
#      0.5 sits 1.96x below the anchor and 1.54x above Rossler. Below ~0.33
#      there is no valid threshold at all. The cost is real and is stated in
#      full: problems 02 and 10 at n=32 have no confident node and fractions
#      of 0.25 and 0.125, i.e. 8 of 32 and 4 of 32 nodes, so the branch no
#      longer fires there. That is the intended reading of the word consensus
#      -- 4 flat windows out of 32 is not "this structure is low-dynamic-
#      range" -- and it restores both problems to the pre-N8
#      `poly_algebraic_min_n` of 64 that docs/POLY_ALGEBRAIC_BENCHMARK.md
#      10.2 records (the node-local gate had silently moved both to 32).
#
#      WHAT THIS FIXES, END TO END. Across all ten round-2 problems at
#      production settings, `poly_algebraic_min_n` under this rule equals the
#      pre-N8 baseline on every one, while the anchor is kept bit-for-bit
#      (196/200 nodes, mean 0.994266). Problem 06 returns from 800 to the
#      recorded 400. Over the 72-configuration non-1D calibration set
#      (grids, discs, Brownian, cubes, Lorenz x k in {6,8,12} x max_radius in
#      {3,4,6}) spurious admissions go 1 -> 0, including the length-6 residue
#      no length threshold could reach.
#
#      NOTE ON CONDITION 0'S STATUS. On every configuration measured, the
#      consensus condition alone also removes all 44 short-window admissions
#      that motivated the min-fit-length gate, and it does so while keeping
#      MORE genuine 1D nodes than the gate allows (164/200 rather than 88/200
#      on problem 01's ring at max_radius=3). Condition 0 is therefore now
#      redundant on the measured set. It is retained deliberately, as an
#      independent node-local floor with its own mechanism argument, because
#      dropping it is a second behaviour change whose only measured benefit is
#      coverage rather than correctness -- see n8d_lengthgate.py for the table
#      that would justify dropping it later.
#
# R^2 itself is left completely untouched -- it is still the honest
# variance-explained number, and sequences with real dynamic range are still
# judged by it alone. This gate only *adds* acceptances in the band where R^2
# is uninformative.
NEAR_CONSTANT_CV = 0.15
NEAR_CONSTANT_SLOPE_BOUND = 0.25
NEAR_CONSTANT_MIN_FIT_LENGTH = 5
NEAR_CONSTANT_CONSENSUS_TOLERANCE = 0.25
NEAR_CONSTANT_CONSENSUS_FRACTION = 0.5

# --- Finite-size saturation guard on the fit window (AR2, round 3) ------------
#
# `max_ball_fraction` is the fraction of the graph a ball may already cover and
# still have its NEXT shell counted as geometry. See `local_dimension`'s
# parameter documentation for the argument; this constant is only the value the
# round-3 AutoResearch measurement settled on for the chaotic-attractor
# configuration, exported so callers and scripts cite one number rather than
# re-deriving it.
#
# THE DEFECT IT ANSWERS. `local_dimension` fits log(shell) vs log(radius) over
# radii 1..max_radius and excludes only radii where the shell is EXACTLY zero
# ("saturated"). Exact zero is the last stage of a process that biases the fit
# long before it arrives. A ball that has already engulfed most of a finite
# sample has nowhere left to grow into, so its next shell is small because the
# sample ran out, not because the geometry says so -- and since that shrinking
# shell sits at the largest radius, i.e. at the high-leverage end of the log-log
# fit, it drags the slope (and hence the dimension) down hard.
#
# This is not hypothetical and it is not a small effect. Measured on the round-2
# Rossler cloud (problem 09, k=15, max_radius=6) with the Theiler window on,
# every sampled node's radius-6 ball covers essentially the WHOLE cloud:
# at n=200 shell sequences read (35,52,45,31,32,4) and (33,50,55,28,31,2),
# summing to 199 of 200 nodes; at n=400, (32,49,88,90,99,41) sums to 399 of 400.
# The trailing 4, 2 and 41 are the sample being exhausted. The resulting
# per-node dimensions are 0.20, 0.00 and 1.40 on a 2.01-dimensional attractor,
# no node reaches r_squared 0.9, and `mean_dimension` returns nan.
#
# The correction is the standard "fit inside the scaling region" discipline of
# correlation-dimension practice (Grassberger & Procaccia 1983; Theiler 1986
# Sec. IV), transposed from radius-in-space to radius-in-graph-hops: fit only
# where the observable is still growing as a power law, not where the finite
# sample has flattened it. It is also the concrete graph-side form of the
# manifold-adaptive idea (Farahmand, Szepesvari & Audibert 2007): the
# neighbourhood the estimate is read off must adapt to the sample actually in
# hand instead of being a fixed max_radius for every n. Note the direction of
# the adaptation -- at large n nothing is truncated at all, so this changes the
# small-n regime and leaves the asymptotic one alone.
#
# 0.5 IS CALIBRATED, NOT ASSUMED
# (scripts/hypergraph_benchmark/round3/ar2_saturation_guard.py section 3 sweeps
# {1.0 (off), 0.75, 0.6, 0.5, 0.35} over all ten round-2 production clouds at
# their production k / max_radius / n_grid / tolerance, with the Theiler window
# both off and on). `poly_algebraic_min_n`, Theiler window on, the setting the
# chaotic problems are measured at:
#
#     frac      01    02    03    04    05    06    07    08    09    10
#     1.0 off   50    64     -     -   400     -     -   800   400    64
#     0.75      50    64     -     -   400     -     -   800   100    64
#     0.6       50    64     -     -   100     -     -   800   100    64
#     0.5       50    64     -     -   100     -     -   800   100    64
#     0.35      50    32     -     -   100     -     -   800   800    32
#
# (03/04/06/07 are dashes in every row including the OFF row -- the Theiler
# window is what breaks them, which AR1 measured and is why that window is
# itself opt-in. Nothing here changes that.)
#
# The reading, with both sides of the bracket stated:
#   * 0.6 and 0.5 are a PLATEAU, identical on all ten problems in both window
#     settings. The value is not perched between two neighbouring measurements.
#   * 0.75 is partially effective: it rescues Rossler but leaves the
#     quasiperiodic torus at 400 (against 100 at 0.5) and, with the window off,
#     only takes Brownian from 400 to 200 rather than to 100. The guard fires
#     too late to reach the shells that are already saturated.
#   * 0.35 is the harmful side and it is harmful in two distinguishable ways.
#     Rossler goes 100 -> 800, i.e. WORSE than not correcting at all, because
#     the truncation has started eating the scaling region itself. And problems
#     02 and 10 go 64 -> 32 -- which is a smaller number and still a regression:
#     32 is the vacuous acceptance the NEAR_CONSTANT_* consensus rule above was
#     built to remove, re-manufactured here by a fit window short enough that
#     R^2 has no residual degrees of freedom left to fail on.
#   * 0.5 additionally has a mechanism argument, which is why it is preferred
#     to 0.6 within the plateau: once a ball holds half the sample, at least
#     half of what its next shell could have reached is already taken, so that
#     shell is at least as much a statement about the remaining budget as about
#     the geometry. It is the largest fraction at which "most of what is left
#     is boundary" is true.
#   * INDEPENDENT GEOMETRIC CHECK, on a structure with an exact answer rather
#     than on the benchmark: on the 15x15 square lattice the centre node's
#     scaling region is exactly radii 1..7 (shell = 4r exactly, then clipped by
#     the patch boundary), and ball(6)/N = 0.3778 while ball(7)/N = 0.5022. Any
#     fraction in (0.3778, 0.5022] cuts precisely at the scaling region's edge,
#     and 0.5 is inside it -- recovering dimension 2.0 and R^2 1.0 exactly,
#     against 1.8132 at R^2 0.9034 uncorrected. See
#     tests/test_hypergraph_dimension_saturation.py, which asserts the bracket
#     rather than the constant.
#
# STRICTLY OPT-IN, for the same reason `pointcloud.theiler_window` is: the
# default 1.0 cannot ever truncate (a ball never exceeds the whole graph), so it
# is the pre-AR2 code path verbatim and every recorded benchmark number
# reproduces bit-for-bit. The guard is a correction for the regime where the
# ball outruns the sample; on the clouds where it does not, turning it on is
# not free -- it shortens fit windows, and a shorter window makes both R^2 and
# the NEAR_CONSTANT_MIN_FIT_LENGTH gate easier to satisfy for the wrong reasons.
SATURATION_BALL_FRACTION = 0.5


@dataclass(frozen=True)
class _FitResult:
    """Internal return type of `_log_log_fit` (see `DimensionEstimate` for the
    public meaning of each flag)."""

    slope: float
    r_squared: float
    degenerate: bool
    near_degenerate: bool = False
    shell_cv: float = 0.0
    slope_bound: float = 0.0


@dataclass(frozen=True)
class DimensionEstimate:
    """Result of a log-log fit of ball volume vs radius.

    `degenerate=True` means the shell sequence being fit was constant (to
    within floating-point noise) -- e.g. a k-NN graph on a smooth closed
    curve, which is an exact circulant ring lattice. In that regime
    `r_squared=1.0` is a *sentinel* meaning "the input had no variation to
    fit," not a measurement of fit quality on a diverse shell sequence. This
    distinction exists because a 10-problem physics benchmark
    (docs/POLY_ALGEBRAIC_BENCHMARK.md, finding F1/N2) found that 6 of 10
    "passes" were exactly this case, misread by the estimator's own API as
    a perfect fit -- `is_well_fit()` alone could not distinguish them.

    `near_degenerate=True` is the same situation one step less extreme: the
    shell sequence is *nearly* constant, so `r_squared` is a real number but
    an uninformative one (finding R2-F4 / next step N8). Unlike `degenerate`,
    `r_squared` is NOT overwritten with a sentinel here -- the honest
    variance-explained value is reported as measured, and the flag records
    that it should not be used as the accept/reject criterion. `slope_bound`
    is the rigorous cap on |dimension - 1| -- NOT on |dimension - truth| --
    implied by the sequence's spread (see the module constants above);
    `shell_cv` is its coefficient of variation. The flag additionally
    requires a fit window of at least NEAR_CONSTANT_MIN_FIT_LENGTH radii,
    because in a shorter window "near-constant" is not distinguishable from
    "noisy" and the branch demonstrably admitted dimension~1 nodes out of
    genuinely 2D structures.

    `near_degenerate` is necessary but NOT sufficient for acceptance. It says
    only that THIS window is flat; whether a flat window is evidence of ~1D
    geometry depends on the structure the node was drawn from, which no single
    node can know. That second, graph-level condition is
    `near_constant_consensus()`, applied by `mean_dimension` over the whole
    sampled set and passed into `is_well_fit`.

    Both flags mean "this node's r_squared is not a measurement of fit
    quality." `is_well_fit` accepts them when the graph consents (the
    dimension value is sound); `is_genuinely_well_fit` rejects them outright
    (the fit-quality number is not evidence).
    """

    source: Node
    radii: tuple[int, ...]
    volumes: tuple[int, ...]
    dimension: float
    r_squared: float
    degenerate: bool = False
    near_degenerate: bool = False
    shell_cv: float = 0.0
    slope_bound: float = 0.0

    def is_well_fit(self, threshold: float = 0.95, *, near_constant_consensus: bool = True) -> bool:
        """Is this node's dimension estimate usable?

        Gates on relative shell spread *in addition to* R^2, not on R^2 alone:
        a near-constant shell sequence over a long enough fit window is
        accepted on the strength of its `slope_bound` guarantee even though
        its R^2 is low, because in that regime R^2 measures the absence of
        variance rather than the presence of misfit. Sequences with real
        dynamic range are still judged purely by R^2 (finding R2-F4 / next
        step N8).

        `slope_bound` guarantees closeness to 1, not closeness to the truth,
        so that acceptance is only as good as the premise that a near-constant
        long window means a genuinely ~1-dimensional neighbourhood. That
        premise is a property of the whole graph, not of one node, so
        `near_constant_consensus` carries the graph's verdict on it: pass the
        result of the module-level `near_constant_consensus()` over all the
        estimates being pooled. `mean_dimension` does this for you.

        The default is True, i.e. "no graph context was supplied, so assume
        the premise holds" -- which is the pre-existing node-local behaviour,
        kept so that examining a single estimate in isolation still works. A
        caller pooling estimates should pass the real value; a caller
        measuring a structure whose dimension is not ~1 should prefer
        `is_genuinely_well_fit`, which refuses this branch outright.
        """
        if self.r_squared >= threshold:
            return True
        return self.near_degenerate and near_constant_consensus

    def is_genuinely_well_fit(
        self, threshold: float = 0.95, *, near_constant_consensus: bool = True
    ) -> bool:
        """Well-fit AND `r_squared` is a real measurement, not a low-information
        artifact of a constant or near-constant shell sequence."""
        return (
            self.is_well_fit(threshold, near_constant_consensus=near_constant_consensus)
            and not self.degenerate
            and not self.near_degenerate
        )


def _log_log_fit(xs: list[float], ys: list[float]) -> _FitResult:
    """Least-squares slope, R^2, and degeneracy flags for log(ys) vs log(xs)."""
    n = len(xs)
    log_x = [math.log(x) for x in xs]
    log_y = [math.log(y) for y in ys]
    mean_x = sum(log_x) / n
    mean_y = sum(log_y) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in log_x)
    if var_x == 0:
        return _FitResult(0.0, 0.0, True)
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in log_y)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_x, log_y, strict=True))
    # ss_tot is exactly 0 for a perfectly constant y-sequence in exact arithmetic,
    # but floating-point rounding in mean_y (sum(log_y) / n) can leave a tiny
    # nonzero residual -- observed as small as ~1e-31 for some fit lengths/values
    # -- that makes a literal `ss_tot > 0` check take the general branch and
    # divide by that residual, collapsing r_squared to ~0 by pure numerical noise
    # even though the underlying fit is exact. A relative-scale tolerance avoids
    # treating that noise as a real residual.
    scale = max(1.0, sum(y * y for y in log_y))
    degenerate = ss_tot <= 1e-24 * scale
    r_squared = 1.0 if degenerate else 1.0 - ss_res / ss_tot

    # Relative spread of the shell counts, and the rigorous cap it puts on
    # |slope| (= |dimension - 1|) for any arrangement of values within the
    # observed band. See the NEAR_CONSTANT_* commentary above for the
    # derivation and for why these two conditions are not redundant.
    mean_y_linear = sum(ys) / n
    variance_y = sum((y - mean_y_linear) ** 2 for y in ys) / n
    shell_cv = math.sqrt(variance_y) / mean_y_linear if mean_y_linear > 0 else float("inf")
    band_width = max(log_y) - min(log_y)
    slope_bound = band_width * sum(abs(x - mean_x) for x in log_x) / (2.0 * var_x)
    # The length condition is listed first because it is a precondition for the
    # other two carrying information at all, not an extra filter on top of them:
    # over a 3-radius window in a k-NN graph, shell(1) is just the degree and a
    # low CV is a statement about the sample rather than about the geometry.
    near_degenerate = (
        not degenerate
        and n >= NEAR_CONSTANT_MIN_FIT_LENGTH
        and shell_cv <= NEAR_CONSTANT_CV
        and slope_bound <= NEAR_CONSTANT_SLOPE_BOUND
    )
    return _FitResult(slope, r_squared, degenerate, near_degenerate, shell_cv, slope_bound)


def near_constant_consensus(
    estimates: Sequence[DimensionEstimate], *, threshold: float = 0.95
) -> bool:
    """Does this graph, as a whole, endorse the near-constant branch?

    The decision the near-constant gate's condition 3 rests on, made once per
    graph over a pooled set of `DimensionEstimate`s (see the NEAR_CONSTANT_*
    commentary at the top of this module for the derivation and the measured
    calibration). A near-constant shell sequence is evidence for "~1D" only if
    the structure it came from is a low-dynamic-range ~1D object, which is a
    claim about the graph rather than about the node.

    Two cases, because "what the rest of the graph says" is only available
    when the rest of the graph says anything at all:

      * Some node is confident (`r_squared >= threshold`, which includes the
        degenerate/constant-shell nodes whose sentinel r_squared is 1.0 and
        whose dimension is exactly 1.0). Then the mean of those confident
        dimensions must be within NEAR_CONSTANT_CONSENSUS_TOLERANCE of 1 --
        the branch may not contradict the graph's own measurement.
      * No node is confident. Then the near-constant nodes are the only
        evidence there is, so they must be the dominant description of the
        graph: their fraction must reach NEAR_CONSTANT_CONSENSUS_FRACTION.

    Returns False for an empty input: no evidence is not consensus.
    """
    estimates = list(estimates)
    if not estimates:
        return False
    confident = [e for e in estimates if e.r_squared >= threshold]
    if confident:
        consensus_dimension = sum(e.dimension for e in confident) / len(confident)
        return abs(consensus_dimension - 1.0) <= NEAR_CONSTANT_CONSENSUS_TOLERANCE
    near_constant = sum(1 for e in estimates if e.near_degenerate)
    return near_constant / len(estimates) >= NEAR_CONSTANT_CONSENSUS_FRACTION


def local_dimension(
    hg: Hypergraph,
    source: Node,
    *,
    max_radius: int = 6,
    min_radius: int = 1,
    max_ball_fraction: float = 1.0,
) -> DimensionEstimate:
    """Estimate the volume-growth dimension around `source`.

    Fits log(shell size) vs log(radius) and reports `slope + 1` as the
    dimension (a d-dimensional lattice has shell size ~ r^(d-1)). Radii past
    saturation (the ball has stopped growing, e.g. the whole connected
    component has been reached, so the shell drops to zero) are excluded --
    a saturated shell has no well-defined log and would otherwise crash or
    silently bias the fit.

    Parameters
    ----------
    max_ball_fraction:
        Finite-size saturation guard (see SATURATION_BALL_FRACTION above for
        the derivation, the measured defect and the calibration). The shell at
        radius `r` is admitted to the fit only if the ball at radius `r - 1`
        covered at most this fraction of the graph's nodes; the first radius
        that fails ends the fit window, exactly as an exactly-zero shell does.

        * ``1.0`` (default) -- OFF. A ball can never exceed the whole graph, so
          no radius is ever excluded and this is the pre-AR2 code path verbatim.
        * any ``0 < f < 1`` -- the guard, with that fraction.

        The comparison is against the whole graph's node count rather than the
        source's connected component, because `local_dimension` deliberately
        does not pay for a full-component BFS. On a graph with several
        components that makes the guard PERMISSIVE (it under-truncates) rather
        than wrong: the fraction is measured against a denominator at least as
        large as the truth, so the guard can only fire later than it ideally
        would, never earlier on geometry that was still genuine.

        Turning this on is a real change of estimator, not a free improvement:
        it shortens the fit window, and a shorter window makes both `r_squared`
        and the near-constant branch's length condition easier to satisfy. Read
        SATURATION_BALL_FRACTION's commentary before enabling it on a cloud
        whose balls do not actually outrun the sample.

    Computes the whole radii=0..max_radius volume sequence in a single
    incremental BFS pass (each radius extends the previous frontier rather
    than recomputing `ball()` from the source each time) -- calling `ball()`
    once per radius, as an earlier version of this function did, redid all
    of radius r-1's work at every step, which compounded with the
    O(hypergraph) cost of an uncached `adjacency()` call into the dominant
    cost of a dimension-vs-n convergence sweep (see `core.py`'s
    `_adjacency_cached`, fixed alongside this).
    """
    # `Hypergraph.nodes` rebuilds its frozenset on every access, so it is taken
    # once and used for both the membership check and the saturation guard's
    # denominator rather than twice.
    nodes = hg.nodes
    if source not in nodes:
        raise ValueError(f"node {source} is not present in this hypergraph")
    if not 0.0 < max_ball_fraction <= 1.0:
        raise ValueError(
            f"max_ball_fraction must lie in (0, 1]; got {max_ball_fraction}. "
            f"1.0 disables the saturation guard."
        )
    n_nodes = len(nodes)
    adj = hg.adjacency()
    volumes = [1]
    visited = {source}
    frontier = {source}
    for _ in range(max_radius):
        next_frontier: set[Node] = set()
        for u in frontier:
            next_frontier |= adj.get(u, set()) - visited
        visited |= next_frontier
        volumes.append(len(visited))
        if not next_frontier:
            volumes.extend([len(visited)] * (max_radius - len(volumes) + 1))
            break
        frontier = next_frontier

    radii = list(range(0, max_radius + 1))
    # shells[i - 1] is the shell size (new nodes) at radius i.
    shells = [volumes[i] - volumes[i - 1] for i in range(1, len(volumes))]

    # `volumes[r - 1]` is the ball at radius r-1, i.e. what had already been
    # consumed BEFORE the shell at radius r was counted. Comparing that (rather
    # than volumes[r]) is the honest test of whether the shell at r had room to
    # grow: a shell is admitted iff at the moment it was measured there was
    # still at least (1 - max_ball_fraction) of the sample left for it to reach.
    ball_budget = max_ball_fraction * n_nodes

    fit_radii, fit_shells = [], []
    for r, shell in zip(radii[1:], shells, strict=True):
        if r < min_radius:
            continue
        if shell <= 0:
            break  # saturated; stop here and beyond
        if volumes[r - 1] > ball_budget:
            break  # finite-size saturation; this shell measures the sample, not the geometry
        fit_radii.append(r)
        fit_shells.append(shell)

    if len(fit_radii) < 2:
        return DimensionEstimate(
            source,
            tuple(radii[1:]),
            tuple(volumes[1:]),
            dimension=0.0,
            r_squared=0.0,
            degenerate=False,
        )

    fit = _log_log_fit([float(r) for r in fit_radii], [float(s) for s in fit_shells])
    return DimensionEstimate(
        source,
        tuple(radii[1:]),
        tuple(volumes[1:]),
        dimension=fit.slope + 1.0,
        r_squared=fit.r_squared,
        degenerate=fit.degenerate,
        near_degenerate=fit.near_degenerate,
        shell_cv=fit.shell_cv,
        slope_bound=fit.slope_bound,
    )


def _sampled_nodes(hg: Hypergraph, samples: int | None) -> list[Node]:
    """The deterministic stride-sampled node subset every aggregator here uses.

    Factored out only so `mean_dimension`, `degenerate_fraction` and
    `near_degenerate_fraction` provably sample the SAME nodes -- comparison.py's
    criterion (c) relies on that, and it was previously three copies of the same
    three lines.
    """
    nodes = sorted(hg.nodes)
    if samples is not None and samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    return nodes


def mean_dimension(
    hg: Hypergraph,
    *,
    samples: int | None = None,
    max_radius: int = 6,
    max_ball_fraction: float = 1.0,
) -> float:
    """Average local dimension over (a sample of) nodes, ignoring poor fits.

    Includes degenerate (constant-shell) fits in the average -- their
    dimension value is still correct, only their r_squared is a sentinel
    (see `DimensionEstimate.degenerate`). Callers who need to distinguish a
    genuine measurement from a degenerate one (e.g. before citing accuracy,
    per docs/POLY_ALGEBRAIC_BENCHMARK.md finding F1) should call
    `local_dimension` directly and check `.degenerate` themselves; this
    convenience wrapper answers "what does the estimator say", not "is that
    answer informative."

    The near-constant branch is admitted here only with the *graph's* consent
    (`near_constant_consensus` over exactly the nodes being pooled), not on
    each node's own say-so -- see the NEAR_CONSTANT_* commentary above. This
    is the only place the decision can honestly be made, because it is the
    only place the whole sample is in hand.

    `max_ball_fraction` is forwarded to `local_dimension` unchanged; 1.0 (the
    default) disables the finite-size saturation guard and reproduces the
    pre-AR2 result exactly.
    """
    nodes = _sampled_nodes(hg, samples)
    estimates = [
        local_dimension(hg, n, max_radius=max_radius, max_ball_fraction=max_ball_fraction)
        for n in nodes
    ]
    consensus = near_constant_consensus(estimates, threshold=0.9)
    well_fit = [
        e.dimension
        for e in estimates
        if e.is_well_fit(threshold=0.9, near_constant_consensus=consensus)
    ]
    if not well_fit:
        return float("nan")
    return sum(well_fit) / len(well_fit)


def degenerate_fraction(
    hg: Hypergraph,
    *,
    samples: int | None = None,
    max_radius: int = 6,
    max_ball_fraction: float = 1.0,
) -> float:
    """Fraction of sampled nodes whose fit was degenerate (constant shell sequence).

    A high value (e.g. all of `exact_periodic` in the physics benchmark) is
    itself informative: it means the r_squared column for this hypergraph is
    largely sentinel-valued, and `mean_dimension`'s apparent "perfect fit"
    should not be cited as evidence of estimator accuracy (finding F1).
    """
    nodes = _sampled_nodes(hg, samples)
    if not nodes:
        return float("nan")
    estimates = [
        local_dimension(hg, n, max_radius=max_radius, max_ball_fraction=max_ball_fraction)
        for n in nodes
    ]
    return sum(1 for e in estimates if e.degenerate) / len(estimates)


def near_degenerate_fraction(
    hg: Hypergraph,
    *,
    samples: int | None = None,
    max_radius: int = 6,
    max_ball_fraction: float = 1.0,
) -> float:
    """Fraction of sampled nodes in the near-constant (low dynamic range) band.

    Companion to `degenerate_fraction` for finding R2-F4 / next step N8: these
    are the nodes whose dimension value is sound (bounded by `slope_bound`)
    but whose `r_squared` is uninformative because the shell sequence has
    almost no variance to explain. A high value means this hypergraph's
    r_squared column should not be cited as fit quality, exactly as a high
    `degenerate_fraction` does.

    This is the per-node candidate count, reported as measured. It is NOT the
    acceptance rule: whether those candidates are actually pooled into
    `mean_dimension` is decided by `near_constant_consensus`, which this
    fraction only feeds in the case where no node is confident.
    """
    nodes = _sampled_nodes(hg, samples)
    if not nodes:
        return float("nan")
    estimates = [
        local_dimension(hg, n, max_radius=max_radius, max_ball_fraction=max_ball_fraction)
        for n in nodes
    ]
    return sum(1 for e in estimates if e.near_degenerate) / len(estimates)


def dimension_profile(
    history: list[Hypergraph], *, samples: int = 8, max_radius: int = 5
) -> list[float]:
    """Mean estimated dimension at each step of an evolution history."""
    return [mean_dimension(hg, samples=samples, max_radius=max_radius) for hg in history]
