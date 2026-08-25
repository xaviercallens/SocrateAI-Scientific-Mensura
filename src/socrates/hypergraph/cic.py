"""MENSURA Certified Interval Criterion (CIC) core.

Tier B. This module is the minimal machinery MENSURA-BENCH v2 needs and does
not have: an estimator that emits ``{verdict, [d_lo, d_hi], certificate}``
instead of a bare point estimate, that chooses its own configuration from the
data, and that is able to say `UNDECIDED`. See `docs/MENSURA_BENCH_V2.md` --
this file implements sections 2.1 (interval output), 2.2 (zero-knob
self-configuration) and 3.1 (the out-of-domain detector), and honours the
measured bias floor of section 6.

It deliberately does NOT touch `comparison.py`: that rewrite is Phase-A work.
Nothing here changes any number already on the record; `dimension.py` and
`baseline.py` are consumed exactly as they stand, through their public API
plus two private helpers (`_sampled_nodes`, `_log_log_fit`) that the existing
diagnostics scripts already import.

THE ONE SUCCESS CRITERION
-------------------------
A `MEASURED` interval that does not contain the declared truth is a
CALIBRATION VIOLATION, and it is the only unforgivable outcome. `UNDECIDED`
never violates and never scores. Owner decision E2 says this first build
optimises for VALIDITY, NOT SCORE: abstain aggressively, and accept that a
valid run may abstain on most rows. Every threshold below is therefore set on
the abstaining side of its measured evidence, and every one of them is
recorded on the certificate so that a later pre-registration can move it
knowingly rather than by accident.

WHAT THE INTERVAL IS, AND WHAT IT IS NOT
----------------------------------------
It is NOT a confidence interval. There is no sampling model, no coverage
probability, and no repeated-experiment guarantee anywhere in this file, and
the word "confidence" is not used for that reason. What it is:

    a BRACKET between two estimators measured to fail in opposite directions
    on the same target, widened by the largest RELATIVE error either estimator
    was measured to make on a target whose dimension is known exactly.

The bracket half is v2 section 2.1's recommendation and is data-backed: on a
2-D uniform support at n=12800 the shell readout gives 2.08-2.20 (high) while
the Grassberger-Procaccia correlation sum gives 1.927 (low) against a truth of
exactly 2.000. Hulling the two is honest by construction and widens exactly
when they disagree.

The widening half exists because that opposite-direction property is an
EMPIRICAL observation on a handful of targets, not a theorem, and it is
already known to fail: the same shell estimator reads -0.053 BELOW truth at
n=800 (v2 section 6). So the hull alone is not safe, and `RHO_BIAS_FLOOR`
below carries the measured systematic error that the hull cannot be relied on
to straddle.

What the interval explicitly does NOT claim:

  * no coverage probability, and no statement about a hypothetical resample;
  * nothing at all about targets outside the estimator's validated domain --
    that is what the abstention detector is for, and when it fires there is no
    interval;
  * no reconciliation of D_0 and D_2. The shell readout tracks D_0 and the
    correlation sum tracks D_2 (v2 section 4). On a target where those differ
    the hull spans the gap, and `READOUT_DIVERGENCE` abstains rather than
    pretending the span is an interval on one number;
  * the calibration evidence behind `RHO_BIAS_FLOOR` is a small number of
    i.i.d.-sampled monofractal supports in 2 and 3 dimensions. It is not
    validated at D >= 4, and v2 section 3.3 already predicts `UNDECIDED`
    there.

ZERO KNOB
---------
`certify()` takes a point cloud and nothing that could change the answer. This
is a correctness requirement, not tidiness: v2 section 2.2 measured one target
moving from -0.39 to +0.99 as `k` alone varied from 10 to 15. A caller
choosing `k` chooses the answer. The selection rule is `select_settings()`,
its output is recorded verbatim on the certificate, and the only arguments
`certify()` accepts besides the data are provenance labels.

SCALE-AWARENESS, AND WHY IT IS A CORRECTNESS PROPERTY
------------------------------------------------------
Dimension is a bi-Lipschitz invariant, so `certify(X)` and `certify(X @ M)`
for an invertible `M` must agree or one of them must abstain. Under plain
Euclidean readouts they did not: v2 section 7.1 records the same uniform
square returning MEASURED [1.5859, 2.2822] and, after rescaling one
coordinate by 0.01, MEASURED [0.6374, 1.3626] -- two DISJOINT measured
intervals for one set, with no signal raised, and reachable by ordinary use
(a Takens delay embedding at lag 1 lands there). Both arms collapsed toward 1
TOGETHER -- the shell arm because the k-NN graph degenerates into a chain
along the long axis, the correlation-sum arm because its radii were fractions
of the raw bounding-box diagonal, which the long axis dominates -- and
`READOUT_DIVERGENCE` fires only when the arms DISAGREE, so it was silent
exactly when they shared the bias.

Both readouts are therefore taken under a converged per-point Mahalanobis
metric estimated from the data (`pointcloud.local_mahalanobis_metric`, and
`baseline.local_mahalanobis_correlation_dimension` for the distance arm).
This is a change to the READOUTS ONLY. Every abstention constant and every
signal rule below is exactly as it was, and that is deliberate: the measured
region grows to ~100:1 anisotropy at n=1600 with any rotation, and beyond
that ceiling -- which is structural, not a tuning shortfall -- the unchanged
gate turns what were silent violations into honest UNDECIDED verdicts.
"""

from __future__ import annotations

import hashlib
import math
import statistics as st
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .baseline import local_mahalanobis_correlation_dimension
from .core import Hypergraph, Node
from .dimension import (
    DimensionEstimate,
    _log_log_fit,
    _sampled_nodes,
    local_dimension,
    near_constant_consensus,
)
from .pointcloud import (
    GLOBAL_RIDGE_FLOOR_FRAC,
    MAX_METRIC_ITERATIONS,
    METRIC_CONVERGENCE_OVERLAP,
    METRIC_RIDGE_FLOOR_FRAC,
    TOPOLOGY_RIDGE_FLOOR_FRAC,
    LocalMetric,
    _bbox_diagonal,
    _duplicate_clusters,
    auto_covariance_window,
    hypergraph_from_neighbours,
    local_mahalanobis_metric,
    mahalanobis_knn_indices,
)

# --------------------------------------------------------------------------
# Contract identity. Bump on ANY change to a constant below or to the
# selection/abstention logic: a certificate is only replayable against the
# contract it was issued under.
#
# cic-1.0 -> cic-1.1: the two readouts became scale-aware. Neighbour selection
# now runs under a converged per-point Mahalanobis metric instead of the raw
# Euclidean one, and the correlation-sum arm measures distances under that
# same metric with an inverse-variance weighted log-log fit instead of radii
# taken as fractions of the raw bounding-box diagonal. NOT ONE ABSTENTION
# CONSTANT OR SIGNAL RULE CHANGED -- see the note above `_detect`.
# --------------------------------------------------------------------------
CIC_CONTRACT_VERSION = "cic-1.1"
DEFAULT_ADAPTER = "local-mahalanobis-knn-shell-growth+local-mahalanobis-correlation-sum"

# --------------------------------------------------------------------------
# MEASURED CONSTANTS. Every one is a number read off a recorded experiment,
# with the citation inline.
#
# PROVENANCE, stated because "measured" is easy to claim and hard to check.
# All of these were fixed BEFORE the known-answer validation was run, and two
# were then revised -- each on a measurement of the DETECTOR's own error rate,
# never on whether a dimension came out near its truth:
#
#   DUPLICATE_FRACTION_MAX     0.01 -> 0.10, after measuring the near-duplicate
#                              detector's FALSE-POSITIVE rate on clean 1-D
#                              clouds (1.75% at n=6400) against its
#                              true-positive rate on the F3 corruption (100%).
#   READOUT_DIVERGENCE_FACTOR  3.0 -> 1.0, after the detector was caught
#                              emitting a CALIBRATION VIOLATION on an i.i.d.
#                              uniform cloud in R^6.
#
# Both are argued at their definitions. No constant here was moved to make a
# particular target's interval contain a particular truth.
# --------------------------------------------------------------------------

# The bias floor, as a RELATIVE error. v2 section 6: on a target of exactly
# known dimension 2.0, with the estimator free to pick any (k, max_radius,
# min_radius) in the ranges tested, the achieved error spans +0.03 to +0.24 --
# 0.24/2.0 = 0.120 relative. Round-3 measured a 3-D uniform cube reaching only
# ~2.47 against truth 3.0 -- 0.53/3.0 = 0.177 relative. RELATIVE rather than
# absolute precisely because those two disagree in absolute terms and agree
# much better in relative terms, which is the v2 observation that the bias
# grows with dimension. The larger of the two, rounded up:
RHO_BIAS_FLOOR = 0.18

# The bias is systematic, k-dependent, and measured NOT to be a boundary or a
# finite-size effect (v2 section 6 refutes all three hypotheses). Do not
# attempt a boundary correction here: interior-only restriction was measured to
# make the error WORSE by a median of +0.046, because boundary nodes are biased
# down, interior nodes up, and the all-node figure is a partial cancellation.

# Fit-quality thresholds. Each is the corresponding estimator's OWN default --
# `dimension.DimensionEstimate.is_well_fit` and
# `baseline.CorrelationDimensionEstimate.is_well_fit` -- not a new choice made
# here. The bracket's two arms are judged by their own authors' criteria.
WELL_FIT_R_SQUARED = 0.95
GP_WELL_FIT_R_SQUARED = 0.90

# Nodes sampled per graph. Matches the recorded diagnostics (E4, F4v used 80),
# so the shell readout here is comparable to the numbers in the v2 document.
SAMPLES = 80

# The k ladder the selector searches. It spans the range over which the recorded
# sweeps measured k-sensitivity (k=8..25 on Cantor dust, k=6..15 on the uniform
# square) and extends ONE RUNG BEYOND EACH END, so that neither the
# "smallest admissible k" rule nor the k-stability test is ever evaluated at a
# ladder boundary -- a rule that always selects the edge of its own search space
# has not selected anything. The upper rung is load-bearing in practice:
# `dimension.NEAR_CONSTANT_CV` = 0.15 requires shell counts large enough for
# their relative fluctuation to fall below it, and a randomly-spaced circle at
# n=1600 was measured to first satisfy that at k=25.
K_LADDER: tuple[int, ...] = (4, 6, 8, 10, 15, 25, 40)

# min_radius. v2 section 6 measured starting the fit window at radius 2 or 3
# instead of 1: at k=6 it makes things markedly worse (+0.05 -> +0.20 -> +0.24),
# at k=10 roughly flat, and only at k=15 does it help. "No single window rule
# improves all k", so the window starts at 1 -- the configuration every recorded
# number was taken under. Fixed, not selected, and recorded as such.
MIN_RADIUS = 1

# Fit-window search range. The floor of 3 is `dimension.MIN_RSQUARED_FIT_LENGTH`
# restated: below it the least-squares line has zero residual degrees of freedom
# and r_squared is 1.0 by algebra (finding N9). The cap of 8 bounds the BFS cost
# and is past the depth at which balls on 2-D and 3-D supports at attainable n
# start eating the sample.
MIN_FIT_RADII = 3
RADIUS_CAP = 8

# Saturation guard. `dimension.SATURATION_BALL_FRACTION`, restated: a shell
# measured after the ball already covered half the sample is measuring the
# sample, not the geometry.
BALL_FRACTION_CAP = 0.5

# --- abstention thresholds ------------------------------------------------

# A k-NN graph whose largest component holds less than this fraction of nodes
# is not one on which graph distance approximates geodesic distance. Measured
# reference point: Cantor dust at n=12800 gave 639 components with the largest
# holding 56-94 nodes, i.e. a largest-component fraction of ~0.007.
CONNECTIVITY_FRACTION = 0.90

# Below this well-fit fraction the reported mean is a mean over a minority
# subpopulation selected by fit quality -- a selection effect, not a
# measurement. Measured reference point: Cantor dust reached as low as 10/80.
WELL_FIT_FRACTION_MIN = 0.50

# `k` instability, as a multiple of the bias floor. If moving the estimator's
# own self-selected knob moves the answer by more than twice its known
# systematic bias, the answer is a property of the knob rather than of the
# data. Measured: the uniform square spans 0.21 across k=6..15; Cantor dust
# spans 1.38 across k=8..25.
K_INSTABILITY_FACTOR = 2.0

# Divergence between the two readouts, as a multiple of the bias floor. The two
# arms must agree to within the largest error either was MEASURED to make on an
# exactly-known target; past that, nothing calibrated on that experience applies
# to this input. Either an estimator is out of its domain, or the target is
# multifractal and the two are tracking D_0 and D_2 (v2 section 4, where the gap
# reaches 1.0 by construction) -- in which case the hull is not an interval on
# one number at all.
#
# MEASURED HERE, and this constant was the one real defect this build found in
# itself. At 3.0 the detector emitted MEASURED [2.61, 5.06] on an i.i.d. uniform
# cloud in R^6 -- truth exactly 6.0, a CALIBRATION VIOLATION -- because the shell
# arm collapses to 3.31 and the correlation sum to 4.37 while the threshold sat
# at 2.07. Their disagreement (1.07) was already far outside anything recorded
# on an exactly-known target. Tightened to 1.0, i.e. "agree to within one bias
# floor or abstain", which abstains from D >= 4 upward -- exactly the outcome v2
# section 3.3 predicts and endorses -- while leaving D <= 3 measured.
READOUT_DIVERGENCE_FACTOR = 1.0

# The chain artifact. A constant shell sequence is the exact signature of a
# ring/chain graph, which is genuinely 1-dimensional -- but it is ALSO what an
# undersampled trajectory produces: measured (round 3, AR1), 100% of k-NN edges
# joined index-adjacent points on a Lorenz cloud at n=100 and 57% at n=400,
# over which range the shell estimator reported 1.01-1.17 on a 2.05-dimensional
# attractor. The correlation sum is built on distances rather than on a
# neighbour RANKING, so it does not share the defect and is used as the
# cross-check.
RING_FRACTION = 0.50
RING_GP_TOLERANCE = 0.25

# Near-duplicate points. `pointcloud.DuplicatePointsError` documents finding F3:
# sampling several periods of a closed orbit puts essentially EVERY point into a
# near-duplicate cluster. Both sides of the threshold were measured here, at
# `pointcloud`'s own calibrated 1e-6 relative tolerance:
#
#   F3 corruption (closed orbit x3, n=1600..6400)   fraction in clusters = 1.000
#   clean 1-D segment, n=6400 (worst clean case)                        = 0.0175
#   clean circle / helix, n=6400                                        <= 0.0088
#   clean 2-D square and Cantor dust, n<=6400                           <= 0.0003
#
# A dense sample of a LOW-dimensional support has genuine near-coincidences at
# that tolerance -- at n=6400 on a segment, 1.75% of points -- so a 1% threshold
# (tried first) abstains on clean 1-D clouds. 10% sits ~6x above the measured
# false-positive rate and 10x below the true-positive one.
DUPLICATE_FRACTION_MAX = 0.10
DUPLICATE_TOLERANCE = 1e-6

# Exactly-constant-shell fraction required for the DEGENERATE-EXACT verdict.
DEGENERATE_EXACT_FRACTION = 0.90


class Verdict:
    """The three CIC verdicts. Plain strings so a certificate round-trips
    through JSON without a codec."""

    MEASURED = "MEASURED"
    DEGENERATE_EXACT = "DEGENERATE-EXACT"
    UNDECIDED = "UNDECIDED"


class Signal:
    """Abstention signals. Any one of them forces `UNDECIDED`.

    Three were already visible in the v2 diagnostic data (`NON_POSITIVE`,
    `K_INSTABILITY`, `FRAGMENTATION`); the rest are added here, each with the
    measured failure it is derived from.
    """

    # --- the three v2 section 3.1 names ---
    NON_POSITIVE = "NON_POSITIVE"  # a dimension estimate <= 0
    K_INSTABILITY = "K_INSTABILITY"  # answer moves with the self-selected k
    FRAGMENTATION = "FRAGMENTATION"  # k-NN graph is not one connected object

    # --- added here ---
    READOUT_DIVERGENCE = "READOUT_DIVERGENCE"  # shell and GP disagree past the bracket
    CHAIN_ARTIFACT = "CHAIN_ARTIFACT"  # ring signature contradicted by GP
    LOW_WELL_FIT = "LOW_WELL_FIT"  # the mean is over a selected minority
    UNDERDETERMINED_WINDOW = "UNDERDETERMINED_WINDOW"  # r_squared true by algebra
    NO_SCALING_REGION = "NO_SCALING_REGION"  # the power-law premise holds at no window
    BALL_SATURATION = "BALL_SATURATION"  # no fit window before the ball eats the sample
    GP_UNRELIABLE = "GP_UNRELIABLE"  # the bracket's other arm did not measure
    INSUFFICIENT_ENSEMBLE = "INSUFFICIENT_ENSEMBLE"  # < 2 admissible k, so no stability test
    NEAR_DUPLICATE_POINTS = "NEAR_DUPLICATE_POINTS"  # finding F3 corruption
    TOO_FEW_POINTS = "TOO_FEW_POINTS"


@dataclass(frozen=True)
class SelectedSettings:
    """What the method chose for itself. A caller may not supply any of it.

    The metric block is as much a setting as `k` is: which points count as
    neighbours is chosen from the data, so the certificate has to say under
    what metric, from how big a covariance window, after how many refinement
    rounds, and at which ridge floors -- otherwise a replay cannot tell
    whether it reproduced the measurement or merely resembled it.
    """

    k: int
    max_radius: int
    min_radius: int
    samples: int
    well_fit_r_squared: float
    admissible_k: tuple[int, ...]
    k_ladder: tuple[int, ...]
    selection_rule: str
    # --- the scale-aware metric, self-selected and recorded (v2 section 8.4) ---
    metric: str = "euclidean"
    covariance_window_k0: int = 0
    metric_iterations: int = 0
    metric_converged: bool = False
    metric_final_overlap: float = 0.0
    topology_ridge_floor: float = 0.0
    metric_ridge_floor: float = 0.0
    global_ridge_floor: float = 0.0


@dataclass(frozen=True)
class Certificate:
    """Everything needed to say what was measured and to measure it again."""

    inputs_hash: str
    adapter: str
    contract_version: str
    settings: SelectedSettings
    replay_command: str
    label: str | None = None
    n_points_input: int = 0
    constants: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Diagnostics:
    """Mandatory, recorded, and never a gate.

    Under v2 everything here -- points-to-converge, compute cost, robustness --
    survives as a diagnostic and wins nothing. Recording it is what lets a
    later analysis ask why a row abstained without re-running it.
    """

    # what was actually measured on
    n_points_used: int
    n_duplicates_dropped: int
    duplicate_fraction: float
    n_nodes: int
    n_sampled_nodes: int

    # the two readouts and the bracket they span
    shell_readout: float
    shell_readout_median: float
    gp_readout: float
    gp_r_squared: float
    hull_lo: float
    hull_hi: float
    allowance: float

    # robustness of the self-selection (the K_INSTABILITY evidence)
    k_ensemble: tuple[tuple[int, float], ...]
    k_spread: float

    # graph structure (the FRAGMENTATION evidence)
    component_counts: tuple[tuple[int, int], ...]
    largest_component_fraction: float
    n_components: int

    # fit quality
    n_well_fit: int
    well_fit_fraction: float
    median_r_squared: float
    median_fit_window: float
    degenerate_fraction: float
    near_degenerate_fraction: float
    mean_slope_bound: float
    near_constant_consensus: bool
    median_ball_fraction: float
    # (radius, well-fit fraction, median ball fraction) for every window the
    # selector examined at the chosen k -- the evidence behind `max_radius`.
    window_trace: tuple[tuple[int, float, float], ...]

    # convergence proxies (recorded, never scored)
    shell_readout_half_sample: float
    half_sample_shift: float

    # cost (recorded, never scored)
    n_graphs_built: int
    n_ball_expansions: int
    n_pair_counts: int
    wall_time_s: float

    # optional provenance-only statistic; see the module docstring on why the
    # Theiler window is NOT auto-applied
    temporal_adjacency_fraction: float | None = None

    # --- the scale-aware metric, recorded, never a gate ---------------------
    # How anisotropic the local metric ended up, and how hard it had to work to
    # get there. `metric_final_overlap` below 1.0 with `metric_iterations` at
    # MAX_METRIC_ITERATIONS means the refinement never settled, which is worth
    # reading on any surprising row -- but it is NOT a signal, because no
    # threshold on it has been measured.
    metric_global_cov_cond: float = float("nan")
    metric_median_topology_cond: float = float("nan")
    metric_median_cond: float = float("nan")
    metric_max_cond: float = float("nan")
    metric_round_overlaps: tuple[float, ...] = ()

    # --- what the weighted correlation-sum fit leaned on --------------------
    # The weights are the observed pair counts, which span five orders of
    # magnitude across the radius grid, so "20 radii were fitted" is not a
    # useful statement on its own. `gp_effective_n_radii` is Kish's effective
    # sample size over those weights and is the honest count.
    gp_ref_scale: float = float("nan")
    gp_n_radii_used: int = 0
    gp_effective_n_radii: float = float("nan")
    gp_weighted_mean_neighbors: float = float("nan")
    gp_min_pair_count: float = float("nan")
    gp_max_pair_count: float = float("nan")


@dataclass(frozen=True)
class CICResult:
    """The CIC output schema.

    `d_lo`/`d_hi` are `None` exactly when `verdict == UNDECIDED`.
    """

    verdict: str
    d_lo: float | None
    d_hi: float | None
    certificate: Certificate
    diagnostics: Diagnostics
    signals: tuple[str, ...]
    interval_basis: str
    notes: str

    @property
    def width(self) -> float | None:
        if self.d_lo is None or self.d_hi is None:
            return None
        return self.d_hi - self.d_lo

    def contains(self, truth: float) -> bool | None:
        """Does the emitted interval contain `truth`? `None` if UNDECIDED.

        The whole criterion in one method: `False` here on a `MEASURED` row is
        a calibration violation.
        """
        if self.d_lo is None or self.d_hi is None:
            return None
        return self.d_lo <= truth <= self.d_hi

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict. Used for the incremental on-disk record (E6)."""
        out = asdict(self)
        out["width"] = self.width
        return _jsonable(out)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------
# (c) ZERO-KNOB SELECTION
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _GraphAtK:
    k: int
    hg: Hypergraph
    n_components: int
    largest_fraction: float


def _components(hg: Hypergraph, n_nodes: int) -> tuple[int, float]:
    """(component count, largest-component fraction) of an undirected graph.

    Via `scipy.sparse.csgraph` rather than a Python BFS: this runs once per
    ladder rung, and on the fragmented graphs it exists to detect there are
    hundreds of components to walk.
    """
    edges = list(hg.edges)
    if not edges:
        return n_nodes, (1.0 / n_nodes if n_nodes else 0.0)
    nodes = sorted(hg.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    rows = np.array([index[e[0]] for e in edges] + [index[e[1]] for e in edges])
    cols = np.array([index[e[1]] for e in edges] + [index[e[0]] for e in edges])
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(nodes), len(nodes))
    )
    n_comp, labels = connected_components(adjacency, directed=False)
    # Isolated points are nodes of the point cloud with no edge at all; they are
    # absent from `hg.nodes` (a Hypergraph is its edge set), so count them.
    isolated = n_nodes - len(nodes)
    sizes = np.bincount(labels)
    return int(n_comp) + isolated, float(sizes.max()) / n_nodes if n_nodes else 0.0


def build_ladder(arr: np.ndarray) -> tuple[tuple[_GraphAtK, ...], LocalMetric | None]:
    """Build the proximity graph at every rung of `K_LADDER`, SCALE-AWARELY,
    and measure its components. Returns `(rungs, metric)`.

    Reads only structure. No dimension estimate is computed here, so this half
    of the selection provably cannot select for an answer.

    THE ONE CHANGE FROM cic-1.0, and everything that follows from it: the
    edges come from `pointcloud.mahalanobis_knn_indices` under the converged
    per-point metric, not from a Euclidean k-NN query. The ladder is then
    prefixes of ONE neighbour query rather than one query per rung -- correct
    because `mahalanobis_knn_indices` returns each row ascending by distance,
    so its first `k` entries are exactly that point's `k` nearest, and it
    keeps the O(n^2 d^2) search to a single pass.

    The query is deliberately made FRESH under `sigma_inv_topology` rather
    than reusing the refinement's last neighbour sets: those were selected by
    the metric one round earlier, and at convergence "nearly the same" is not
    "the same". That gap was measured to decide rung admissibility on a
    Lorenz cloud (stale sets: no admissible rung anywhere on the ladder;
    fresh query under the reported metric: k=25 admissible). The metric a
    certificate reports must be the metric that selected the edges.
    """
    n = len(arr)
    if n < 4:
        return (), None
    metric = local_mahalanobis_metric(arr, k0=auto_covariance_window(n, arr.shape[1]))
    query_len = min(max(max(K_LADDER), metric.k0), n - 1)
    neighbours = mahalanobis_knn_indices(arr, metric.sigma_inv_topology, query_len)
    rungs: list[_GraphAtK] = []
    for k in K_LADDER:
        if k >= n or k > query_len:
            continue
        hg = hypergraph_from_neighbours(neighbours[:, :k])
        n_comp, largest = _components(hg, n)
        rungs.append(_GraphAtK(k, hg, n_comp, largest))
    return tuple(rungs), metric


def select_max_radius(
    at_cap: Sequence[DimensionEstimate], n_nodes: int
) -> tuple[int | None, tuple[tuple[int, float, float], ...]]:
    """Choose the fit window's upper end: the LONGEST window the estimator accepts.

    THE RULE. A window `[1, r]`, `r` in `[MIN_FIT_RADII, RADIUS_CAP]`, is
    admissible iff both:

      * SATURATION -- the median ball at radius `r-1` over sampled nodes covers
        no more than `BALL_FRACTION_CAP` of the graph. This is
        `dimension.SATURATION_BALL_FRACTION`'s comparison, made once for the
        graph so the window is a recorded setting rather than a per-node
        accident. A shell counted after the ball has already eaten half the
        sample measures the sample, not the geometry.
      * ACCEPTANCE -- at least `WELL_FIT_FRACTION_MIN` of the sampled nodes are
        `is_well_fit` on that window, with the graph-level
        `near_constant_consensus` supplied exactly as `mean_dimension` supplies
        it. The estimator's whole premise is that shell size is a power of
        radius; the window is the range over which that premise is observably
        true for the median node.

    Take the LONGEST admissible window. Longest rather than best-scoring: a
    shorter window fits a line more easily for trivial reasons (at
    `MIN_FIT_RADII` there is exactly one residual degree of freedom), so
    maximising fit quality over window length walks straight into the
    `underdetermined` failure of finding N9.

    WHY ACCEPTANCE IS `is_well_fit` AND NOT RAW R^2, which was the first thing
    tried and is wrong: on a ring graph the shell sequence is (near-)constant,
    so its total variance is ~0 and R^2 is low BY CONSTRUCTION even though the
    fit is perfect. Gating on R^2 alone rejects every window on a circle and
    abstains on a target the estimator handles exactly. `is_well_fit` is the
    rule `dimension.py` already derived for precisely this, near-constant
    branch and all, so reusing it introduces no new constant and no new
    calibration.

    WHY SELECTING ON THE DATA IS ADMISSIBLE HERE, since that is exactly what v2
    section 2.2 warns about: fit quality measures how straight the log-log
    relation is and is invariant to its SLOPE -- a window fitting dimension 1.2
    perfectly and one fitting 2.8 perfectly score identically -- so the rule
    cannot pull the estimate toward any particular value. The one place that
    invariance leaks is a constant shell sequence, which scores perfectly and
    does encode dimension ~1. That leak is real, and it is why `CHAIN_ARTIFACT`
    exists as a separate abstention signal rather than being folded in here.

    Returns `(max_radius or None, per-radius (r, well-fit fraction, median ball
    fraction) trace)`. `None` means no window on this graph satisfies the
    estimator's own premise -- there is no scaling region, which makes the rung
    inadmissible.
    """
    trace: list[tuple[int, float, float]] = []
    if not at_cap or n_nodes <= 0:
        return None, ()
    budget = BALL_FRACTION_CAP * n_nodes
    best: int | None = None
    for r in range(MIN_FIT_RADII, RADIUS_CAP + 1):
        # `estimate.volumes[i]` is the ball at radius i+1, so the ball at radius
        # r-1 is volumes[r - 2] for r >= 2.
        prev = [float(e.volumes[min(r - 2, len(e.volumes) - 1)]) for e in at_cap]
        ball_fraction = st.median(prev) / n_nodes
        refits = [_refit(e, r) for e in at_cap]
        consensus = near_constant_consensus(refits, threshold=WELL_FIT_R_SQUARED)
        accepted = sum(
            1
            for e in refits
            if e.is_well_fit(WELL_FIT_R_SQUARED, near_constant_consensus=consensus)
        )
        fraction = accepted / len(refits)
        trace.append((r, fraction, ball_fraction))
        if st.median(prev) > budget:
            break
        if fraction >= WELL_FIT_FRACTION_MIN:
            best = r
    return best, tuple(trace)


_SELECTION_RULE = (
    "k = smallest ADMISSIBLE rung of K_LADDER, where a rung is admissible iff "
    f"(i) its k-NN graph's largest connected component covers >= {CONNECTIVITY_FRACTION} "
    "of the points -- connectivity is the precondition for graph distance to track "
    "geodesic distance -- and (ii) it has a scaling region, i.e. some window of at "
    f"least {MIN_FIT_RADII} radii on which >= {WELL_FIT_FRACTION_MIN} of sampled nodes "
    "are is_well_fit (dimension.py's own gate, near-constant branch included, with "
    "graph consensus). SMALLEST, because v2 section 6 measured the shell bias growing "
    "with k (+0.004 at k=6 to +0.21 at k=15). max_radius = the LONGEST such window that "
    f"also keeps median ball(r-1) <= {BALL_FRACTION_CAP} of the graph. min_radius fixed "
    "at 1: v2 section 6 measured no window rule that improves all k, so the window "
    "starts where every recorded number started. NEIGHBOURS are selected under a "
    "per-point Mahalanobis metric, itself selected from the data: covariance window "
    "k0 = auto_covariance_window(n, d), bootstrapped from the whole-cloud covariance "
    "(never from a Euclidean neighbourhood, which under anisotropy is already the "
    "artifact being corrected), then refined by re-selection until consecutive rounds "
    f"agree on >= {METRIC_CONVERGENCE_OVERLAP} of every neighbour set or "
    f"{MAX_METRIC_ITERATIONS} rounds have run."
)


def _settings_from(
    admissible: Sequence[int],
    max_radius: int,
    metric: LocalMetric | None,
) -> SelectedSettings:
    """Assemble the recorded settings. One function so `select_settings()` and
    `certify()` cannot drift apart -- a test asserts they agree exactly."""
    return SelectedSettings(
        k=admissible[0] if admissible else 0,
        max_radius=max_radius,
        min_radius=MIN_RADIUS,
        samples=SAMPLES,
        well_fit_r_squared=WELL_FIT_R_SQUARED,
        admissible_k=tuple(admissible),
        k_ladder=K_LADDER,
        selection_rule=_SELECTION_RULE,
        metric="local-mahalanobis" if metric is not None else "euclidean",
        covariance_window_k0=metric.k0 if metric is not None else 0,
        metric_iterations=metric.iterations if metric is not None else 0,
        metric_converged=metric.converged if metric is not None else False,
        metric_final_overlap=metric.final_overlap if metric is not None else 0.0,
        topology_ridge_floor=TOPOLOGY_RIDGE_FLOOR_FRAC if metric is not None else 0.0,
        metric_ridge_floor=METRIC_RIDGE_FLOOR_FRAC if metric is not None else 0.0,
        global_ridge_floor=GLOBAL_RIDGE_FLOOR_FRAC if metric is not None else 0.0,
    )


def select_settings(points: Sequence[Sequence[float]] | np.ndarray) -> SelectedSettings:
    """The whole zero-knob choice, exposed on its own so it can be surveyed.

    `certify()` runs exactly this logic. It is separate only so that "what does
    the method pick on input X" is answerable on its own.
    """
    arr = _as_array(points)
    arr, _, _ = _dedupe(arr)
    rungs, metric = build_ladder(arr) if len(arr) > min(K_LADDER) else ((), None)
    arms, _ = _shell_arms(rungs)
    admissible = tuple(sorted(arms))
    k = admissible[0] if admissible else 0
    return _settings_from(admissible, arms[k].max_radius if k else 0, metric)


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _as_array(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"points must be a 2-D array of shape (n, d); got {arr.shape}")
    return arr


def _hash_points(arr: np.ndarray) -> str:
    """Content hash of the exact float64 bytes actually measured on, plus shape."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return "sha256:" + h.hexdigest()


def _dedupe(arr: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Drop near-duplicate clusters once, up front.

    Doing it here rather than per-`knn_hypergraph`-call matters: the dedupe
    renumbers nodes, so letting each ladder rung dedupe independently would
    give the rungs different node sets and make the k-ensemble incomparable.
    The count is returned so a heavy-duplicate cloud (finding F3) can abstain
    rather than be silently repaired.
    """
    n = len(arr)
    if n < 2:
        return arr, 0, 0.0
    diag = _bbox_diagonal(arr)
    threshold = DUPLICATE_TOLERANCE * diag if diag > 0 else DUPLICATE_TOLERANCE
    clusters = _duplicate_clusters(cKDTree(arr), n, threshold)
    if not clusters:
        return arr, 0, 0.0
    drop: set[int] = set()
    touched = 0
    for cluster in clusters:
        touched += len(cluster)
        drop.update(cluster[1:])
    keep = [i for i in range(n) if i not in drop]
    return arr[keep], len(drop), touched / n


def _refit(estimate: DimensionEstimate, max_radius: int) -> DimensionEstimate:
    """Re-run the log-log fit of an estimate taken at RADIUS_CAP on a shorter window.

    Exactly reproduces `local_dimension(hg, node, max_radius=max_radius)` --
    asserted against it in the test suite -- but reads the ball volumes that
    were already computed instead of paying for a second BFS. The saturation
    guard is left at its default (off), so this is `dimension.py`'s recorded
    code path with a different window, not a different estimator.
    """
    volumes = [1] + list(estimate.volumes)
    shells = [volumes[i] - volumes[i - 1] for i in range(1, len(volumes))]
    fit_radii: list[float] = []
    fit_shells: list[float] = []
    for r, shell in zip(range(1, len(volumes)), shells, strict=True):
        if r > max_radius:
            break
        if r < MIN_RADIUS:
            continue
        if shell <= 0:
            break
        fit_radii.append(float(r))
        fit_shells.append(float(shell))

    radii = tuple(range(1, max_radius + 1))
    kept = tuple(volumes[1 : max_radius + 1])
    if len(fit_radii) < 2:
        return DimensionEstimate(
            estimate.source, radii, kept, dimension=0.0, r_squared=0.0, underdetermined=True
        )
    fit = _log_log_fit(fit_radii, fit_shells)
    return DimensionEstimate(
        estimate.source,
        radii,
        kept,
        dimension=fit.slope + 1.0,
        r_squared=fit.r_squared,
        degenerate=fit.degenerate,
        near_degenerate=fit.near_degenerate,
        shell_cv=fit.shell_cv,
        slope_bound=fit.slope_bound,
        underdetermined=fit.underdetermined,
    )


@dataclass(frozen=True)
class _ShellReadout:
    """The shell arm at one `k`, with everything the detector needs from it."""

    k: int
    max_radius: int
    readout: float
    readout_median: float
    readout_half: float
    estimates: tuple[DimensionEstimate, ...]
    well_fit: tuple[float, ...]
    consensus: bool
    n_sampled: int
    median_ball_fraction: float
    window_trace: tuple[tuple[int, float, float], ...]

    @property
    def well_fit_fraction(self) -> float:
        return len(self.well_fit) / self.n_sampled if self.n_sampled else 0.0

    @property
    def degenerate_fraction(self) -> float:
        if not self.estimates:
            return 0.0
        return sum(1 for e in self.estimates if e.degenerate) / len(self.estimates)

    @property
    def near_degenerate_fraction(self) -> float:
        if not self.estimates:
            return 0.0
        return sum(1 for e in self.estimates if e.near_degenerate) / len(self.estimates)

    @property
    def underdetermined_fraction(self) -> float:
        if not self.estimates:
            return 1.0
        return sum(1 for e in self.estimates if e.underdetermined) / len(self.estimates)

    @property
    def mean_slope_bound(self) -> float:
        ring = [e.slope_bound for e in self.estimates if e.degenerate or e.near_degenerate]
        return st.mean(ring) if ring else 0.0

    @property
    def median_r_squared(self) -> float:
        return st.median([e.r_squared for e in self.estimates]) if self.estimates else 0.0

    @property
    def median_fit_window(self) -> float:
        windows = []
        for e in self.estimates:
            volumes = [1] + list(e.volumes)
            count = 0
            for i in range(1, len(volumes)):
                if volumes[i] - volumes[i - 1] <= 0:
                    break
                count += 1
            windows.append(count)
        return st.median(windows) if windows else 0.0


_WindowTrace = tuple[tuple[int, float, float], ...]


def _shell_arms(
    rungs: Sequence[_GraphAtK],
) -> tuple[dict[int, _ShellReadout], dict[int, _WindowTrace]]:
    """The shell readout at every ADMISSIBLE rung, keyed by `k`, plus every trace.

    A rung is admissible iff its graph is connected enough AND it has a scaling
    region. Inadmissible rungs are simply absent from the readouts -- they
    contribute neither an estimate nor a vote to the stability test, because a
    number taken outside the estimator's premise is not evidence about that
    number's stability. Their window traces are kept regardless, so a run that
    abstains can still say WHY every rung was rejected.
    """
    arms: dict[int, _ShellReadout] = {}
    traces: dict[int, _WindowTrace] = {}
    for rung in rungs:
        if rung.largest_fraction < CONNECTIVITY_FRACTION:
            continue
        arm, trace = _shell_arm(rung.hg, rung.k)
        traces[rung.k] = trace
        if arm is not None:
            arms[rung.k] = arm
    return arms, traces


def _shell_arm(hg: Hypergraph, k: int) -> tuple[_ShellReadout | None, _WindowTrace]:
    """The shell readout at one `k`, on that graph's own self-selected window."""
    n_nodes = len(hg.nodes)
    nodes: list[Node] = _sampled_nodes(hg, SAMPLES)
    if not nodes:
        return None, ()
    at_cap = [
        local_dimension(hg, node, max_radius=RADIUS_CAP, min_radius=MIN_RADIUS) for node in nodes
    ]
    max_radius, trace = select_max_radius(at_cap, n_nodes)
    if max_radius is None:
        return None, trace
    estimates = tuple(_refit(e, max_radius) for e in at_cap)
    consensus = near_constant_consensus(estimates, threshold=WELL_FIT_R_SQUARED)
    well_fit = tuple(
        e.dimension
        for e in estimates
        if e.is_well_fit(WELL_FIT_R_SQUARED, near_constant_consensus=consensus)
    )
    half = tuple(
        e.dimension
        for e in estimates[::2]
        if e.is_well_fit(WELL_FIT_R_SQUARED, near_constant_consensus=consensus)
    )
    ball_fractions = [e.volumes[-1] / n_nodes for e in estimates if e.volumes]
    return _ShellReadout(
        k=k,
        max_radius=max_radius,
        readout=st.mean(well_fit) if well_fit else float("nan"),
        readout_median=st.median(well_fit) if well_fit else float("nan"),
        readout_half=st.mean(half) if half else float("nan"),
        estimates=estimates,
        well_fit=well_fit,
        consensus=consensus,
        n_sampled=len(nodes),
        median_ball_fraction=st.median(ball_fractions) if ball_fractions else 0.0,
        window_trace=trace,
    ), trace


def _temporal_adjacency_fraction(hg: Hypergraph, times: np.ndarray | None) -> float | None:
    """Fraction of k-NN edges joining trajectory-adjacent samples.

    RECORDED ONLY, never a gate, and that is a decision the data forces.
    `pointcloud`'s module docstring measures both sides of this statistic: it
    must fire on an undersampled Lorenz cloud (100% at n=100) and must NOT fire
    on a singly-sampled closed orbit, where temporal neighbours genuinely ARE
    spatial neighbours -- and the discriminating inflation factor separates
    those two sides by only 1.55x versus 2.39x, too narrow to promote into an
    automatic rule. The dangerous half of that ambiguity is caught instead by
    `CHAIN_ARTIFACT`, which asks the correlation sum -- an estimator built on
    distances rather than on a neighbour ranking, so not subject to the defect.
    """
    if times is None:
        return None
    edges = list(hg.edges)
    if not edges:
        return None
    adjacent = sum(1 for a, b in edges if abs(int(times[a]) - int(times[b])) <= 1)
    return adjacent / len(edges)


# --------------------------------------------------------------------------
# (b) THE INTERVAL BUILDER and (d) THE ABSTENTION DETECTOR
# --------------------------------------------------------------------------


def build_interval(
    shell: float,
    gp: float,
    *,
    k_spread: float,
    ring_bound: float | None = None,
) -> tuple[float, float, float, float, float, str]:
    """(d_lo, d_hi, hull_lo, hull_hi, allowance, basis).

    THE CONSTRUCTION, stated precisely.

      1. HULL. `[min(shell, gp), max(shell, gp)]`. v2 section 2.1: the two
         readouts were measured to fail in opposite directions on the same
         target (shell 2.08-2.20 high, correlation sum 1.927 low, truth
         2.000). The hull of two opposite-signed errors straddles the truth
         whenever that property holds, and widens exactly when the two
         disagree -- it is not an error bar bolted onto a point estimate.

      2. RING BOUND, when the shell sequence is (near-)constant. Then
         `DimensionEstimate.slope_bound` is a RIGOROUS cap on |dimension - 1|
         implied by the observed shell band: no arrangement of values within
         that band gives a slope outside it. Averaging the per-node bounds
         bounds their mean, so `[1 - B, 1 + B]` is a derived bound rather than
         an empirical allowance, and it joins the hull.

      3. ALLOWANCE. The hull's straddling property is EMPIRICAL and is already
         known to fail -- the same shell estimator reads 0.053 BELOW truth at
         n=800.

         MEASURED HERE, and this is the finding a pre-registration most needs:
         across the 45 MEASURED rows of `cic_known_answer.json`, the raw hull
         contains the truth on only 31. It holds on every 1-D row, on 4/6
         uniform-square rows, on 3/6 swiss-roll rows, and on 0/6 uniform-cube
         rows -- where BOTH arms read low together (shell ~2.5, correlation sum
         ~2.9, truth 3.0). So under zero-knob selection the two arms are NOT
         reliably on opposite sides, and on roughly a third of rows the
         interval is carried entirely by the allowance below rather than by the
         bracket v2 section 2.1 anticipated. The bracket is real but it is not
         sufficient, and an interval built from the hull alone would have
         produced 14 calibration violations.

         So the hull is widened by

             eps = max(RHO_BIAS_FLOOR * max(d_mid, 1), 0.5 * k_spread)

         The first term is the measured systematic error on exactly-known
         targets (see `RHO_BIAS_FLOOR`), scaled by the estimate because the
         bias was measured to grow with dimension. The `max(., 1)` floor keeps
         the allowance from vanishing as the estimate approaches 0, where the
         relative form carries no information. The second term makes residual
         self-selection sensitivity that passed the abstention gate widen the
         interval instead of being discarded.

      4. CLAMP at 0. A box-counting dimension is non-negative; that is a fact
         about the quantity, not about this estimator.

    What this is NOT: a confidence interval. No coverage probability is
    claimed or computed. See the module docstring.
    """
    hull_lo = min(shell, gp)
    hull_hi = max(shell, gp)
    basis = "bracket(shell,gp)"
    if ring_bound is not None:
        hull_lo = min(hull_lo, 1.0 - ring_bound)
        hull_hi = max(hull_hi, 1.0 + ring_bound)
        basis = "bracket(shell,gp)+slope-bound"
    d_mid = 0.5 * (hull_lo + hull_hi)
    allowance = max(RHO_BIAS_FLOOR * max(d_mid, 1.0), 0.5 * k_spread)
    return max(0.0, hull_lo - allowance), hull_hi + allowance, hull_lo, hull_hi, allowance, basis


def _detect(
    *,
    selected: _ShellReadout | None,
    ensemble: Sequence[_ShellReadout],
    gp_dimension: float,
    gp_r_squared: float,
    k_chosen: int | None,
    largest_fraction: float,
    duplicate_fraction: float,
    n_points: int,
    admissible: Sequence[int],
    saturated_everywhere: bool,
) -> list[str]:
    """Every abstention signal, evaluated. Any hit forces `UNDECIDED`.

    Ordered structural-first, because a fragmented graph makes every downstream
    number meaningless and the report is more useful naming the cause than the
    symptom.

    DO NOT LOOSEN ANYTHING IN HERE TO LET A RESULT THROUGH. This gate is what
    caught every calibration violation this project has found, including two
    that only the final pre-integration gate discovered (a 500:1 rescale
    missing truth by 0.0044, and the 1000:1 family at n=1440 missing by
    0.090 -- the second found by perturbing n, not the rescale factor).
    Making the scale-aware readouts pass was the job of the READOUTS; the
    thresholds and factors here are exactly those of cic-1.0, deliberately.
    Loosening a calibration gate so wins pass is the anti-pattern recorded in
    docs/LL.md lessons 8-9.
    """
    signals: list[str] = []

    if n_points < max(K_LADDER) + 1:
        signals.append(Signal.TOO_FEW_POINTS)
    if duplicate_fraction >= DUPLICATE_FRACTION_MAX:
        signals.append(Signal.NEAR_DUPLICATE_POINTS)
    if largest_fraction < CONNECTIVITY_FRACTION:
        signals.append(Signal.FRAGMENTATION)
    if selected is None or k_chosen is None:
        # No rung was both connected and in possession of a scaling region.
        if Signal.FRAGMENTATION not in signals:
            signals.append(
                Signal.BALL_SATURATION if saturated_everywhere else Signal.NO_SCALING_REGION
            )
        return signals
    if len(admissible) < 2:
        signals.append(Signal.INSUFFICIENT_ENSEMBLE)

    readouts = [r.readout for r in ensemble if math.isfinite(r.readout)]
    k_spread = (max(readouts) - min(readouts)) if len(readouts) >= 2 else 0.0

    gp_ok = math.isfinite(gp_dimension)
    if not gp_ok or gp_r_squared < GP_WELL_FIT_R_SQUARED:
        signals.append(Signal.GP_UNRELIABLE)

    shell = selected.readout
    if not math.isfinite(shell):
        signals.append(Signal.LOW_WELL_FIT)
        return signals

    # scale for the relative thresholds: the midpoint of the bracket, floored
    # at 1 for the same reason the allowance is.
    if gp_ok:
        scale = max(1.0, 0.5 * (min(shell, gp_dimension) + max(shell, gp_dimension)))
    else:
        scale = max(1.0, shell)

    if shell <= 0.0 or any(r <= 0.0 for r in readouts) or (gp_ok and gp_dimension <= 0.0):
        signals.append(Signal.NON_POSITIVE)
    if k_spread > K_INSTABILITY_FACTOR * RHO_BIAS_FLOOR * scale:
        signals.append(Signal.K_INSTABILITY)
    if gp_ok and abs(shell - gp_dimension) > READOUT_DIVERGENCE_FACTOR * RHO_BIAS_FLOOR * scale:
        signals.append(Signal.READOUT_DIVERGENCE)
    if selected.well_fit_fraction < WELL_FIT_FRACTION_MIN:
        signals.append(Signal.LOW_WELL_FIT)
    # Defensive: the window rule already guarantees a well-fit majority, and an
    # `underdetermined` node can never be well-fit, so neither arm of this can
    # fire under the current selector. It is kept because it is the guard that
    # finding N9 earns -- an r_squared of 1.0 that is true by algebra rather
    # than by fit -- and it must not silently disappear if the selector changes.
    if (
        selected.underdetermined_fraction > 1.0 - WELL_FIT_FRACTION_MIN
        or selected.median_fit_window < MIN_FIT_RADII
    ):
        signals.append(Signal.UNDERDETERMINED_WINDOW)

    ring = selected.degenerate_fraction + selected.near_degenerate_fraction
    if ring >= RING_FRACTION and gp_ok and abs(gp_dimension - 1.0) > RING_GP_TOLERANCE:
        signals.append(Signal.CHAIN_ARTIFACT)

    return signals


def certify(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    label: str | None = None,
    adapter: str = DEFAULT_ADAPTER,
    replay_command: str | None = None,
    time_indices: Sequence[int] | None = None,
) -> CICResult:
    """Emit a CIC verdict, interval and certificate for one point cloud.

    ZERO KNOB: there is deliberately no `k`, `max_radius`, `min_radius`,
    `samples` or fit-threshold argument. Everything that could change the
    answer is chosen by `select_settings()` from the data and recorded on the
    certificate. `label`, `adapter` and `replay_command` are provenance;
    `time_indices` feeds one recorded-only diagnostic and is never consumed by
    the estimate (see `_temporal_adjacency_fraction`).
    """
    t0 = time.perf_counter()
    arr_in = _as_array(points)
    inputs_hash = _hash_points(arr_in)
    n_input = len(arr_in)
    arr, n_dropped, duplicate_fraction = _dedupe(arr_in)
    # Node ids index the DEDUPED cloud, so the recorded-only temporal statistic
    # is only meaningful when nothing was dropped and one index was supplied per
    # point. Otherwise it is simply not reported.
    times = None
    if time_indices is not None and n_dropped == 0 and len(time_indices) == len(arr):
        times = np.asarray(time_indices)

    rungs, metric = build_ladder(arr) if len(arr) > min(K_LADDER) else ((), None)
    largest_fraction = max((r.largest_fraction for r in rungs), default=0.0)
    arms, traces = _shell_arms(rungs)
    admissible = tuple(sorted(arms))
    k_chosen = admissible[0] if admissible else None
    ensemble = [arms[k] for k in admissible]
    selected = arms.get(k_chosen) if k_chosen is not None else None
    n_ball_expansions = sum(a.n_sampled for a in ensemble)
    # Distinguish "the ball ate the sample before any window opened" from "no
    # window satisfied the power-law premise": every connected rung whose very
    # first candidate window was already past the saturation cap.
    saturated_everywhere = bool(traces) and all(
        t and t[0][2] > BALL_FRACTION_CAP for t in traces.values()
    )

    # The correlation-sum arm: the other side of the bracket, measured under
    # the SAME per-point metric the edges were selected under -- but floored at
    # METRIC_RIDGE_FLOOR rather than TOPOLOGY_RIDGE_FLOOR, because this arm has
    # no graph to grow spurious shortcut edges in and so tolerates a much more
    # aggressive correction. See the floor commentary in `pointcloud`.
    gp_dimension, gp_r_squared, n_pair_counts = float("nan"), 0.0, 0
    gp: Any | None = None
    if metric is not None and len(arr) >= 3:
        try:
            gp = local_mahalanobis_correlation_dimension(arr, metric.sigma_inv_metric)
            gp_dimension, gp_r_squared = gp.dimension, gp.r_squared
            n_pair_counts = gp.n_pair_counts
        except ValueError:
            gp = None

    signals = _detect(
        selected=selected,
        ensemble=ensemble,
        gp_dimension=gp_dimension,
        gp_r_squared=gp_r_squared,
        k_chosen=k_chosen,
        largest_fraction=largest_fraction,
        duplicate_fraction=duplicate_fraction,
        n_points=len(arr),
        admissible=admissible,
        saturated_everywhere=saturated_everywhere,
    )

    readouts = [r.readout for r in ensemble if math.isfinite(r.readout)]
    k_spread = (max(readouts) - min(readouts)) if len(readouts) >= 2 else 0.0
    selected_hg = next((r.hg for r in rungs if r.k == k_chosen), None)

    settings = _settings_from(admissible, selected.max_radius if selected else 0, metric)

    # --- verdict and interval ---
    d_lo: float | None = None
    d_hi: float | None = None
    hull_lo = hull_hi = allowance = float("nan")
    basis = "none"
    if signals:
        verdict = Verdict.UNDECIDED
        notes = "out of domain: " + ", ".join(signals)
    else:
        assert selected is not None
        # DEGENERATE-EXACT is the only verdict in the schema that claims
        # exactness, so it takes four independent conditions, all of which must
        # hold. (1) At the SELECTED configuration the shell sequence is exactly
        # constant for essentially every sampled node -- for a proximity graph
        # that is the structural identity of a ring/chain lattice, whose growth
        # dimension is 1 by construction, and it is an identity rather than a
        # fit, so the RHO_BIAS_FLOOR that widens a fitted interval does not
        # apply. (2) The graph as a whole endorses the near-constant reading.
        # (3) The identity survives the method's own k selection: every
        # admissible rung reads ~1. (4) The correlation sum corroborates -- it
        # is built on distances rather than on a neighbour RANKING, so it is
        # not subject to the chain artifact that makes an undersampled
        # trajectory look like a ring (measured, round 3: shell 1.01-1.17 on a
        # 2.05-dimensional attractor). Condition (4) is what stands between
        # this verdict and a zero-width calibration violation.
        exact_here = selected.degenerate_fraction >= DEGENERATE_EXACT_FRACTION
        stable_at_one = all(
            math.isfinite(r.readout) and abs(r.readout - 1.0) <= RING_GP_TOLERANCE for r in ensemble
        )
        gp_agrees_with_one = math.isfinite(gp_dimension) and (
            abs(gp_dimension - 1.0) <= RING_GP_TOLERANCE
        )
        if exact_here and selected.consensus and stable_at_one and gp_agrees_with_one:
            verdict = Verdict.DEGENERATE_EXACT
            d_lo = d_hi = 1.0
            hull_lo = hull_hi = 1.0
            allowance = 0.0
            basis = "structural-identity(constant-shell ring lattice)"
            notes = (
                f"shell sequence exactly constant for {selected.degenerate_fraction:.0%} of "
                f"sampled nodes at k={selected.k}; the readout is a structural identity, "
                "not a log-log fit, so the measured fit bias does not apply. Stable at 1 "
                f"across all {len(ensemble)} admissible k, and corroborated by the "
                f"correlation sum at {gp_dimension:.4f}."
            )
        else:
            verdict = Verdict.MEASURED
            ring_bound = None
            if (
                selected.degenerate_fraction + selected.near_degenerate_fraction
            ) >= RING_FRACTION and selected.consensus:
                ring_bound = selected.mean_slope_bound
            d_lo, d_hi, hull_lo, hull_hi, allowance, basis = build_interval(
                selected.readout,
                gp_dimension,
                k_spread=k_spread,
                ring_bound=ring_bound,
            )
            notes = (
                f"bracket [{hull_lo:.4f}, {hull_hi:.4f}] from shell={selected.readout:.4f} "
                f"and correlation-sum={gp_dimension:.4f}, widened by {allowance:.4f} "
                f"(measured bias floor rho={RHO_BIAS_FLOOR}). Not a confidence interval."
            )

    diagnostics = Diagnostics(
        n_points_used=len(arr),
        n_duplicates_dropped=n_dropped,
        duplicate_fraction=duplicate_fraction,
        n_nodes=len(selected_hg.nodes) if selected_hg is not None else 0,
        n_sampled_nodes=selected.n_sampled if selected else 0,
        shell_readout=selected.readout if selected else float("nan"),
        shell_readout_median=selected.readout_median if selected else float("nan"),
        gp_readout=gp_dimension,
        gp_r_squared=gp_r_squared,
        hull_lo=hull_lo,
        hull_hi=hull_hi,
        allowance=allowance,
        k_ensemble=tuple((r.k, r.readout) for r in ensemble),
        k_spread=k_spread,
        component_counts=tuple((r.k, r.n_components) for r in rungs),
        largest_component_fraction=(
            next((r.largest_fraction for r in rungs if r.k == k_chosen), largest_fraction)
        ),
        n_components=next((r.n_components for r in rungs if r.k == k_chosen), 0),
        n_well_fit=len(selected.well_fit) if selected else 0,
        well_fit_fraction=selected.well_fit_fraction if selected else 0.0,
        median_r_squared=selected.median_r_squared if selected else 0.0,
        median_fit_window=selected.median_fit_window if selected else 0.0,
        degenerate_fraction=selected.degenerate_fraction if selected else 0.0,
        near_degenerate_fraction=selected.near_degenerate_fraction if selected else 0.0,
        mean_slope_bound=selected.mean_slope_bound if selected else 0.0,
        near_constant_consensus=selected.consensus if selected else False,
        median_ball_fraction=selected.median_ball_fraction if selected else 0.0,
        window_trace=selected.window_trace if selected else (),
        shell_readout_half_sample=selected.readout_half if selected else float("nan"),
        half_sample_shift=(
            abs(selected.readout - selected.readout_half)
            if selected and math.isfinite(selected.readout) and math.isfinite(selected.readout_half)
            else float("nan")
        ),
        n_graphs_built=len(rungs),
        n_ball_expansions=n_ball_expansions,
        n_pair_counts=n_pair_counts,
        wall_time_s=time.perf_counter() - t0,
        temporal_adjacency_fraction=(
            _temporal_adjacency_fraction(selected_hg, times) if selected_hg is not None else None
        ),
        metric_global_cov_cond=metric.global_cov_cond if metric else float("nan"),
        metric_median_topology_cond=metric.median_topology_cond if metric else float("nan"),
        metric_median_cond=metric.median_metric_cond if metric else float("nan"),
        metric_max_cond=metric.max_metric_cond if metric else float("nan"),
        metric_round_overlaps=metric.round_overlaps if metric else (),
        gp_ref_scale=gp.ref_scale if gp else float("nan"),
        gp_n_radii_used=gp.n_radii_used if gp else 0,
        gp_effective_n_radii=gp.effective_n_radii if gp else float("nan"),
        gp_weighted_mean_neighbors=gp.weighted_mean_neighbors_per_point if gp else float("nan"),
        gp_min_pair_count=gp.min_pair_count if gp else float("nan"),
        gp_max_pair_count=gp.max_pair_count if gp else float("nan"),
    )

    certificate = Certificate(
        inputs_hash=inputs_hash,
        adapter=adapter,
        contract_version=CIC_CONTRACT_VERSION,
        settings=settings,
        replay_command=replay_command
        or (
            'python -c "import numpy as np, json; '
            "from socrates.hypergraph.cic import certify; "
            f"print(json.dumps(certify(np.load('POINTS.npy'), label={label!r}).to_dict()))\"  "
            f"# expects {inputs_hash}"
        ),
        label=label,
        n_points_input=n_input,
        constants=(
            ("RHO_BIAS_FLOOR", RHO_BIAS_FLOOR),
            ("CONNECTIVITY_FRACTION", CONNECTIVITY_FRACTION),
            ("WELL_FIT_R_SQUARED", WELL_FIT_R_SQUARED),
            ("WELL_FIT_FRACTION_MIN", WELL_FIT_FRACTION_MIN),
            ("K_INSTABILITY_FACTOR", K_INSTABILITY_FACTOR),
            ("READOUT_DIVERGENCE_FACTOR", READOUT_DIVERGENCE_FACTOR),
            ("RING_FRACTION", RING_FRACTION),
            ("RING_GP_TOLERANCE", RING_GP_TOLERANCE),
            ("DUPLICATE_FRACTION_MAX", DUPLICATE_FRACTION_MAX),
            ("SAMPLES", float(SAMPLES)),
            ("RADIUS_CAP", float(RADIUS_CAP)),
            ("BALL_FRACTION_CAP", BALL_FRACTION_CAP),
            ("GLOBAL_RIDGE_FLOOR_FRAC", GLOBAL_RIDGE_FLOOR_FRAC),
            ("TOPOLOGY_RIDGE_FLOOR_FRAC", TOPOLOGY_RIDGE_FLOOR_FRAC),
            ("METRIC_RIDGE_FLOOR_FRAC", METRIC_RIDGE_FLOOR_FRAC),
            ("METRIC_CONVERGENCE_OVERLAP", METRIC_CONVERGENCE_OVERLAP),
            ("MAX_METRIC_ITERATIONS", float(MAX_METRIC_ITERATIONS)),
        ),
    )

    return CICResult(
        verdict=verdict,
        d_lo=d_lo,
        d_hi=d_hi,
        certificate=certificate,
        diagnostics=diagnostics,
        signals=tuple(signals),
        interval_basis=basis,
        notes=notes,
    )
