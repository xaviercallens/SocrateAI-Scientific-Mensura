"""SCALE-AWARE ESTIMATOR PROTOTYPE -- design spike for the anisotropy refutation.

See docs/MENSURA_BENCH_V2.md section 7 for the defect this answers. Summary:
dimension is a bi-Lipschitz invariant, so `certify(A)` and `certify(A @ M)`
for an invertible linear `M` must agree or abstain. They do not: rescaling
one axis of a passing uniform-square row by 0.01 flips MEASURED [1.59, 2.28]
to MEASURED [0.64, 1.36] -- disjoint, zero signals. Mechanism: the shell
(k-NN) arm's proximity graph degenerates into a near-chain along the long
axis, and the correlation-sum arm's radii are hard-coded fractions of the
bounding-box diagonal, which the long axis dominates -- both arms collapse
toward dimension 1 TOGETHER, so `READOUT_DIVERGENCE` (which only fires on
DISAGREEMENT) stays silent.

THE OWNER'S CHOSEN FIX (explicit, over two cheaper alternatives): make the
estimators themselves scale-aware via local-metric-adaptive neighbourhoods,
not a global normalise-and-declare adapter and not an anisotropy-abstains
gate. This module is that design, prototyped and validated against the
known-answer battery and the refutation case. It does NOT touch
dimension.py, baseline.py, pointcloud.py or cic.py -- those are imported
and reused as libraries (exactly as cic.py itself reuses dimension.py's
private helpers), never edited.

LITERATURE THIS IS GROUNDED IN
-------------------------------
* Roweis & Saul (2000), "Nonlinear Dimensionality Reduction by Locally
  Linear Embedding", Science 290(5500) -- the local-covariance-from-
  k-nearest-neighbours construction (`local_covariances` below) is LLE's
  local-linear-patch step verbatim: assume each point's neighbourhood is
  well approximated by a local affine patch, and estimate that patch's
  second-moment structure from the neighbours. LLE's regularisation of the
  local covariance (a ridge term when the neighbourhood undersamples the
  ambient dimension) is the direct ancestor of `LOCAL_RIDGE_FLOOR_FRAC` below.
* Farahmand, Szepesvari & Audibert (2007), "Manifold-adaptive dimension
  estimation" -- already cited in this codebase (dimension.py's
  SATURATION_BALL_FRACTION commentary, MENSURA_BENCHMARK.md) for the
  general principle that a neighbourhood should adapt to the sample in
  hand rather than being a fixed, unexamined radius. This module is the
  same principle applied to the METRIC rather than to the radius: the
  neighbourhood should adapt to the local SHAPE of the sample, not just
  its size.
* Levina & Bickel (2004), "Maximum Likelihood Estimation of Intrinsic
  Dimension" and TwoNN (Facco et al. 2017) are the codebase's standing
  cross-checks (docs/LL.md, MENSURA_BENCH_V2.md section 5) for a k-NN-based
  estimator's answer on a KNOWN target; they are not re-implemented here,
  but the known-answer battery below is the same discipline applied to this
  prototype.
* Theiler (1986) is the existing codebase's precedent for "a hard-coded
  scale computed from raw coordinates silently encodes an assumption" --
  the same shape of defect this module fixes, one abstraction level over
  (there it was TIME scale; here it is SPACE scale).

DESIGN, IN ONE PARAGRAPH
-------------------------
For each point, estimate a local covariance matrix from an initial
neighbourhood, use its inverse as a local Mahalanobis metric, re-select
neighbours under that metric, and iterate (a fixed-point / EM-like
refinement, same shape as LLE's own local-patch iteration). The refined
neighbour sets feed dimension.py's existing Hypergraph + shell-growth
machinery UNCHANGED -- only the EDGE SET changes, not the estimator reading
it. The correlation-sum side gets the analogous fix: radii expressed as
Mahalanobis distance (locally, per point, or globally) instead of a
fraction of the raw bounding-box diagonal.

A REAL, MEASURED WRINKLE: naively bootstrapping the very first covariance
estimate from a plain EUCLIDEAN k-NN neighbourhood on the ANISOTROPIC data
does not work under strong (~100x) anisotropy -- see `bootstrap="euclidean"`
below and the worked argument in `scale_aware_neighbors`'s docstring. The
Euclidean neighbourhood IS the chain artifact this whole construction exists
to correct, so bootstrapping the local metric from it is circular: the
first-round covariance comes out with the axes backwards. `bootstrap="global"`
(a single global covariance estimate, computed with no neighbour selection
at all) escapes the trap and is this module's default; both are implemented
and measured so the failure is shown, not just asserted.

E6 (this repo's standing lesson: a prior workflow lost agents to API errors
and only on-disk artifacts survived): every result is written to
`scale_aware_prototype.json` incrementally, after each row.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from socrates.hypergraph import dimension as dim_mod  # noqa: E402
from socrates.hypergraph.baseline import _log_log_fit, correlation_dimension  # noqa: E402
from socrates.hypergraph.cic import build_interval, certify  # noqa: E402
from socrates.hypergraph.core import Hypergraph  # noqa: E402

import targets  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "scale_aware_prototype.json"
SWEEP_PATH = OUT_DIR / "scale_aware_prototype_sweep.json"

# --------------------------------------------------------------------------
# OPTIMIZATION PASS (owner decision 8.5 item 3) -- pure memoization of
# immutable Hypergraph instances.
#
# PROFILING FINDING (cProfile, uniform-square n=1600, 44.9s total): the
# dominant cost was NOT the O(n^2 d^2) Mahalanobis distances (6.4s) but
# production `Hypergraph.nodes` -- an UNCACHED property rebuilding a
# frozenset over all edges on every call (25.1s across 824 calls from
# `dimension.local_dimension` via `ball()`), plus the dataclass-generated
# `__hash__` re-hashing the full ~20k-edge tuple on every `adjacency()`
# lru_cache lookup (2.0s).
#
# Production files must not be edited (task constraint), so this scratch
# module installs runtime-only per-instance caches on the (frozen, immutable)
# class at import time. This is PURE MEMOIZATION: the cached value is exactly
# the value the original code computes, computed once. `Hypergraph` is a
# frozen dataclass, so nothing can mutate `edges` after construction and no
# cache can ever go stale. Numerical output is byte-identical by
# construction; the full-battery equivalence check (optimize_results.json)
# verifies it end to end.
# --------------------------------------------------------------------------

_CHUNK = 256  # row-chunk size for batched pairwise Mahalanobis computations


def _install_hypergraph_caches() -> None:
    if getattr(Hypergraph, "_scratch_caches_installed", False):
        return

    orig_nodes = Hypergraph.nodes.fget

    def _nodes(self: Hypergraph) -> frozenset[int]:
        try:
            return object.__getattribute__(self, "_cache_nodes")
        except AttributeError:
            val = orig_nodes(self)
            object.__setattr__(self, "_cache_nodes", val)
            return val

    Hypergraph.nodes = property(_nodes)

    orig_hash = Hypergraph.__hash__

    def _hash(self: Hypergraph) -> int:
        try:
            return object.__getattribute__(self, "_cache_hash")
        except AttributeError:
            val = orig_hash(self)
            object.__setattr__(self, "_cache_hash", val)
            return val

    Hypergraph.__hash__ = _hash

    orig_adjacency = Hypergraph.adjacency

    def _adjacency(self: Hypergraph) -> dict:
        try:
            return object.__getattribute__(self, "_cache_adj")
        except AttributeError:
            val = orig_adjacency(self)  # still goes through the module lru_cache,
            object.__setattr__(self, "_cache_adj", val)  # preserving shared-dict semantics
            return val

    Hypergraph.adjacency = _adjacency
    Hypergraph._scratch_caches_installed = True


_install_hypergraph_caches()

# --------------------------------------------------------------------------
# Local covariance estimation (Roweis & Saul 2000 local-linear-patch step).
# --------------------------------------------------------------------------

# Eigenvalues of a local covariance are clipped to at least this fraction of
# the neighbourhood's own largest eigenvalue before inverting it.
#
# WHY THIS IS NEEDED, PRECISELY. A neighbourhood drawn from a genuinely
# lower-dimensional manifold (a circle, a curve) has a local covariance with
# one or more near-zero eigenvalues BY CONSTRUCTION -- that near-zero
# eigenvalue is real local structure (there is almost no spread transverse to
# the curve), not noise. Inverting it unregularised sends the Mahalanobis
# metric's transverse weight to near-infinity, so the SLIGHTEST transverse
# offset (sampling noise, or the curve's own curvature) dominates every
# distance and the re-selected "neighbours" become whichever points happen to
# lie closest to the tangent line -- not a re-derivation of the true local
# geometry but a new, worse artifact of the same shape as the one being
# fixed. LLE's own construction carries the identical ridge term for the
# identical reason (Roweis & Saul 2000, section on solving for
# reconstruction weights when the neighbourhood count exceeds the ambient
# dimension).
#
# A REAL DEFECT FOUND WHILE PROTOTYPING, kept documented rather than quietly
# fixed, because it is itself one of this design's open questions (see the
# final report). The first version of this module used ONE floor_frac for
# both the global bootstrap covariance and every per-point local covariance.
# That is wrong, and measurably so: a single 0.05 floor caps the correctable
# anisotropy ratio at 20:1 EVERYWHERE, including on the global covariance --
# but the refutation case's true ratio is 10000:1 (0.01 squared), so the
# global bootstrap was silently only ever correcting a 20:1 slice of a
# 10000:1 problem, and every local round inherited the same cap. Measured
# effect: with one shared floor, `gp_global_mahalanobis` on the refutation
# case read 1.34 (barely moved from the unfixed 0.64-1.36 bracket); splitting
# the floors below moves it to ~1.9 (see the companion report).
#
# The two floors serve genuinely different purposes and should NOT be forced
# to share a value:
#
#   GLOBAL_RIDGE_FLOOR_FRAC -- regularises the WHOLE-CLOUD covariance, which
#   is estimated from all n points and is therefore a low-variance, reliable
#   statistic. Its only job is to prevent literal singularity (an exactly
#   degenerate direction, e.g. a coordinate that is a constant); it should
#   NOT cap how much real anisotropy gets corrected, so it is set close to
#   machine-precision-relative rather than to a geometrically meaningful
#   fraction.
#
#   LOCAL_RIDGE_FLOOR_FRAC -- regularises a PER-POINT covariance estimated
#   from only k0 neighbours, which is noisy AND may be estimating a
#   genuinely lower-dimensional local patch (a curve's transverse spread).
#   Here a real, geometrically-motivated floor is wanted, for the reason
#   argued above (LLE's own regularisation, Roweis & Saul 2000) -- 0.05 is
#   kept as the (uncalibrated) choice for this one.
GLOBAL_RIDGE_FLOOR_FRAC = 1e-9
LOCAL_RIDGE_FLOOR_FRAC = 0.05


def regularize_covariance(cov: np.ndarray, floor_frac: float) -> np.ndarray:
    """Clip `cov`'s eigenvalues to `floor_frac` of its own largest eigenvalue."""
    evals, evecs = np.linalg.eigh(cov)
    floor = max(float(evals.max()), 1e-300) * floor_frac
    evals = np.maximum(evals, floor)
    return (evecs * evals) @ evecs.T


def regularize_covariance_batch(covs: np.ndarray, floor_frac: float) -> np.ndarray:
    """Batched `regularize_covariance` over a (n, d, d) stack.

    OPTIMIZATION (8.5 item 3): replaces n separate `np.linalg.eigh` calls
    (per-call overhead dominated the arithmetic at d=2-3) with one batched
    gufunc call. Bit-identical to the scalar version: batched `eigh` runs the
    same LAPACK routine per slice (verified `np.array_equal` on the full
    battery inputs), `evals[..., -1]` is the max eigenvalue because eigh
    returns ascending order, and the reconstruction `(evecs * evals) @
    evecs.T` is the same contraction per slice.
    """
    evals, evecs = np.linalg.eigh(covs)
    floor = np.maximum(evals[..., -1], 1e-300) * floor_frac
    evals = np.maximum(evals, floor[..., None])
    return np.matmul(evecs * evals[..., None, :], np.swapaxes(evecs, -1, -2))


def global_covariance(arr: np.ndarray, *, floor_frac: float = GLOBAL_RIDGE_FLOOR_FRAC) -> np.ndarray:
    """The whole cloud's covariance -- needs no neighbour selection at all,
    which is exactly why it is safe to use as a bootstrap (see module
    docstring's "REAL, MEASURED WRINKLE")."""
    centered = arr - arr.mean(axis=0)
    cov = (centered.T @ centered) / len(arr)
    return regularize_covariance(cov, floor_frac)


def local_covariances(
    arr: np.ndarray, neighbor_idx: list[np.ndarray], *, floor_frac: float = LOCAL_RIDGE_FLOOR_FRAC
) -> np.ndarray:
    """Per-point local covariance from each point's current neighbour set.

    OPTIMIZATION (8.5 item 3): when every neighbour row has the same length
    (the only case the pipeline produces -- `mahalanobis_knn_full` and
    `euclidean_knn` return exactly k indices per point), the gather, the
    centering and the (d,k)@(k,d) products are batched. Verified bit-identical
    to the original per-point loop (`np.array_equal` on battery inputs):
    `mean(axis=1)` reduces each row in the same order as the original
    per-row `mean(axis=0)`, and batched `matmul` runs the same product per
    slice. The ragged case keeps the original loop.
    """
    n, d = arr.shape
    if len({len(ix) for ix in neighbor_idx}) == 1:
        idx = np.asarray(neighbor_idx)
        pts = arr[idx]  # (n, k, d)
        centered = pts - pts.mean(axis=1, keepdims=True)
        covs = np.matmul(centered.transpose(0, 2, 1), centered) / idx.shape[1]
        return regularize_covariance_batch(covs, floor_frac)
    covs = np.empty((n, d, d))
    for i, idx_row in enumerate(neighbor_idx):
        pts = arr[idx_row]
        centered = pts - pts.mean(axis=0)
        cov = (centered.T @ centered) / len(idx_row)
        covs[i] = regularize_covariance(cov, floor_frac)
    return covs


def whiten(arr: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Transform so that Euclidean distance in the output = Mahalanobis
    distance under `cov` in the input. Used only for the GLOBAL correlation-
    sum variant, where a single metric lets the existing tree-based
    `baseline.correlation_dimension` be reused unchanged."""
    evals, evecs = np.linalg.eigh(cov)
    w = evecs @ np.diag(evals**-0.5) @ evecs.T
    return (arr - arr.mean(axis=0)) @ w.T


def euclidean_knn(arr: np.ndarray, k: int) -> list[np.ndarray]:
    tree = cKDTree(arr)
    _, idx = tree.query(arr, k=k + 1)
    return [row[row != i][:k] for i, row in enumerate(idx)]


def mahalanobis_knn_full(arr: np.ndarray, sigma_inv: np.ndarray, k: int) -> list[np.ndarray]:
    """Exact k nearest neighbours of every point under a Mahalanobis metric,
    searched over the FULL population rather than a Euclidean shortlist.

    THIS IS DELIBERATE, NOT A MISSED OPTIMISATION. Pre-filtering candidates
    with a Euclidean k-NN query (a standard "widen then re-rank" trick, and
    the first thing tried here) is UNSAFE under the anisotropy this module
    exists to correct: under a 100x compression, a point's true isotropic
    neighbours are demonstrably NOT among its top few hundred Euclidean
    neighbours (worked out in the module docstring's bootstrap discussion),
    so a shortlist built that way would silently exclude the very points the
    Mahalanobis re-ranking is supposed to surface. Searching the full
    population is the correctness fix; it is also the module's dominant cost
    (O(n^2 d^2) per round) -- see the final report for the measured
    trade-off.

    `sigma_inv` is either one (d, d) matrix (broadcast to every point, the
    GLOBAL-bootstrap case) or one (n, d, d) matrix per point (the per-point
    LOCAL case).

    OPTIMIZATION (8.5 item 3): the per-point quadratic form is computed for
    _CHUNK query points at a time with one batched einsum instead of one
    einsum call per point (the per-call einsum overhead, not the flops,
    dominated at d=2-3). The einsum subscripts contract over the same (d, e)
    index order as the original per-point call, and the chunked result was
    verified `np.array_equal` (bit-identical) against the per-point loop on
    the full battery inputs, global and per-point sigma both. Neighbour
    SELECTION (inf-ing the diagonal, argpartition, argsort of the top-k) is
    the original per-row code operating on the identical d2 row.
    """
    n = len(arr)
    per_point = sigma_inv.ndim == 3
    out: list[np.ndarray] = [np.empty(0, dtype=np.intp)] * n
    for start in range(0, n, _CHUNK):
        end = min(start + _CHUNK, n)
        diffs = arr[None, :, :] - arr[start:end, None, :]  # (c, n, d)
        if per_point:
            s = sigma_inv[start:end]
        else:
            s = np.broadcast_to(sigma_inv, (end - start, *sigma_inv.shape))
        d2c = np.einsum("cnd,cde,cne->cn", diffs, s, diffs)
        for ci in range(end - start):
            i = start + ci
            d2 = d2c[ci]
            d2[i] = np.inf
            idx = np.argpartition(d2, k)[:k]
            idx = idx[np.argsort(d2[idx])]
            out[i] = idx
    return out


@dataclass
class ScaleAwareResult:
    neighbor_idx: list[np.ndarray]  # final, length k (graph degree)
    sigma: np.ndarray  # (n, d, d) final local covariances (window k0)
    sigma_inv: np.ndarray
    diagnostics: dict[str, Any]


def scale_aware_neighbors(
    arr: np.ndarray,
    k: int,
    *,
    k0: int,
    iterations: int,
    bootstrap: str = "global",
) -> ScaleAwareResult:
    """The core construction: local-metric-adaptive neighbour selection.

    Parameters
    ----------
    k : final graph degree (what the shell-growth k-NN graph is built with).
    k0 : covariance-estimation window -- must be well above `k` for a stable
        covariance in d dimensions (needs at least d(d+1)/2 points to be
        non-degenerate even before regularisation; comfortably more for a
        low-variance estimate). Kept separate from `k` deliberately: using
        `k` itself for both would couple graph sparsity to covariance
        stability for no reason.
    iterations : number of local Mahalanobis refinement rounds AFTER the
        bootstrap. 0 means "use the bootstrap metric only" (for
        `bootstrap="global"` this is a single global whitening -- see the
        final report for why that is not, by itself, "declare and
        normalise": it is the FIRST round of a construction that keeps
        refining locally, and demonstrably keeps refining on curved targets
        -- but it is worth testing on its own as the cheapest member of this
        family).
    bootstrap : "global" (default) -- the whole cloud's covariance, no
        neighbour selection needed, escapes the chicken-and-egg trap
        described in the module docstring. "euclidean" -- the FIRST thing
        tried, kept here as a documented NEGATIVE CONTROL: bootstrap the
        initial covariance from a plain Euclidean k0-NN on the (possibly
        anisotropic) raw data. Under strong anisotropy this is circular (the
        candidate neighbourhood already IS the chain artifact), and the
        battery below measures whether that circularity actually breaks the
        result or self-corrects under iteration.
    """
    n, d = arr.shape
    t0 = time.perf_counter()
    diag: dict[str, Any] = {"bootstrap": bootstrap, "k": k, "k0": k0, "iterations": iterations, "rounds": []}

    if bootstrap == "global":
        g_cov = global_covariance(arr)
        sigma_inv = np.linalg.inv(g_cov)
        diag["bootstrap_cov_eigs"] = np.linalg.eigvalsh(g_cov).tolist()
    elif bootstrap == "euclidean":
        neigh0 = euclidean_knn(arr, k0)
        covs0 = local_covariances(arr, neigh0)
        sigma_inv = np.linalg.inv(covs0)
    else:
        raise ValueError(f"bootstrap must be 'global' or 'euclidean', got {bootstrap!r}")

    # Reference point for the report: what does a PLAIN Euclidean k0-NN
    # neighbourhood's local covariance look like on this data, before any
    # Mahalanobis correction at all? On an isotropic cloud this should be
    # close to 1 (near-circular local neighbourhoods); comparing it against
    # `final_cond_number_*` below is the evidence for whether the iterative
    # reselection is introducing anisotropy that was not there originally
    # (see the module's "REAL, MEASURED WRINKLE" and the final report).
    euclid_cond = np.linalg.cond(local_covariances(arr, euclidean_knn(arr, k0)))
    diag["round0_euclidean_local_cond_median"] = float(np.median(euclid_cond))
    diag["round0_euclidean_local_cond_max"] = float(np.max(euclid_cond))

    neighbor_idx = mahalanobis_knn_full(arr, sigma_inv, k0)
    prev_sets = [set(x.tolist()) for x in neighbor_idx]
    diag["rounds"].append({"round": 0})

    final_covs = local_covariances(arr, neighbor_idx)
    for it in range(1, iterations + 1):
        sigma_inv = np.linalg.inv(final_covs)
        new_idx = mahalanobis_knn_full(arr, sigma_inv, k0)
        new_sets = [set(x.tolist()) for x in new_idx]
        overlap = float(np.mean([len(a & b) / k0 for a, b in zip(prev_sets, new_sets, strict=True)]))
        cond = np.linalg.cond(final_covs)
        diag["rounds"].append(
            {
                "round": it,
                "neighbor_set_overlap_with_prev_fraction": overlap,
                "median_local_cond_number": float(np.median(cond)),
                "max_local_cond_number": float(np.max(cond)),
            }
        )
        neighbor_idx = new_idx
        prev_sets = new_sets
        final_covs = local_covariances(arr, neighbor_idx)

    sigma_inv_final = np.linalg.inv(final_covs)
    final_idx = [idx[:k] for idx in neighbor_idx]
    diag["wall_time_s"] = time.perf_counter() - t0
    diag["final_cond_number_median"] = float(np.median(np.linalg.cond(final_covs)))
    diag["final_cond_number_max"] = float(np.max(np.linalg.cond(final_covs)))
    # How much the per-point metric VARIES across the cloud -- a global
    # (single-matrix) fix would show ~0 spread here; genuine local adaptation
    # (e.g. on a curved manifold) should show real spread. Measured as the
    # spread of each point's covariance's dominant-eigenvector ANGLE (2-D
    # only; recorded as nan otherwise).
    if d == 2:
        dom_vecs = np.array([np.linalg.eigh(final_covs[i])[1][:, -1] for i in range(n)])
        angs = np.arctan2(dom_vecs[:, 1], dom_vecs[:, 0])
        # fold to [0, pi) since eigenvectors are sign-ambiguous
        angs = np.mod(angs, math.pi)
        diag["dominant_axis_angle_std_rad"] = float(np.std(angs))
    return ScaleAwareResult(final_idx, final_covs, sigma_inv_final, diag)


def build_hypergraph(neighbor_idx: list[np.ndarray]) -> Hypergraph:
    """OPTIMIZATION (8.5 item 3): the Python set-of-tuples loop is replaced by
    a vectorized min/max + `np.unique(axis=0)` when all rows have equal
    length. `np.unique` returns lexicographically sorted unique rows, which is
    exactly `sorted(set(...))` over (lo, hi) int pairs -- the resulting edge
    tuple (Python ints via `.tolist()`) is identical, so the Hypergraph
    compares and hashes equal. Verified on the full battery. Ragged rows keep
    the original loop."""
    if len({len(r) for r in neighbor_idx}) == 1:
        n = len(neighbor_idx)
        rows = np.asarray(neighbor_idx, dtype=np.int64)
        i_col = np.repeat(np.arange(n, dtype=np.int64), rows.shape[1])
        j_col = rows.reshape(-1)
        lo, hi = np.minimum(i_col, j_col), np.maximum(i_col, j_col)
        pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
        return Hypergraph(tuple(map(tuple, pairs.tolist())))
    edges: set[tuple[int, int]] = set()
    for i, row in enumerate(neighbor_idx):
        for j in row:
            j = int(j)
            edges.add((i, j) if i < j else (j, i))
    return Hypergraph(tuple(sorted(edges)))


# --------------------------------------------------------------------------
# Shell (k-NN / ball-volume) arm: unchanged dimension.py machinery on a
# scale-aware edge set.
# --------------------------------------------------------------------------


def shell_estimate(hg: Hypergraph, *, max_radius: int, samples: int) -> dict[str, float]:
    return {
        "mean_dimension": dim_mod.mean_dimension(hg, samples=samples, max_radius=max_radius),
        "degenerate_fraction": dim_mod.degenerate_fraction(hg, samples=samples, max_radius=max_radius),
        "near_degenerate_fraction": dim_mod.near_degenerate_fraction(
            hg, samples=samples, max_radius=max_radius
        ),
    }


# --------------------------------------------------------------------------
# Correlation-sum (Grassberger-Procaccia) arm: two scale-aware variants.
# --------------------------------------------------------------------------


def gp_baseline(points: list[tuple[float, ...]]) -> dict[str, float]:
    est = correlation_dimension(points)
    return {"dimension": est.dimension, "r_squared": est.r_squared}


def gp_global_mahalanobis(arr: np.ndarray) -> dict[str, float]:
    """Replace the raw bounding-box diagonal with a Mahalanobis-whitened one:
    whiten by the CLOUD'S global covariance, then hand the existing
    correlation_dimension() the whitened coordinates unchanged. Cheap
    (O(n log n), same as the original), and fixes exactly the mechanism the
    refutation names for a globally-homogeneous anisotropy -- but it is a
    single global metric, so it is the variant closest in spirit to the
    "declare and normalise" option the owner did NOT choose; kept here as the
    cheap end of the design space, not the recommended one.
    """
    cov = global_covariance(arr)
    w = whiten(arr, cov)
    est = correlation_dimension([tuple(row) for row in w])
    return {"dimension": est.dimension, "r_squared": est.r_squared}


def gp_local_mahalanobis(
    arr: np.ndarray,
    sigma_inv: np.ndarray,
    *,
    n_radii: int = 20,
    r_min_frac: float = 0.01,
    r_max_frac: float = 0.5,
    ref_neighbors: int = 20,
) -> dict[str, float]:
    """Locally metric-aware correlation sum: each point counts neighbours
    under ITS OWN Mahalanobis metric (`sigma_inv[i]`, e.g. reused from the
    shell arm's converged local covariances), and radii are fractions of a
    DATA-DRIVEN reference scale (the median, over points, of that point's
    own `ref_neighbors`-th Mahalanobis-neighbour distance) rather than
    baseline.py's hard-coded fraction of the raw bounding-box diagonal --
    which is precisely the quantity the refutation names as the mechanism of
    failure (the long axis dominates the diagonal, so a fixed fraction of it
    is the wrong scale on every other axis).

    C(r) = (1/n) * sum_i [ (1/(n-1)) * #{j != i : d_Mahalanobis_i(x_i,x_j) < r} ]

    This is a per-point-metric generalisation of the classical correlation
    integral, not the classical one itself (each term uses a different
    metric), so it is a NEW statistic, not literally the same D_2 -- flagged
    explicitly here rather than presented as a drop-in replacement. It is
    also the module's most expensive routine: O(n^2 d^2) for the distance
    computation, and the reported wall time should be read as a real,
    measured cost, not a rough guess.

    OPTIMIZATION (8.5 item 3): one chunked-einsum pass computes each row's
    squared distances (bit-identical to the per-point einsum, see
    `mahalanobis_knn_full`), sorted once per row and reused for both the
    reference-scale pass and the counting pass (the original recomputed the
    identical d2 row in each pass). Equivalences, each verified
    `np.array_equal` on the full battery: the kth order statistic of d2 ==
    `np.partition(d2, kth)[kth]`; sqrt is monotone so sorting d2 then taking
    sqrt gives the sorted d values elementwise; `np.searchsorted(sorted_d,
    r, side="left")` == `np.count_nonzero(d < r)` (an INTEGER, so summation
    order cannot matter -- counts stay exact int64 until the original's
    single float division).
    """
    n = len(arr)
    kth = min(ref_neighbors, n - 2)
    d2_sorted = np.empty((n, n))
    for start in range(0, n, _CHUNK):
        end = min(start + _CHUNK, n)
        diffs = arr[None, :, :] - arr[start:end, None, :]
        d2c = np.einsum("cnd,cde,cne->cn", diffs, sigma_inv[start:end], diffs)
        d2c[np.arange(end - start), np.arange(start, end)] = np.inf
        d2_sorted[start:end] = np.sort(d2c, axis=1)
    ref_dists = np.array([math.sqrt(float(d2_sorted[i, kth])) for i in range(n)])
    ref_scale = float(np.median(ref_dists))
    if ref_scale <= 0:
        return {"dimension": float("nan"), "r_squared": 0.0, "n_radii_used": 0, "ref_scale": ref_scale}

    radii = np.logspace(math.log10(r_min_frac * ref_scale), math.log10(r_max_frac * ref_scale), n_radii)
    d_sorted = np.sqrt(d2_sorted, out=d2_sorted)  # in place; d2_sorted not reused below
    counts_int = np.zeros(n_radii, dtype=np.int64)
    for i in range(n):
        counts_int += np.searchsorted(d_sorted[i], radii, side="left")
    counts = counts_int.astype(float)
    c = counts / (n * (n - 1))

    log_r, log_c = [], []
    for r, ci in zip(radii, c, strict=True):
        if 0 < ci < 1:
            log_r.append(math.log(r))
            log_c.append(math.log(ci))
    if len(log_r) < 2:
        return {
            "dimension": float("nan"),
            "r_squared": 0.0,
            "n_radii_used": len(log_r),
            "ref_scale": ref_scale,
        }
    slope, r2 = _log_log_fit(log_r, log_c)
    return {"dimension": slope, "r_squared": r2, "n_radii_used": len(log_r), "ref_scale": ref_scale}


# --------------------------------------------------------------------------
# Bracket construction (reuses cic.build_interval UNCHANGED -- this is the
# only piece of cic.py this module calls, and only as a library function).
# --------------------------------------------------------------------------


def bracket(shell: float, gp: float) -> dict[str, Any]:
    d_lo, d_hi, hull_lo, hull_hi, allowance, basis = build_interval(shell, gp, k_spread=0.0)
    return {
        "d_lo": d_lo,
        "d_hi": d_hi,
        "hull_lo": hull_lo,
        "hull_hi": hull_hi,
        "allowance": allowance,
        "basis": basis,
    }


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["d_lo"] <= b["d_hi"] and b["d_lo"] <= a["d_hi"]


# --------------------------------------------------------------------------
# Lorenz / Takens delay embedding (reproduces the module docstring's cited
# lag-1 failure; independent re-implementation, no generator script survived
# from the run that produced cic_delay_embedding.json).
# --------------------------------------------------------------------------


def lorenz_x_series(n_raw: int, *, dt: float = 0.004, burn_in: int = 2000) -> np.ndarray:
    x = np.array([1.0, 1.0, 1.0])
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    out = np.empty(n_raw)
    j = 0
    steps = n_raw + burn_in
    for i in range(steps):
        dx = np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])
        x = x + dt * dx
        if i >= burn_in:
            out[j] = x[0]
            j += 1
    return out


def takens_embedding(x_series: np.ndarray, *, m: int, tau: int, n: int) -> np.ndarray:
    need = (m - 1) * tau + n
    if len(x_series) < need:
        raise ValueError(f"need {need} raw samples, got {len(x_series)}")
    idx = np.arange(n)
    cols = [x_series[idx + j * tau] for j in range(m)]
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------
# The comparison battery
# --------------------------------------------------------------------------


def _flush(rows: list[dict], sweep: list[dict]) -> None:
    RESULTS_PATH.write_text(json.dumps({"rows": rows}, indent=2, default=str))
    SWEEP_PATH.write_text(json.dumps({"rows": sweep}, indent=2, default=str))


def run_case(
    name: str,
    points: np.ndarray,
    truth: float | None,
    *,
    k: int,
    k0: int,
    iterations: int,
    max_radius: int,
    samples: int,
    bootstrap: str = "global",
    run_local_gp: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    arr = np.asarray(points, dtype=float)
    cloud = [tuple(row) for row in arr]
    n = len(arr)

    # --- BEFORE: production cic.certify() (official, self-selecting) ---
    cic_result = certify(arr, label=name)
    before_official = {
        "verdict": cic_result.verdict,
        "d_lo": cic_result.d_lo,
        "d_hi": cic_result.d_hi,
        "shell": cic_result.diagnostics.shell_readout,
        "gp": cic_result.diagnostics.gp_readout,
        "signals": list(cic_result.signals),
        "k_selected": cic_result.certificate.settings.k,
    }

    # --- BEFORE, fixed-settings Euclidean (apples-to-apples with "after") ---
    euclid_idx = euclidean_knn(arr, k)
    euclid_hg = build_hypergraph(euclid_idx)
    before_shell = shell_estimate(euclid_hg, max_radius=max_radius, samples=samples)
    before_gp = gp_baseline(cloud)
    before_bracket = bracket(before_shell["mean_dimension"], before_gp["dimension"])

    # --- AFTER: scale-aware local-metric-adaptive construction ---
    sa = scale_aware_neighbors(arr, k, k0=k0, iterations=iterations, bootstrap=bootstrap)
    sa_hg = build_hypergraph(sa.neighbor_idx)
    after_shell = shell_estimate(sa_hg, max_radius=max_radius, samples=samples)
    after_gp_global = gp_global_mahalanobis(arr)
    after_gp_local = (
        gp_local_mahalanobis(arr, sa.sigma_inv) if run_local_gp else {"dimension": float("nan"), "r_squared": 0.0}
    )
    after_bracket_global = bracket(after_shell["mean_dimension"], after_gp_global["dimension"])
    after_bracket_local = bracket(after_shell["mean_dimension"], after_gp_local["dimension"])

    row = {
        "case": name,
        "n": n,
        "d": arr.shape[1],
        "truth": truth,
        "settings": {"k": k, "k0": k0, "iterations": iterations, "max_radius": max_radius, "samples": samples, "bootstrap": bootstrap},
        "before_official_certify": before_official,
        "before_fixed_euclidean": {
            "shell": before_shell,
            "gp": before_gp,
            "bracket": before_bracket,
            "contains_truth": (before_bracket["d_lo"] <= truth <= before_bracket["d_hi"]) if truth is not None else None,
        },
        "after_scale_aware": {
            "shell": after_shell,
            "gp_global_mahalanobis": after_gp_global,
            "gp_local_mahalanobis": after_gp_local,
            "bracket_global_gp": after_bracket_global,
            "bracket_local_gp": after_bracket_local,
            "contains_truth_global_gp": (
                (after_bracket_global["d_lo"] <= truth <= after_bracket_global["d_hi"]) if truth is not None else None
            ),
            "contains_truth_local_gp": (
                (after_bracket_local["d_lo"] <= truth <= after_bracket_local["d_hi"]) if truth is not None else None
            ),
            "diagnostics": sa.diagnostics,
        },
        "wall_time_s": time.perf_counter() - t0,
    }
    return row


def main() -> int:
    rows: list[dict[str, Any]] = []
    sweep: list[dict[str, Any]] = []
    _flush(rows, sweep)

    K = 8
    K0 = 30
    ITER = 2
    MAX_RADIUS = 6
    SAMPLES = 80

    print("=" * 100)
    print("PART A -- known-answer battery, before/after (n=1600)")
    print("=" * 100)

    known_cases = [
        ("uniform-square-2d/n=1600/seed=11", targets.uniform_square(1600, 11), 2.0),
        ("circle-1d/n=1600/seed=11", targets.circle(1600, 11), 1.0),
        ("uniform-cube-3d/n=1600/seed=11", targets.uniform_cube(1600, 11), 3.0),
    ]
    for name, pts, truth in known_cases:
        row = run_case(name, pts, truth, k=K, k0=K0, iterations=ITER, max_radius=MAX_RADIUS, samples=SAMPLES)
        rows.append(row)
        _flush(rows, sweep)
        bb = row["before_fixed_euclidean"]["bracket"]
        ab_g = row["after_scale_aware"]["bracket_global_gp"]
        ab_l = row["after_scale_aware"]["bracket_local_gp"]
        print(
            f"{name:<38} truth={truth:<5.3f} "
            f"BEFORE shell={row['before_fixed_euclidean']['shell']['mean_dimension']:.3f} "
            f"gp={row['before_fixed_euclidean']['gp']['dimension']:.3f} "
            f"[{bb['d_lo']:.2f},{bb['d_hi']:.2f}] contains={row['before_fixed_euclidean']['contains_truth']}  |  "
            f"AFTER shell={row['after_scale_aware']['shell']['mean_dimension']:.3f} "
            f"gp_glob={row['after_scale_aware']['gp_global_mahalanobis']['dimension']:.3f} "
            f"[{ab_g['d_lo']:.2f},{ab_g['d_hi']:.2f}] contains={row['after_scale_aware']['contains_truth_global_gp']}  "
            f"gp_loc={row['after_scale_aware']['gp_local_mahalanobis']['dimension']:.3f} "
            f"[{ab_l['d_lo']:.2f},{ab_l['d_hi']:.2f}] contains={row['after_scale_aware']['contains_truth_local_gp']}",
            flush=True,
        )

    print("=" * 100)
    print("PART B -- the refutation case: A, A@diag(100,100), A@diag(1,0.01)")
    print("=" * 100)

    a = np.random.default_rng(11).random((1600, 2))
    aniso_cases = [
        ("A = uniform_square(1600,11)", a, 2.0),
        ("A @ diag(100,100) [isotropic control]", a * 100.0, 2.0),
        ("A @ diag(1,0.01) [refutation: y in km]", a * np.array([1.0, 0.01]), 2.0),
    ]
    aniso_rows = []
    for name, pts, truth in aniso_cases:
        row = run_case(name, pts, truth, k=K, k0=K0, iterations=ITER, max_radius=MAX_RADIUS, samples=SAMPLES)
        rows.append(row)
        aniso_rows.append(row)
        _flush(rows, sweep)
        bb = row["before_fixed_euclidean"]["bracket"]
        ab_g = row["after_scale_aware"]["bracket_global_gp"]
        ab_l = row["after_scale_aware"]["bracket_local_gp"]
        print(
            f"{name:<40}\n"
            f"  official certify(): {row['before_official_certify']['verdict']} "
            f"[{row['before_official_certify']['d_lo']}, {row['before_official_certify']['d_hi']}] "
            f"signals={row['before_official_certify']['signals']}\n"
            f"  BEFORE (fixed k={K}, Euclidean): shell={row['before_fixed_euclidean']['shell']['mean_dimension']:.4f} "
            f"gp={row['before_fixed_euclidean']['gp']['dimension']:.4f} -> [{bb['d_lo']:.4f},{bb['d_hi']:.4f}]\n"
            f"  AFTER (scale-aware, global-boot, {ITER} local rounds): shell={row['after_scale_aware']['shell']['mean_dimension']:.4f} "
            f"gp_global={row['after_scale_aware']['gp_global_mahalanobis']['dimension']:.4f} -> [{ab_g['d_lo']:.4f},{ab_g['d_hi']:.4f}] "
            f"gp_local={row['after_scale_aware']['gp_local_mahalanobis']['dimension']:.4f} -> [{ab_l['d_lo']:.4f},{ab_l['d_hi']:.4f}]\n",
            flush=True,
        )

    ident_row = aniso_rows[0]
    aniso_row = aniso_rows[2]
    ident_after = ident_row["after_scale_aware"]["bracket_local_gp"]
    aniso_after = aniso_row["after_scale_aware"]["bracket_local_gp"]
    compatible = overlaps(ident_after, aniso_after)
    refutation_verdict = {
        "before_disjoint": not overlaps(ident_row["before_fixed_euclidean"]["bracket"], aniso_row["before_fixed_euclidean"]["bracket"]),
        "after_local_gp_compatible": compatible,
        "ident_after_local": ident_after,
        "aniso_after_local": aniso_after,
    }
    print(json.dumps(refutation_verdict, indent=2, default=str))
    rows.append({"case": "REFUTATION_VERDICT_SUMMARY", "result": refutation_verdict})
    _flush(rows, sweep)

    print("=" * 100)
    print("PART C -- naive-local bootstrap negative control (refutation case only)")
    print("=" * 100)
    naive_row = run_case(
        "A @ diag(1,0.01), bootstrap=euclidean (negative control)",
        a * np.array([1.0, 0.01]),
        2.0,
        k=K,
        k0=K0,
        iterations=ITER,
        max_radius=MAX_RADIUS,
        samples=SAMPLES,
        bootstrap="euclidean",
    )
    rows.append(naive_row)
    _flush(rows, sweep)
    nb_g = naive_row["after_scale_aware"]["bracket_global_gp"]
    nb_l = naive_row["after_scale_aware"]["bracket_local_gp"]
    print(
        f"naive euclidean-bootstrap: shell={naive_row['after_scale_aware']['shell']['mean_dimension']:.4f} "
        f"gp_global={naive_row['after_scale_aware']['gp_global_mahalanobis']['dimension']:.4f} -> [{nb_g['d_lo']:.4f},{nb_g['d_hi']:.4f}] "
        f"contains_2={naive_row['after_scale_aware']['contains_truth_global_gp']}  "
        f"gp_local={naive_row['after_scale_aware']['gp_local_mahalanobis']['dimension']:.4f} -> [{nb_l['d_lo']:.4f},{nb_l['d_hi']:.4f}] "
        f"contains_2={naive_row['after_scale_aware']['contains_truth_local_gp']}",
        flush=True,
    )

    print("=" * 100)
    print("PART D -- Takens/Lorenz delay embedding at lag=1 (m=3, n=1600)")
    print("=" * 100)
    x_series = lorenz_x_series(1600 + 2 + 4000, burn_in=2000)
    lorenz_pts = takens_embedding(x_series, m=3, tau=1, n=1600)
    lorenz_pts = lorenz_pts / 30.0  # match targets.py's normalisation convention
    lorenz_row = run_case(
        "Lorenz-x Takens embedding, tau=1, m=3, n=1600",
        lorenz_pts,
        2.06,
        k=K,
        k0=K0,
        iterations=ITER,
        max_radius=MAX_RADIUS,
        samples=SAMPLES,
        run_local_gp=True,
    )
    rows.append(lorenz_row)
    _flush(rows, sweep)
    lb = lorenz_row["before_fixed_euclidean"]["bracket"]
    la_g = lorenz_row["after_scale_aware"]["bracket_global_gp"]
    la_l = lorenz_row["after_scale_aware"]["bracket_local_gp"]
    print(
        f"official certify(): {lorenz_row['before_official_certify']['verdict']} "
        f"[{lorenz_row['before_official_certify']['d_lo']}, {lorenz_row['before_official_certify']['d_hi']}] "
        f"shell={lorenz_row['before_official_certify']['shell']:.4f} gp={lorenz_row['before_official_certify']['gp']:.4f}\n"
        f"BEFORE (fixed euclid): shell={lorenz_row['before_fixed_euclidean']['shell']['mean_dimension']:.4f} "
        f"gp={lorenz_row['before_fixed_euclidean']['gp']['dimension']:.4f} -> [{lb['d_lo']:.4f},{lb['d_hi']:.4f}]\n"
        f"AFTER (scale-aware): shell={lorenz_row['after_scale_aware']['shell']['mean_dimension']:.4f} "
        f"gp_global={lorenz_row['after_scale_aware']['gp_global_mahalanobis']['dimension']:.4f} -> [{la_g['d_lo']:.4f},{la_g['d_hi']:.4f}] "
        f"gp_local={lorenz_row['after_scale_aware']['gp_local_mahalanobis']['dimension']:.4f} -> [{la_l['d_lo']:.4f},{la_l['d_hi']:.4f}]",
        flush=True,
    )

    print("=" * 100)
    print("PART E -- iteration-count / k0 sweep on the refutation case (open question: how many rounds are enough?)")
    print("=" * 100)
    for iters in (0, 1, 2, 3):
        sa = scale_aware_neighbors(a * np.array([1.0, 0.01]), K, k0=K0, iterations=iters, bootstrap="global")
        hg = build_hypergraph(sa.neighbor_idx)
        se = shell_estimate(hg, max_radius=MAX_RADIUS, samples=SAMPLES)
        gp = gp_local_mahalanobis(a * np.array([1.0, 0.01]), sa.sigma_inv)
        entry = {
            "param": "iterations",
            "value": iters,
            "shell": se["mean_dimension"],
            "gp_local": gp["dimension"],
            "wall_time_s": sa.diagnostics["wall_time_s"],
            "final_cond_median": sa.diagnostics.get("final_cond_number_median"),
        }
        sweep.append(entry)
        _flush(rows, sweep)
        print(f"  iterations={iters}: shell={se['mean_dimension']:.4f} gp_local={gp['dimension']:.4f} "
              f"wall={sa.diagnostics['wall_time_s']:.2f}s cond_med={sa.diagnostics.get('final_cond_number_median')}")

    for k0v in (10, 20, 30, 50):
        sa = scale_aware_neighbors(a * np.array([1.0, 0.01]), K, k0=k0v, iterations=ITER, bootstrap="global")
        hg = build_hypergraph(sa.neighbor_idx)
        se = shell_estimate(hg, max_radius=MAX_RADIUS, samples=SAMPLES)
        gp = gp_local_mahalanobis(a * np.array([1.0, 0.01]), sa.sigma_inv)
        entry = {
            "param": "k0",
            "value": k0v,
            "shell": se["mean_dimension"],
            "gp_local": gp["dimension"],
            "wall_time_s": sa.diagnostics["wall_time_s"],
        }
        sweep.append(entry)
        _flush(rows, sweep)
        print(f"  k0={k0v}: shell={se['mean_dimension']:.4f} gp_local={gp['dimension']:.4f} wall={sa.diagnostics['wall_time_s']:.2f}s")

    _flush(rows, sweep)
    print(f"\nwritten: {RESULTS_PATH}\nwritten: {SWEEP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
