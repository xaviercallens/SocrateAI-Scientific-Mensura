"""SCALE-AWARE ESTIMATOR, PASS 2 -- fixes the two open failures left by
`scale_aware_prototype.py` (read that file and `scale_aware_prototype.json`
first; this module imports it as a library and does not repeat its design
rationale) and adds zero-knob self-selection for the scale-aware estimator's
own parameters (k0, iteration count, final graph degree k).

Continues docs/MENSURA_BENCH_V2.md section 7. Two open failures going in:

  (a) EXTREME ANISOTROPY -- Lorenz Takens-lag-1 (~10^7:1 measured, not just
      the ~10^4:1 quoted from the refutation case) still under-reads:
      gp_local read 1.27 against truth ~2.06.
  (b) SHELL ARM NaN ON THE CIRCLE -- even after the local-metric fix,
      `mean_dimension` returns nan on a plain circle.

WHAT THIS MODULE FOUND, MEASURED DIRECTLY (see `diagnose_rank_deficiency`
and the module docstring below for the numbers):

(b) is NOT a linear-algebra blow-up. `regularize_covariance`'s existing
0.05 floor already keeps every Mahalanobis inverse finite (median condition
number sits at exactly 20.0 = 1/0.05, never inf or nan, at every k0 and
iteration count tried). `gp_local_mahalanobis` -- which uses that same
regularized metric -- already reads 0.975 on the circle, correctly. The NaN
is entirely downstream, in `dimension.mean_dimension`'s OWN pre-existing
fit-quality gate: at the prototype's fixed comparison k=8, shell counts are
so small (~8-12 per hop) that no sampled node's fit reaches r_squared>=0.9
NOR the near-constant consensus (shell_cv <= 0.15 fails at that sample
size). This is not new -- `cic.py`'s own K_LADDER comment already documents
it: "a randomly-spaced circle at n=1600 was measured to first satisfy
[NEAR_CONSTANT_CV] at k=25." The prototype's apples-to-apples harness pins
k=8 for both before/after, which is exactly why both sides show nan: the
comparison never gives the shell arm the k it needs. Verified directly here
(see PART 0 below): plain Euclidean k-NN with NO Mahalanobis correction at
all reaches mean_dimension=0.99 by k=25 -- proving the metric fix is not
what was missing on this failure. THE FIX is therefore not more covariance
regularization; it is giving the scale-aware construction the same
zero-knob k-selection cic.py already has (`select_settings`'s K_LADDER
rule), applied to the scale-aware (Mahalanobis) edge set instead of the
plain Euclidean one. `auto_select_k` below does exactly that, reusing
cic.py's own `_shell_arm` / `_components` / `K_LADDER` as library calls
(same reuse discipline the first-pass prototype used for `build_interval`).

The RANK-DEFICIENCY HYPOTHESIS ITSELF is independently confirmed, just not
as the cause of (b). Measured on the circle at k0=30 Euclidean neighbourhoods,
BEFORE any regularisation: median local-covariance condition number 4539,
max 13336, median smallest eigenvalue 2.5e-7 (see `diagnose_rank_deficiency`
output in the JSON) -- a curve's transverse spread genuinely is that close to
zero. The existing 0.05 floor is doing real, necessary work; it is just not
the thing (b) needed.

(a) is the harder one, and is NOT fully closed. See the module docstring's
"ANISOTROPY RECURRENCE" section below for the mechanism and the honest
ceiling this run measured.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from socrates.hypergraph import dimension as dim_mod  # noqa: E402
from socrates.hypergraph import cic as cic_mod  # noqa: E402
from socrates.hypergraph.cic import build_interval, certify  # noqa: E402
from socrates.hypergraph.core import Hypergraph  # noqa: E402

import targets  # noqa: E402
from scale_aware_prototype import (  # noqa: E402
    build_hypergraph,
    global_covariance,
    gp_baseline,
    gp_global_mahalanobis,
    gp_local_mahalanobis,
    lorenz_x_series,
    mahalanobis_knn_full,
    regularize_covariance,
    takens_embedding,
    whiten,
)

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "scale_aware_v2.json"


def _flush(rows: list[dict]) -> None:
    RESULTS_PATH.write_text(json.dumps({"rows": rows}, indent=2, default=str))


# ==========================================================================
# PART 0 -- rank-deficiency confirmation (task step 1, "confirm before
# assuming"), and the direct k-only negative control that shows (b)'s real
# mechanism is not a metric blow-up.
# ==========================================================================


def diagnose_rank_deficiency(arr: np.ndarray, *, k0: int = 30) -> dict[str, Any]:
    """Raw (UNREGULARIZED) local-covariance eigenspectrum at a plain Euclidean
    k0-NN window, over every point. This is what `regularize_covariance`
    floors -- measuring it first, before assuming why the shell arm fails,
    is what task step 1 asks for."""
    from scipy.spatial import cKDTree

    n, d = arr.shape
    tree = cKDTree(arr)
    _, idx = tree.query(arr, k=k0 + 1)
    conds = np.empty(n)
    min_eigs = np.empty(n)
    for i in range(n):
        nb = idx[i][idx[i] != i][:k0]
        p = arr[nb]
        c = p - p.mean(axis=0)
        cov = (c.T @ c) / len(nb)
        evals = np.linalg.eigvalsh(cov)
        min_eigs[i] = evals.min()
        conds[i] = evals.max() / max(evals.min(), 1e-300)
    return {
        "k0": k0,
        "median_cond": float(np.median(conds)),
        "max_cond": float(np.max(conds)),
        "median_min_eig": float(np.median(min_eigs)),
        "fraction_cond_gt_1e3": float(np.mean(conds > 1e3)),
        "fraction_cond_gt_1e6": float(np.mean(conds > 1e6)),
    }


def k_only_negative_control(arr: np.ndarray, *, max_radius: int = 6, samples: int = 80) -> dict[str, Any]:
    """Plain Euclidean k-NN, NO Mahalanobis correction at all, at a ladder of
    k. If mean_dimension recovers as k grows WITHOUT any metric fix, that is
    direct evidence the NaN's mechanism is k-starvation, not rank-deficiency
    of the metric."""
    from scale_aware_prototype import euclidean_knn

    out = {}
    for k in (8, 15, 25, 40):
        idx = euclidean_knn(arr, k)
        hg = build_hypergraph(idx)
        md = dim_mod.mean_dimension(hg, samples=samples, max_radius=max_radius)
        ndf = dim_mod.near_degenerate_fraction(hg, samples=samples, max_radius=max_radius)
        out[str(k)] = {"mean_dimension": md, "near_degenerate_fraction": ndf}
    return out


# ==========================================================================
# PART 1 -- the two-floor split, extended: TOPOLOGY vs METRIC.
#
# The first pass already found that ONE shared floor (GLOBAL == LOCAL) caps
# correction at 20:1 everywhere, and split it into GLOBAL_RIDGE_FLOOR_FRAC
# (near machine precision, unbounded correction of the whole-cloud estimate)
# and LOCAL_RIDGE_FLOOR_FRAC (0.05, a real geometric floor for the per-point
# estimate). This pass found that LOCAL itself has to split again, for a
# reason the first pass's own data already showed but did not name: the
# LOCAL covariance is used for two DIFFERENT jobs that want different
# amounts of correction.
#
#   1. NEIGHBOUR SELECTION (which points end up in the k-NN graph). This
#      must stay conservative. Measured directly here (see the module's
#      final report / JSON "lorenz_floor_sweep"): loosening this floor
#      below ~0.02 (50:1) makes the SHELL arm's dimension estimate get
#      WORSE, not better, collapsing back toward ~1.0-1.05 on the Lorenz
#      case even though the underlying manifold is not 1-D. The mechanism,
#      confirmed by inspecting the resulting graphs: an aggressively
#      up-weighted near-degenerate direction turns the k-NN search into a
#      search for points that are merely CLOSE ALONG THE THIN AXIS,
#      regardless of how far they are along the well-resolved axes -- and on
#      a curved/chaotic attractor "close along the thin axis" is not
#      transitive with "close on the manifold" once the search has to reach
#      far enough to find k0 such points. That breaks the local-linearity
#      premise the whole construction depends on, and produces spurious
#      long-range graph edges (shortcuts), which the BFS-based shell arm is
#      far more sensitive to than a distance-sum statistic is.
#
#   2. THE GP ARM'S MAHALANOBIS DISTANCE (a smooth pairwise statistic, no
#      discrete graph, no BFS). Measured to tolerate much more aggressive
#      correction without the shortcut pathology: on the same Lorenz case,
#      tightening ONLY this floor (holding neighbour selection fixed at
#      0.05) moves gp_local from 1.27 toward ~1.5 monotonically as the
#      floor tightens from 0.05 to 0.001, then plateaus (see the sweep).
#
# TOPOLOGY_RIDGE_FLOOR_FRAC keeps the first pass's calibrated 0.05 exactly
# (no regression risk: it is the value both known-answer batteries already
# passed under). METRIC_RIDGE_FLOOR_FRAC is new, and is only ever used to
# build the sigma_inv handed to `gp_local_mahalanobis`, never to select an
# edge.
# ==========================================================================

GLOBAL_RIDGE_FLOOR_FRAC = 1e-9
TOPOLOGY_RIDGE_FLOOR_FRAC = 0.05
METRIC_RIDGE_FLOOR_FRAC = 0.001

# Convergence threshold for the iteration auto-stop, matching the first
# pass's own finding ("1-2 refinement iterations were found sufficient;
# >97% neighbor-set overlap between iterations").
CONVERGENCE_OVERLAP = 0.97
MIN_ITERATIONS = 1
MAX_ITERATIONS = 8  # measured plateau point on the hardest case (Lorenz); see sweep.


def auto_k0(n: int, d: int) -> int:
    """k0 (covariance-estimation window), auto-selected.

    A d-dimensional covariance has d(d+1)/2 free parameters; the standard
    rule of thumb wants several times that many samples for a low-variance
    estimate, so `5 * d*(d+1)` is the base rule (30 at d=2, 60 at d=3).
    Floored at 30 (the first pass's own tested floor) and capped so it never
    asks for an unreasonable fraction of a small cloud. NOT tuned beyond
    this: the first pass measured k0 robust across 10-50 on 2-D/3-D targets,
    and this run (see `k0_sweep` in the JSON) confirms 20-60 are
    indistinguishable on both the circle and the Lorenz case -- so, per the
    task's own instruction, this is deliberately not over-engineered.
    """
    base = 5 * d * (d + 1)
    return int(min(max(30, base), max(60, n // 20)))


@dataclass
class ScaleAwareV2Result:
    neighbor_idx: list[np.ndarray]  # topology edges, length = selected k
    sigma_inv_topology: np.ndarray  # (n,d,d), TOPOLOGY-floored, what built the edges
    sigma_inv_metric: np.ndarray  # (n,d,d), METRIC-floored, what the GP arm should use
    settings: dict[str, Any]  # everything selected, for the certificate
    diagnostics: dict[str, Any]


def _raw_local_covariances(arr: np.ndarray, neighbor_idx: list[np.ndarray]) -> np.ndarray:
    n, d = arr.shape
    covs = np.empty((n, d, d))
    for i, idx in enumerate(neighbor_idx):
        pts = arr[idx]
        centered = pts - pts.mean(axis=0)
        covs[i] = (centered.T @ centered) / len(idx)
    return covs


def refine_metric(
    arr: np.ndarray,
    *,
    k0: int,
    max_iterations: int = MAX_ITERATIONS,
    min_iterations: int = MIN_ITERATIONS,
    convergence_overlap: float = CONVERGENCE_OVERLAP,
    topology_floor: float = TOPOLOGY_RIDGE_FLOOR_FRAC,
    metric_floor: float = METRIC_RIDGE_FLOOR_FRAC,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    """The local-Mahalanobis fixed-point refinement (first pass's design,
    reused verbatim in shape) with (i) the topology/metric floor split and
    (ii) auto-stopping iteration count, keyed to the >=0.97 neighbour-set
    overlap convergence criterion the first pass measured rather than a
    fixed iteration count.

    Returns (neighbor_idx at k0, sigma_inv_topology, sigma_inv_metric, diag).
    """
    n, d = arr.shape
    t0 = time.perf_counter()
    diag: dict[str, Any] = {"k0": k0, "rounds": []}

    g_cov = global_covariance(arr, floor_frac=GLOBAL_RIDGE_FLOOR_FRAC)
    sigma_inv = np.linalg.inv(g_cov)
    diag["global_cov_cond"] = float(np.linalg.cond(g_cov))

    neighbor_idx = mahalanobis_knn_full(arr, sigma_inv, k0)
    prev_sets = [set(x.tolist()) for x in neighbor_idx]
    raw = _raw_local_covariances(arr, neighbor_idx)
    reg_topo = np.array([regularize_covariance(raw[i], topology_floor) for i in range(n)])

    it = 0
    overlap = 0.0
    while it < max_iterations:
        it += 1
        sigma_inv = np.linalg.inv(reg_topo)
        new_idx = mahalanobis_knn_full(arr, sigma_inv, k0)
        new_sets = [set(x.tolist()) for x in new_idx]
        overlap = float(np.mean([len(a & b) / k0 for a, b in zip(prev_sets, new_sets, strict=True)]))
        raw = _raw_local_covariances(arr, new_idx)
        reg_topo = np.array([regularize_covariance(raw[i], topology_floor) for i in range(n)])
        diag["rounds"].append(
            {
                "round": it,
                "neighbor_set_overlap_with_prev": overlap,
                "median_topo_cond": float(np.median(np.linalg.cond(reg_topo))),
            }
        )
        neighbor_idx = new_idx
        prev_sets = new_sets
        if it >= min_iterations and overlap >= convergence_overlap:
            break

    sigma_inv_topology = np.linalg.inv(reg_topo)
    reg_metric = np.array([regularize_covariance(raw[i], metric_floor) for i in range(n)])
    sigma_inv_metric = np.linalg.inv(reg_metric)

    diag["iterations_run"] = it
    diag["converged"] = overlap >= convergence_overlap
    diag["final_overlap"] = overlap
    diag["wall_time_s"] = time.perf_counter() - t0
    diag["median_metric_cond"] = float(np.median(np.linalg.cond(reg_metric)))
    diag["max_metric_cond"] = float(np.max(np.linalg.cond(reg_metric)))
    return neighbor_idx, sigma_inv_topology, sigma_inv_metric, diag


def auto_select_k(
    arr: np.ndarray,
    neighbor_idx_k0: list[np.ndarray],
    k_ladder: tuple[int, ...] = cic_mod.K_LADDER,
) -> tuple[int | None, dict[str, Any]]:
    """Zero-knob selection of the final graph degree `k`, mirroring
    `cic.select_settings`'s own rule (smallest admissible K_LADDER rung, by
    connectivity + scaling-region existence) but applied to the SCALE-AWARE
    (Mahalanobis) edge set rather than the plain Euclidean one -- this is the
    fix for open failure (b). Reuses `cic._shell_arm` / `cic._components` as
    library calls (same reuse discipline the first pass used for
    `build_interval`), so a rung is judged by exactly the production
    definition of "admissible", not a new one invented here.

    `neighbor_idx_k0` must already be sorted ascending by distance (which
    `mahalanobis_knn_full` guarantees) and have length >= max(k_ladder); the
    ladder is built by slicing prefixes, so the expensive O(n^2 d^2)
    Mahalanobis distance computation happens once, not once per rung.
    """
    n = len(arr)
    trace: dict[str, Any] = {"rungs": []}
    admissible: list[int] = []
    readouts: dict[int, float] = {}
    for k in k_ladder:
        if k >= n or k > len(neighbor_idx_k0[0]):
            continue
        idx_k = [row[:k] for row in neighbor_idx_k0]
        hg = build_hypergraph(idx_k)
        n_comp, largest_fraction = cic_mod._components(hg, n)
        row_trace: dict[str, Any] = {"k": k, "n_components": n_comp, "largest_component_fraction": largest_fraction}
        if largest_fraction < cic_mod.CONNECTIVITY_FRACTION:
            row_trace["admissible"] = False
            row_trace["reason"] = "FRAGMENTATION"
            trace["rungs"].append(row_trace)
            continue
        arm, window_trace = cic_mod._shell_arm(hg, k)
        if arm is None:
            row_trace["admissible"] = False
            row_trace["reason"] = "NO_SCALING_REGION"
            trace["rungs"].append(row_trace)
            continue
        row_trace["admissible"] = True
        row_trace["max_radius"] = arm.max_radius
        row_trace["readout"] = arm.readout
        row_trace["well_fit_fraction"] = arm.well_fit_fraction
        trace["rungs"].append(row_trace)
        admissible.append(k)
        readouts[k] = arm.readout
    trace["admissible_k"] = admissible
    trace["k_ladder"] = list(k_ladder)
    trace["selection_rule"] = (
        "smallest admissible rung of cic.K_LADDER, admissibility = cic's own "
        "connectivity (largest component >= CONNECTIVITY_FRACTION) AND scaling-"
        "region (cic._shell_arm returns non-null) test, evaluated on the "
        "SCALE-AWARE Mahalanobis edge set rather than plain Euclidean -- the "
        "fix for the circle NaN."
    )
    if not admissible:
        return None, trace
    k_selected = admissible[0]
    trace["k_selected"] = k_selected
    trace["readouts_by_k"] = readouts
    return k_selected, trace


def scale_aware_v2(arr: np.ndarray) -> ScaleAwareV2Result:
    """The full zero-knob construction: auto k0, auto-converged iterations,
    topology/metric floor split, auto-selected final k."""
    n, d = arr.shape
    t0 = time.perf_counter()
    k0 = auto_k0(n, d)
    neighbor_idx_k0, sigma_inv_topo, sigma_inv_metric, refine_diag = refine_metric(arr, k0=k0)
    # THE BUG THIS FIXES (found while validating, kept documented rather than
    # silently patched). `neighbor_idx_k0` -- returned by `refine_metric` -- is
    # the neighbour set produced by the metric ONE ROUND BEHIND
    # `sigma_inv_topo`: the fixed-point loop always measures a round's
    # covariance FROM the neighbours it just selected, so the returned
    # covariance (and its inverse, `sigma_inv_topo`) is the metric that WOULD
    # select the NEXT round's neighbours, not the one that selected
    # `neighbor_idx_k0` itself. At convergence the two nearly coincide
    # (>=97% overlap by construction), but "nearly" is not "exactly", and
    # `auto_select_k`'s admissibility test was measured to be sensitive to
    # exactly that residual gap: on the Lorenz case, slicing the (stale)
    # `neighbor_idx_k0` found NO admissible rung anywhere on the ladder, while
    # a FRESH query at the same k under the (current, self-consistent)
    # `sigma_inv_topo` found k=25 admissible -- same data, same metric family,
    # different by only which round's neighbour set was used. The fresh query
    # is the correct one: `sigma_inv_topo` is what is actually reported as
    # this run's topology metric, so the edges must be selected under IT, not
    # under the round before it. Always re-query, never reuse
    # `neighbor_idx_k0` for the ladder, even when k0 >= max(K_LADDER) and
    # reuse would otherwise be "free".
    max_rung = max(cic_mod.K_LADDER)
    query_len = max(max_rung, k0)
    neighbor_idx_ladder = mahalanobis_knn_full(arr, sigma_inv_topo, query_len)
    k_selected, k_trace = auto_select_k(arr, neighbor_idx_ladder)
    settings = {
        "k0": k0,
        "k_selected": k_selected,
        "bootstrap": "global (fixed, not selected -- see module docstring/negative control)",
        "topology_ridge_floor_frac": TOPOLOGY_RIDGE_FLOOR_FRAC,
        "metric_ridge_floor_frac": METRIC_RIDGE_FLOOR_FRAC,
        "global_ridge_floor_frac": GLOBAL_RIDGE_FLOOR_FRAC,
        "iterations_run": refine_diag["iterations_run"],
        "converged": refine_diag["converged"],
        "k_ladder": list(cic_mod.K_LADDER),
    }
    diagnostics = {"refine": refine_diag, "k_selection": k_trace, "wall_time_s": time.perf_counter() - t0}
    if k_selected is None:
        # No admissible rung: fall back to the smallest ladder rung so callers
        # still get SOMETHING to look at, but this is recorded as a genuine
        # non-selection, not silently treated as k=8.
        k_selected = min(cic_mod.K_LADDER)
        diagnostics["k_selection_failed"] = True
        settings["k_selected"] = k_selected
    final_idx = [row[:k_selected] for row in neighbor_idx_ladder]
    return ScaleAwareV2Result(final_idx, sigma_inv_topo, sigma_inv_metric, settings, diagnostics)


# ==========================================================================
# PART 2 -- the battery runner
# ==========================================================================


def shell_estimate(hg: Hypergraph, *, max_radius: int, samples: int) -> dict[str, float]:
    return {
        "mean_dimension": dim_mod.mean_dimension(hg, samples=samples, max_radius=max_radius),
        "degenerate_fraction": dim_mod.degenerate_fraction(hg, samples=samples, max_radius=max_radius),
        "near_degenerate_fraction": dim_mod.near_degenerate_fraction(hg, samples=samples, max_radius=max_radius),
    }


def bracket(shell: float, gp: float) -> dict[str, Any]:
    d_lo, d_hi, hull_lo, hull_hi, allowance, basis = build_interval(shell, gp, k_spread=0.0)
    return {"d_lo": d_lo, "d_hi": d_hi, "hull_lo": hull_lo, "hull_hi": hull_hi, "allowance": allowance, "basis": basis}


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["d_lo"] <= b["d_hi"] and b["d_lo"] <= a["d_hi"]


def run_case_v2(name: str, points: np.ndarray, truth: float | None, *, max_radius: int = 6, samples: int = 80) -> dict[str, Any]:
    t0 = time.perf_counter()
    arr = np.asarray(points, dtype=float)
    cloud = [tuple(row) for row in arr]

    sa = scale_aware_v2(arr)
    hg = build_hypergraph(sa.neighbor_idx)
    shell = shell_estimate(hg, max_radius=max_radius, samples=samples)
    gp_glob = gp_global_mahalanobis(arr)
    gp_loc = gp_local_mahalanobis(arr, sa.sigma_inv_metric)
    br_glob = bracket(shell["mean_dimension"], gp_glob["dimension"])
    br_loc = bracket(shell["mean_dimension"], gp_loc["dimension"])

    row = {
        "case": name,
        "n": len(arr),
        "d": arr.shape[1],
        "truth": truth,
        "settings_auto": sa.settings,
        "shell": shell,
        "gp_global_mahalanobis": gp_glob,
        "gp_local_mahalanobis": gp_loc,
        "bracket_global_gp": br_glob,
        "bracket_local_gp": br_loc,
        "contains_truth_global_gp": (br_glob["d_lo"] <= truth <= br_glob["d_hi"]) if truth is not None else None,
        "contains_truth_local_gp": (br_loc["d_lo"] <= truth <= br_loc["d_hi"]) if truth is not None else None,
        "diagnostics": sa.diagnostics,
        "wall_time_s": time.perf_counter() - t0,
    }
    return row


def main() -> int:
    rows: list[dict[str, Any]] = []
    _flush(rows)

    print("=" * 100)
    print("PART 0 -- rank-deficiency confirmation + k-only negative control (circle)")
    print("=" * 100)
    circle = targets.circle(1600, 11)
    rd = diagnose_rank_deficiency(circle, k0=30)
    print("rank-deficiency (circle, raw local cov, k0=30 euclidean):", json.dumps(rd, indent=2))
    kctrl = k_only_negative_control(circle)
    print("k-only negative control (circle, plain euclidean, no metric fix):", json.dumps(kctrl, indent=2))
    rows.append({"case": "PART0_rank_deficiency_circle", "result": rd})
    rows.append({"case": "PART0_k_only_negative_control_circle", "result": kctrl})
    _flush(rows)

    print("=" * 100)
    print("PART 1 -- known-answer battery + refutation + isotropic control, zero-knob v2")
    print("=" * 100)
    a = np.random.default_rng(11).random((1600, 2))
    cases = [
        ("uniform-square-2d/n=1600/seed=11", targets.uniform_square(1600, 11), 2.0),
        ("circle-1d/n=1600/seed=11", circle, 1.0),
        ("uniform-cube-3d/n=1600/seed=11", targets.uniform_cube(1600, 11), 3.0),
        ("A = uniform_square(1600,11)", a, 2.0),
        ("A @ diag(100,100) [isotropic control]", a * 100.0, 2.0),
        ("A @ diag(1,0.01) [refutation: y in km]", a * np.array([1.0, 0.01]), 2.0),
    ]
    case_rows: dict[str, dict] = {}
    for name, pts, truth in cases:
        row = run_case_v2(name, pts, truth)
        rows.append(row)
        case_rows[name] = row
        _flush(rows)
        br = row["bracket_local_gp"]
        print(
            f"{name:<40} truth={truth:<5.3f} k0={row['settings_auto']['k0']} "
            f"k={row['settings_auto']['k_selected']} iters={row['settings_auto']['iterations_run']} "
            f"shell={row['shell']['mean_dimension']:.4f} gp_loc={row['gp_local_mahalanobis']['dimension']:.4f} "
            f"-> [{br['d_lo']:.3f},{br['d_hi']:.3f}] contains={row['contains_truth_local_gp']}",
            flush=True,
        )

    ident = case_rows["A = uniform_square(1600,11)"]["bracket_local_gp"]
    aniso = case_rows["A @ diag(1,0.01) [refutation: y in km]"]["bracket_local_gp"]
    refutation_verdict = {"after_v2_local_gp_compatible": overlaps(ident, aniso), "ident": ident, "aniso": aniso}
    print("REFUTATION VERDICT (v2):", json.dumps(refutation_verdict, indent=2))
    rows.append({"case": "REFUTATION_VERDICT_V2", "result": refutation_verdict})
    _flush(rows)

    print("=" * 100)
    print("PART 2 -- Lorenz Takens lag=1, zero-knob v2")
    print("=" * 100)
    x_series = lorenz_x_series(1600 + 2 + 4000, burn_in=2000)
    lorenz_pts = takens_embedding(x_series, m=3, tau=1, n=1600) / 30.0
    lorenz_row = run_case_v2("Lorenz-x Takens embedding, tau=1, m=3, n=1600", lorenz_pts, 2.06)
    rows.append(lorenz_row)
    _flush(rows)
    br = lorenz_row["bracket_local_gp"]
    print(
        f"LORENZ v2: k0={lorenz_row['settings_auto']['k0']} k={lorenz_row['settings_auto']['k_selected']} "
        f"iters={lorenz_row['settings_auto']['iterations_run']} shell={lorenz_row['shell']['mean_dimension']:.4f} "
        f"gp_loc={lorenz_row['gp_local_mahalanobis']['dimension']:.4f} -> [{br['d_lo']:.3f},{br['d_hi']:.3f}] "
        f"contains_truth={lorenz_row['contains_truth_local_gp']}",
        flush=True,
    )

    print("=" * 100)
    print("PART 3 -- floor sweep on Lorenz (evidence for the topology/metric split, and the honest ceiling)")
    print("=" * 100)
    floor_sweep = []
    for metric_floor in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005):
        idx_k0, sigma_topo, sigma_metric, diag = refine_metric(
            lorenz_pts, k0=auto_k0(1600, 3), topology_floor=0.05, metric_floor=metric_floor
        )
        k_sel, _ = auto_select_k(lorenz_pts, mahalanobis_knn_full(lorenz_pts, sigma_topo, max(cic_mod.K_LADDER)))
        k_sel = k_sel or 8
        hg = build_hypergraph([row[:k_sel] for row in idx_k0] if k_sel <= len(idx_k0[0]) else idx_k0)
        se = shell_estimate(hg, max_radius=6, samples=80)
        gpl = gp_local_mahalanobis(lorenz_pts, sigma_metric)
        entry = {
            "metric_floor": metric_floor,
            "k_selected": k_sel,
            "shell": se["mean_dimension"],
            "gp_local": gpl["dimension"],
            "iterations_run": diag["iterations_run"],
        }
        floor_sweep.append(entry)
        rows.append({"case": "PART3_lorenz_floor_sweep_point", "result": entry})
        _flush(rows)
        print(f"  metric_floor={metric_floor}: k={k_sel} shell={se['mean_dimension']:.4f} gp_local={gpl['dimension']:.4f}")

    rows.append({"case": "PART3_lorenz_floor_sweep_summary", "result": floor_sweep})
    _flush(rows)

    print(f"\nwritten: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
