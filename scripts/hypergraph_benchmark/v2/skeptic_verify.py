"""INDEPENDENT SKEPTIC VERIFICATION of scale_aware_v2.py's round-2 claims.

This is deliberately NOT a rerun of scale_aware_v2.main() or
scale_aware_prototype.main(). It imports the library functions under test
(scale_aware_v2, refine_metric, auto_select_k, gp_local_mahalanobis, shell
estimate, build_interval/bracket via cic.py) exactly as any caller would, but
drives them with:

  - independently generated point clouds (different RNG seeds / different
    Lorenz initial conditions than the prototype used, an independent RK4
    Lorenz integrator instead of the prototype's Euler one),
  - adversarial cases the prototype never tried (rescale sweeps, rotation,
    2-axis/3-axis anisotropy, generalization targets),
  - independent containment checks (plain float comparison, not trusting the
    prototype's own "contains_truth" field),
  - a direct empirical test of the claimed stale-neighbor-set bug by calling
    refine_metric() directly and comparing admissibility under the STALE
    returned neighbor set vs a FRESH re-query under the reported metric.

Each stage writes its own JSON immediately (E6 lesson: on-disk artifacts
survive, agents don't). Run with `--stage NAME`; `--stage all` runs
everything sequentially (slow, ~15-20 min).

Does not import or modify dimension.py, baseline.py, pointcloud.py, cic.py
(read-only library use only, same discipline as the prototype files).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from socrates.hypergraph import cic as cic_mod  # noqa: E402
from socrates.hypergraph.cic import build_interval  # noqa: E402

import targets  # noqa: E402
from scale_aware_prototype import (  # noqa: E402
    build_hypergraph,
    euclidean_knn,
    gp_local_mahalanobis,
    mahalanobis_knn_full,
    regularize_covariance,
    shell_estimate,
)
from scale_aware_v2 import (  # noqa: E402
    auto_k0,
    auto_select_k,
    refine_metric,
    scale_aware_v2,
    GLOBAL_RIDGE_FLOOR_FRAC,
    TOPOLOGY_RIDGE_FLOOR_FRAC,
    METRIC_RIDGE_FLOOR_FRAC,
)

OUT_DIR = Path(__file__).resolve().parent


def _write(name: str, obj: Any) -> None:
    path = OUT_DIR / f"skeptic_{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  [written {path.name}]", flush=True)


def bracket(shell: float, gp: float) -> dict[str, Any]:
    d_lo, d_hi, hull_lo, hull_hi, allowance, basis = build_interval(shell, gp, k_spread=0.0)
    return {"d_lo": d_lo, "d_hi": d_hi, "hull_lo": hull_lo, "hull_hi": hull_hi, "allowance": allowance, "basis": basis}


def contains(br: dict[str, Any], truth: float) -> bool:
    # independent containment check -- plain float comparison, not trusting
    # any field the system under test computed for us.
    return float(br["d_lo"]) <= float(truth) <= float(br["d_hi"])


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["d_lo"] <= b["d_hi"] and b["d_lo"] <= a["d_hi"]


def run_full_pipeline(arr: np.ndarray, *, max_radius: int = 6, samples: int = 80) -> dict[str, Any]:
    """One full zero-knob scale-aware-v2 pass, independent driver (not
    run_case_v2): computes shell + gp_local + bracket ourselves from the
    returned ScaleAwareV2Result, so nothing about the interval or containment
    check is inherited from the prototype's own reporting code."""
    t0 = time.perf_counter()
    sa = scale_aware_v2(arr)
    hg = build_hypergraph(sa.neighbor_idx)
    se = shell_estimate(hg, max_radius=max_radius, samples=samples)
    gpl = gp_local_mahalanobis(arr, sa.sigma_inv_metric)
    br = bracket(se["mean_dimension"], gpl["dimension"])
    return {
        "n": len(arr),
        "d": arr.shape[1],
        "settings_auto": sa.settings,
        "shell_mean_dimension": se["mean_dimension"],
        "shell_near_degenerate_fraction": se["near_degenerate_fraction"],
        "gp_local_dimension": gpl["dimension"],
        "gp_local_r_squared": gpl["r_squared"],
        "bracket": br,
        "wall_time_s": time.perf_counter() - t0,
        "diagnostics_summary": {
            "iterations_run": sa.diagnostics["refine"]["iterations_run"],
            "converged": sa.diagnostics["refine"]["converged"],
            "k_selection_failed": sa.diagnostics.get("k_selection_failed", False),
        },
    }


# ==========================================================================
# STAGE A -- independent Lorenz replication (claim 5)
# ==========================================================================


def lorenz_rk4(x0: np.ndarray, n_steps: int, *, dt: float = 0.005) -> np.ndarray:
    """Independent RK4 Lorenz integrator (prototype used forward Euler at
    dt=0.004; this is a different, higher-order method at a different step
    size -- deliberately not the same code path)."""
    s, r, b = 10.0, 28.0, 8.0 / 3.0

    def deriv(x: np.ndarray) -> np.ndarray:
        return np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])

    out = np.empty((n_steps, 3))
    x = x0.copy()
    for i in range(n_steps):
        k1 = deriv(x)
        k2 = deriv(x + 0.5 * dt * k1)
        k3 = deriv(x + 0.5 * dt * k2)
        k4 = deriv(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = x
    return out


def lorenz_trajectory(seed: int, n_raw: int, *, burn_in: int = 6000, dt: float = 0.005) -> np.ndarray:
    """Independent initial condition per seed (prototype used a fixed
    (1,1,1) IC, no randomization at all -- every seed here is a genuinely
    different trajectory, not a resample of the same one)."""
    rng = np.random.default_rng(seed)
    x0 = np.array([1.0, 1.0, 1.0]) + rng.uniform(-15.0, 15.0, size=3)
    traj = lorenz_rk4(x0, burn_in + n_raw, dt=dt)
    return traj[burn_in:]


def takens(x: np.ndarray, *, m: int, tau: int, n: int) -> np.ndarray:
    need = (m - 1) * tau + n
    if len(x) < need:
        raise ValueError(f"need {need}, got {len(x)}")
    idx = np.arange(n)
    return np.stack([x[idx + j * tau] for j in range(m)], axis=1)


D_KY_LORENZ = 2.062152


def stage_lorenz(seeds: list[int], lags: list[int], n: int = 1600) -> None:
    print("=" * 100)
    print(f"STAGE A -- independent Lorenz replication: seeds={seeds} lags={lags} n={n}")
    print("=" * 100)
    rows: list[dict[str, Any]] = []
    _write("lorenz", {"rows": rows, "status": "starting"})
    for seed in seeds:
        raw = lorenz_trajectory(seed, n_raw=n + max(lags) * 4 + 50, burn_in=6000, dt=0.005)
        x_series = raw[:, 0] / 30.0  # match prototype's magnitude convention; global isotropic scale, no effect on dimension
        for tau in lags:
            pts = takens(x_series, m=3, tau=tau, n=n)
            result = run_full_pipeline(pts)
            br = result["bracket"]
            ok = contains(br, D_KY_LORENZ)
            row = {
                "seed": seed,
                "lag": tau,
                "n": n,
                "truth_D_KY": D_KY_LORENZ,
                **result,
                "contains_truth_independent_check": ok,
            }
            rows.append(row)
            _write("lorenz", {"rows": rows, "status": "in_progress"})
            print(
                f"  seed={seed} lag={tau}: k0={result['settings_auto']['k0']} k={result['settings_auto']['k_selected']} "
                f"shell={result['shell_mean_dimension']:.4f} gp_loc={result['gp_local_dimension']:.4f} "
                f"-> [{br['d_lo']:.3f},{br['d_hi']:.3f}] contains_D_KY={ok} wall={result['wall_time_s']:.1f}s",
                flush=True,
            )
    _write("lorenz", {"rows": rows, "status": "done"})
    n_ok = sum(1 for r in rows if r["contains_truth_independent_check"])
    print(f"\nSTAGE A SUMMARY: {n_ok}/{len(rows)} (seed,lag) combinations contain D_KY={D_KY_LORENZ}")


# ==========================================================================
# STAGE B -- try to break the refutation-case fix (claim 1)
# ==========================================================================


def rotation_2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotation_3d_random(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # random rotation via QR of a random Gaussian matrix
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def stage_break_refutation() -> None:
    print("=" * 100)
    print("STAGE B -- adversarial rescale/rotation cases the prototype did not try")
    print("=" * 100)
    rows: list[dict[str, Any]] = []
    _write("rescale_battery", {"rows": rows, "status": "starting"})

    a2 = np.random.default_rng(11).random((1600, 2))  # same seed as prototype's ident case for direct comparability
    ident_result = run_full_pipeline(a2)
    ident_br = ident_result["bracket"]
    rows.append({"case": "IDENT_2D_seed11", "truth": 2.0, **ident_result, "contains_truth": contains(ident_br, 2.0)})
    _write("rescale_battery", {"rows": rows, "status": "in_progress"})
    print(f"  IDENT: shell={ident_result['shell_mean_dimension']:.4f} gp={ident_result['gp_local_dimension']:.4f} -> [{ident_br['d_lo']:.3f},{ident_br['d_hi']:.3f}]")

    cases: list[tuple[str, np.ndarray, float]] = []
    for factor in (0.1, 0.001, 0.0001):
        cases.append((f"rescale_y_factor={factor}", a2 * np.array([1.0, factor]), 2.0))
    cases.append(("two_axis_different_rescale_0.1_0.001", a2 * np.array([0.1, 0.001]), 2.0))
    center = a2 - a2.mean(axis=0)
    rotated_aniso = (center * np.array([1.0, 0.01])) @ rotation_2d(0.6435).T  # ~37 degrees, long axis NOT coordinate-aligned
    cases.append(("rescale_0.01_then_rotate_37deg", rotated_aniso, 2.0))

    a3 = np.random.default_rng(11).random((1600, 3))
    cases.append(("3d_anisotropic_diag(1,0.01,0.0001)", a3 * np.array([1.0, 0.01, 0.0001]), 3.0))
    center3 = a3 - a3.mean(axis=0)
    r3 = rotation_3d_random(seed=99)
    cases.append(("3d_anisotropic_then_random_rotation", (center3 * np.array([1.0, 0.01, 0.0001])) @ r3.T, 3.0))

    for name, pts, truth in cases:
        result = run_full_pipeline(pts, max_radius=6, samples=80)
        br = result["bracket"]
        ok = contains(br, truth)
        overlap_with_ident = overlaps(ident_br, br) if pts.shape[1] == 2 else None
        row = {"case": name, "truth": truth, **result, "contains_truth": ok, "overlaps_ident_2d": overlap_with_ident}
        rows.append(row)
        _write("rescale_battery", {"rows": rows, "status": "in_progress"})
        print(
            f"  {name:<38} shell={result['shell_mean_dimension']:.4f} gp={result['gp_local_dimension']:.4f} "
            f"-> [{br['d_lo']:.3f},{br['d_hi']:.3f}] contains_truth={ok} overlaps_ident={overlap_with_ident} "
            f"wall={result['wall_time_s']:.1f}s",
            flush=True,
        )
    _write("rescale_battery", {"rows": rows, "status": "done"})
    n_ok = sum(1 for r in rows if r.get("contains_truth"))
    print(f"\nSTAGE B SUMMARY: {n_ok}/{len(rows)} adversarial rescale/rotation cases contain truth")


# ==========================================================================
# STAGE C -- does the circle k-starvation fix generalize? (claim 2)
# ==========================================================================


def elongated_ellipse(n: int, seed: int, aspect: float) -> np.ndarray:
    t = np.sort(np.random.default_rng(seed).random(n)) * 2.0 * math.pi
    return np.c_[np.cos(t), aspect * np.sin(t)]


def stage_circle_generalization() -> None:
    print("=" * 100)
    print("STAGE C -- k-starvation fix generalization: line, elongated ellipse, helix")
    print("=" * 100)
    rows: list[dict[str, Any]] = []
    _write("circle_generalization", {"rows": rows, "status": "starting"})

    cases = [
        ("line_segment_1d", targets.uniform_interval(1600, 23), 1.0),
        ("elongated_ellipse_aspect_0.05", elongated_ellipse(1600, 23, 0.05), 1.0),
        ("elongated_ellipse_aspect_0.01", elongated_ellipse(1600, 23, 0.01), 1.0),
        ("helix_1d_in_3d", targets.helix(1600, 23), 1.0),
    ]
    for name, pts, truth in cases:
        result = run_full_pipeline(pts)
        br = result["bracket"]
        ok = contains(br, truth)
        row = {"case": name, "truth": truth, **result, "contains_truth": ok}
        rows.append(row)
        _write("circle_generalization", {"rows": rows, "status": "in_progress"})
        print(
            f"  {name:<32} k0={result['settings_auto']['k0']} k={result['settings_auto']['k_selected']} "
            f"shell={result['shell_mean_dimension']:.4f} gp={result['gp_local_dimension']:.4f} "
            f"-> [{br['d_lo']:.3f},{br['d_hi']:.3f}] contains={ok} wall={result['wall_time_s']:.1f}s",
            flush=True,
        )
    _write("circle_generalization", {"rows": rows, "status": "done"})
    n_ok = sum(1 for r in rows if r["contains_truth"])
    print(f"\nSTAGE C SUMMARY: {n_ok}/{len(rows)} low-dimension generalization cases contain truth")


# ==========================================================================
# STAGE D -- direct empirical test of the claimed stale-neighbor-set bug
# (claim 4), by calling refine_metric() ourselves and comparing STALE vs
# FRESH admissibility, rather than trusting the module's own inline comment.
# ==========================================================================


def stage_stale_neighbor_bug() -> None:
    print("=" * 100)
    print("STAGE D -- empirical test of the stale-neighbor-set bug claim")
    print("=" * 100)
    rows: list[dict[str, Any]] = []
    _write("stale_neighbor_bug", {"rows": rows, "status": "starting"})

    cube = targets.uniform_cube(1600, 11)
    raw_lorenz = lorenz_trajectory(seed=777, n_raw=1600 + 20, burn_in=6000, dt=0.005)
    lorenz_pts = takens(raw_lorenz[:, 0] / 30.0, m=3, tau=1, n=1600)

    for name, arr in [("uniform-cube-3d/n=1600/seed=11", cube), ("lorenz-independent-seed=777-tau=1", lorenz_pts)]:
        n, d = arr.shape
        k0 = auto_k0(n, d)
        neighbor_idx_k0, sigma_inv_topo, sigma_inv_metric, refine_diag = refine_metric(arr, k0=k0)

        # STALE: slice the neighbor set refine_metric() actually returned
        # (selected under the PREVIOUS round's metric), padded/sliced to the
        # ladder as auto_select_k expects.
        max_rung = max(cic_mod.K_LADDER)
        if len(neighbor_idx_k0[0]) >= max_rung:
            stale_ladder_source = neighbor_idx_k0
        else:
            # can't slice past what's available; query fresh at k0 length under
            # the STALE metric that selected neighbor_idx_k0 is not directly
            # recoverable (refine_metric doesn't return it), so this measures
            # what "reuse neighbor_idx_k0 directly" would give when k0 itself
            # is >= max_rung; report explicitly when it is not.
            stale_ladder_source = neighbor_idx_k0
        k_stale, trace_stale = auto_select_k(arr, stale_ladder_source, k_ladder=cic_mod.K_LADDER)

        # FRESH: re-query under the metric actually reported (sigma_inv_topo).
        query_len = max(max_rung, k0)
        fresh_ladder_source = mahalanobis_knn_full(arr, sigma_inv_topo, query_len)
        k_fresh, trace_fresh = auto_select_k(arr, fresh_ladder_source, k_ladder=cic_mod.K_LADDER)

        # Direct measurement of the claimed mechanism: how different are the
        # two neighbor sets actually (fraction of shared neighbors per point,
        # at the shortest common length)?
        common_len = min(len(neighbor_idx_k0[0]), len(fresh_ladder_source[0]))
        overlap_frac = float(
            np.mean(
                [
                    len(set(neighbor_idx_k0[i][:common_len].tolist()) & set(fresh_ladder_source[i][:common_len].tolist())) / common_len
                    for i in range(n)
                ]
            )
        )

        row = {
            "case": name,
            "n": n,
            "d": d,
            "k0": k0,
            "k0_ge_max_rung": len(neighbor_idx_k0[0]) >= max_rung,
            "refine_iterations_run": refine_diag["iterations_run"],
            "refine_final_overlap": refine_diag["final_overlap"],
            "stale_admissible_k": trace_stale["admissible_k"],
            "stale_k_selected": trace_stale.get("k_selected"),
            "fresh_admissible_k": trace_fresh["admissible_k"],
            "fresh_k_selected": trace_fresh.get("k_selected"),
            "stale_vs_fresh_neighbor_overlap_fraction": overlap_frac,
            "bug_confirmed_this_case": (trace_stale.get("k_selected") != trace_fresh.get("k_selected")),
        }
        rows.append(row)
        _write("stale_neighbor_bug", {"rows": rows, "status": "in_progress"})
        print(
            f"  {name}: k0={k0} (>=max_rung={row['k0_ge_max_rung']}) "
            f"STALE admissible={trace_stale['admissible_k']} selected={trace_stale.get('k_selected')} | "
            f"FRESH admissible={trace_fresh['admissible_k']} selected={trace_fresh.get('k_selected')} | "
            f"neighbor_overlap={overlap_frac:.4f} | differs={row['bug_confirmed_this_case']}",
            flush=True,
        )
    _write("stale_neighbor_bug", {"rows": rows, "status": "done"})


# ==========================================================================
# STAGE E -- known-answer battery, fresh seeds never used by the prototype
# ==========================================================================


def stage_known_answer_refresh(seed: int = 90210) -> None:
    print("=" * 100)
    print(f"STAGE E -- known-answer battery refresh, fresh seed={seed} (prototype used seed=11 throughout)")
    print("=" * 100)
    rows: list[dict[str, Any]] = []
    _write("known_answer_refresh", {"rows": rows, "status": "starting"})
    cases = [
        ("uniform-square-2d", targets.uniform_square(1600, seed), 2.0),
        ("circle-1d", targets.circle(1600, seed), 1.0),
        ("uniform-cube-3d", targets.uniform_cube(1600, seed), 3.0),
    ]
    for name, pts, truth in cases:
        result = run_full_pipeline(pts)
        br = result["bracket"]
        ok = contains(br, truth)
        row = {"case": name, "seed": seed, "truth": truth, **result, "contains_truth": ok}
        rows.append(row)
        _write("known_answer_refresh", {"rows": rows, "status": "in_progress"})
        print(
            f"  {name:<20} shell={result['shell_mean_dimension']:.4f} gp={result['gp_local_dimension']:.4f} "
            f"-> [{br['d_lo']:.3f},{br['d_hi']:.3f}] contains={ok} wall={result['wall_time_s']:.1f}s",
            flush=True,
        )
    _write("known_answer_refresh", {"rows": rows, "status": "done"})
    n_ok = sum(1 for r in rows if r["contains_truth"])
    print(f"\nSTAGE E SUMMARY: {n_ok}/{len(rows)} fresh-seed known-answer cases contain truth")


# ==========================================================================
# STAGE F -- scalability: does cost stay ~constant-factor or does it
# compound as n grows (claim re: 100x slowdown measured only at n=1600)?
# ==========================================================================


def stage_scalability() -> None:
    print("=" * 100)
    print("STAGE F -- scalability check, n=1600 vs n=3200")
    print("=" * 100)
    rows: list[dict[str, Any]] = []
    _write("scalability", {"rows": rows, "status": "starting"})
    for n in (1600, 3200):
        pts = targets.uniform_square(n, 555)
        t0 = time.perf_counter()
        result = run_full_pipeline(pts)
        wall = time.perf_counter() - t0
        br = result["bracket"]
        row = {"n": n, "wall_time_s": wall, **{k: v for k, v in result.items() if k != "wall_time_s"}, "contains_truth": contains(br, 2.0)}
        rows.append(row)
        _write("scalability", {"rows": rows, "status": "in_progress"})
        print(f"  n={n}: wall={wall:.1f}s shell={result['shell_mean_dimension']:.4f} gp={result['gp_local_dimension']:.4f} -> [{br['d_lo']:.3f},{br['d_hi']:.3f}]")
    if len(rows) == 2:
        ratio = rows[1]["wall_time_s"] / rows[0]["wall_time_s"]
        print(f"\nSTAGE F SUMMARY: n doubled (1600->3200), wall time ratio = {ratio:.2f}x (linear=2x, quadratic=4x)")
        _write("scalability", {"rows": rows, "status": "done", "n_doubling_wall_ratio": ratio})


# ==========================================================================
# STAGE G -- zero-knob auto-selection vs hand-picked fixed parameters
# (claim 6): same 7-case battery, k0=30/iterations=2/k=8 fixed (the
# prototype's own hand-picked defaults) vs scale_aware_v2's auto selection.
# ==========================================================================


def hand_picked_pipeline(arr: np.ndarray, *, k0: int = 30, iterations: int = 2, k: int = 8, max_radius: int = 6, samples: int = 80) -> dict[str, Any]:
    t0 = time.perf_counter()
    neighbor_idx_k0, sigma_topo, sigma_metric, diag = refine_metric(
        arr, k0=k0, max_iterations=iterations, min_iterations=iterations, convergence_overlap=2.0  # force exactly `iterations` rounds
    )
    k_eff = min(k, len(neighbor_idx_k0[0]))
    final_idx = mahalanobis_knn_full(arr, sigma_topo, k_eff)
    hg = build_hypergraph(final_idx)
    se = shell_estimate(hg, max_radius=max_radius, samples=samples)
    gpl = gp_local_mahalanobis(arr, sigma_metric)
    br = bracket(se["mean_dimension"], gpl["dimension"])
    return {
        "settings_fixed": {"k0": k0, "iterations": iterations, "k": k},
        "shell_mean_dimension": se["mean_dimension"],
        "gp_local_dimension": gpl["dimension"],
        "bracket": br,
        "wall_time_s": time.perf_counter() - t0,
    }


def stage_zero_knob_vs_handpicked() -> None:
    print("=" * 100)
    print("STAGE G -- zero-knob auto-selection vs hand-picked fixed params, 7-case battery")
    print("=" * 100)
    rows: list[dict[str, Any]] = []
    _write("zero_knob_vs_handpicked", {"rows": rows, "status": "starting"})

    a = np.random.default_rng(11).random((1600, 2))
    raw_lorenz = lorenz_trajectory(seed=42, n_raw=1600 + 20, burn_in=6000, dt=0.005)
    lorenz_pts = takens(raw_lorenz[:, 0] / 30.0, m=3, tau=1, n=1600)
    cases = [
        ("uniform-square", targets.uniform_square(1600, 11), 2.0),
        ("circle", targets.circle(1600, 11), 1.0),
        ("uniform-cube", targets.uniform_cube(1600, 11), 3.0),
        ("refutation-A", a, 2.0),
        ("isotropic-control", a * 100.0, 2.0),
        ("refutation-rescaled", a * np.array([1.0, 0.01]), 2.0),
        ("lorenz-tau1-independent-seed", lorenz_pts, D_KY_LORENZ),
    ]
    for name, pts, truth in cases:
        auto = run_full_pipeline(pts)
        hand = hand_picked_pipeline(pts, k0=30, iterations=2, k=8)
        auto_br, hand_br = auto["bracket"], hand["bracket"]
        row = {
            "case": name,
            "truth": truth,
            "auto": {"shell": auto["shell_mean_dimension"], "gp": auto["gp_local_dimension"], "bracket": auto_br, "settings": auto["settings_auto"], "contains": contains(auto_br, truth)},
            "hand_picked_k0=30_iter=2_k=8": {"shell": hand["shell_mean_dimension"], "gp": hand["gp_local_dimension"], "bracket": hand_br, "contains": contains(hand_br, truth)},
            "brackets_overlap": overlaps(auto_br, hand_br),
            "shell_abs_diff": abs(auto["shell_mean_dimension"] - hand["shell_mean_dimension"]),
            "gp_abs_diff": abs(auto["gp_local_dimension"] - hand["gp_local_dimension"]),
        }
        rows.append(row)
        _write("zero_knob_vs_handpicked", {"rows": rows, "status": "in_progress"})
        print(
            f"  {name:<20} AUTO k={auto['settings_auto']['k_selected']} shell={auto['shell_mean_dimension']:.4f} gp={auto['gp_local_dimension']:.4f} [{auto_br['d_lo']:.3f},{auto_br['d_hi']:.3f}]  |  "
            f"HAND k=8 shell={hand['shell_mean_dimension']:.4f} gp={hand['gp_local_dimension']:.4f} [{hand_br['d_lo']:.3f},{hand_br['d_hi']:.3f}]  "
            f"overlap={row['brackets_overlap']} shell_diff={row['shell_abs_diff']:.4f} gp_diff={row['gp_abs_diff']:.4f}",
            flush=True,
        )
    _write("zero_knob_vs_handpicked", {"rows": rows, "status": "done"})
    n_overlap = sum(1 for r in rows if r["brackets_overlap"])
    n_shell_close = sum(1 for r in rows if r["shell_abs_diff"] < 0.02)
    print(f"\nSTAGE G SUMMARY: {n_overlap}/{len(rows)} auto-vs-hand brackets overlap; {n_shell_close}/{len(rows)} shell readouts within 0.02 (near-'identical')")


# ==========================================================================


STAGES = {
    "lorenz": lambda: stage_lorenz(seeds=[101, 202, 303, 404, 505], lags=[1, 2, 5]),
    "break_refutation": stage_break_refutation,
    "circle_generalization": stage_circle_generalization,
    "stale_neighbor_bug": stage_stale_neighbor_bug,
    "known_answer_refresh": stage_known_answer_refresh,
    "scalability": stage_scalability,
    "zero_knob_vs_handpicked": stage_zero_knob_vs_handpicked,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES.keys()) + ["all"])
    args = ap.parse_args()
    stage_names = list(STAGES.keys()) if args.stage == "all" else [args.stage]
    for name in stage_names:
        STAGES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
