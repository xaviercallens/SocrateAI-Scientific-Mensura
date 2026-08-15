"""FINAL PRE-INTEGRATION VERIFICATION GATE (docs/MENSURA_BENCH_V2.md section 8.5
item 1). Independent skeptical verification round; treats every prior report as
claims to test. Writes incrementally to final_gate_*.json (E6 discipline).

Stages (run via `python final_gate_runner.py <stage>`):

  stage4  independent spot-check of the optimizer's bit-identity claim against
          PRE-optimization artifacts (scale_aware_v2.json, skeptic_rescale_battery.json)
  stage1  zero-knob vs hand-picked equivalence (claim 6) + perturbation
          robustness of the auto-selector (seed change, n +/- 10%)
  stage3  the production-gate composition claim (8.3), verified by EXECUTING
          cic._detect / cic.build_interval (the real signal stack) on the
          scale-aware readouts -- not by arithmetic
  stage2  scalability WITH correctness at n=3200 (+ n=6400 direction check)
  stage5  calibration-violation hunt in the unexplored transition zone
          (rescale factors 0.05..0.002, each +/- a random rotation), with the
          real production gate verdict for every row

Nothing here touches production sources. All heavy machinery is imported from
the existing scratch modules and from cic.py as library calls, so the thing
being tested is exactly the code that would be integrated.
"""

from __future__ import annotations

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

import targets  # noqa: E402
from scale_aware_prototype import (  # noqa: E402
    build_hypergraph,
    gp_global_mahalanobis,
    gp_local_mahalanobis,
    mahalanobis_knn_full,
)
from scale_aware_v2 import (  # noqa: E402
    bracket,
    refine_metric,
    scale_aware_v2,
    shell_estimate,
)

OUT_DIR = Path(__file__).resolve().parent


def _write(name: str, obj: Any) -> None:
    (OUT_DIR / f"final_gate_{name}.json").write_text(json.dumps(obj, indent=1, default=str))


def contains(br: dict[str, Any], truth: float) -> bool:
    return bool(br["d_lo"] <= truth <= br["d_hi"])


# ==========================================================================
# Scratch pipeline (identical shape to skeptic_verify.run_full_pipeline /
# run_case_v2: the construction whose numbers the prior artifacts recorded).
# ==========================================================================


def run_scratch(arr: np.ndarray, *, max_radius: int = 6, samples: int = 80, with_global: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    sa = scale_aware_v2(arr)
    hg = build_hypergraph(sa.neighbor_idx)
    se = shell_estimate(hg, max_radius=max_radius, samples=samples)
    gpl = gp_local_mahalanobis(arr, sa.sigma_inv_metric)
    br = bracket(se["mean_dimension"], gpl["dimension"])
    row: dict[str, Any] = {
        "n": len(arr),
        "d": int(arr.shape[1]),
        "settings_auto": sa.settings,
        "shell_mean_dimension": se["mean_dimension"],
        "shell_degenerate_fraction": se["degenerate_fraction"],
        "shell_near_degenerate_fraction": se["near_degenerate_fraction"],
        "gp_local_dimension": gpl["dimension"],
        "gp_local_r_squared": gpl["r_squared"],
        "bracket": br,
        "k_selection_failed": bool(sa.diagnostics.get("k_selection_failed", False)),
        "wall_time_s": time.perf_counter() - t0,
    }
    if with_global:
        gpg = gp_global_mahalanobis(arr)
        row["gp_global_dimension"] = gpg["dimension"]
        row["gp_global_r_squared"] = gpg["r_squared"]
        row["bracket_global_gp"] = bracket(se["mean_dimension"], gpg["dimension"])
    return row


def run_hand_picked(arr: np.ndarray, *, k0: int, iterations: int, k: int, max_radius: int = 6, samples: int = 80) -> dict[str, Any]:
    """Hand-picked parameters, mirroring skeptic_verify.hand_picked_pipeline but
    parameterized by the ROUND-2 RECORDED values (k0 by dim, recorded k and
    iteration count) rather than the old fixed defaults."""
    t0 = time.perf_counter()
    _idx_k0, sigma_topo, sigma_metric, _diag = refine_metric(
        arr, k0=k0, max_iterations=iterations, min_iterations=iterations, convergence_overlap=2.0
    )
    final_idx = mahalanobis_knn_full(arr, sigma_topo, k)
    hg = build_hypergraph(final_idx)
    se = shell_estimate(hg, max_radius=max_radius, samples=samples)
    gpl = gp_local_mahalanobis(arr, sigma_metric)
    br = bracket(se["mean_dimension"], gpl["dimension"])
    return {
        "settings_fixed": {"k0": k0, "iterations": iterations, "k": k},
        "shell_mean_dimension": se["mean_dimension"],
        "gp_local_dimension": gpl["dimension"],
        "gp_local_r_squared": gpl["r_squared"],
        "bracket": br,
        "wall_time_s": time.perf_counter() - t0,
    }


# ==========================================================================
# Production signal stack, EXECUTED (stage 3's core). Everything below calls
# cic.py's real code: _components, _shell_arm, _detect, build_interval,
# _dedupe, and the real constants. The only substitution is the one the
# integration itself proposes (8.4 item 1): the edge set at each K_LADDER
# rung comes from the scale-aware Mahalanobis metric instead of Euclidean.
# ==========================================================================


def _gate_margins(shell: float, gp: float, k_spread: float) -> dict[str, Any]:
    """The READOUT_DIVERGENCE / K_INSTABILITY arithmetic exactly as _detect
    executes it, exposed so margins can be reported, not just pass/fail."""
    gp_ok = math.isfinite(gp)
    if gp_ok:
        scale = max(1.0, 0.5 * (min(shell, gp) + max(shell, gp)))
    else:
        scale = max(1.0, shell)
    div_threshold = cic_mod.READOUT_DIVERGENCE_FACTOR * cic_mod.RHO_BIAS_FLOOR * scale
    arm_gap = abs(shell - gp) if gp_ok else float("nan")
    k_threshold = cic_mod.K_INSTABILITY_FACTOR * cic_mod.RHO_BIAS_FLOOR * scale
    return {
        "shell": shell,
        "gp": gp,
        "arm_gap": arm_gap,
        "scale": scale,
        "readout_divergence_threshold": div_threshold,
        "readout_divergence_margin_gap_minus_threshold": (arm_gap - div_threshold) if gp_ok else float("nan"),
        "readout_divergence_fires": bool(gp_ok and arm_gap > div_threshold),
        "k_spread": k_spread,
        "k_instability_threshold": k_threshold,
        "k_instability_fires": bool(k_spread > k_threshold),
    }


def production_stack(arr: np.ndarray, *, scratch_max_radius: int = 6, scratch_samples: int = 80) -> dict[str, Any]:
    """Run the scale-aware construction, then push its readouts through the
    REAL production signal stack (cic._detect) and the REAL verdict/interval
    logic (mirroring certify()'s own branch structure line for line), for both
    the local-Mahalanobis gp arm and the global-Mahalanobis gp arm."""
    t0 = time.perf_counter()
    n = len(arr)
    sa = scale_aware_v2(arr)

    # Scratch-style bracket (what the prior artifacts recorded), for continuity.
    hg_scratch = build_hypergraph(sa.neighbor_idx)
    se = shell_estimate(hg_scratch, max_radius=scratch_max_radius, samples=scratch_samples)
    gp_loc = gp_local_mahalanobis(arr, sa.sigma_inv_metric)
    gp_glob = gp_global_mahalanobis(arr)
    scratch_br = bracket(se["mean_dimension"], gp_loc["dimension"])

    # The K_LADDER of scale-aware graphs, shell arms via cic's own machinery.
    query_len = max(max(cic_mod.K_LADDER), sa.settings["k0"])
    idx_ladder = mahalanobis_knn_full(arr, sa.sigma_inv_topology, query_len)
    arms: dict[int, Any] = {}
    traces: dict[int, Any] = {}
    largest_fraction = 0.0
    for k in cic_mod.K_LADDER:
        if k >= n or k > len(idx_ladder[0]):
            continue
        hg = build_hypergraph([row[:k] for row in idx_ladder])
        _n_comp, lf = cic_mod._components(hg, n)
        largest_fraction = max(largest_fraction, lf)
        if lf < cic_mod.CONNECTIVITY_FRACTION:
            continue
        arm, trace = cic_mod._shell_arm(hg, k)
        traces[k] = trace
        if arm is not None:
            arms[k] = arm
    admissible = tuple(sorted(arms))
    k_chosen = admissible[0] if admissible else None
    ensemble = [arms[k] for k in admissible]
    selected = arms.get(k_chosen) if k_chosen is not None else None
    saturated_everywhere = bool(traces) and all(
        t and t[0][2] > cic_mod.BALL_FRACTION_CAP for t in traces.values()
    )
    _arr2, _n_dropped, duplicate_fraction = cic_mod._dedupe(arr)

    readouts = [r.readout for r in ensemble if math.isfinite(r.readout)]
    k_spread = (max(readouts) - min(readouts)) if len(readouts) >= 2 else 0.0

    row: dict[str, Any] = {
        "n": n,
        "d": int(arr.shape[1]),
        "settings_auto": sa.settings,
        "scratch": {
            "shell_mean_dimension": se["mean_dimension"],
            "shell_near_degenerate_fraction": se["near_degenerate_fraction"],
            "gp_local_dimension": gp_loc["dimension"],
            "gp_local_r_squared": gp_loc["r_squared"],
            "gp_global_dimension": gp_glob["dimension"],
            "gp_global_r_squared": gp_glob["r_squared"],
            "bracket_local_gp": scratch_br,
            "bracket_global_gp": bracket(se["mean_dimension"], gp_glob["dimension"]),
        },
        "production": {
            "admissible_k": list(admissible),
            "k_chosen": k_chosen,
            "shell_readout_at_k_chosen": selected.readout if selected is not None else float("nan"),
            "readouts_by_k": {str(k): arms[k].readout for k in admissible},
            "k_spread": k_spread,
            "largest_fraction": largest_fraction,
            "duplicate_fraction": duplicate_fraction,
            "saturated_everywhere": saturated_everywhere,
        },
        "arms": {},
    }

    for arm_name, gp in (("local_gp", gp_loc), ("global_gp", gp_glob)):
        signals = cic_mod._detect(
            selected=selected,
            ensemble=ensemble,
            gp_dimension=gp["dimension"],
            gp_r_squared=gp["r_squared"],
            k_chosen=k_chosen,
            largest_fraction=largest_fraction,
            duplicate_fraction=duplicate_fraction,
            n_points=n,
            admissible=admissible,
            saturated_everywhere=saturated_everywhere,
        )
        entry: dict[str, Any] = {"signals": list(signals)}
        if selected is not None:
            entry["margins"] = _gate_margins(selected.readout, gp["dimension"], k_spread)
            entry["chain_artifact_inputs"] = {
                "ring_fraction": selected.degenerate_fraction + selected.near_degenerate_fraction,
                "gp_minus_1_abs": abs(gp["dimension"] - 1.0) if math.isfinite(gp["dimension"]) else float("nan"),
                "fires": "CHAIN_ARTIFACT" in signals,
            }
        # Verdict logic mirroring certify() exactly.
        if signals:
            entry["verdict"] = cic_mod.Verdict.UNDECIDED
        else:
            exact_here = selected.degenerate_fraction >= cic_mod.DEGENERATE_EXACT_FRACTION
            stable_at_one = all(
                math.isfinite(r.readout) and abs(r.readout - 1.0) <= cic_mod.RING_GP_TOLERANCE
                for r in ensemble
            )
            gp_agrees_with_one = math.isfinite(gp["dimension"]) and (
                abs(gp["dimension"] - 1.0) <= cic_mod.RING_GP_TOLERANCE
            )
            if exact_here and selected.consensus and stable_at_one and gp_agrees_with_one:
                entry["verdict"] = cic_mod.Verdict.DEGENERATE_EXACT
                entry["interval"] = {"d_lo": 1.0, "d_hi": 1.0}
            else:
                ring_bound = None
                if (
                    selected.degenerate_fraction + selected.near_degenerate_fraction
                ) >= cic_mod.RING_FRACTION and selected.consensus:
                    ring_bound = selected.mean_slope_bound
                d_lo, d_hi, hull_lo, hull_hi, allowance, basis = cic_mod.build_interval(
                    selected.readout, gp["dimension"], k_spread=k_spread, ring_bound=ring_bound
                )
                entry["verdict"] = cic_mod.Verdict.MEASURED
                entry["interval"] = {
                    "d_lo": d_lo, "d_hi": d_hi, "hull_lo": hull_lo, "hull_hi": hull_hi,
                    "allowance": allowance, "basis": basis,
                }
        row["arms"][arm_name] = entry

    row["wall_time_s"] = time.perf_counter() - t0
    return row


# ==========================================================================
# Case generators (shared)
# ==========================================================================


def rotation_2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def case_points(kind: str, n: int, seed: int) -> tuple[np.ndarray, float]:
    if kind == "square":
        return targets.uniform_square(n, seed), 2.0
    if kind == "circle":
        return targets.circle(n, seed), 1.0
    if kind == "cube":
        return targets.uniform_cube(n, seed), 3.0
    if kind == "refutation_100to1":
        return np.random.default_rng(seed).random((n, 2)) * np.array([1.0, 0.01]), 2.0
    if kind == "rescale_1000to1":
        return np.random.default_rng(seed).random((n, 2)) * np.array([1.0, 0.001]), 2.0
    raise ValueError(kind)


# ==========================================================================
# STAGE 4 -- independent bit-identity spot check
# ==========================================================================


def stage4() -> None:
    print("=" * 100)
    print("STAGE 4 -- independent spot-check of the bit-identity claim (current code vs PRE-optimization artifacts)")
    print("=" * 100)
    v2_art = json.loads((OUT_DIR / "scale_aware_v2.json").read_text())
    sk_art = json.loads((OUT_DIR / "skeptic_rescale_battery.json").read_text())
    v2_rows = {r["case"]: r for r in v2_art["rows"]}
    sk_rows = {r["case"]: r for r in sk_art["rows"]}

    out: dict[str, Any] = {"status": "starting", "protocol": (
        "3 battery cases re-run with the CURRENT (optimized) scratch code and compared field-by-field "
        "against numbers recorded by the UNOPTIMIZED code in scale_aware_v2.json (round 2) and "
        "skeptic_rescale_battery.json (stopped skeptic). Floats compared for exact equality (== on float)."
    ), "cases": []}
    _write("stage4_bitidentity", out)

    a = np.random.default_rng(11).random((1600, 2))

    def cmp(label: str, got: float | int | bool | None, ref: float | int | bool | None) -> dict[str, Any]:
        exact = (got == ref) or (got is None and ref is None)
        return {"field": label, "current": got, "original": ref, "exact_match": bool(exact)}

    # case 1: identity square A (in BOTH artifacts)
    r = run_scratch(a)
    ref_v2 = v2_rows["A = uniform_square(1600,11)"]
    ref_sk = sk_rows["IDENT_2D_seed11"]
    fields = [
        cmp("shell_mean_dimension(vs v2.json)", r["shell_mean_dimension"], ref_v2["shell"]["mean_dimension"]),
        cmp("gp_local_dimension(vs v2.json)", r["gp_local_dimension"], ref_v2["gp_local_mahalanobis"]["dimension"]),
        cmp("bracket.d_lo(vs v2.json)", r["bracket"]["d_lo"], ref_v2["bracket_local_gp"]["d_lo"]),
        cmp("bracket.d_hi(vs v2.json)", r["bracket"]["d_hi"], ref_v2["bracket_local_gp"]["d_hi"]),
        cmp("k_selected(vs v2.json)", r["settings_auto"]["k_selected"], ref_v2["settings_auto"]["k_selected"]),
        cmp("iterations(vs v2.json)", r["settings_auto"]["iterations_run"], ref_v2["settings_auto"]["iterations_run"]),
        cmp("shell_mean_dimension(vs skeptic)", r["shell_mean_dimension"], ref_sk["shell_mean_dimension"]),
        cmp("gp_local_dimension(vs skeptic)", r["gp_local_dimension"], ref_sk["gp_local_dimension"]),
        cmp("bracket.d_lo(vs skeptic)", r["bracket"]["d_lo"], ref_sk["bracket"]["d_lo"]),
        cmp("bracket.d_hi(vs skeptic)", r["bracket"]["d_hi"], ref_sk["bracket"]["d_hi"]),
    ]
    out["cases"].append({"case": "A = uniform_square(1600,11) == IDENT_2D_seed11", "fields": fields, "current_row": r})
    _write("stage4_bitidentity", out)
    print(f"  ident square: {sum(f['exact_match'] for f in fields)}/{len(fields)} exact")

    # case 2: the pinned 8.1 violation row (skeptic artifact)
    r = run_scratch(a * np.array([1.0, 0.001]))
    ref = sk_rows["rescale_y_factor=0.001"]
    fields = [
        cmp("shell_mean_dimension", r["shell_mean_dimension"], ref["shell_mean_dimension"]),
        cmp("shell_near_degenerate_fraction", r["shell_near_degenerate_fraction"], ref["shell_near_degenerate_fraction"]),
        cmp("gp_local_dimension", r["gp_local_dimension"], ref["gp_local_dimension"]),
        cmp("gp_local_r_squared", r["gp_local_r_squared"], ref["gp_local_r_squared"]),
        cmp("bracket.d_lo", r["bracket"]["d_lo"], ref["bracket"]["d_lo"]),
        cmp("bracket.d_hi", r["bracket"]["d_hi"], ref["bracket"]["d_hi"]),
        cmp("k_selected", r["settings_auto"]["k_selected"], ref["settings_auto"]["k_selected"]),
        cmp("iterations", r["settings_auto"]["iterations_run"], ref["settings_auto"]["iterations_run"]),
    ]
    out["cases"].append({"case": "rescale_y_factor=0.001 [pinned 8.1 violation row]", "fields": fields, "current_row": r})
    _write("stage4_bitidentity", out)
    print(f"  1000:1 row: {sum(f['exact_match'] for f in fields)}/{len(fields)} exact")

    # case 3: 3-D cube (round-2 artifact, includes the global gp arm)
    r = run_scratch(targets.uniform_cube(1600, 11), with_global=True)
    ref = v2_rows["uniform-cube-3d/n=1600/seed=11"]
    fields = [
        cmp("shell_mean_dimension", r["shell_mean_dimension"], ref["shell"]["mean_dimension"]),
        cmp("gp_local_dimension", r["gp_local_dimension"], ref["gp_local_mahalanobis"]["dimension"]),
        cmp("gp_global_dimension", r["gp_global_dimension"], ref["gp_global_mahalanobis"]["dimension"]),
        cmp("bracket.d_lo", r["bracket"]["d_lo"], ref["bracket_local_gp"]["d_lo"]),
        cmp("bracket.d_hi", r["bracket"]["d_hi"], ref["bracket_local_gp"]["d_hi"]),
        cmp("k0", r["settings_auto"]["k0"], ref["settings_auto"]["k0"]),
        cmp("k_selected", r["settings_auto"]["k_selected"], ref["settings_auto"]["k_selected"]),
        cmp("iterations", r["settings_auto"]["iterations_run"], ref["settings_auto"]["iterations_run"]),
    ]
    out["cases"].append({"case": "uniform-cube-3d/n=1600/seed=11", "fields": fields, "current_row": r})

    all_fields = [f for c in out["cases"] for f in c["fields"]]
    out["summary"] = {
        "n_fields": len(all_fields),
        "n_exact": sum(f["exact_match"] for f in all_fields),
        "mismatches": [f for f in all_fields if not f["exact_match"]],
    }
    out["status"] = "done"
    _write("stage4_bitidentity", out)
    print(f"  cube: {sum(f['exact_match'] for f in out['cases'][-1]['fields'])}/{len(out['cases'][-1]['fields'])} exact")
    print(f"STAGE 4 SUMMARY: {out['summary']['n_exact']}/{out['summary']['n_fields']} fields exact; mismatches: {len(out['summary']['mismatches'])}")


# ==========================================================================
# STAGE 1 -- zero-knob vs hand-picked + perturbation robustness
# ==========================================================================

# Round-2 recorded hand-picked values (k0 by dim; k/iterations as recorded in
# scale_aware_v2.json and skeptic_rescale_battery.json).
HAND_PICKED = {
    "square": {"k0": 30, "iterations": 4, "k": 4},
    "circle": {"k0": 30, "iterations": 1, "k": 25},
    "cube": {"k0": 60, "iterations": 5, "k": 4},
    "refutation_100to1": {"k0": 30, "iterations": 2, "k": 10},
    "rescale_1000to1": {"k0": 30, "iterations": 2, "k": 25},
}


def stage1() -> None:
    print("=" * 100)
    print("STAGE 1 -- zero-knob vs hand-picked equivalence + auto-selector perturbation robustness")
    print("=" * 100)
    out: dict[str, Any] = {"status": "starting", "equivalence": [], "perturbations": []}
    _write("stage1_zeroknob", out)

    for kind in HAND_PICKED:
        pts, truth = case_points(kind, 1600, 11)
        auto = run_scratch(pts)
        hand = run_hand_picked(pts, **HAND_PICKED[kind])
        row = {
            "case": kind, "truth": truth,
            "auto": {k: auto[k] for k in ("settings_auto", "shell_mean_dimension", "gp_local_dimension", "bracket")},
            "hand": {k: hand[k] for k in ("settings_fixed", "shell_mean_dimension", "gp_local_dimension", "bracket")},
            "auto_contains": contains(auto["bracket"], truth),
            "hand_contains": contains(hand["bracket"], truth),
            "auto_selected_equals_hand_picked": (
                auto["settings_auto"]["k0"] == HAND_PICKED[kind]["k0"]
                and auto["settings_auto"]["k_selected"] == HAND_PICKED[kind]["k"]
                and auto["settings_auto"]["iterations_run"] == HAND_PICKED[kind]["iterations"]
            ),
            "shell_exact_equal": auto["shell_mean_dimension"] == hand["shell_mean_dimension"],
            "gp_exact_equal": auto["gp_local_dimension"] == hand["gp_local_dimension"],
            "bracket_exact_equal": auto["bracket"]["d_lo"] == hand["bracket"]["d_lo"] and auto["bracket"]["d_hi"] == hand["bracket"]["d_hi"],
            "shell_abs_diff": abs(auto["shell_mean_dimension"] - hand["shell_mean_dimension"]),
            "gp_abs_diff": abs(auto["gp_local_dimension"] - hand["gp_local_dimension"]),
        }
        out["equivalence"].append(row)
        _write("stage1_zeroknob", out)
        print(
            f"  {kind:<18} AUTO(k0={auto['settings_auto']['k0']},k={auto['settings_auto']['k_selected']},it={auto['settings_auto']['iterations_run']}) "
            f"[{auto['bracket']['d_lo']:.3f},{auto['bracket']['d_hi']:.3f}]  HAND [{hand['bracket']['d_lo']:.3f},{hand['bracket']['d_hi']:.3f}] "
            f"exact_bracket={row['bracket_exact_equal']} shell_diff={row['shell_abs_diff']:.2e} gp_diff={row['gp_abs_diff']:.2e}",
            flush=True,
        )

    # perturbation robustness of auto-selection
    perturbs = [("seed=12,n=1600", 1600, 12), ("seed=11,n=1440", 1440, 11), ("seed=11,n=1760", 1760, 11)]
    for kind in HAND_PICKED:
        base = next(r for r in out["equivalence"] if r["case"] == kind)
        base_sa = base["auto"]["settings_auto"]
        base_br = base["auto"]["bracket"]
        for label, n, seed in perturbs:
            pts, truth = case_points(kind, n, seed)
            r = run_scratch(pts)
            sa = r["settings_auto"]
            mid_shift = abs(
                0.5 * (r["bracket"]["d_lo"] + r["bracket"]["d_hi"]) - 0.5 * (base_br["d_lo"] + base_br["d_hi"])
            )
            row = {
                "case": kind, "perturbation": label, "truth": truth,
                "settings_auto": sa,
                "bracket": r["bracket"],
                "contains_truth": contains(r["bracket"], truth),
                "k_flipped_from_base": sa["k_selected"] != base_sa["k_selected"],
                "base_k": base_sa["k_selected"],
                "iterations_changed_from_base": sa["iterations_run"] != base_sa["iterations_run"],
                "bracket_midpoint_shift_from_base": mid_shift,
                "shell": r["shell_mean_dimension"], "gp": r["gp_local_dimension"],
            }
            out["perturbations"].append(row)
            _write("stage1_zeroknob", out)
            print(
                f"  {kind:<18} {label:<15} k={sa['k_selected']}(base {base_sa['k_selected']}) it={sa['iterations_run']} "
                f"[{r['bracket']['d_lo']:.3f},{r['bracket']['d_hi']:.3f}] contains={row['contains_truth']} midshift={mid_shift:.3f}",
                flush=True,
            )

    eq = out["equivalence"]
    out["summary"] = {
        "n_cases": len(eq),
        "n_auto_matches_hand_settings": sum(r["auto_selected_equals_hand_picked"] for r in eq),
        "n_bracket_exact_equal": sum(r["bracket_exact_equal"] for r in eq),
        "n_perturbation_rows": len(out["perturbations"]),
        "n_perturbation_contains": sum(r["contains_truth"] for r in out["perturbations"]),
        "k_flips": [
            {"case": r["case"], "perturbation": r["perturbation"], "k": r["settings_auto"]["k_selected"], "base_k": r["base_k"], "midshift": r["bracket_midpoint_shift_from_base"], "contains": r["contains_truth"]}
            for r in out["perturbations"] if r["k_flipped_from_base"]
        ],
    }
    out["status"] = "done"
    _write("stage1_zeroknob", out)
    print("STAGE 1 SUMMARY:", json.dumps(out["summary"], indent=1, default=str))


# ==========================================================================
# STAGE 3 -- production gate composition, executed
# ==========================================================================


def stage3() -> None:
    print("=" * 100)
    print("STAGE 3 -- production signal stack (cic._detect) EXECUTED on scale-aware readouts, n=1600")
    print("=" * 100)
    out: dict[str, Any] = {"status": "starting", "salvaged_number_margins": [], "rows": []}

    # (i) salvaged 8.1 numbers through the real gate arithmetic (constants and
    # expression imported from cic, same code path as _gate_margins uses).
    sk = json.loads((OUT_DIR / "skeptic_rescale_battery.json").read_text())
    for r in sk["rows"]:
        m = _gate_margins(r["shell_mean_dimension"], r["gp_local_dimension"], 0.0)
        out["salvaged_number_margins"].append({"case": r["case"], **m})
    _write("stage3_production_gate", out)

    a = np.random.default_rng(11).random((1600, 2))
    cases = [
        ("square_ident(A)", a, 2.0),
        ("circle", targets.circle(1600, 11), 1.0),
        ("cube", targets.uniform_cube(1600, 11), 3.0),
        ("isotropic_control(A@diag(100,100))", a * 100.0, 2.0),
        ("rescale_10to1(A@diag(1,0.1))", a * np.array([1.0, 0.1]), 2.0),
        ("rescale_100to1(A@diag(1,0.01))", a * np.array([1.0, 0.01]), 2.0),
        ("rescale_1000to1(A@diag(1,0.001))", a * np.array([1.0, 0.001]), 2.0),
        ("rescale_10000to1(A@diag(1,0.0001))", a * np.array([1.0, 0.0001]), 2.0),
    ]
    for name, pts, truth in cases:
        row = production_stack(pts)
        row["case"] = name
        row["truth"] = truth
        row["scratch_bracket_contains_truth"] = contains(row["scratch"]["bracket_local_gp"], truth)
        for arm in ("local_gp", "global_gp"):
            e = row["arms"][arm]
            if e["verdict"] == cic_mod.Verdict.MEASURED:
                e["interval_contains_truth"] = bool(e["interval"]["d_lo"] <= truth <= e["interval"]["d_hi"])
        out["rows"].append(row)
        _write("stage3_production_gate", out)
        loc = row["arms"]["local_gp"]
        glob = row["arms"]["global_gp"]
        print(
            f"  {name:<38} local_gp={loc['verdict']:<10} signals={','.join(loc['signals']) or '-'} | "
            f"global_gp={glob['verdict']:<10} signals={','.join(glob['signals']) or '-'} "
            f"(prod shell={row['production']['shell_readout_at_k_chosen']:.4f} gp_loc={row['scratch']['gp_local_dimension']:.4f} k_spread={row['production']['k_spread']:.3f})",
            flush=True,
        )
    out["status"] = "done"
    _write("stage3_production_gate", out)
    print("STAGE 3 done.")


# ==========================================================================
# STAGE 2 -- scalability with correctness at n=3200 (+ 6400 direction check)
# ==========================================================================


def stage2() -> None:
    print("=" * 100)
    print("STAGE 2 -- correctness at n=3200 (and one n=6400 direction check), with production gate")
    print("=" * 100)
    out: dict[str, Any] = {"status": "starting", "rows": []}
    _write("stage2_scale", out)

    a32 = np.random.default_rng(11).random((3200, 2))
    cases = [
        ("uniform-square-2d/n=3200/seed=555", targets.uniform_square(3200, 555), 2.0),
        ("uniform-cube-3d/n=3200/seed=555", targets.uniform_cube(3200, 555), 3.0),
        ("circle-1d/n=3200/seed=11", targets.circle(3200, 11), 1.0),
        ("A32_ident/n=3200/seed=11", a32, 2.0),
        ("A32@diag(1,0.01)/n=3200", a32 * np.array([1.0, 0.01]), 2.0),
        ("A32@diag(1,0.001)/n=3200", a32 * np.array([1.0, 0.001]), 2.0),
    ]
    for name, pts, truth in cases:
        row = production_stack(pts)
        row["case"] = name
        row["truth"] = truth
        row["scratch_bracket_contains_truth"] = contains(row["scratch"]["bracket_local_gp"], truth)
        for arm in ("local_gp", "global_gp"):
            e = row["arms"][arm]
            if e["verdict"] == cic_mod.Verdict.MEASURED:
                e["interval_contains_truth"] = bool(e["interval"]["d_lo"] <= truth <= e["interval"]["d_hi"])
        out["rows"].append(row)
        _write("stage2_scale", out)
        br = row["scratch"]["bracket_local_gp"]
        print(
            f"  {name:<36} k={row['settings_auto']['k_selected']} shell={row['scratch']['shell_mean_dimension']:.4f} "
            f"gp_loc={row['scratch']['gp_local_dimension']:.4f} [{br['d_lo']:.3f},{br['d_hi']:.3f}] "
            f"contains={row['scratch_bracket_contains_truth']} verdict_loc={row['arms']['local_gp']['verdict']} "
            f"wall={row['wall_time_s']:.1f}s",
            flush=True,
        )

    # cross-check the two rows optimize_results.json also ran (same code, same
    # inputs -- consistency check on the optimizer's own numbers)
    opt = json.loads((OUT_DIR / "optimize_results.json").read_text())["after"]
    xchk = []
    for name in ("uniform-square-2d/n=3200/seed=555", "uniform-cube-3d/n=3200/seed=555"):
        mine = next(r for r in out["rows"] if r["case"] == name)
        ref = opt[name]
        xchk.append({
            "case": name,
            "shell_exact": mine["scratch"]["shell_mean_dimension"] == ref["shell"]["mean_dimension"],
            "gp_exact": mine["scratch"]["gp_local_dimension"] == ref["gp_local_mahalanobis"]["dimension"],
            "bracket_exact": mine["scratch"]["bracket_local_gp"]["d_lo"] == ref["bracket_local_gp"]["d_lo"]
            and mine["scratch"]["bracket_local_gp"]["d_hi"] == ref["bracket_local_gp"]["d_hi"],
        })
    out["optimizer_crosscheck"] = xchk
    _write("stage2_scale", out)
    print("  optimizer cross-check:", json.dumps(xchk))

    # n=6400 direction check (scratch pipeline only -- direction, not gate)
    t0 = time.perf_counter()
    r = run_scratch(targets.uniform_square(6400, 555))
    r["case"] = "uniform-square-2d/n=6400/seed=555"
    r["truth"] = 2.0
    r["contains_truth"] = contains(r["bracket"], 2.0)
    out["n6400_direction_check"] = r
    out["status"] = "done"
    _write("stage2_scale", out)
    print(
        f"  n=6400 square: k={r['settings_auto']['k_selected']} shell={r['shell_mean_dimension']:.4f} "
        f"gp={r['gp_local_dimension']:.4f} [{r['bracket']['d_lo']:.3f},{r['bracket']['d_hi']:.3f}] "
        f"contains={r['contains_truth']} wall={time.perf_counter()-t0:.1f}s"
    )
    print("STAGE 2 done.")


# ==========================================================================
# STAGE 5 -- transition-zone violation hunt with real gate verdicts
# ==========================================================================


def stage5() -> None:
    print("=" * 100)
    print("STAGE 5 -- transition zone rescale factors {0.05,0.02,0.01,0.005,0.002}, +/- random rotation, n=1600")
    print("=" * 100)
    out: dict[str, Any] = {"status": "starting", "rows": []}
    _write("stage5_transition", out)

    a = np.random.default_rng(11).random((1600, 2))
    factors = [0.05, 0.02, 0.01, 0.005, 0.002]
    jobs: list[tuple[str, np.ndarray, float]] = []
    for i, f in enumerate(factors):
        jobs.append((f"rescale({f})", a * np.array([1.0, f]), f))
        theta = float(np.random.default_rng(1000 + i).uniform(0.1, math.pi / 2 - 0.1))
        jobs.append((f"rescale({f})+rot({math.degrees(theta):.1f}deg)", (a * np.array([1.0, f])) @ rotation_2d(theta).T, f))

    for name, pts, f in jobs:
        row = production_stack(pts)
        row["case"] = name
        row["rescale_factor"] = f
        row["anisotropy_ratio"] = 1.0 / f
        row["truth"] = 2.0
        row["scratch_bracket_contains_truth"] = contains(row["scratch"]["bracket_local_gp"], 2.0)
        for arm in ("local_gp", "global_gp"):
            e = row["arms"][arm]
            if e["verdict"] == cic_mod.Verdict.MEASURED:
                e["interval_contains_truth"] = bool(e["interval"]["d_lo"] <= 2.0 <= e["interval"]["d_hi"])
        loc = row["arms"]["local_gp"]
        row["violation_that_survives_gate"] = bool(
            loc["verdict"] == cic_mod.Verdict.MEASURED and not loc.get("interval_contains_truth", True)
        )
        out["rows"].append(row)
        _write("stage5_transition", out)
        br = row["scratch"]["bracket_local_gp"]
        print(
            f"  {name:<28} shell={row['scratch']['shell_mean_dimension']:.4f} gp={row['scratch']['gp_local_dimension']:.4f} "
            f"[{br['d_lo']:.3f},{br['d_hi']:.3f}] scratch_contains={row['scratch_bracket_contains_truth']} "
            f"gate={loc['verdict']} signals={','.join(loc['signals']) or '-'} "
            f"SURVIVING_VIOLATION={row['violation_that_survives_gate']}",
            flush=True,
        )

    out["summary"] = {
        "n_rows": len(out["rows"]),
        "n_scratch_violations": sum(1 for r in out["rows"] if not r["scratch_bracket_contains_truth"]),
        "n_gate_undecided": sum(1 for r in out["rows"] if r["arms"]["local_gp"]["verdict"] == "UNDECIDED"),
        "n_violations_surviving_gate": sum(1 for r in out["rows"] if r["violation_that_survives_gate"]),
        "boundary_map": [
            {"case": r["case"], "ratio": r["anisotropy_ratio"], "verdict": r["arms"]["local_gp"]["verdict"],
             "scratch_contains": r["scratch_bracket_contains_truth"],
             "surviving_violation": r["violation_that_survives_gate"]}
            for r in out["rows"]
        ],
    }
    out["status"] = "done"
    _write("stage5_transition", out)
    print("STAGE 5 SUMMARY:", json.dumps(out["summary"], indent=1, default=str))


STAGES = {"stage4": stage4, "stage1": stage1, "stage3": stage3, "stage2": stage2, "stage5": stage5}


def main() -> int:
    names = sys.argv[1:] or ["stage4", "stage1", "stage3", "stage2", "stage5"]
    for name in names:
        STAGES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
