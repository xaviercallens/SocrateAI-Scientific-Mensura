"""VALIDATION of the gp_local WLS fix (see gp_bias_fix.py) against the full
battery AND the real production gate.

Every row runs the scale-aware construction ONCE and then reads out BOTH gp
variants -- `before` = the current `gp_local_mahalanobis` (OLS), `after` =
`gp_local_mahalanobis_wls` (inverse-variance weighted) -- on the SAME
distances, the same metric and the same shell arm, so any difference between
the two columns is attributable to the fit and to nothing else.

The verdict for each is produced by EXECUTING `cic._detect` / `cic.build_interval`
/ `cic._shell_arm` (via `final_gate_runner.production_stack`'s own structure,
reused here rather than re-derived), so this is the real gate, not an
arithmetic imitation of it.

Stages:
  battery   the full case list at n=1600 and n=3200, before/after, with gate
            margins and containment for every row
  seeds     estimator-variance check: 5 seeds of the isotropic square, since
            the weighted fit has a smaller effective radius count than OLS
  cost      runtime attribution of the fit change at n=1600 and n=3200
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

import targets  # noqa: E402
from final_gate_runner import _gate_margins, rotation_2d  # noqa: E402
from gp_bias_fix import gp_local_mahalanobis_wls, write_json  # noqa: E402
from scale_aware_prototype import (  # noqa: E402
    build_hypergraph,
    gp_global_mahalanobis,
    gp_local_mahalanobis,
    mahalanobis_knn_full,
)
from scale_aware_v2 import bracket, scale_aware_v2, shell_estimate  # noqa: E402

from socrates.hypergraph import cic as cic_mod  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def _verdict_for(
    gp: dict[str, float],
    *,
    selected: Any,
    ensemble: list[Any],
    k_chosen: int | None,
    largest_fraction: float,
    duplicate_fraction: float,
    n: int,
    admissible: tuple[int, ...],
    saturated_everywhere: bool,
    k_spread: float,
    truth: float,
) -> dict[str, Any]:
    """certify()'s branch structure, executed against cic's real _detect /
    build_interval. Copied in shape from final_gate_runner.production_stack so
    the two agree line for line."""
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
    entry: dict[str, Any] = {
        "signals": list(signals),
        "gp_dimension": gp["dimension"],
        "gp_r_squared": gp["r_squared"],
    }
    if selected is not None:
        entry["margins"] = _gate_margins(selected.readout, gp["dimension"], k_spread)
    if signals or selected is None:
        entry["verdict"] = str(cic_mod.Verdict.UNDECIDED)
        entry["violation"] = False
        return entry
    exact_here = selected.degenerate_fraction >= cic_mod.DEGENERATE_EXACT_FRACTION
    stable_at_one = all(
        math.isfinite(r.readout) and abs(r.readout - 1.0) <= cic_mod.RING_GP_TOLERANCE
        for r in ensemble
    )
    gp_agrees_with_one = math.isfinite(gp["dimension"]) and (
        abs(gp["dimension"] - 1.0) <= cic_mod.RING_GP_TOLERANCE
    )
    if exact_here and selected.consensus and stable_at_one and gp_agrees_with_one:
        entry["verdict"] = str(cic_mod.Verdict.DEGENERATE_EXACT)
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
        entry["verdict"] = str(cic_mod.Verdict.MEASURED)
        entry["interval"] = {
            "d_lo": d_lo, "d_hi": d_hi, "hull_lo": hull_lo, "hull_hi": hull_hi,
            "allowance": allowance, "basis": basis,
        }
    if entry.get("interval") is not None:
        iv = entry["interval"]
        entry["interval_contains_truth"] = bool(iv["d_lo"] <= truth <= iv["d_hi"])
    entry["violation"] = bool(
        entry["verdict"] == str(cic_mod.Verdict.MEASURED)
        and not entry.get("interval_contains_truth", True)
    )
    return entry


def run_row(name: str, arr: np.ndarray, truth: float) -> dict[str, Any]:
    """One case: one scale-aware construction, three gp readouts (before /
    after / global), all three through the real gate."""
    t_all = time.perf_counter()
    arr = np.ascontiguousarray(np.asarray(arr, dtype=float))
    n = len(arr)
    t0 = time.perf_counter()
    sa = scale_aware_v2(arr)
    t_construction = time.perf_counter() - t0

    hg_scratch = build_hypergraph(sa.neighbor_idx)
    se = shell_estimate(hg_scratch, max_radius=6, samples=80)

    t0 = time.perf_counter()
    gp_before = gp_local_mahalanobis(arr, sa.sigma_inv_metric)
    t_before = time.perf_counter() - t0
    t0 = time.perf_counter()
    gp_after = gp_local_mahalanobis_wls(arr, sa.sigma_inv_metric, weighting="pair_count")
    t_after = time.perf_counter() - t0
    t0 = time.perf_counter()
    gp_glob = gp_global_mahalanobis(arr)
    t_global = time.perf_counter() - t0

    # Production ladder / shell arm, exactly as final_gate_runner does it.
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
    _a2, _nd, duplicate_fraction = cic_mod._dedupe(arr)
    readouts = [r.readout for r in ensemble if math.isfinite(r.readout)]
    k_spread = (max(readouts) - min(readouts)) if len(readouts) >= 2 else 0.0

    common = dict(
        selected=selected, ensemble=ensemble, k_chosen=k_chosen, largest_fraction=largest_fraction,
        duplicate_fraction=duplicate_fraction, n=n, admissible=admissible,
        saturated_everywhere=saturated_everywhere, k_spread=k_spread, truth=truth,
    )
    row: dict[str, Any] = {
        "case": name, "n": n, "d": int(arr.shape[1]), "truth": truth,
        "settings_auto": sa.settings,
        "scratch": {
            "shell_mean_dimension": se["mean_dimension"],
            "gp_before": gp_before["dimension"], "gp_before_r2": gp_before["r_squared"],
            "gp_after": gp_after["dimension"], "gp_after_r2": gp_after["r_squared"],
            "gp_global": gp_glob["dimension"],
            "bracket_before": bracket(se["mean_dimension"], gp_before["dimension"]),
            "bracket_after": bracket(se["mean_dimension"], gp_after["dimension"]),
            "wls_effective_n_radii": gp_after.get("effective_n_radii"),
            "wls_weighted_mean_neighbors": gp_after.get("weighted_mean_neighbors_per_point"),
        },
        "production": {
            "admissible_k": list(admissible), "k_chosen": k_chosen,
            "shell_readout": selected.readout if selected is not None else float("nan"),
            "k_spread": k_spread, "largest_fraction": largest_fraction,
            "duplicate_fraction": duplicate_fraction, "saturated_everywhere": saturated_everywhere,
        },
        "gate": {
            "before": _verdict_for(gp_before, **common),
            "after": _verdict_for(gp_after, **common),
            "global": _verdict_for(gp_glob, **common),
        },
        "timing": {
            "construction_s": t_construction, "gp_before_s": t_before,
            "gp_after_s": t_after, "gp_global_s": t_global, "total_s": time.perf_counter() - t_all,
        },
    }
    return row


def elongated_ellipse(n: int, seed: int, aspect: float) -> np.ndarray:
    t = np.sort(np.random.default_rng(seed).random(n)) * 2.0 * math.pi
    return np.c_[np.cos(t), aspect * np.sin(t)]


LADDER = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 10000.0]


def battery_cases(n: int) -> list[tuple[str, np.ndarray, float]]:
    """Exact case definitions reused from final_gate_runner (seed 11 base
    cloud A, and the same rotation-seed rule stage5 used) so before/after
    numbers are comparable with the recorded artifacts."""
    a = np.random.default_rng(11).random((n, 2))
    cases: list[tuple[str, np.ndarray, float]] = [
        ("uniform_square(A)", a, 2.0),
        ("circle", targets.circle(n, 11), 1.0),
        ("cube", targets.uniform_cube(n, 11), 3.0),
        ("line_segment_1d", targets.uniform_interval(n, 23), 1.0),
        ("ellipse_aspect_0.05", elongated_ellipse(n, 23, 0.05), 1.0),
        ("ellipse_aspect_0.01", elongated_ellipse(n, 23, 0.01), 1.0),
        ("helix_1d_in_3d", targets.helix(n, 23), 1.0),
        ("isotropic_control(A@diag(100,100))", a * 100.0, 2.0),
    ]
    for i, ratio in enumerate(LADDER):
        f = 1.0 / ratio
        cases.append((f"rescale_{ratio:g}to1", a * np.array([1.0, f]), 2.0))
        theta = float(np.random.default_rng(2000 + i).uniform(0.1, math.pi / 2 - 0.1))
        cases.append(
            (
                f"rescale_{ratio:g}to1+rot({math.degrees(theta):.1f}deg)",
                (a * np.array([1.0, f])) @ rotation_2d(theta).T,
                2.0,
            )
        )
    return cases


def stage_battery(ns: list[int], tag: str) -> None:
    out: dict[str, Any] = {"status": "starting", "rows": []}
    write_json(tag, out)
    for n in ns:
        for name, pts, truth in battery_cases(n):
            row = run_row(name, pts, truth)
            out["rows"].append(row)
            write_json(tag, out)
            b, a_, g = row["gate"]["before"], row["gate"]["after"], row["gate"]["global"]
            key = "readout_divergence_margin_gap_minus_threshold"
            mb = b.get("margins", {}).get(key, float("nan"))
            ma = a_.get("margins", {}).get(key, float("nan"))
            print(
                f"  n={n} {name:<40} shell={row['production']['shell_readout']:.4f} "
                f"gp {row['scratch']['gp_before']:.4f}->{row['scratch']['gp_after']:.4f} | "
                f"BEFORE {b['verdict']:<9} m={mb:+.3f} "
                f"contains={b.get('interval_contains_truth')} | "
                f"AFTER {a_['verdict']:<9} m={ma:+.3f} "
                f"contains={a_.get('interval_contains_truth')} | "
                f"GLOBAL {g['verdict']:<9} contains={g.get('interval_contains_truth')} "
                f"{'*** VIOLATION_AFTER ***' if a_.get('violation') else ''}",
                flush=True,
            )
    out["summary"] = _summarize(out["rows"])
    out["status"] = "done"
    write_json(tag, out)
    print(json.dumps(out["summary"], indent=1, default=str))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def tally(which: str) -> dict[str, Any]:
        meas = [r for r in rows if r["gate"][which]["verdict"] == "MEASURED"]
        return {
            "n_measured": len(meas),
            "n_undecided": sum(1 for r in rows if r["gate"][which]["verdict"] == "UNDECIDED"),
            "n_degenerate_exact": sum(
                1 for r in rows if r["gate"][which]["verdict"] == "DEGENERATE_EXACT"
            ),
            "n_measured_containing_truth": sum(
                1 for r in meas if r["gate"][which].get("interval_contains_truth")
            ),
            "violations": [
                {
                    "case": r["case"], "n": r["n"],
                    "interval": r["gate"][which]["interval"], "truth": r["truth"],
                }
                for r in rows
                if r["gate"][which].get("violation")
            ],
        }

    return {
        "n_rows": len(rows),
        "before": tally("before"),
        "after": tally("after"),
        "global": tally("global"),
    }


def stage_seeds() -> None:
    """Estimator variance: the weighted fit leans on fewer effective radii, so
    check it is not merely trading bias for variance."""
    out: dict[str, Any] = {"status": "starting", "rows": []}
    for seed in (11, 12, 13, 14, 15):
        for kind in ("square", "cube"):
            n = 1600
            arr = (
                np.random.default_rng(seed).random((n, 2))
                if kind == "square"
                else np.asarray(targets.uniform_cube(n, seed), dtype=float)
            )
            truth = 2.0 if kind == "square" else 3.0
            sa = scale_aware_v2(arr)
            before = gp_local_mahalanobis(arr, sa.sigma_inv_metric)
            after = gp_local_mahalanobis_wls(arr, sa.sigma_inv_metric, weighting="pair_count")
            out["rows"].append(
                {
                    "case": kind, "seed": seed, "truth": truth,
                    "gp_before": before["dimension"], "gp_after": after["dimension"],
                    "effective_n_radii": after["effective_n_radii"],
                    "r_squared_after": after["r_squared"],
                }
            )
            write_json("seeds", out)
            print(
                f"  {kind} seed={seed}: before={before['dimension']:.4f} "
                f"after={after['dimension']:.4f}",
                flush=True,
            )
    for kind in ("square", "cube"):
        vals_b = [r["gp_before"] for r in out["rows"] if r["case"] == kind]
        vals_a = [r["gp_after"] for r in out["rows"] if r["case"] == kind]
        truth = 2.0 if kind == "square" else 3.0
        out[f"{kind}_summary"] = {
            "before_mean": float(np.mean(vals_b)), "before_std": float(np.std(vals_b, ddof=1)),
            "before_rmse": float(np.sqrt(np.mean((np.array(vals_b) - truth) ** 2))),
            "after_mean": float(np.mean(vals_a)), "after_std": float(np.std(vals_a, ddof=1)),
            "after_rmse": float(np.sqrt(np.mean((np.array(vals_a) - truth) ** 2))),
        }
    out["status"] = "done"
    write_json("seeds", out)
    summaries = {k: v for k, v in out.items() if k.endswith("_summary")}
    print(json.dumps(summaries, indent=1, default=str))


def stage_isotropic_identity() -> None:
    """The pinned property: A and A@diag(100,100) must produce IDENTICAL
    readouts under the fixed estimator, exactly as under the old one."""
    out: dict[str, Any] = {"rows": []}
    for n in (1600, 3200):
        a = np.random.default_rng(11).random((n, 2))
        ra = run_row("A", a, 2.0)
        rb = run_row("A@diag(100,100)", a * 100.0, 2.0)
        fields = [
            ("shell", ra["production"]["shell_readout"], rb["production"]["shell_readout"]),
            ("gp_before", ra["scratch"]["gp_before"], rb["scratch"]["gp_before"]),
            ("gp_after", ra["scratch"]["gp_after"], rb["scratch"]["gp_after"]),
            ("gp_after_r2", ra["scratch"]["gp_after_r2"], rb["scratch"]["gp_after_r2"]),
            ("k_spread", ra["production"]["k_spread"], rb["production"]["k_spread"]),
            ("verdict_after", ra["gate"]["after"]["verdict"], rb["gate"]["after"]["verdict"]),
        ]
        if ra["gate"]["after"].get("interval") and rb["gate"]["after"].get("interval"):
            iva, ivb = ra["gate"]["after"]["interval"], rb["gate"]["after"]["interval"]
            fields.append(("d_lo_after", iva["d_lo"], ivb["d_lo"]))
            fields.append(("d_hi_after", iva["d_hi"], ivb["d_hi"]))
        rows = [
            {"n": n, "field": f, "A": x, "A_scaled": y, "exact_equal": bool(x == y)}
            for f, x, y in fields
        ]
        out["rows"].extend(rows)
        write_json("isoidentity", out)
        for r in rows:
            print(
                f"  n={n} {r['field']:<16} {r['A']} == {r['A_scaled']} -> {r['exact_equal']}",
                flush=True,
            )
    out["all_exact_equal"] = all(r["exact_equal"] for r in out["rows"])
    write_json("isoidentity", out)
    print("ALL EXACT EQUAL:", out["all_exact_equal"])


def stage_transition() -> None:
    """Active violation hunt in the zone where the final gate found the only
    two violations nobody had found before: the 500:1 rescale (missed truth by
    0.0044) and the 1000:1 family at n=1440 (missed by 0.090). Reuses stage5's
    exact factor list and rotation-seed rule, and adds the n-perturbation axis
    that found the second one -- the fix must not turn either into a MEASURED
    row with an interval missing the truth."""
    out: dict[str, Any] = {"status": "starting", "rows": []}
    write_json("transition", out)
    factors = [0.05, 0.02, 0.01, 0.005, 0.002]
    jobs: list[tuple[str, np.ndarray, float]] = []
    for n in (1440, 1600, 1760):
        a = np.random.default_rng(11).random((n, 2))
        for i, f in enumerate(factors):
            jobs.append((f"n={n}/rescale({f})", a * np.array([1.0, f]), 2.0))
            theta = float(np.random.default_rng(1000 + i).uniform(0.1, math.pi / 2 - 0.1))
            jobs.append(
                (
                    f"n={n}/rescale({f})+rot({math.degrees(theta):.1f}deg)",
                    (a * np.array([1.0, f])) @ rotation_2d(theta).T,
                    2.0,
                )
            )
    for name, pts, truth in jobs:
        row = run_row(name, pts, truth)
        out["rows"].append(row)
        write_json("transition", out)
        b, a_ = row["gate"]["before"], row["gate"]["after"]
        print(
            f"  {name:<40} shell={row['production']['shell_readout']:.4f} "
            f"gp {row['scratch']['gp_before']:.4f}->{row['scratch']['gp_after']:.4f} | "
            f"BEFORE {b['verdict']:<9} contains={b.get('interval_contains_truth')} | "
            f"AFTER {a_['verdict']:<9} contains={a_.get('interval_contains_truth')} "
            f"{'*** VIOLATION_AFTER ***' if a_.get('violation') else ''}",
            flush=True,
        )
    out["summary"] = _summarize(out["rows"])
    out["status"] = "done"
    write_json("transition", out)
    print(json.dumps(out["summary"], indent=1, default=str))


def stage_pinned() -> None:
    """The two named regression rows from MENSURA_BENCH_V2 section 9.1 -- the
    only two calibration violations the final gate found -- plus section 8.1's
    original 1000:1 row. All three must stay UNDECIDED, or come out MEASURED
    with an interval that contains 2.0."""
    out: dict[str, Any] = {"status": "starting", "rows": []}
    jobs = []
    pinned = (
        (1600, 0.002, "9.1a_500to1_n1600"),
        (1440, 0.001, "9.1b_1000to1_n1440"),
        (1600, 0.001, "8.1_1000to1_n1600"),
    )
    for n, f, label in pinned:
        a = np.random.default_rng(11).random((n, 2))
        jobs.append((label, a * np.array([1.0, f]), 2.0))
    for name, pts, truth in jobs:
        row = run_row(name, pts, truth)
        out["rows"].append(row)
        write_json("pinned", out)
        b, a_ = row["gate"]["before"], row["gate"]["after"]
        print(
            f"  {name:<24} shell={row['production']['shell_readout']:.4f} "
            f"gp {row['scratch']['gp_before']:.4f}->{row['scratch']['gp_after']:.4f} "
            f"scratch_bracket_after={row['scratch']['bracket_after']} | "
            f"BEFORE {b['verdict']} | AFTER {a_['verdict']} "
            f"contains={a_.get('interval_contains_truth')} "
            f"{'*** VIOLATION_AFTER ***' if a_.get('violation') else ''}",
            flush=True,
        )
    out["summary"] = _summarize(out["rows"])
    out["status"] = "done"
    write_json("pinned", out)
    print(json.dumps(out["summary"], indent=1, default=str))


STAGES = {
    "battery1600": lambda: stage_battery([1600], "battery_n1600"),
    "pinned": stage_pinned,
    "battery3200": lambda: stage_battery([3200], "battery_n3200"),
    "seeds": stage_seeds,
    "isoidentity": stage_isotropic_identity,
    "transition": stage_transition,
}


def main() -> int:
    for name in sys.argv[1:] or list(STAGES):
        print("=" * 110)
        print("STAGE", name)
        print("=" * 110, flush=True)
        STAGES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
