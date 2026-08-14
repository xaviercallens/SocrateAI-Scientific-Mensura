"""Battery runner for the O(n^2 d^2) optimization pass (owner decision 8.5
item 3). Runs the full 7-case battery through the same pipeline that produced
the section-8 evidence (scale_aware_v2 -> shell_estimate -> gp_local /
gp_global -> bracket), records every number, and writes incrementally to
optimize_results.json under a caller-chosen tag ("before" / "after").

Usage: python optimize_battery.py <tag> [--n3200]

The 1000:1 rescale row must reproduce the pinned section-8.1 evidence
bracket [0.749, 1.989] exactly (it came from skeptic_verify.py's
run_full_pipeline, which this mirrors: shell samples=80, max_radius=6).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import targets  # noqa: E402
from scale_aware_prototype import (  # noqa: E402
    build_hypergraph,
    gp_global_mahalanobis,
    gp_local_mahalanobis,
    shell_estimate,
)
from scale_aware_v2 import bracket, scale_aware_v2  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "optimize_results.json"


def _load() -> dict[str, Any]:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def _save(obj: dict[str, Any]) -> None:
    RESULTS_PATH.write_text(json.dumps(obj, indent=2, default=str))


def run_case(name: str, arr: np.ndarray, truth: float | None) -> dict[str, Any]:
    t0 = time.perf_counter()
    sa = scale_aware_v2(arr)
    t_sa = time.perf_counter() - t0
    hg = build_hypergraph(sa.neighbor_idx)
    t1 = time.perf_counter()
    se = shell_estimate(hg, max_radius=6, samples=80)
    t_shell = time.perf_counter() - t1
    t2 = time.perf_counter()
    gpl = gp_local_mahalanobis(arr, sa.sigma_inv_metric)
    t_gpl = time.perf_counter() - t2
    t3 = time.perf_counter()
    gpg = gp_global_mahalanobis(arr)
    t_gpg = time.perf_counter() - t3
    br_loc = bracket(se["mean_dimension"], gpl["dimension"])
    br_glob = bracket(se["mean_dimension"], gpg["dimension"])
    return {
        "case": name,
        "n": len(arr),
        "d": int(arr.shape[1]),
        "truth": truth,
        "settings_auto": sa.settings,
        "shell": se,
        "gp_local_mahalanobis": gpl,
        "gp_global_mahalanobis": gpg,
        "bracket_local_gp": br_loc,
        "bracket_global_gp": br_glob,
        "contains_truth_local_gp": (
            (br_loc["d_lo"] <= truth <= br_loc["d_hi"]) if truth is not None else None
        ),
        "refine_rounds": sa.diagnostics["refine"]["rounds"],
        "wall_time_s": time.perf_counter() - t0,
        "wall_breakdown_s": {
            "scale_aware_v2": t_sa,
            "shell_estimate": t_shell,
            "gp_local": t_gpl,
            "gp_global": t_gpg,
        },
    }


def battery_cases(include_n3200: bool) -> list[tuple[str, np.ndarray, float | None]]:
    a = np.random.default_rng(11).random((1600, 2))
    cases: list[tuple[str, np.ndarray, float | None]] = [
        ("uniform-square-2d/n=1600/seed=11", targets.uniform_square(1600, 11), 2.0),
        ("circle-1d/n=1600/seed=11", targets.circle(1600, 11), 1.0),
        ("uniform-cube-3d/n=1600/seed=11", targets.uniform_cube(1600, 11), 3.0),
        ("A = uniform_square(1600,11)", a, 2.0),
        ("A @ diag(1,0.01) [refutation: y in km]", a * np.array([1.0, 0.01]), 2.0),
        ("A @ diag(100,100) [isotropic control]", a * 100.0, 2.0),
        ("rescale_y_factor=0.001 [pinned 8.1 violation row]", a * np.array([1.0, 0.001]), 2.0),
    ]
    if include_n3200:
        cases.append(("uniform-square-2d/n=3200/seed=555", targets.uniform_square(3200, 555), 2.0))
        cases.append(("uniform-cube-3d/n=3200/seed=555", targets.uniform_cube(3200, 555), 3.0))
    return cases


def main() -> int:
    tag = sys.argv[1]
    include_n3200 = "--n3200" in sys.argv
    results = _load()
    results.setdefault(tag, {})
    for name, arr, truth in battery_cases(include_n3200):
        if name in results[tag]:
            print(f"[skip, already present] {name}", flush=True)
            continue
        row = run_case(name, np.asarray(arr, dtype=float), truth)
        results[tag][name] = row
        _save(results)
        br = row["bracket_local_gp"]
        print(
            f"{name:<52} shell={row['shell']['mean_dimension']:.6f} "
            f"gp_loc={row['gp_local_mahalanobis']['dimension']:.6f} "
            f"-> [{br['d_lo']:.4f},{br['d_hi']:.4f}] "
            f"k0={row['settings_auto']['k0']} k={row['settings_auto']['k_selected']} "
            f"iters={row['settings_auto']['iterations_run']} wall={row['wall_time_s']:.2f}s",
            flush=True,
        )
    print(f"written: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
