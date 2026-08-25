"""DIAGNOSIS of the `gp_local_mahalanobis` low-read on near-isotropic clouds
(docs/MENSURA_BENCH_V2.md section 9.2 blocker).

Measure before fixing. Nothing here changes behaviour; every routine below
either re-runs the existing scratch estimator with a SUBSTITUTED metric, or
instruments its internals, so that the mechanism can be attributed rather
than assumed.

The owner's hypothesis under test: per-point covariance ESTIMATION NOISE on
isotropic data smears the pairwise distance distribution and drags the fitted
log-log slope down. Its four predictions:

  (a) bias shrinks as k0 grows
  (b) bias vanishes if each M_i is replaced by the global-covariance metric
  (c) eigenvalue-ratio spread of the local covariances is >> 1 on isotropic data
  (d) bias is worse at small n and higher ambient d

Prediction (b) is the sharp one, and it is run FIRST, because it separates
"the per-point metric is the problem" from "the radius/fit rule is the
problem": `gp_local_mahalanobis` does NOT use the same fit window as
`baseline.correlation_dimension` (which the global arm goes through). The
local arm fits r in [0.01, 0.5] * (median 20th-neighbour distance); the
global arm fits r in [0.01, 0.2] * (bounding-box diagonal). Feeding a
CONSTANT metric through the local arm's own radius rule isolates which of
the two differences is responsible.
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
from scale_aware_prototype import (  # noqa: E402
    _CHUNK,
    global_covariance,
    gp_global_mahalanobis,
)
from scale_aware_v2 import (  # noqa: E402
    METRIC_RIDGE_FLOOR_FRAC,
    auto_k0,
    refine_metric,
    scale_aware_v2,
)

OUT_DIR = Path(__file__).resolve().parent


def _write(name: str, obj: Any) -> None:
    """MERGE into the on-disk artifact rather than overwrite it: the stages are
    run in separate processes (E6 discipline -- a stage that dies must not take
    a completed stage's evidence with it)."""
    path = OUT_DIR / f"gp_bias_fix_{name}.json"
    merged: dict[str, Any] = {}
    if path.exists():
        try:
            merged = json.loads(path.read_text())
        except json.JSONDecodeError:
            merged = {}
    merged.update(obj)
    path.write_text(json.dumps(merged, indent=1, default=str))


# --------------------------------------------------------------------------
# Instrumented copy of gp_local_mahalanobis: same arithmetic, but it also
# returns the radii, the raw counts and the per-interval local slopes, so the
# SHAPE of the log-log curve can be inspected rather than only its fitted
# slope.
# --------------------------------------------------------------------------


def gp_local_curve(
    arr: np.ndarray,
    sigma_inv: np.ndarray,
    *,
    n_radii: int = 20,
    r_min_frac: float = 0.01,
    r_max_frac: float = 0.5,
    ref_neighbors: int = 20,
) -> dict[str, Any]:
    n = len(arr)
    kth = min(ref_neighbors, n - 2)
    d2_sorted = np.empty((n, n))
    for start in range(0, n, _CHUNK):
        end = min(start + _CHUNK, n)
        diffs = arr[None, :, :] - arr[start:end, None, :]
        d2c = np.einsum("cnd,cde,cne->cn", diffs, sigma_inv[start:end], diffs)
        d2c[np.arange(end - start), np.arange(start, end)] = np.inf
        d2_sorted[start:end] = np.sort(d2c, axis=1)
    ref_dists = np.sqrt(d2_sorted[:, kth])
    ref_scale = float(np.median(ref_dists))
    radii = np.logspace(
        math.log10(r_min_frac * ref_scale), math.log10(r_max_frac * ref_scale), n_radii
    )
    d_sorted = np.sqrt(d2_sorted, out=d2_sorted)
    counts_int = np.zeros(n_radii, dtype=np.int64)
    for i in range(n):
        counts_int += np.searchsorted(d_sorted[i], radii, side="left")
    c = counts_int.astype(float) / (n * (n - 1))

    log_r, log_c, used = [], [], []
    for j, (r, ci) in enumerate(zip(radii, c, strict=True)):
        if 0 < ci < 1:
            log_r.append(math.log(r))
            log_c.append(math.log(ci))
            used.append(j)
    slopes = [
        (log_c[i + 1] - log_c[i]) / (log_r[i + 1] - log_r[i]) for i in range(len(log_r) - 1)
    ]
    if len(log_r) >= 2:
        design = np.vstack([np.array(log_r), np.ones(len(log_r))]).T
        coef, *_ = np.linalg.lstsq(design, np.array(log_c), rcond=None)
        slope = float(coef[0])
        pred = design @ coef
        ss_res = float(np.sum((np.array(log_c) - pred) ** 2))
        ss_tot = float(np.sum((np.array(log_c) - np.mean(log_c)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope, r2 = float("nan"), 0.0
    return {
        "dimension": slope,
        "r_squared": r2,
        "ref_scale": ref_scale,
        "radii": radii.tolist(),
        "pair_counts": counts_int.tolist(),
        "mean_neighbors_per_point": (counts_int / n).tolist(),
        "radii_used_idx": used,
        "n_radii_used": len(used),
        "local_slopes": slopes,
        "slope_first_half": float(np.mean(slopes[: len(slopes) // 2])) if slopes else float("nan"),
        "slope_second_half": float(np.mean(slopes[len(slopes) // 2 :])) if slopes else float("nan"),
    }


def _const_metric(sigma_inv_one: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(sigma_inv_one[None, :, :], n, axis=0)


def raw_local_covs(arr: np.ndarray, k0: int) -> np.ndarray:
    """UNREGULARIZED per-point covariance at a plain Euclidean k0 window."""
    from scipy.spatial import cKDTree

    tree = cKDTree(arr)
    _, idx = tree.query(arr, k=k0 + 1)
    rows = np.array([r[r != i][:k0] for i, r in enumerate(idx)])
    pts = arr[rows]
    centered = pts - pts.mean(axis=1, keepdims=True)
    return np.matmul(centered.transpose(0, 2, 1), centered) / k0


# ==========================================================================
# D1 -- the sharp discriminator (owner's prediction (b)), plus a decomposition
# of the per-point metric into its SCALE part and its SHAPE part.
# ==========================================================================


def d1_metric_substitution(arr: np.ndarray, label: str) -> dict[str, Any]:
    n, d = arr.shape
    sa = scale_aware_v2(arr)
    sig = sa.sigma_inv_metric  # (n,d,d) per-point, METRIC-floored
    cov_i = np.linalg.inv(sig)  # back to covariance space

    gcov = global_covariance(arr)
    ginv = np.linalg.inv(gcov)

    # SHAPE-ONLY noise: each local covariance rescaled to the same determinant
    # as the global one -> per-point orientation/aspect noise kept, per-point
    # SCALE noise removed.
    dets = np.linalg.det(cov_i)
    gdet = float(np.linalg.det(gcov))
    shape_only = cov_i * ((gdet / dets) ** (1.0 / d))[:, None, None]

    # SCALE-ONLY noise: isotropic matrix with each point's own mean eigenvalue
    # -> per-point scale noise kept, shape noise removed.
    tr = np.trace(cov_i, axis1=1, axis2=2) / d
    eye = np.eye(d)
    scale_only = tr[:, None, None] * eye[None, :, :]

    variants: dict[str, np.ndarray] = {
        "per_point_local (production)": sig,
        "global_cov_broadcast": _const_metric(ginv, n),
        "identity_broadcast": _const_metric(eye, n),
        "shape_noise_only": np.linalg.inv(shape_only),
        "scale_noise_only": np.linalg.inv(scale_only),
    }
    out: dict[str, Any] = {"case": label, "n": n, "d": d, "k0": sa.settings["k0"], "variants": {}}
    for vname, s in variants.items():
        t0 = time.perf_counter()
        cur = gp_local_curve(arr, s)
        cur["wall_time_s"] = time.perf_counter() - t0
        out["variants"][vname] = cur
    out["gp_global_arm (baseline radius rule)"] = gp_global_mahalanobis(arr)
    return out


# ==========================================================================
# D2 -- the radius/fit-window probe. Same per-point metric, but the local
# arm's fit window widened toward the one the global arm actually uses.
# ==========================================================================


def d2_window_probe(arr: np.ndarray, label: str) -> dict[str, Any]:
    sa = scale_aware_v2(arr)
    sig = sa.sigma_inv_metric
    ginv = np.linalg.inv(global_covariance(arr))
    const = _const_metric(ginv, len(arr))
    rows = []
    for r_min, r_max in [
        (0.01, 0.5),  # production
        (0.05, 0.5),
        (0.1, 1.0),
        (0.2, 2.0),
        (0.5, 4.0),
        (1.0, 8.0),
    ]:
        for mname, s in (("per_point_local", sig), ("global_broadcast", const)):
            cur = gp_local_curve(arr, s, r_min_frac=r_min, r_max_frac=r_max)
            rows.append(
                {
                    "case": label,
                    "metric": mname,
                    "r_min_frac": r_min,
                    "r_max_frac": r_max,
                    "dimension": cur["dimension"],
                    "r_squared": cur["r_squared"],
                    "n_radii_used": cur["n_radii_used"],
                    "min_mean_neighbors": min(cur["mean_neighbors_per_point"]),
                    "max_mean_neighbors": max(cur["mean_neighbors_per_point"]),
                    "slope_first_half": cur["slope_first_half"],
                    "slope_second_half": cur["slope_second_half"],
                }
            )
    return {"case": label, "rows": rows}


# ==========================================================================
# D3 -- owner's prediction (a): k0 sweep.  D4 -- prediction (c): eigenvalue
# ratio spread.  D5 -- prediction (d): n and d dependence.
# ==========================================================================


def d3_k0_sweep(arr: np.ndarray, label: str, k0s: list[int]) -> dict[str, Any]:
    rows = []
    for k0 in k0s:
        if k0 >= len(arr) - 2:
            continue
        t0 = time.perf_counter()
        _idx, _stopo, smetric, diag = refine_metric(arr, k0=k0)
        cur = gp_local_curve(arr, smetric)
        cov_i = np.linalg.inv(smetric)
        ev = np.linalg.eigvalsh(cov_i)
        ratio = ev[:, -1] / np.maximum(ev[:, 0], 1e-300)
        rows.append(
            {
                "case": label,
                "k0": k0,
                "iterations_run": diag["iterations_run"],
                "dimension": cur["dimension"],
                "r_squared": cur["r_squared"],
                "median_eig_ratio": float(np.median(ratio)),
                "p90_eig_ratio": float(np.percentile(ratio, 90)),
                "wall_time_s": time.perf_counter() - t0,
            }
        )
    return {"case": label, "rows": rows}


def d4_eig_spread(arr: np.ndarray, label: str) -> dict[str, Any]:
    n, d = arr.shape
    rows = []
    for k0 in (15, 30, 60, 120, 240):
        if k0 >= n - 2:
            continue
        covs = raw_local_covs(arr, k0)
        ev = np.linalg.eigvalsh(covs)
        ratio = ev[:, -1] / np.maximum(ev[:, 0], 1e-300)
        tr = ev.mean(axis=1)
        rows.append(
            {
                "case": label,
                "k0": k0,
                "median_eig_ratio_raw": float(np.median(ratio)),
                "p90_eig_ratio_raw": float(np.percentile(ratio, 90)),
                "max_eig_ratio_raw": float(np.max(ratio)),
                "scale_cv": float(np.std(tr) / np.mean(tr)),
                "theory_iso_expectation": "ratio -> 1 as k0 -> inf if truly isotropic",
            }
        )
    return {"case": label, "d": d, "rows": rows}


def d5_n_d_dependence() -> dict[str, Any]:
    rows = []
    for n in (400, 800, 1600, 3200):
        for kind, d in (("square", 2), ("cube", 3)):
            arr = targets.uniform_square(n, 11) if kind == "square" else targets.uniform_cube(n, 11)
            arr = np.asarray(arr, dtype=float)
            truth = 2.0 if kind == "square" else 3.0
            sa = scale_aware_v2(arr)
            loc = gp_local_curve(arr, sa.sigma_inv_metric)
            ginv = np.linalg.inv(global_covariance(arr))
            glob = gp_local_curve(arr, _const_metric(ginv, n))
            rows.append(
                {
                    "case": kind,
                    "n": n,
                    "d": d,
                    "truth": truth,
                    "k0": sa.settings["k0"],
                    "gp_local": loc["dimension"],
                    "gp_const_metric_same_window": glob["dimension"],
                    "bias_local": loc["dimension"] - truth,
                    "bias_const": glob["dimension"] - truth,
                    "noise_attributable": loc["dimension"] - glob["dimension"],
                }
            )
            print(
                f"  D5 {kind} n={n}: local={loc['dimension']:.4f} const={glob['dimension']:.4f}",
                flush=True,
            )
    return {"rows": rows}


def main() -> int:
    which = sys.argv[1:] or ["d1", "d2", "d3", "d4", "d5"]
    a = np.random.default_rng(11).random((1600, 2))
    cube = np.asarray(targets.uniform_cube(1600, 11), dtype=float)
    aniso100 = a * np.array([1.0, 0.01])
    out: dict[str, Any] = {"status": "starting"}

    if "d1" in which:
        out["d1_metric_substitution"] = []
        for label, arr in (("square(A)", a), ("cube", cube), ("rescale_100to1", aniso100)):
            r = d1_metric_substitution(arr, label)
            out["d1_metric_substitution"].append(r)
            _write("diag", out)
            print(f"D1 {label}:", flush=True)
            for vn, v in r["variants"].items():
                print(
                    f"    {vn:<30} D={v['dimension']:.4f} r2={v['r_squared']:.4f} "
                    f"used={v['n_radii_used']} nbrs=[{min(v['mean_neighbors_per_point']):.3f},"
                    f"{max(v['mean_neighbors_per_point']):.1f}] "
                    f"slope_lo={v['slope_first_half']:.3f} slope_hi={v['slope_second_half']:.3f}",
                    flush=True,
                )
            gdim = r["gp_global_arm (baseline radius rule)"]["dimension"]
            print(f"    global arm (own radius rule) D={gdim:.4f}", flush=True)

    if "d2" in which:
        out["d2_window_probe"] = []
        for label, arr in (("square(A)", a), ("rescale_100to1", aniso100)):
            r = d2_window_probe(arr, label)
            out["d2_window_probe"].append(r)
            _write("diag", out)
            print(f"D2 {label}:", flush=True)
            for row in r["rows"]:
                print(
                    f"    win=[{row['r_min_frac']},{row['r_max_frac']}] {row['metric']:<18} "
                    f"D={row['dimension']:.4f} r2={row['r_squared']:.4f} "
                    f"used={row['n_radii_used']} "
                    f"nbrs=[{row['min_mean_neighbors']:.3f},{row['max_mean_neighbors']:.1f}]",
                    flush=True,
                )

    if "d3" in which:
        out["d3_k0_sweep"] = []
        for label, arr in (("square(A)", a),):
            r = d3_k0_sweep(arr, label, [15, 30, 60, 120, 240, 480])
            out["d3_k0_sweep"].append(r)
            _write("diag", out)
            print(f"D3 {label}:", flush=True)
            for row in r["rows"]:
                print(
                    f"    k0={row['k0']:<4} D={row['dimension']:.4f} "
                    f"med_eig_ratio={row['median_eig_ratio']:.3f} "
                    f"it={row['iterations_run']} t={row['wall_time_s']:.1f}s",
                    flush=True,
                )

    if "d4" in which:
        out["d4_eig_spread"] = []
        for label, arr in (("square(A)", a), ("cube", cube)):
            r = d4_eig_spread(arr, label)
            out["d4_eig_spread"].append(r)
            _write("diag", out)
            print(f"D4 {label}:", flush=True)
            for row in r["rows"]:
                print(
                    f"    k0={row['k0']:<4} median_eig_ratio={row['median_eig_ratio_raw']:.3f} "
                    f"p90={row['p90_eig_ratio_raw']:.3f} max={row['max_eig_ratio_raw']:.2f} "
                    f"scale_cv={row['scale_cv']:.4f}",
                    flush=True,
                )

    if "d5" in which:
        out["d5_n_d"] = d5_n_d_dependence()
        _write("diag", out)

    out["metric_ridge_floor_frac"] = METRIC_RIDGE_FLOOR_FRAC
    out["auto_k0_at_1600_2d"] = auto_k0(1600, 2)
    out["status"] = "done"
    _write("diag", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
