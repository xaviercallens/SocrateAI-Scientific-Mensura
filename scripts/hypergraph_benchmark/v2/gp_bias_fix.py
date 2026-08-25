"""FIX for the `gp_local_mahalanobis` low-read on near-isotropic clouds
(docs/MENSURA_BENCH_V2.md section 9.2 blocker), plus the harness that
validates it against the full battery and the REAL production gate.

WHAT THE DIAGNOSIS FOUND (gp_bias_fix_diag.json, gp_bias_diag.py)
-----------------------------------------------------------------
The owner's hypothesis -- per-point covariance ESTIMATION NOISE smears the
pairwise distance distribution and drags the slope down -- is REFUTED as the
dominant mechanism. Measured on the canonical isotropic square
(`default_rng(11).random((1600,2))`, truth 2.0), feeding a CONSTANT metric
through `gp_local_mahalanobis`'s OWN radius rule:

    per-point local metric   D = 1.6525   <- production
    global covariance metric D = 1.6895   <- all per-point noise removed
    identity metric          D = 1.6855   <- no metric at all
    global arm, own rule     D = 1.9343

Removing 100% of the per-point metric noise recovers 0.037 of the 0.348
deficit -- about 10%. The k0 sweep confirms it independently: k0 15 -> 480
drops the median local eigenvalue ratio 4.84 -> 2.31 while D stays flat
(1.757, 1.653, 1.695, 1.651, 1.640, 1.669). Prediction (a) fails.

THE ACTUAL MECHANISM is the FIT WINDOW, and it is visible in the curve
itself. `gp_local_mahalanobis` fits r in [0.01, 0.5] * ref_scale where
ref_scale is the median 20th-Mahalanobis-neighbour distance. Since a radius
of f * r_20 contains on average 20 * f^d neighbours, that window spans mean
neighbour counts of 0.009 to 4.5 per point at n=1600 -- the ENTIRE fit lives
at or below the point-spacing scale. At the bottom of the window the total
pair count is ~14 out of 2.6M, and the `0 < C(r) < 1` filter silently
conditions on "at least one pair survived", which floors log C and FLATTENS
the log-log curve exactly where the sample is empty. Measured directly: on
the square the mean local slope over the lower half of the window is 1.168
while over the upper half it is 1.976 -- i.e. the top of the window already
reads the truth, and an unweighted OLS through both halves averages the
truth with an artifact.

The global arm does not have this problem because it goes through
`baseline.correlation_dimension`, whose window is [0.01, 0.2] * bounding-box
diagonal -- which at n=1600 spans mean neighbour counts of ~1 to ~400, i.e.
sits two decades HIGHER in the count scale.

THE FIX (zero-knob, no new constants)
-------------------------------------
The estimator is fitting log C(r) by ORDINARY least squares, which assumes
the errors in log C are homoscedastic. They are not: C(r) is a pair count,
so Var(log C_j) ~= 1/P_j to leading order, and P_j varies over five orders of
magnitude across this window (14 pairs at the bottom, 7.2M at the top).
The correct fit is inverse-variance weighted least squares with w_j = P_j.

This is a statistical correction, not a tuning choice:

  * it introduces NO constant, NO threshold and NO caller-facing knob -- the
    weights are the observed pair counts;
  * it leaves the radius grid, the metric, the reference scale and the
    distance arithmetic bit-for-bit unchanged, so the anisotropy correction
    that section 8.2 established is untouched by construction;
  * it is exactly scale-equivariant -- rescaling the cloud rescales every
    radius and leaves every count identical -- so the pinned isotropic-control
    byte-equality property is preserved by construction;
  * it removes the survivorship/depletion artifact automatically, because a
    radius holding 14 pairs gets 2e-6 of the weight of one holding 7.2M.

`weighted_fit_diagnostics` records the effective number of radii
(Kish's (sum w)^2 / sum w^2), the weight-weighted mean neighbour count and
the pair count at the weight median, so the certificate can state what the
fit actually used rather than only that a fit happened.

SHRINKAGE, evaluated and NOT adopted as the primary fix. The owner's
proposed Ledoit-Wolf shrinkage of the local covariance toward the global one
is implemented here (`shrunk_local_covariances`) and measured, because the
proposal deserves a number rather than an opinion. It addresses the 10%
term, not the 90% term: its best case is bounded above by the
`global_cov_broadcast` row (D = 1.6895), which still fails the gate. It is
reported as a measured negative result, and can be composed with the WLS fix
independently if ever wanted.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scale_aware_prototype import _CHUNK, global_covariance  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def write_json(name: str, obj: Any) -> None:
    (OUT_DIR / f"gp_bias_fix_{name}.json").write_text(json.dumps(obj, indent=1, default=str))


# --------------------------------------------------------------------------
# The pairwise-distance pass, lifted verbatim from `gp_local_mahalanobis` so
# that every variant below is compared on IDENTICAL distances and the only
# thing that differs is the fit.
# --------------------------------------------------------------------------


def _sorted_distances(arr: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    n = len(arr)
    d2_sorted = np.empty((n, n))
    for start in range(0, n, _CHUNK):
        end = min(start + _CHUNK, n)
        diffs = arr[None, :, :] - arr[start:end, None, :]
        d2c = np.einsum("cnd,cde,cne->cn", diffs, sigma_inv[start:end], diffs)
        d2c[np.arange(end - start), np.arange(start, end)] = np.inf
        d2_sorted[start:end] = np.sort(d2c, axis=1)
    return np.sqrt(d2_sorted, out=d2_sorted)


def _pair_counts(d_sorted: np.ndarray, radii: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(radii), dtype=np.int64)
    for i in range(len(d_sorted)):
        counts += np.searchsorted(d_sorted[i], radii, side="left")
    return counts


def _wls(log_r: np.ndarray, log_c: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least squares slope + weighted R^2. `w` unnormalized."""
    w = w / w.sum()
    sw = np.sqrt(w)
    a = np.vstack([log_r, np.ones_like(log_r)]).T
    coef, *_ = np.linalg.lstsq(a * sw[:, None], log_c * sw, rcond=None)
    pred = a @ coef
    ss_res = float(np.sum(w * (log_c - pred) ** 2))
    mean_w = float(np.sum(w * log_c))
    ss_tot = float(np.sum(w * (log_c - mean_w) ** 2))
    return float(coef[0]), (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def gp_local_mahalanobis_wls(
    arr: np.ndarray,
    sigma_inv: np.ndarray,
    *,
    n_radii: int = 20,
    r_min_frac: float = 0.01,
    r_max_frac: float = 0.5,
    ref_neighbors: int = 20,
    weighting: str = "pair_count",
) -> dict[str, float]:
    """`gp_local_mahalanobis` with the OLS log-log fit replaced by the
    inverse-variance weighted fit described in the module docstring.

    Everything up to and including `counts` is the original routine, so the
    two differ in the final four lines and nowhere else. `weighting="none"`
    reproduces the original estimator exactly and is kept as the control arm
    of every comparison below.
    """
    n = len(arr)
    kth = min(ref_neighbors, n - 2)
    d_sorted = _sorted_distances(arr, sigma_inv)
    ref_scale = float(np.median(d_sorted[:, kth]))
    if ref_scale <= 0:
        return {
            "dimension": float("nan"),
            "r_squared": 0.0,
            "n_radii_used": 0,
            "ref_scale": ref_scale,
        }

    radii = np.logspace(
        math.log10(r_min_frac * ref_scale), math.log10(r_max_frac * ref_scale), n_radii
    )
    counts = _pair_counts(d_sorted, radii)
    c = counts.astype(float) / (n * (n - 1))

    keep = (c > 0) & (c < 1)
    if int(keep.sum()) < 2:
        return {
            "dimension": float("nan"),
            "r_squared": 0.0,
            "n_radii_used": int(keep.sum()),
            "ref_scale": ref_scale,
        }
    log_r = np.log(radii[keep])
    log_c = np.log(c[keep])
    pairs = counts[keep].astype(float)
    if weighting == "none":
        w = np.ones_like(pairs)
    elif weighting == "pair_count":
        w = pairs
    elif weighting == "sqrt_pair_count":
        w = np.sqrt(pairs)
    else:
        raise ValueError(weighting)
    slope, r2 = _wls(log_r, log_c, w)
    wn = w / w.sum()
    return {
        "dimension": slope,
        "r_squared": r2,
        "n_radii_used": int(keep.sum()),
        "ref_scale": ref_scale,
        "weighting": weighting,
        # Certificate fields: what the fit actually leaned on.
        "effective_n_radii": float(1.0 / np.sum(wn**2)),
        "weighted_mean_neighbors_per_point": float(np.sum(wn * pairs) / n),
        "min_pair_count": float(pairs.min()),
        "max_pair_count": float(pairs.max()),
    }


# --------------------------------------------------------------------------
# The owner's proposed shrinkage, implemented so it can be measured.
# --------------------------------------------------------------------------


def shrunk_local_covariances(
    arr: np.ndarray, covs: np.ndarray, neighbor_idx: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ledoit-Wolf-style shrinkage of each per-point covariance toward the
    GLOBAL covariance, with the intensity estimated from the data per point.

    Sigma_i_shrunk = (1 - a_i) * Sigma_i + a_i * T,  T = global covariance
    scaled to Sigma_i's own trace (so the shrinkage moves SHAPE, not SCALE --
    a target with the wrong overall scale would otherwise inject a spurious
    per-point rescaling, which is the one thing the local metric must not do).

    a_i = min(1, (1/k^2 * sum_j ||y_j y_j^T - Sigma_i||_F^2) / ||Sigma_i - T||_F^2)
    is Ledoit-Wolf (2004) eq. (2)-(6) applied to the point's own neighbour
    window: numerator = estimated variance of the sample covariance,
    denominator = squared bias of the target. Returns (shrunk, intensities).
    """
    n, d = arr.shape
    pts = arr[neighbor_idx]
    y = pts - pts.mean(axis=1, keepdims=True)  # (n, k, d)
    k = y.shape[1]
    tr = np.trace(covs, axis1=1, axis2=2) / d
    t_tr = float(np.trace(target)) / d
    tgt = target[None, :, :] * (tr / t_tr)[:, None, None]  # (n, d, d)

    # sum_j || y_j y_j^T - Sigma_i ||_F^2  =  sum_j (|y_j|^4) - k*||Sigma_i||_F^2
    y2 = np.einsum("nkd,nkd->nk", y, y)
    num = (np.sum(y2**2, axis=1) - k * np.sum(covs**2, axis=(1, 2))) / (k * k)
    den = np.sum((covs - tgt) ** 2, axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(den > 0, num / den, 1.0)
    a = np.clip(a, 0.0, 1.0)
    shrunk = (1.0 - a)[:, None, None] * covs + a[:, None, None] * tgt
    return shrunk, a


def global_target(arr: np.ndarray) -> np.ndarray:
    return global_covariance(arr)
