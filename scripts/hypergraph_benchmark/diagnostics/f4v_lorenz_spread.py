"""F4 verification (e): how big is D0 - D2 for Lorenz and Rossler, really?

DO NOT try to measure D_0 on these attractors directly and trust it -- published
direct box counts for Lorenz span roughly 1.97 to 2.31, and several of them fall
BELOW the published D_2, which is impossible (D_0 >= D_1 >= D_2).  Takens' box
algorithm is known not to converge on Lorenz.  A direct box count here would
just add another unreliable number.

Instead the gap is BRACKETED, using two well-conditioned quantities and two
standard results:

  * Kaplan-Yorke / Lyapunov dimension D_L, from the full Lyapunov spectrum by
    Benettin QR.  This is by far the best-conditioned dimension for these
    systems (Sprott: D_KY "believed accurate to the four significant digits",
    against capacity and correlation dimensions that are "considerably less
    reliable").
  * D_2 from the library's own Grassberger-Procaccia estimator, on a long
    strided sample of the same orbit.

  UPPER BOUND.  Box dimension is bounded by the Lyapunov dimension for an
  attractor (Constantin-Foias-Temam / Ledrappier; Leonov-Kuznetsov for these
  systems specifically):        D_0 <= D_L
  and D_2 <= D_1 <= D_0 always.  Therefore

        0  <=  D_0 - D_2  <=  D_L - D_2

  so measuring D_L and D_2 pins the ENTIRE spread from both sides without ever
  measuring D_0.  That is the number the F4 caveat turns on.

A direct box count is still reported alongside, with per-level local slopes, but
purely as a sanity check -- it is expected to be poor and is not used for the
verdict.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
RA, RB, RC = 0.2, 0.2, 5.7


def lorenz_block(M, out):
    """M is (3, 1+m): column 0 = state, rest = tangent vectors."""
    x, y, z = M[0, 0], M[1, 0], M[2, 0]
    out[0, 0] = SIGMA * (y - x)
    out[1, 0] = x * (RHO - z) - y
    out[2, 0] = x * y - BETA * z
    V = M[:, 1:]
    out[0, 1:] = -SIGMA * V[0] + SIGMA * V[1]
    out[1, 1:] = (RHO - z) * V[0] - V[1] - x * V[2]
    out[2, 1:] = y * V[0] + x * V[1] - BETA * V[2]
    return out


def rossler_block(M, out):
    x, y, z = M[0, 0], M[1, 0], M[2, 0]
    out[0, 0] = -y - z
    out[1, 0] = x + RA * y
    out[2, 0] = RB + z * (x - RC)
    V = M[:, 1:]
    out[0, 1:] = -V[1] - V[2]
    out[1, 1:] = V[0] + RA * V[1]
    out[2, 1:] = z * V[0] + (x - RC) * V[2]
    return out


def rk4_block(f, M, h, buf):
    k1, k2, k3, k4, tmp = buf
    f(M, k1)
    np.multiply(k1, 0.5 * h, out=tmp); np.add(M, tmp, out=tmp); f(tmp, k2)
    np.multiply(k2, 0.5 * h, out=tmp); np.add(M, tmp, out=tmp); f(tmp, k3)
    np.multiply(k3, h, out=tmp); np.add(M, tmp, out=tmp); f(tmp, k4)
    return M + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def lyapunov(f, x0, *, h, t_total, t_renorm=1.0, t_transient=200.0):
    M = np.zeros((3, 4))
    M[:, 0] = x0
    M[:, 1:] = np.eye(3)
    buf = [np.empty((3, 4)) for _ in range(5)]
    for _ in range(int(t_transient / h)):
        M = rk4_block(f, M, h, buf)
    M[:, 1:] = np.eye(3)
    acc = np.zeros(3)
    per = int(round(t_renorm / h))
    n_ren = int(round(t_total / t_renorm))
    for _ in range(n_ren):
        for _ in range(per):
            M = rk4_block(f, M, h, buf)
        Q, R = np.linalg.qr(M[:, 1:])
        d = np.sign(np.diag(R)); d[d == 0] = 1.0
        acc += np.log(np.abs(np.diag(R)))
        M[:, 1:] = Q * d
    return np.sort(acc / (n_ren * t_renorm))[::-1]


def kaplan_yorke(l):
    c = np.cumsum(l)
    idx = np.where(c >= 0)[0]
    if len(idx) == 0:
        return 0.0
    j = int(idx.max())
    if j == len(l) - 1:
        return float(len(l))
    return (j + 1) + float(c[j] / abs(l[j + 1]))


def orbit(f, x0, *, h, n, sub, t_transient=200.0):
    M = np.zeros((3, 1))
    M[:, 0] = x0
    buf = [np.empty((3, 1)) for _ in range(5)]
    for _ in range(int(t_transient / h)):
        M = rk4_block(f, M, h, buf)
    out = np.empty((n, 3))
    for i in range(n):
        for _ in range(sub):
            M = rk4_block(f, M, h, buf)
        out[i] = M[:, 0]
    return out


def boxcount(pts, n_levels=13, occ_frac=0.05):
    lo = pts.min(axis=0)
    span = float((pts.max(axis=0) - lo).max())
    n = len(pts)
    xs, ys, occ = [], [], []
    for m in range(1, n_levels + 1):
        idx = np.floor((pts - lo) / (span * 2.0**-m)).astype(np.int64)
        k = len(np.unique(idx, axis=0))
        xs.append(m * math.log(2.0)); ys.append(math.log(k)); occ.append(k)
    local = [round((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]), 3) for i in range(len(xs) - 1)]
    keep = [i for i, o in enumerate(occ) if 30 <= o <= occ_frac * n]
    d0 = float(np.polyfit([xs[i] for i in keep], [ys[i] for i in keep], 1)[0]) \
        if len(keep) >= 2 else float("nan")
    return d0, occ, local, [i + 1 for i in keep]


def main() -> None:
    for name, f, x0, h, dt_orbit, target in (
        ("Lorenz  (sigma=10, rho=28, beta=8/3)", lorenz_block, [1.0, 1.0, 1.0], 0.004, 0.02, 2.05),
        ("Rossler (a=0.2, b=0.2, c=5.7)", rossler_block, [1.0, 1.0, 1.0], 0.008, 0.05, 2.01),
    ):
        print("=" * 100, flush=True)
        print(name)
        print("=" * 100, flush=True)
        lam = lyapunov(f, x0, h=h, t_total=3000.0)
        lam2 = lyapunov(f, x0, h=h / 2, t_total=1500.0)
        dl, dl2 = kaplan_yorke(lam), kaplan_yorke(lam2)
        print(f"  Lyapunov spectrum (h={h},   T=3000): {np.round(lam, 5)}  sum {lam.sum():+.4f}"
              f"   -> D_L = {dl:.5f}", flush=True)
        print(f"  Lyapunov spectrum (h={h/2}, T=1500): {np.round(lam2, 5)}  sum {lam2.sum():+.4f}"
              f"   -> D_L = {dl2:.5f}", flush=True)

        pts = orbit(f, x0, h=h, n=400_000, sub=max(1, int(round(dt_orbit / h))))
        d0, occ, local, keep = boxcount(pts)
        print(f"  box count on {len(pts)} points (SANITY ONLY -- known unreliable here):")
        print(f"    occupied per halving level: {occ}")
        print(f"    local slopes              : {local}")
        print(f"    D_0 over levels {keep} = {d0:.4f}", flush=True)

        sub = pts[:: max(1, len(pts) // 20000)][:20000]
        gp_wide = correlation_dimension(list(map(tuple, sub)), n_radii=40)
        gp_tight = correlation_dimension(list(map(tuple, sub)), n_radii=40,
                                         r_min_frac=0.005, r_max_frac=0.05)
        print(f"  GP D_2 (n={len(sub)}, r in [0.01,0.2]*diag)  = {gp_wide.dimension:.5f}"
              f"  R2={gp_wide.r_squared:.4f}")
        print(f"  GP D_2 (n={len(sub)}, r in [0.005,0.05]*diag)= {gp_tight.dimension:.5f}"
              f"  R2={gp_tight.r_squared:.4f}", flush=True)

        print()
        for lbl, d2 in (("wide-r GP", gp_wide.dimension), ("tight-r GP", gp_tight.dimension)):
            print(f"  BRACKET with {lbl}:  0 <= D_0 - D_2 <= D_L - D_2 = "
                  f"{dl - d2:+.4f}")
        print(f"  benchmark scoring target {target}, tolerance +/- 0.5  -> the whole "
              f"D_0-D_2 spread is at most {abs(dl - gp_tight.dimension) / 0.5 * 100:.1f}% "
              f"of the tolerance")
        print()


if __name__ == "__main__":
    main()
