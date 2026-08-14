"""F3 diagnostic: verify the C workhorse reproduces the NumPy reference exactly.

The known-answer battery lives in f3_lyapunov_ref.py and validates the NumPy
implementation. The long integrations are run by f3_lyap.c. This script closes
the gap between the two: identical short runs, both implementations, agreement
required to ~1e-12 on every exponent. Without it, "validated" would apply to
code that never produced a reported number.

Also re-runs the CHEAP half of the known-answer battery through the C binary
directly (V1 linear spectrum with its 1/t extrapolation, V2 degenerate pair,
V3 zero pair, V4 Van der Pol), so the binary is not trusted purely by
inheritance from the reference.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

import f3_lyapunov_ref as ref

HERE = Path(__file__).resolve().parent
BIN = HERE / "f3_lyap"


def run_c(sys_name, dt, t_total, t_reorth, t_transient, x0, n_out=1):
    out = subprocess.run(
        [str(BIN), sys_name, repr(dt), repr(t_total), repr(t_reorth), repr(t_transient),
         *[repr(float(v)) for v in x0], str(n_out)],
        capture_output=True, text=True, check=True,
    ).stdout
    final = [ln for ln in out.splitlines() if ln.startswith("#FINAL")][0].split()
    n = (len(final) - 6) // 1
    lam = [float(v) for v in final[2 : 2 + n]]
    mean_div = float(final[final.index("mean_div") + 1])
    mean_x = float(final[final.index("mean_x") + 1])
    lam = [float(v) for v in final[2 : final.index("mean_div")]]
    return np.array(lam), mean_div, mean_x


SYSTEMS = {
    "lorenz": (ref.lorenz_f, ref.lorenz_j, [1.0, 1.0, 1.0], 3),
    "rossler": (ref.rossler_f, ref.rossler_j, [1.0, 1.0, 1.0], 3),
    "lin3": (ref.lin3_f, ref.lin3_j, [1.0, 1.0, 1.0], 3),
    "vdp": (ref.vdp_f, ref.vdp_j, [2.0, 0.0, 0.0], 2),
}


def main() -> None:
    print("=" * 78)
    print("CROSS-CHECK: C workhorse vs NumPy reference (identical short runs)")
    print("=" * 78)
    worst = 0.0
    for name, (f, j, x0, n) in SYSTEMS.items():
        dt, t_total, t_reorth, t_trans = 0.005, 40.0, 0.5, 5.0
        lam_c, div_c, _ = run_c(name, dt, t_total, t_reorth, t_trans, x0)
        r = ref.lyapunov_spectrum(f, j, np.array(x0[:n]), dt=dt, t_total=t_total,
                                  t_reorth=t_reorth, t_transient=t_trans, n_out=1)
        lam_p = r["exponents"]
        err = float(np.max(np.abs(lam_c - lam_p)))
        derr = abs(div_c - r["mean_divergence"])
        worst = max(worst, err, derr)
        print(f"  {name:8s} C   {'  '.join(f'{v:+.12f}' for v in lam_c)}")
        print(f"  {'':8s} py  {'  '.join(f'{v:+.12f}' for v in lam_p)}")
        print(f"  {'':8s} max|C-py| = {err:.3e}   |div diff| = {derr:.3e}")
    print(f"\n  worst disagreement across all systems: {worst:.3e}")
    print(f"  CROSS-CHECK {'PASSED' if worst < 1e-12 else 'FAILED'} (bar 1e-12)")

    print()
    print("=" * 78)
    print("KNOWN-ANSWER CASES RE-RUN THROUGH THE C BINARY ITSELF")
    print("=" * 78)

    # V1: exact spectrum {0.5,-0.2,-1.3}; the residual is the initial-basis
    # offset, which is O(1/t) and dt-independent -- verified, then extrapolated.
    print("\n  V1  3D non-normal linear flow, exact exponents (0.5, -0.2, -1.3)")
    rows = []
    for T in (200.0, 2000.0, 20000.0):
        lam, div, _ = run_c("lin3", 0.001, T, 0.1, 0.0, [1.0, 1.0, 1.0])
        lam = np.sort(lam)[::-1]
        rows.append((T, lam))
        err = lam - np.array([0.5, -0.2, -1.3])
        print(f"      T={T:>8.0f}  {'  '.join(f'{v:+.8f}' for v in lam)}   "
              f"max err {np.max(np.abs(err)):.3e}   sum={lam.sum():+.10f} (exact -1.0)")
    (t1, l1), (t2, l2) = rows[1], rows[2]
    # lam(t) = lam_inf + c/t  =>  lam_inf = (t2*l2 - t1*l1) / (t2 - t1)
    extrap = (t2 * l2 - t1 * l1) / (t2 - t1)
    exact = np.array([0.5, -0.2, -1.3])
    print(f"      1/t-extrapolated from T=2000,20000: "
          f"{'  '.join(f'{v:+.10f}' for v in extrap)}")
    print(f"      max |extrapolated - exact| = {np.max(np.abs(extrap - exact)):.3e}   "
          f"[{'PASS' if np.max(np.abs(extrap - exact)) < 1e-8 else 'FAIL'}]")
    lam_a, _, _ = run_c("lin3", 0.001, 2000.0, 0.1, 0.0, [1.0, 1.0, 1.0])
    lam_b, _, _ = run_c("lin3", 0.00025, 2000.0, 0.1, 0.0, [1.0, 1.0, 1.0])
    print(f"      dt-independence of the residual (dt 0.001 vs 0.00025): "
          f"max diff {np.max(np.abs(np.sort(lam_a) - np.sort(lam_b))):.3e}  "
          f"-> residual is the initial-basis offset, not integrator error")

    # V2 / V3: exactly degenerate and exactly zero pairs.
    for name, x0, expect, label in (
        ("rot", [1.0, 0.0, 0.0], [-0.3, -0.3], "V2  2D rotation+contraction (degenerate pair)"),
        ("harm", [1.0, 0.0, 0.0], [0.0, 0.0], "V3  harmonic oscillator (both exactly 0)"),
    ):
        lam, _, _ = run_c(name, 0.001, 500.0, 0.1, 0.0, x0)
        lam = np.sort(lam)[::-1]
        err = float(np.max(np.abs(lam - np.array(expect))))
        print(f"\n  {label}")
        print(f"      {'  '.join(f'{v:+.12f}' for v in lam)}   max err {err:.3e}   "
              f"[{'PASS' if err < 1e-9 else 'FAIL'}]")

    # V4: nonlinear flow with a genuine zero exponent, plus the exponent-sum
    # identity checked against an independently time-averaged divergence.
    print("\n  V4  Van der Pol mu=1 limit cycle (nonlinear, one exponent exactly 0)")
    for T in (2000.0, 20000.0):
        lam, div, _ = run_c("vdp", 0.0005, T, 0.1, 200.0, [2.0, 0.0, 0.0])
        lam = np.sort(lam)[::-1]
        print(f"      T={T:>8.0f}  lam1={lam[0]:+.3e} (exact 0)   "
              f"sum={lam.sum():+.10f}  vs <div f>={div:+.10f}   "
              f"|diff|={abs(lam.sum()-div):.3e}")

    # V5: chaotic map, published lambda_1 and an analytically exact sum.
    print("\n  V5  Henon map (a=1.4,b=0.3): published lam1~0.41922, exact sum=ln(0.3)")
    r = ref.lyapunov_spectrum_map(ref.henon_step, ref.henon_jac, np.array([0.1, 0.1]),
                                  n_iter=2_000_000, n_transient=10_000)
    lam = np.sort(r["exponents"])[::-1]
    print(f"      lam = {'  '.join(f'{v:+.8f}' for v in lam)}")
    print(f"      lam1 - 0.41922 = {lam[0]-0.41922:+.2e}   "
          f"sum - ln(0.3) = {lam.sum()-math.log(0.3):+.2e}")


if __name__ == "__main__":
    main()
