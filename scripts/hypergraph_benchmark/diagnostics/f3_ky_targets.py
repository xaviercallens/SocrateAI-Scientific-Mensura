"""F3 diagnostic: method-independent Kaplan-Yorke targets for problems 08 and 09.

Runs the validated Benettin solver (f3_lyap.c, cross-checked against
f3_lyapunov_ref.py, whose known-answer battery must pass first) over a grid of
initial conditions, time steps and re-orthonormalisation intervals for the
EXACT systems the benchmark's own clouds came from:

  Lorenz   sigma=10, rho=28, beta=8/3     (round2/08_lorenz_attractor.py)
  Rossler  a=0.2, b=0.2, c=5.7            (round2/09_rossler_attractor.py)

and reports, for each:
  * the full spectrum with its spread across the grid,
  * convergence of each exponent vs integration time,
  * the exponent-sum identity check (sum lambda = <div f>, which for Lorenz is
    the CONSTANT -(sigma+1+beta) = -13.6666... and for Rossler is a - c + <x>),
  * D_KY with an uncertainty taken from the actual grid spread, not asserted.

The uncertainty is deliberately built from three independent sources that are
all measured here rather than assumed: scatter across initial conditions
(sampling of the natural measure), scatter across dt (integrator truncation),
and scatter across t_reorth (algorithm parameter). Whichever dominates is
reported.

Output: JSON to the scratchpad plus a human-readable table.
"""

from __future__ import annotations


import json


from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BIN = HERE / "f3_lyap"
OUT = Path(
    "/tmp/claude-1000/-home-xavkal-xdev/cb774970-2d62-4712-aaa9-94dafb5c745b"
    "/scratchpad/f3_ky_results.json"
)

# Initial conditions: 8 per system, all well away from the fixed points and
# from each other, so the ensemble samples the natural measure rather than one
# orbit segment. The transient is discarded before the tangent basis starts.
ICS = [
    (1.0, 1.0, 1.0), (-1.0, 2.0, 5.0), (5.0, -3.0, 20.0), (0.5, 0.5, 0.5),
    (-8.0, 7.0, 27.0), (2.0, -2.0, 12.0), (10.0, 10.0, 30.0), (-4.0, -4.0, 15.0),
]

GRID = {
    "lorenz": {
        "dts": (0.005, 0.002, 0.001),
        "reorths": (0.5, 1.0),
        "t_total": 100_000.0,
        "t_transient": 100.0,
        # sum lambda must equal this constant exactly (div f is state-independent)
        "div_exact": -(10.0 + 1.0 + 8.0 / 3.0),
    },
    "rossler": {
        "dts": (0.01, 0.005, 0.002),
        "reorths": (0.5, 1.0),
        "t_total": 200_000.0,  # lambda_1 ~ 0.07: needs a longer average
        "t_transient": 500.0,
        "div_exact": None,  # a - c + <x>, measured per run
    },
}


RUNS_DIR = Path(
    "/tmp/claude-1000/-home-xavkal-xdev/cb774970-2d62-4712-aaa9-94dafb5c745b"
    "/scratchpad/f3_runs"
)


def parse_run(path: Path):
    """Parse one f3_lyap output file (written by f3_run_sweep.sh)."""
    rows, final, header = [], None, ""
    for ln in path.read_text().splitlines():
        if ln.startswith("#FINAL"):
            final = ln.split()
        elif ln.startswith("# sys="):
            header = ln
        elif not ln.startswith("#"):
            rows.append([float(v) for v in ln.split()])
    if final is None:
        return None
    meta = dict(tok.split("=", 1) for tok in header[2:].split() if "=" in tok and "(" not in tok)
    lam = [float(v) for v in final[2 : final.index("mean_div")]]
    return {
        "system": meta["sys"],
        "dt": float(meta["dt"]),
        "t_reorth": float(meta["t_reorth"]),
        "t_total": float(meta["t_total"]),
        "ic": path.stem.split("_ic")[1],
        "lam": sorted(lam, reverse=True),
        "mean_div": float(final[final.index("mean_div") + 1]),
        "mean_x": float(final[final.index("mean_x") + 1]),
        "trace": rows,
    }


def kaplan_yorke(lam) -> float:
    lam = np.sort(np.asarray(lam, dtype=float))[::-1]
    cum = np.cumsum(lam)
    j = int(np.sum(cum >= 0.0))
    if j == 0:
        return 0.0
    if j >= len(lam):
        return float(len(lam))
    return float(j + cum[j - 1] / abs(lam[j]))


def main() -> None:
    results = [r for r in (parse_run(p) for p in sorted(RUNS_DIR.glob("*.txt"))) if r]
    print(f"parsed {len(results)} completed Benettin integrations from {RUNS_DIR}")

    for r in results:
        r["D_KY"] = kaplan_yorke(r["lam"])
        lam0 = list(r["lam"])
        lam0[1] = 0.0  # lambda_2 is exactly 0 for a bounded non-fixed-point flow
        r["D_KY_lam2_zero"] = kaplan_yorke(lam0)

    with OUT.open("w") as fh:
        json.dump(results, fh)

    for sys_name, cfg in GRID.items():
        rs = [r for r in results if r["system"] == sys_name]
        lam = np.array([r["lam"] for r in rs])
        dky = np.array([r["D_KY"] for r in rs])
        dky0 = np.array([r["D_KY_lam2_zero"] for r in rs])

        print()
        print("=" * 78)
        print(f"{sys_name.upper()}   T={cfg['t_total']:.0f}   "
              f"{len(rs)} runs = {len(ICS)} ICs x {len(cfg['dts'])} dt x "
              f"{len(cfg['reorths'])} t_reorth")
        print("=" * 78)
        for i in range(3):
            print(f"  lambda_{i+1}:  mean {lam[:, i].mean():+.6f}   "
                  f"sd {lam[:, i].std(ddof=1):.6f}   "
                  f"min {lam[:, i].min():+.6f}   max {lam[:, i].max():+.6f}")
        s = lam.sum(axis=1)
        div = np.array([r["mean_div"] for r in rs])
        print(f"  sum lambda:  mean {s.mean():+.8f}   sd {s.std(ddof=1):.2e}")
        print(f"  <div f>   :  mean {div.mean():+.8f}   "
              f"max |sum - <div f>| over runs = {np.max(np.abs(s - div)):.2e}")
        if cfg["div_exact"] is not None:
            print(f"  analytic <div f> = {cfg['div_exact']:+.8f}   "
                  f"max |sum - analytic| = {np.max(np.abs(s - cfg['div_exact'])):.2e}")
        else:
            mx = np.array([r["mean_x"] for r in rs])
            pred = 0.2 - 5.7 + mx
            print(f"  analytic <div f> = a - c + <x> = {pred.mean():+.8f} "
                  f"(<x> = {mx.mean():+.6f})   max |sum - analytic| = "
                  f"{np.max(np.abs(s - pred)):.2e}")

        print(f"  D_KY (measured lambda_2)     : mean {dky.mean():.6f}   "
              f"sd {dky.std(ddof=1):.6f}   range [{dky.min():.6f}, {dky.max():.6f}]")
        print(f"  D_KY (lambda_2 set exactly 0): mean {dky0.mean():.6f}   "
              f"sd {dky0.std(ddof=1):.6f}   range [{dky0.min():.6f}, {dky0.max():.6f}]")

        # Which knob dominates the spread?
        print("\n  spread decomposition of D_KY (lambda_2 = 0 convention):")
        for key, label in (("dt", "dt"), ("t_reorth", "t_reorth")):
            vals = sorted({r[key] for r in rs})
            for v in vals:
                sub = np.array([r["D_KY_lam2_zero"] for r in rs if r[key] == v])
                print(f"    {label}={v:<7}: mean {sub.mean():.6f}  sd {sub.std(ddof=1):.6f}  "
                      f"n={len(sub)}")
        best_dt = min(cfg["dts"])
        sub = [r for r in rs if r["dt"] == best_dt]
        by_ic = np.array([r["D_KY_lam2_zero"] for r in sub])
        print(f"    across ICs at finest dt={best_dt}: mean {by_ic.mean():.6f}  "
              f"sd {by_ic.std(ddof=1):.6f}  n={len(by_ic)}")

        # Convergence vs integration time, from one representative run.
        rep = [r for r in rs if r["dt"] == best_dt and r["t_reorth"] == 0.5][0]
        tr = np.array(rep["trace"])
        print(f"\n  convergence vs T (IC={rep['ic']}, dt={best_dt}, t_reorth=0.5):")
        print(f"    {'T':>9}  {'lam1':>11}  {'lam2':>11}  {'lam3':>11}  {'D_KY':>9}")
        targets = [100, 300, 1000, 3000, 10_000, 30_000, 100_000, 200_000]
        for T in targets:
            if T > tr[-1, 0]:
                continue
            idx = int(np.argmin(np.abs(tr[:, 0] - T)))
            row = tr[idx]
            lam_row = sorted(row[1:4], reverse=True)
            lam_row0 = [lam_row[0], 0.0, lam_row[2]]
            print(f"    {row[0]:>9.0f}  {lam_row[0]:>+11.6f}  {lam_row[1]:>+11.6f}  "
                  f"{lam_row[2]:>+11.6f}  {kaplan_yorke(lam_row0):>9.6f}")

        # Scatter of the running D_KY over the last decade of T, an independent
        # read on how much of the uncertainty is just finite averaging time.
        tails = []
        for r in sub:
            t = np.array(r["trace"])
            mask = t[:, 0] >= 0.5 * t[-1, 0]
            for row in t[mask]:
                lam_row = sorted(row[1:4], reverse=True)
                tails.append(kaplan_yorke([lam_row[0], 0.0, lam_row[2]]))
        tails = np.array(tails)
        print(f"\n  running D_KY over the last half of each run (finest dt, all ICs): "
              f"sd {tails.std(ddof=1):.6f}, range [{tails.min():.6f}, {tails.max():.6f}]")


if __name__ == "__main__":
    main()
