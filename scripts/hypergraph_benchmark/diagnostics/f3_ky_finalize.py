"""F3 diagnostic: reduce the Benettin sweep to one target value + uncertainty each.

Reads f3_ky_results.json (written by f3_ky_targets.py) and writes
f3_ky_targets_final.json, which f3_rescore.py consumes. The reduction rule is
stated here rather than chosen after seeing the answer:

  central value  mean of D_KY over the runs at the FINEST dt in the sweep
                 (integrator truncation is the one error source that does not
                 average out, so the coarse-dt runs are diagnostics for the
                 dt-sensitivity, not contributions to the estimate);
  lambda_2       set to exactly 0, its analytic value for a bounded flow that
                 is not at a fixed point. The measured lambda_2 is reported
                 alongside as a check that it IS small, and the difference
                 between the two conventions is folded into the uncertainty.
  uncertainty    the LARGEST of: the spread across initial conditions, the
                 shift between the two finest dt, the shift between the two
                 t_reorth values, and the lambda_2-convention difference.
                 Taking the max rather than a quadrature sum is deliberate:
                 these are systematic knobs, not independent random draws.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCRATCH = Path(
    "/tmp/claude-1000/-home-xavkal-xdev/cb774970-2d62-4712-aaa9-94dafb5c745b/scratchpad"
)
SRC = SCRATCH / "f3_ky_results.json"
DST = SCRATCH / "f3_ky_targets_final.json"

# Published Kaplan-Yorke values carried from general knowledge of the
# literature, for comparison only -- NOTHING below is computed from them.
PUBLISHED = {
    "lorenz": {
        "lam": (0.9056, 0.0, -14.5723),
        "D_KY": 2.0 + 0.9056 / 14.5723,
        "src": "Wolf, Swift, Swinney & Vastano (1985), Physica D 16, 285 -- the "
               "standard quoted spectrum for sigma=10, rho=28, beta=8/3",
    },
    "rossler": {
        "lam": (0.0714, 0.0, -5.3943),
        "D_KY": 2.0 + 0.0714 / 5.3943,
        "src": "Wolf et al. (1985) / Sprott, Chaos and Time-Series Analysis -- "
               "standard quoted spectrum for a=b=0.2, c=5.7",
    },
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
    with SRC.open() as fh:
        runs = json.load(fh)

    out = {}
    for system in ("lorenz", "rossler"):
        rs = [r for r in runs if r["system"] == system]
        dts = sorted({r["dt"] for r in rs})
        fine, next_fine = dts[0], dts[1]

        def dky(r, lam2_zero=True):
            lam = list(r["lam"])
            if lam2_zero:
                lam[1] = 0.0
            return kaplan_yorke(lam)

        fine_runs = [r for r in rs if r["dt"] == fine]
        central = float(np.mean([dky(r) for r in fine_runs]))

        u_ic = float(np.std([dky(r) for r in fine_runs], ddof=1))
        u_dt = abs(central - float(np.mean([dky(r) for r in rs if r["dt"] == next_fine])))
        reorths = sorted({r["t_reorth"] for r in rs})
        u_reorth = abs(
            float(np.mean([dky(r) for r in fine_runs if r["t_reorth"] == reorths[0]]))
            - float(np.mean([dky(r) for r in fine_runs if r["t_reorth"] == reorths[1]]))
        )
        u_lam2 = abs(central - float(np.mean([dky(r, lam2_zero=False) for r in fine_runs])))
        sd = max(u_ic, u_dt, u_reorth, u_lam2)

        lam_mean = np.mean([r["lam"] for r in fine_runs], axis=0)
        lam_sd = np.std([r["lam"] for r in fine_runs], axis=0, ddof=1)

        pub = PUBLISHED[system]
        out[system] = {
            "D_KY": central,
            "sd": sd,
            "uncertainty_budget": {
                "across_initial_conditions": u_ic,
                "dt_shift_finest_to_next": u_dt,
                "t_reorth_shift": u_reorth,
                "lambda2_convention": u_lam2,
            },
            "lambda_mean": lam_mean.tolist(),
            "lambda_sd": lam_sd.tolist(),
            "dt_used": fine,
            "n_runs_used": len(fine_runs),
            "t_total": fine_runs[0]["t_total"],
            "published_D_KY": pub["D_KY"],
            "published_lambda": list(pub["lam"]),
            "published_source": pub["src"],
        }

        print("=" * 78)
        print(f"{system.upper()}")
        print("=" * 78)
        print(f"  spectrum (dt={fine}, T={fine_runs[0]['t_total']:.0f}, "
              f"{len(fine_runs)} runs):")
        for i in range(3):
            print(f"    lambda_{i+1} = {lam_mean[i]:+.6f} +- {lam_sd[i]:.6f}   "
                  f"(published {pub['lam'][i]:+.4f})")
        print(f"  D_KY = {central:.5f} +- {sd:.5f}")
        print(f"    budget: IC {u_ic:.5f} | dt {u_dt:.5f} | t_reorth {u_reorth:.5f} "
              f"| lambda_2 convention {u_lam2:.5f}")
        print(f"  published D_KY = {pub['D_KY']:.5f}   "
              f"difference = {central - pub['D_KY']:+.5f}  "
              f"({abs(central - pub['D_KY']) / sd:.1f} of our own uncertainty)")
        print(f"    source: {pub['src']}")

    with DST.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
