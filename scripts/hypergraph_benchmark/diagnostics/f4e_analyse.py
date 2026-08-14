"""F4, step E: score the sweep. Which candidate does each estimator track?

Reads f4b_results.json and answers three questions with numbers rather than
inspection:

  E1  ACCURACY. Across all targets and configurations, RMSE of each estimator
      against each of D_0, D_1, D_2. The candidate an estimator tracks is the
      one it is closest to *everywhere*, not on a favourable subset.

  E2  SENSITIVITY, the decisive test. Within a support D_0 is held EXACTLY
      fixed while D_2 is swept by ~1. Regressing each estimator's output on
      D_2 within a support gives a slope that must be ~1 for a D_2 estimator
      and ~0 for a D_0 estimator. This cannot be faked by a bias: any additive
      offset cancels in a slope.

  E3  MATCHED-D_2 ARM. Three supports carrying measures with IDENTICAL D_2 and
      D_0 of 2.000 / 1.585 / 1.262. A D_2 estimator must return one number
      across all three; a D_0 estimator must reproduce the ordering.

Rows where the shell estimator is not measuring anything -- nan, or the
totally-disconnected Cantor-dust support where step A found it returns negative
dimensions -- are reported separately rather than averaged in.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROWS = json.loads((Path(__file__).resolve().parent / "f4b_results.json").read_text())


def rmse(pairs):
    vals = [(a - b) ** 2 for a, b in pairs if math.isfinite(a)]
    return math.sqrt(sum(vals) / len(vals)) if vals else float("nan")


def main() -> None:
    usable = [r for r in ROWS if r["support"] != "CD" and math.isfinite(r["shell"])]
    dropped_cd = [r for r in ROWS if r["support"] == "CD"]
    dropped_nan = [r for r in ROWS if r["support"] != "CD" and not math.isfinite(r["shell"])]
    print(
        f"{len(ROWS)} rows; {len(usable)} usable "
        f"({len(dropped_cd)} Cantor-dust rows and {len(dropped_nan)} nan rows held out)\n"
    )

    print("=" * 96)
    print("E1  RMSE of each estimator against each candidate, over all usable rows")
    print("=" * 96)
    per_row_gp = {(r["target"], r["n"]): r for r in ROWS if r["support"] != "CD"}.values()
    print(f"{'estimator':<28}{'vs D_0':>10}{'vs D_1':>10}{'vs D_2':>10}   verdict")
    for label, key, rows in (
        ("shell-growth (mean_dim)", "shell", usable),
        ("Grassberger-Procaccia", "gp", per_row_gp),
        ("ref: mean log ball count", "d1_ref", per_row_gp),
        ("ref: log mean ball count", "d2_ref", per_row_gp),
    ):
        rows = list(rows)
        e = {c: rmse([(r[key], r[c]) for r in rows]) for c in ("D0", "D1", "D2")}
        best = min(e, key=e.get)
        print(f"{label:<28}{e['D0']:>10.4f}{e['D1']:>10.4f}{e['D2']:>10.4f}   tracks {best}")

    print()
    print("=" * 96)
    print("E2  SENSITIVITY within a support (D_0 held EXACTLY fixed, D_2 swept)")
    print("    slope of estimator vs D_2 across the targets on that support.")
    print("    A D_2 estimator must give ~1.00; a D_0 estimator must give ~0.00.")
    print("=" * 96)
    for support in ("SQ", "GA", "ST"):
        rows = [r for r in usable if r["support"] == support]
        if not rows:
            continue
        d0 = rows[0]["D0"]
        by_cfg = defaultdict(list)
        for r in rows:
            by_cfg[(r["n"], r["k"], r["max_radius"])].append(r)
        slopes = []
        for cfg, group in by_cfg.items():
            if len({round(g["D2"], 6) for g in group}) >= 3:
                slopes.append(
                    np.polyfit([g["D2"] for g in group], [g["shell"] for g in group], 1)[0]
                )
        gp_rows = {(r["target"], r["n"]): r for r in rows}.values()
        by_n = defaultdict(list)
        for r in gp_rows:
            by_n[r["n"]].append(r)
        gp_slopes = [
            np.polyfit([g["D2"] for g in grp], [g["gp"] for g in grp], 1)[0]
            for grp in by_n.values()
            if len({round(g["D2"], 6) for g in grp}) >= 3
        ]
        d2_span = max(r["D2"] for r in rows) - min(r["D2"] for r in rows)
        shell_span = max(r["shell"] for r in rows) - min(r["shell"] for r in rows)
        print(
            f"  support {support} (D_0 = {d0:.4f} exactly, D_2 spans {d2_span:.3f}):\n"
            f"     shell d(est)/d(D_2) = {np.mean(slopes):+.3f} "
            f"(sd {np.std(slopes):.3f}, {len(slopes)} configs)\n"
            f"     GP    d(est)/d(D_2) = {np.mean(gp_slopes):+.3f} "
            f"(sd {np.std(gp_slopes):.3f}, {len(gp_slopes)} n-values)\n"
            f"     total shell spread over every target/config on this support: {shell_span:.3f}"
        )

    print()
    print("=" * 96)
    print("E3  MATCHED-D_2 ARM: same D_2 (1.1844), three different D_0")
    print("=" * 96)
    print(f"{'target':<12}{'D0':>8}{'D2':>8} | {'shell mean':>12}{'sd':>7}{'n cfg':>7} | {'GP':>8}")
    for name in ("SQ-matched", "GA-mild", "ST-matched", "CD-matched"):
        rows = [r for r in ROWS if r["target"] == name and math.isfinite(r["shell"])]
        if not rows:
            print(f"{name:<12}  (no usable rows -- shell estimator returns nan/negative here)")
            continue
        s = [r["shell"] for r in rows]
        gp = np.mean([r["gp"] for r in rows])
        flag = "  <- shell estimator is broken on this support (step A)" if name[:2] == "CD" else ""
        print(
            f"{name:<12}{rows[0]['D0']:>8.4f}{rows[0]['D2']:>8.4f} | "
            f"{np.mean(s):>12.4f}{np.std(s):>7.4f}{len(s):>7} | {gp:>8.4f}{flag}"
        )

    print()
    print("=" * 96)
    print("E4  FULL TABLE, averaged over the k x max_radius grid at each (target, n)")
    print("=" * 96)
    print(
        f"{'target':<12}{'n':>7}{'D0':>7}{'D1':>7}{'D2':>7} | {'shell':>8}{'sd':>7}{'wf/n':>8} | "
        f"{'GP':>8}{'d1ref':>8}{'d2ref':>8} | nearest"
    )
    for name in dict.fromkeys(r["target"] for r in ROWS):
        for n in sorted({r["n"] for r in ROWS}):
            rows = [r for r in ROWS if r["target"] == name and r["n"] == n]
            if not rows:
                continue
            s = [r["shell"] for r in rows if math.isfinite(r["shell"])]
            r0 = rows[0]
            if not s:
                print(f"{name:<12}{n:>7}{r0['D0']:>7.3f}{r0['D1']:>7.3f}{r0['D2']:>7.3f} |"
                      f"{'all nan':>16}")
                continue
            m = float(np.mean(s))
            errs = {c: abs(m - r0[c]) for c in ("D0", "D1", "D2")}
            wf = np.mean([r["n_well_fit"] / r["n_sampled"] for r in rows])
            print(
                f"{name:<12}{n:>7}{r0['D0']:>7.3f}{r0['D1']:>7.3f}{r0['D2']:>7.3f} | "
                f"{m:>8.4f}{np.std(s):>7.4f}{wf:>8.2f} | {r0['gp']:>8.4f}"
                f"{r0['d1_ref']:>8.4f}{r0['d2_ref']:>8.4f} | {min(errs, key=errs.get)}"
            )


if __name__ == "__main__":
    main()
