"""F4 verification: independent re-analysis of f4b_results.json.

Written from scratch by the verification pass. Does NOT import f4e_analyse or
f4_multifractal_targets -- the theoretical D0/D1/D2 are RE-DERIVED here from the
IFS parameters stated in the target names, so a wrong closed form in the
original module cannot propagate.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# --- independent re-derivation of the closed forms ---------------------------
# Equal-ratio IFS, N maps, ratio r, OSC. Self-similar measure with probs p.
#   sum_i p_i^q r^{-tau(q)} = 1  =>  tau(q) = log(sum_i p_i^q) / log r
#   D_q = tau(q)/(q-1)
# so with L = log(1/r) > 0:
#   D0 = log N / L ; D1 = -sum p log p / L ; D2 = -log(sum p^2) / L


def dq(n_maps: int, r: float, p: tuple[float, ...]) -> tuple[float, float, float]:
    L = math.log(1.0 / r)
    assert len(p) == n_maps and abs(sum(p) - 1.0) < 1e-12, (n_maps, p, sum(p))
    d0 = math.log(n_maps) / L
    d1 = -sum(pi * math.log(pi) for pi in p if pi > 0) / L
    d2 = -math.log(sum(pi * pi for pi in p)) / L
    return d0, d1, d2


def solve_b(n: int, r: float, want_d2: float) -> float:
    """p = (1-(n-1)b, b, ..., b) with the given D2.  Closed form, not bisection."""
    s = math.exp(-want_d2 * math.log(1.0 / r))  # = sum p_i^2
    # (1-(n-1)b)^2 + (n-1)b^2 = s
    #  -> b^2[(n-1)^2 + (n-1)] - 2(n-1)b + (1-s) = 0
    A = (n - 1) ** 2 + (n - 1)
    B = -2.0 * (n - 1)
    C = 1.0 - s
    disc = B * B - 4 * A * C
    assert disc >= 0, (n, r, want_d2, disc)
    return (-B - math.sqrt(disc)) / (2 * A)  # the root in [0, 1/n]


MATCHED_D2 = dq(3, 0.5, (0.6, 0.2, 0.2))[2]

SUP = {"SQ": (4, 0.5), "GA": (3, 0.5), "CD": (4, 1 / 3), "ST": (4, 0.5)}

PROBS = {
    "SQ-unif": (4, 0.5, (0.25,) * 4),
    "GA-unif": (3, 0.5, (1 / 3,) * 3),
    "CD-unif": (4, 1 / 3, (0.25,) * 4),
    "ST-unif": (4, 0.5, (0.25,) * 4),
    "SQ-mild": (4, 0.5, (0.4, 0.2, 0.2, 0.2)),
    "ST-mild": (4, 0.5, (0.4, 0.2, 0.2, 0.2)),
    "SQ-strong": (4, 0.5, (0.7, 0.1, 0.1, 0.1)),
    "ST-strong": (4, 0.5, (0.7, 0.1, 0.1, 0.1)),
    "CD-strong": (4, 1 / 3, (0.7, 0.1, 0.1, 0.1)),
    "GA-mild": (3, 0.5, (0.6, 0.2, 0.2)),
    "GA-strong": (3, 0.5, (0.8, 0.1, 0.1)),
}
for nm, sup in (("SQ-matched", "SQ"), ("CD-matched", "CD"), ("ST-matched", "ST")):
    n, r = SUP[sup]
    b = solve_b(n, r, MATCHED_D2)
    PROBS[nm] = (n, r, (1.0 - (n - 1) * b,) + (b,) * (n - 1))

MINE = {nm: dq(*v) for nm, v in PROBS.items()}


def main() -> None:
    rows = json.loads((HERE / "f4b_results.json").read_text())
    print(f"{len(rows)} records\n")

    # --- 1. do the JSON's stored D0/D1/D2 match my independent derivation? ----
    print("=" * 96)
    print("CHECK A: stored closed forms vs independently re-derived closed forms")
    print("=" * 96)
    print(f"{'target':<12} {'D0 store':>9} {'D0 mine':>9} {'D1 store':>9} {'D1 mine':>9} "
          f"{'D2 store':>9} {'D2 mine':>9}  {'agree':>5}")
    seen = {}
    for rec in rows:
        seen.setdefault(rec["target"], (rec["D0"], rec["D1"], rec["D2"]))
    worst = 0.0
    for nm, (d0, d1, d2) in seen.items():
        m0, m1, m2 = MINE[nm]
        err = max(abs(d0 - m0), abs(d1 - m1), abs(d2 - m2))
        worst = max(worst, err)
        print(f"{nm:<12} {d0:9.5f} {m0:9.5f} {d1:9.5f} {m1:9.5f} {d2:9.5f} {m2:9.5f}  {err:.1e}")
    print(f"\nworst absolute disagreement across all targets/quantities: {worst:.3e}\n")

    # --- 2. the identification table at the largest n ------------------------
    by = defaultdict(list)
    for rec in rows:
        by[(rec["target"], rec["n"])].append(rec)

    for n in sorted({r["n"] for r in rows}):
        print("=" * 118)
        print(f"CHECK B: identification at n = {n}   (median over k x max_radius, "
              f"{len(by[('SQ-unif', n)])} configs each)")
        print("=" * 118)
        hdr = (f"{'target':<12} {'D0':>7} {'D1':>7} {'D2':>7} | {'shell':>7} {'sh_unf':>7} "
               f"{'wf/80':>6} | {'gp':>7} {'gp_r2':>6} | {'d1ref':>7} {'d2ref':>7} "
               f"| {'sh-D0':>7} {'sh-D1':>7} {'sh-D2':>7} | {'gp-D2':>7}")
        print(hdr)
        for nm in PROBS:
            recs = by.get((nm, n))
            if not recs:
                continue
            d0, d1, d2 = MINE[nm]
            sh = float(np.nanmedian([r["shell"] for r in recs]))
            shu = float(np.median([r["shell_unfiltered"] for r in recs]))
            wf = float(np.median([r["n_well_fit"] for r in recs]))
            gp = float(np.median([r["gp"] for r in recs]))
            gpr2 = float(np.median([r["gp_r2"] for r in recs]))
            d1r = float(np.median([r["d1_ref"] for r in recs]))
            d2r = float(np.median([r["d2_ref"] for r in recs]))
            print(f"{nm:<12} {d0:7.4f} {d1:7.4f} {d2:7.4f} | {sh:7.4f} {shu:7.4f} {wf:6.0f} "
                  f"| {gp:7.4f} {gpr2:6.3f} | {d1r:7.4f} {d2r:7.4f} "
                  f"| {sh-d0:+7.3f} {sh-d1:+7.3f} {sh-d2:+7.3f} | {gp-d2:+7.3f}")
        print()

    # --- 3. ARM 1: does shell stay put along a fixed-support row? ------------
    print("=" * 100)
    print("CHECK C: ARM 1 -- fixed support (D0 constant), D2 swept. n=12800")
    print("=" * 100)
    for sup, members in (
        ("SQ", ["SQ-unif", "SQ-mild", "SQ-matched", "SQ-strong"]),
        ("ST", ["ST-unif", "ST-mild", "ST-matched", "ST-strong"]),
        ("GA", ["GA-unif", "GA-mild", "GA-strong"]),
        ("CD", ["CD-unif", "CD-matched", "CD-strong"]),
    ):
        print(f"  support {sup}: D0 = {MINE[members[0]][0]:.4f} fixed")
        for nm in members:
            recs = by.get((nm, 12800))
            if not recs:
                continue
            sh = float(np.nanmedian([r["shell"] for r in recs]))
            gp = float(np.median([r["gp"] for r in recs]))
            d2 = MINE[nm][2]
            print(f"    {nm:<12} D2={d2:6.4f}  shell={sh:7.4f}  gp={gp:7.4f}")
        recs_sh = [float(np.nanmedian([r["shell"] for r in by[(nm, 12800)]])) for nm in members
                   if (nm, 12800) in by]
        recs_d2 = [MINE[nm][2] for nm in members if (nm, 12800) in by]
        print(f"    -> D2 range {max(recs_d2)-min(recs_d2):.4f} ; "
              f"shell range {max(recs_sh)-min(recs_sh):.4f}\n")

    # --- 4. ARM 2: same D2, three/four supports ------------------------------
    print("=" * 100)
    print("CHECK D: ARM 2 -- D2 pinned at %.5f, D0 varied. n=12800" % MATCHED_D2)
    print("=" * 100)
    for nm in ["SQ-matched", "ST-matched", "GA-mild", "CD-matched"]:
        recs = by.get((nm, 12800))
        if not recs:
            continue
        sh = float(np.nanmedian([r["shell"] for r in recs]))
        gp = float(np.median([r["gp"] for r in recs]))
        d2r = float(np.median([r["d2_ref"] for r in recs]))
        print(f"  {nm:<12} D0={MINE[nm][0]:6.4f} D1={MINE[nm][1]:6.4f} D2={MINE[nm][2]:6.4f}"
              f"  shell={sh:7.4f}  gp={gp:7.4f}  d2ref={d2r:7.4f}")
    print()

    # --- 5. which hypothesis wins, scored globally --------------------------
    print("=" * 100)
    print("CHECK E: global scoring at n=12800 -- RMS error of each estimator vs each candidate")
    print("=" * 100)
    for excl in ([], ["CD"]):
        tag = "all targets" if not excl else "excluding CD (Cantor dust)"
        acc = defaultdict(list)
        for nm in PROBS:
            if any(nm.startswith(e) for e in excl):
                continue
            recs = by.get((nm, 12800))
            if not recs:
                continue
            sh = float(np.nanmedian([r["shell"] for r in recs]))
            gp = float(np.median([r["gp"] for r in recs]))
            d1r = float(np.median([r["d1_ref"] for r in recs]))
            d2r = float(np.median([r["d2_ref"] for r in recs]))
            d0, d1, d2 = MINE[nm]
            for est, val in (("shell", sh), ("gp", gp), ("d1_ref", d1r), ("d2_ref", d2r)):
                for cand, tv in (("D0", d0), ("D1", d1), ("D2", d2)):
                    acc[(est, cand)].append(val - tv)
        print(f"  [{tag}]")
        print(f"    {'estimator':<9} {'RMSE vs D0':>11} {'RMSE vs D1':>11} {'RMSE vs D2':>11}"
              f"  {'bias D0':>9} {'bias D1':>9} {'bias D2':>9}   verdict")
        for est in ("shell", "gp", "d1_ref", "d2_ref"):
            rmse = {c: math.sqrt(np.mean(np.square(acc[(est, c)]))) for c in ("D0", "D1", "D2")}
            bias = {c: float(np.mean(acc[(est, c)])) for c in ("D0", "D1", "D2")}
            best = min(rmse, key=rmse.get)
            print(f"    {est:<9} {rmse['D0']:11.4f} {rmse['D1']:11.4f} {rmse['D2']:11.4f}"
                  f"  {bias['D0']:+9.4f} {bias['D1']:+9.4f} {bias['D2']:+9.4f}   -> {best}")
        print()


if __name__ == "__main__":
    main()


def robustness() -> None:
    """CHECK F/G/H: is the identification an artifact of the median, the R^2
    filter, or a particular (k, max_radius)?"""
    rows = json.loads((HERE / "f4b_results.json").read_text())
    by = defaultdict(list)
    for rec in rows:
        by[(rec["target"], rec["n"])].append(rec)

    print("=" * 118)
    print("CHECK F: per-configuration SPREAD of the shell estimator at n=12800 "
          "(the median hides this)")
    print("=" * 118)
    print(f"{'target':<12} {'D0':>6} {'D2':>6} | {'min':>7} {'q25':>7} {'med':>7} {'q75':>7} "
          f"{'max':>7} {'#nan':>4} | {'closer to D0':>12} {'closer to D2':>12}")
    for nm in PROBS:
        recs = by.get((nm, 12800))
        if not recs:
            continue
        v = np.array([r["shell"] for r in recs], dtype=float)
        nn = int(np.sum(np.isnan(v)))
        vv = v[~np.isnan(v)]
        d0, _, d2 = MINE[nm]
        c0 = int(np.sum(np.abs(vv - d0) < np.abs(vv - d2)))
        c2 = len(vv) - c0
        print(f"{nm:<12} {d0:6.3f} {d2:6.3f} | {vv.min():7.3f} {np.percentile(vv,25):7.3f} "
              f"{np.median(vv):7.3f} {np.percentile(vv,75):7.3f} {vv.max():7.3f} {nn:4d} "
              f"| {c0:>7}/{len(vv)} {c2:>7}/{len(vv)}")

    print()
    print("=" * 118)
    print("CHECK G: does the R^2 filter make the answer?  Same scoring on the "
          "UNFILTERED per-node mean")
    print("=" * 118)
    for field in ("shell", "shell_unfiltered"):
        for excl in ([], ["CD"]):
            acc = defaultdict(list)
            for nm in PROBS:
                if any(nm.startswith(e) for e in excl):
                    continue
                recs = by.get((nm, 12800))
                if not recs:
                    continue
                val = float(np.nanmedian([r[field] for r in recs]))
                d0, d1, d2 = MINE[nm]
                for cand, tv in (("D0", d0), ("D1", d1), ("D2", d2)):
                    acc[cand].append(val - tv)
            rmse = {c: math.sqrt(np.mean(np.square(acc[c]))) for c in ("D0", "D1", "D2")}
            best = min(rmse, key=rmse.get)
            tag = "all" if not excl else "no CD"
            print(f"  {field:<18} [{tag:<5}]  RMSE  D0={rmse['D0']:.4f}  D1={rmse['D1']:.4f}  "
                  f"D2={rmse['D2']:.4f}   -> {best}")

    print()
    print("=" * 118)
    print("CHECK H: ARM-1 invariance test, per (k, max_radius) rather than pooled.")
    print("  If shell tracks D0 it must be FLAT along a fixed-support row; if it tracks D2 it "
          "must span the D2 range.")
    print("=" * 118)
    for sup, members in (("SQ", ["SQ-unif", "SQ-mild", "SQ-matched", "SQ-strong"]),
                         ("ST", ["ST-unif", "ST-mild", "ST-matched", "ST-strong"]),
                         ("GA", ["GA-unif", "GA-mild", "GA-strong"])):
        d2span = max(MINE[m][2] for m in members) - min(MINE[m][2] for m in members)
        print(f"  support {sup}  (D0 fixed at {MINE[members[0]][0]:.3f}; true D2 span "
              f"{d2span:.3f})")
        cfgs = sorted({(r["k"], r["max_radius"]) for r in by[(members[0], 12800)]})
        spans, gpspans = [], []
        for k, mr in cfgs:
            vals, gps = [], []
            for nm in members:
                rr = [r for r in by[(nm, 12800)] if r["k"] == k and r["max_radius"] == mr]
                if rr:
                    vals.append(rr[0]["shell"])
                    gps.append(rr[0]["gp"])
            if any(math.isnan(v) for v in vals):
                continue
            spans.append(max(vals) - min(vals))
            gpspans.append(max(gps) - min(gps))
        print(f"    shell span along the row, over {len(spans)} configs: "
              f"min {min(spans):.3f}  median {np.median(spans):.3f}  max {max(spans):.3f}")
        print(f"    gp span    along the row                          : "
              f"median {np.median(gpspans):.3f}   (true D2 span {d2span:.3f})")
    print()


if __name__ != "__main__":
    pass
