"""Read `cic_known_answer.json` and answer the two questions that decide whether
the interval construction is sound.

1. VALIDITY. Did any `MEASURED` row miss its truth? That is the only failure.
2. IS THE BRACKET DOING THE WORK? v2 section 2.1 proposed bracketing from two
   readouts that fail in OPPOSITE directions. If the raw hull already contains
   the truth, the bracket is load-bearing and the allowance is insurance. If it
   does not, the interval is being carried entirely by the measured bias floor,
   which is a different -- and weaker -- claim, and the pre-registration needs
   to know which one it is buying.

    python scripts/hypergraph_benchmark/v2/cic_analyse.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    data = json.loads((HERE / "cic_known_answer.json").read_text())
    rows = data["rows"]

    print("=" * 96)
    print("1. VALIDITY -- a MEASURED interval missing its truth is the only failure")
    print("=" * 96)
    violations = [r for r in rows if r["calibration_violation"]]
    measured = [r for r in rows if r["result"]["verdict"] != "UNDECIDED"]
    undecided = [r for r in rows if r["result"]["verdict"] == "UNDECIDED"]
    print(f"  rows                    {len(rows)}")
    print(f"  MEASURED / DEGENERATE   {len(measured)}")
    print(f"  UNDECIDED               {len(undecided)}")
    print(f"  CALIBRATION VIOLATIONS  {len(violations)}")
    for r in violations:
        print(f"    !! {r['target']} n={r['n']} seed={r['seed']} {r['result']['d_lo']}..")

    print()
    print("=" * 96)
    print("2. IS THE BRACKET DOING THE WORK, or is the allowance carrying the interval?")
    print("=" * 96)
    print(
        f"  {'target':<26}{'rows':>5}{'hull holds truth':>18}{'median width':>14}"
        f"{'median hull width':>19}"
    )
    by_target: dict[str, list[dict]] = {}
    for r in measured:
        if r["truth"] is not None:
            by_target.setdefault(r["target"], []).append(r)
    hull_hits = hull_total = 0
    for target, group in by_target.items():
        hits = 0
        widths, hull_widths = [], []
        for r in group:
            diag = r["result"]["diagnostics"]
            lo, hi, truth = diag["hull_lo"], diag["hull_hi"], r["truth"]
            if lo is not None and hi is not None:
                hits += int(lo <= truth <= hi)
                hull_widths.append(hi - lo)
            widths.append(r["result"]["width"])
        hull_hits += hits
        hull_total += len(group)
        print(
            f"  {target:<26}{len(group):>5}{f'{hits}/{len(group)}':>18}"
            f"{st.median(widths):>14.3f}"
            f"{(st.median(hull_widths) if hull_widths else float('nan')):>19.3f}"
        )
    print(
        f"\n  overall: the raw two-readout hull contains the truth on "
        f"{hull_hits}/{hull_total} MEASURED rows."
    )
    print("  Every row where it does not is a row the measured bias floor alone saved.")

    print()
    print("=" * 96)
    print("3. WHY EACH ABSTENTION FIRED")
    print("=" * 96)
    counts: dict[str, int] = {}
    for r in undecided:
        for s in r["result"]["signals"]:
            counts[s] = counts.get(s, 0) + 1
    for signal, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        targets = sorted({r["target"] for r in undecided if signal in r["result"]["signals"]})
        print(f"  {signal:<26}{count:>4}   {', '.join(targets)}")

    print()
    print("=" * 96)
    print("4. WHAT ZERO-KNOB SELECTION PICKED")
    print("=" * 96)
    print(f"  {'target':<26}{'n':>6}  {'k':>3}{'max_radius':>12}{'admissible k'}")
    seen = set()
    for r in rows:
        key = (r["target"], r["n"])
        if key in seen:
            continue
        seen.add(key)
        s = r["result"]["certificate"]["settings"]
        print(
            f"  {r['target']:<26}{r['n']:>6}  {s['k']:>3}{s['max_radius']:>12}"
            f"    {s['admissible_k']}"
        )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
