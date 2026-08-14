"""CIC known-answer validation.

The one thing this checks is the one criterion v2 has: a `MEASURED` interval
that does not contain the declared truth is a CALIBRATION VIOLATION. Owner
decision E2 says this build optimises for VALIDITY, not score -- so a row that
abstains is a pass of the validity condition, and only a violation is a
failure.

E6: results are written to disk after EVERY row, not at the end. A prior
workflow lost three agents to API errors and only its on-disk artifacts were
recoverable.

    python scripts/hypergraph_benchmark/v2/cic_known_answer.py            # core 3 rows
    python scripts/hypergraph_benchmark/v2/cic_known_answer.py --all      # full suite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from targets import ALL_TARGETS, CORE_TARGETS  # noqa: E402

from socrates.hypergraph.cic import CIC_CONTRACT_VERSION, Verdict, certify  # noqa: E402

OUT = Path(__file__).resolve().parent / "cic_known_answer.json"
SEEDS = (11, 20260814, 7)
SIZES = (1600, 3200)


def _flush(payload: dict) -> None:
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run the full suite, not just the core 3")
    ap.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args()

    targets = ALL_TARGETS if args.all else CORE_TARGETS
    payload: dict = {
        "contract_version": CIC_CONTRACT_VERSION,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "criterion": (
            "a MEASURED interval that does not contain the declared truth is a "
            "CALIBRATION VIOLATION and is the only failure; UNDECIDED never violates"
        ),
        "sizes": args.sizes,
        "seeds": args.seeds,
        "rows": [],
        "summary": {},
    }
    _flush(payload)

    violations = 0
    measured = 0
    undecided = 0

    header = (
        f"{'target':<26}{'n':>6}{'seed':>10} {'verdict':<16}"
        f"{'d_lo':>8}{'d_hi':>8}{'w':>7}{'truth':>8}  signals"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for target in targets:
        for n in args.sizes:
            for seed in args.seeds:
                points = target.generator(n, seed)
                t0 = time.perf_counter()
                result = certify(
                    points,
                    label=f"{target.name}/n={n}/seed={seed}",
                    replay_command=(
                        "python scripts/hypergraph_benchmark/v2/cic_known_answer.py --all "
                        f"--sizes {n} --seeds {seed}   # row {target.name}"
                    ),
                )
                elapsed = time.perf_counter() - t0

                contains = result.contains(target.truth) if target.truth is not None else None
                violated = bool(
                    result.verdict != Verdict.UNDECIDED
                    and target.truth is not None
                    and contains is False
                )
                if result.verdict == Verdict.UNDECIDED:
                    undecided += 1
                else:
                    measured += 1
                violations += int(violated)

                row = {
                    "target": target.name,
                    "n": n,
                    "seed": seed,
                    "truth": target.truth,
                    "truth_kind": target.truth_kind,
                    "expected_verdict": target.expect,
                    "contains_truth": contains,
                    "calibration_violation": violated,
                    "elapsed_s": elapsed,
                    "result": result.to_dict(),
                }
                payload["rows"].append(row)
                payload["summary"] = {
                    "rows": len(payload["rows"]),
                    "measured": measured,
                    "undecided": undecided,
                    "calibration_violations": violations,
                    "valid": violations == 0,
                }
                _flush(payload)

                lo = "  --  " if result.d_lo is None else f"{result.d_lo:>8.4f}"
                hi = "  --  " if result.d_hi is None else f"{result.d_hi:>8.4f}"
                w = "  --  " if result.width is None else f"{result.width:>7.3f}"
                truth = "  --  " if target.truth is None else f"{target.truth:>8.4f}"
                flag = "  <<< CALIBRATION VIOLATION" if violated else ""
                print(
                    f"{target.name:<26}{n:>6}{seed:>10} {result.verdict:<16}"
                    f"{lo}{hi}{w}{truth}  {','.join(result.signals) or '-'}{flag}"
                )

    print("=" * 100)
    print(json.dumps(payload["summary"], indent=2))
    print(f"written: {OUT}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
