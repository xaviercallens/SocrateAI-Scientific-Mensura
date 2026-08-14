"""What does ZERO-KNOB selection actually pick?

Deliverable (c). v2 section 2.2 is the reason this has to be shown rather than
asserted: on one target, varying only `k` moved the estimate from -0.39 to
+0.99, so a caller choosing `k` chooses the answer. The rule must therefore be
inspectable -- what it picks, on what evidence, on inputs whose right answers
differ from each other.

Writes incrementally (E6).

    python scripts/hypergraph_benchmark/v2/cic_selection_survey.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from targets import ALL_TARGETS  # noqa: E402

from socrates.hypergraph.cic import (  # noqa: E402
    CIC_CONTRACT_VERSION,
    K_LADDER,
    _as_array,
    _dedupe,
    _shell_arms,
    build_ladder,
    select_settings,
)

OUT = Path(__file__).resolve().parent / "cic_selection_survey.json"
N = 1600
SEED = 11


def main() -> int:
    payload: dict = {
        "contract_version": CIC_CONTRACT_VERSION,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n": N,
        "seed": SEED,
        "k_ladder": list(K_LADDER),
        "rows": [],
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"zero-knob selection, n={N}, seed={SEED}, ladder={K_LADDER}\n")
    print(
        f"{'target':<26}{'k':>4}{'mr':>4}{'minr':>6}  {'admissible k':<26}"
        f"{'largest-comp frac by k'}"
    )
    print("-" * 118)

    for target in ALL_TARGETS:
        points = target.generator(N, SEED)
        settings = select_settings(points)

        arr, dropped, dup_fraction = _dedupe(_as_array(points))
        rungs = build_ladder(arr)
        arms, traces = _shell_arms(rungs)

        row = {
            "target": target.name,
            "truth": target.truth,
            "selected": {
                "k": settings.k,
                "max_radius": settings.max_radius,
                "min_radius": settings.min_radius,
                "samples": settings.samples,
                "well_fit_r_squared": settings.well_fit_r_squared,
                "admissible_k": list(settings.admissible_k),
            },
            "evidence": {
                "n_duplicates_dropped": dropped,
                "duplicate_fraction": dup_fraction,
                "components_by_k": {
                    str(r.k): {
                        "n_components": r.n_components,
                        "largest_fraction": r.largest_fraction,
                    }
                    for r in rungs
                },
                "window_trace_by_k": {
                    str(k): [
                        {"radius": r, "well_fit_fraction": f, "ball_fraction": b}
                        for r, f, b in trace
                    ]
                    for k, trace in traces.items()
                },
                "readout_by_k": {str(k): a.readout for k, a in arms.items()},
            },
        }
        payload["rows"].append(row)
        OUT.write_text(json.dumps(payload, indent=2))

        fracs = " ".join(f"{r.k}:{r.largest_fraction:.2f}" for r in rungs)
        print(
            f"{target.name:<26}{settings.k:>4}{settings.max_radius:>4}"
            f"{settings.min_radius:>6}  {str(list(settings.admissible_k)):<26}{fracs}"
        )

    print(f"\nwritten: {OUT}")
    print(
        "\nselection rule:\n  " + select_settings(ALL_TARGETS[0].generator(400, 1)).selection_rule
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
