"""THE REFUTATION: two disjoint MEASURED intervals for one point set.

Written by an independent reviewer. Run it and read the last line.

Dimension -- Hausdorff, box/D_0, information/D_1, correlation/D_2, every notion
MENSURA-BENCH v2 names -- is invariant under an invertible linear map of the
ambient space. Such a map is bi-Lipschitz, and every one of those dimensions is
a bi-Lipschitz invariant.

Take the EXACT point set of the build's own passing row
`uniform-square-2d / n=1600 / seed=11`, and apply `diag(1, 0.01)`: the same
1600 samples, in different units on the second coordinate. This is what happens
when one column of a dataset is recorded in metres and the other in kilometres.

    certify(A)                -> MEASURED [1.5859, 2.2822]
    certify(A @ diag(1,0.01)) -> MEASURED [0.6374, 1.3626]

The two intervals are DISJOINT, and neither run raised a single abstention
signal. No assignment of a truth value to the underlying set can put it in both,
so at least one of these is a `MEASURED` interval that does not contain the
truth -- a CALIBRATION VIOLATION -- and this conclusion needs no agreement about
what the dimension of a thin rectangle "really" is. The identity row is the one
that contains 2.0, so the rescaled row is the violation.

Control: an ISOTROPIC rescaling (x100) changes nothing, so the failure is
specifically anisotropy, not scale, and is not a numerical-conditioning artifact.

E6: written to disk as it goes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.cic import certify  # noqa: E402

OUT = Path(__file__).resolve().parent / "cic_refutation_affine.json"


def main() -> int:
    # byte-identical to targets.uniform_square(1600, 11)
    a = np.random.default_rng(11).random((1600, 2))
    cases = {
        "A = uniform_square(1600, seed=11)": a,
        "A @ diag(100, 100)  [isotropic control]": a * 100.0,
        "A @ diag(1, 0.01)   [y in km not m]": a * np.array([1.0, 0.01]),
    }
    rows = []
    for name, pts in cases.items():
        r = certify(pts, label=name)
        rows.append(
            {
                "case": name,
                "verdict": r.verdict,
                "d_lo": r.d_lo,
                "d_hi": r.d_hi,
                "contains_2": r.contains(2.0),
                "signals": list(r.signals),
                "shell": r.diagnostics.shell_readout,
                "gp": r.diagnostics.gp_readout,
                "k": r.certificate.settings.k,
                "max_radius": r.certificate.settings.max_radius,
                "inputs_hash": r.certificate.inputs_hash,
            }
        )
        OUT.write_text(json.dumps({"rows": rows}, indent=2))
        print(
            f"{name:<42} {r.verdict:<10} "
            f"[{r.d_lo:.4f}, {r.d_hi:.4f}]" if r.d_lo is not None else f"{name:<42} {r.verdict}",
            f" contains 2.0 = {r.contains(2.0)}  signals={','.join(r.signals) or 'NONE'}",
            flush=True,
        )

    ident = next(r for r in rows if r["case"].startswith("A ="))
    aniso = next(r for r in rows if "0.01" in r["case"])
    disjoint = (
        ident["verdict"] != "UNDECIDED"
        and aniso["verdict"] != "UNDECIDED"
        and (ident["d_lo"] > aniso["d_hi"] or aniso["d_lo"] > ident["d_hi"])
    )
    verdict = {
        "two_measured_intervals_are_disjoint": bool(disjoint),
        "conclusion": (
            "CALIBRATION VIOLATION: one point set, a dimension-preserving linear map, "
            "two disjoint MEASURED intervals, zero abstention signals."
            if disjoint
            else "no refutation on this pair"
        ),
    }
    OUT.write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2))
    print(json.dumps(verdict, indent=2))
    return 1 if disjoint else 0


if __name__ == "__main__":
    raise SystemExit(main())
