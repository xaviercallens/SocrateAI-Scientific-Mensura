"""ADVERSARIAL probe 2: sensitivity attacks on the CIC core (reviewer-written).

Four attacks that the target battery in `cic_adversarial.py` cannot make:

  A. PERMUTATION. `_sampled_nodes` stride-samples SORTED node ids, and node ids
     are row indices of the input array. So the 80 nodes measured depend on the
     ORDER of the rows, not only on the point SET. If a permutation of the same
     cloud can move the verdict or the interval, the answer has a dependence the
     certificate hashes but the science does not license.
  B. n-SWEEP. Every recorded row is at n in {1600, 3200}. Sweep 200..6400.
  C. SMALL n, right at the TOO_FEW_POINTS boundary (max(K_LADDER)+1 = 41).
  D. TRAJECTORY. The recorded `undersampled-trajectory` row lands 0.004 from a
     violation and is excluded from scoring as INDEPENDENT-D1. For Lorenz,
     D_0, D_1 = 2.0622 and D_2 all agree to about 0.01, so that exclusion is
     worth testing across sampling strides.

E6: written to disk after every row.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.cic import certify  # noqa: E402

OUT = Path(__file__).resolve().parent / "cic_adversarial2.json"


def square(n, seed=11):
    return np.random.default_rng(seed).random((n, 2))


def cube(n, seed=11):
    return np.random.default_rng(seed).random((n, 3))


def circle(n, seed=7):
    t = np.sort(np.random.default_rng(seed).random(n)) * 2 * math.pi
    return np.c_[np.cos(t), np.sin(t)]


def swiss(n, seed=11):
    rng = np.random.default_rng(seed)
    t = 1.5 * math.pi * (1 + 2 * rng.random(n))
    h = 12 * rng.random(n)
    return np.c_[t * np.cos(t), h, t * np.sin(t)] / 12


def lorenz(n, stride, seed=0):
    del seed
    dt = 0.002
    x = np.array([1.0, 1.0, 1.0])
    out = []
    for i in range(n * stride + 20000):
        s, r, b = 10.0, 28.0, 8.0 / 3.0
        dx = np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])
        x = x + dt * dx
        if i >= 20000 and (i - 20000) % stride == 0:
            out.append(x.copy())
    return np.asarray(out[:n]) / 30.0


ROWS: list[dict] = []
PAYLOAD: dict = {"purpose": "CIC sensitivity attacks", "rows": ROWS, "summary": {}}


def record(kind, name, points, truth, out: Path, scored=True):
    t0 = time.perf_counter()
    res = certify(points, label=name)
    el = time.perf_counter() - t0
    contains = res.contains(truth) if truth is not None else None
    viol = bool(res.verdict != "UNDECIDED" and scored and contains is False)
    ROWS.append(
        {
            "attack": kind,
            "name": name,
            "n": int(len(points)),
            "truth": truth,
            "scored": scored,
            "verdict": res.verdict,
            "d_lo": res.d_lo,
            "d_hi": res.d_hi,
            "width": res.width,
            "contains_truth": contains,
            "calibration_violation": viol,
            "shell": res.diagnostics.shell_readout,
            "gp": res.diagnostics.gp_readout,
            "k": res.certificate.settings.k,
            "max_radius": res.certificate.settings.max_radius,
            "admissible_k": list(res.certificate.settings.admissible_k),
            "k_spread": res.diagnostics.k_spread,
            "signals": list(res.signals),
            "elapsed_s": el,
        }
    )
    PAYLOAD["summary"] = {
        "rows": len(ROWS),
        "violations": sum(r["calibration_violation"] for r in ROWS),
    }
    out.write_text(json.dumps(PAYLOAD, indent=2))
    lo = "  -  " if res.d_lo is None else f"{res.d_lo:.3f}"
    hi = "  -  " if res.d_hi is None else f"{res.d_hi:.3f}"
    print(
        f"{kind:<6}{name:<34}{res.verdict:<15}[{lo},{hi}] truth={truth} "
        f"sh={res.diagnostics.shell_readout:.3f} gp={res.diagnostics.gp_readout:.3f} "
        f"k={res.certificate.settings.k} mr={res.certificate.settings.max_radius} "
        f"{','.join(res.signals) or '-'}{'  <<< VIOLATION' if viol else ''}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="*", default=["A", "B", "C", "D"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.write_text(json.dumps(PAYLOAD, indent=2))

    if "A" in args.attacks:
        for name, gen, truth in (
            ("square", square, 2.0),
            ("cube", cube, 3.0),
            ("circle", circle, 1.0),
            ("swiss", swiss, 2.0),
        ):
            base = gen(1600)
            for p in range(6):
                pts = base if p == 0 else base[np.random.default_rng(100 + p).permutation(len(base))]
                record("A", f"{name}/perm{p}", pts, truth, out)

    if "B" in args.attacks:
        for n in (200, 400, 800, 1600, 3200, 6400):
            record("B", f"square/n={n}", square(n), 2.0, out)
            record("B", f"cube/n={n}", cube(n), 3.0, out)
            record("B", f"circle/n={n}", circle(n), 1.0, out)

    if "C" in args.attacks:
        for n in (41, 45, 50, 60, 80, 100, 150):
            record("C", f"square/n={n}", square(n), 2.0, out)
            record("C", f"circle/n={n}", circle(n), 1.0, out)

    if "D" in args.attacks:
        # Lorenz: D_0 ~ D_1 = 2.0622 ~ D_2, all within ~0.01 of each other.
        for stride in (2, 4, 8, 16, 32, 64):
            record("D", f"lorenz/stride={stride}/n=1600", lorenz(1600, stride), 2.062, out)
        for n in (800, 3200):
            record("D", f"lorenz/stride=16/n={n}", lorenz(n, 16), 2.062, out)

    print(json.dumps(PAYLOAD["summary"], indent=2))
    return 1 if PAYLOAD["summary"]["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
