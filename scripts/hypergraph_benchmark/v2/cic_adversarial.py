"""ADVERSARIAL probe of the CIC core, written by an independent reviewer.

The single question: is there ANY point cloud with an exactly-known dimension on
which `certify()` returns MEASURED (or DEGENERATE-EXACT) with an interval that
does not contain the truth? One such cloud is a refutation.

Every truth here is EXACT by construction (a manifold's dimension, or an IFS
attractor's similarity dimension with equal ratios and equal probabilities, so
D_0 = D_1 = D_2 and there is no measure confound). Targets whose truth is a
literature value or whose D_0 != D_2 are marked `truth_kind != EXACT` and are
recorded but NOT counted as violations.

E6: written to disk after every row.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.cic import CIC_CONTRACT_VERSION, Verdict, certify  # noqa: E402

OUT = Path(__file__).resolve().parent / "cic_adversarial.json"


# ---------------------------------------------------------------- IFS helper
def _ifs(vertices: np.ndarray, ratio: float, n: int, seed: int, probs=None) -> np.ndarray:
    depth = int(math.ceil(60.0 / math.log2(1.0 / ratio)))
    rng = np.random.default_rng(seed)
    addr = rng.choice(len(vertices), size=(n, depth), p=probs)
    weights = (1.0 - ratio) * ratio ** np.arange(depth)
    return np.einsum("ndc,d->nc", vertices[addr], weights)


def _grid(m: int, d: int) -> np.ndarray:
    """Corner set of a d-dim m^d subdivision, as an (m^d, d) array of ints."""
    return np.stack(np.meshgrid(*[np.arange(m)] * d, indexing="ij"), axis=-1).reshape(-1, d)


_V3 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])

# Sierpinski carpet: the 9-cell 3x3 grid minus its centre -> 8 maps, ratio 1/3.
_CARPET = np.array([c for c in _grid(3, 2) if not (c[0] == 1 and c[1] == 1)], dtype=float)
# Vicsek: the plus-shape, 5 maps, ratio 1/3.
_VICSEK = np.array([c for c in _grid(3, 2) if (c[0] == 1) or (c[1] == 1)], dtype=float)
# Menger sponge: 3x3x3 minus 6 face centres and the body centre -> 20 maps.
_MENGER = np.array(
    [c for c in _grid(3, 3) if sum(1 for v in c if v == 1) <= 1],
    dtype=float,
)


# ---------------------------------------------------------------- generators
def torus_2d(n, seed):
    """2-torus surface in R^3 (R=1, r=0.4), sampled by rejection so the surface
    measure is uniform. Intrinsically 2-dimensional, and has NO BOUNDARY."""
    rng = np.random.default_rng(seed)
    R, r = 1.0, 0.4
    out = []
    while len(out) < n:
        u = rng.random(4 * n) * 2 * math.pi
        v = rng.random(4 * n) * 2 * math.pi
        keep = rng.random(4 * n) < (R + r * np.cos(v)) / (R + r)
        u, v = u[keep], v[keep]
        pts = np.c_[(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)]
        out.extend(pts.tolist())
    return np.asarray(out[:n])


def flat_torus(d):
    """Flat d-torus in R^(2d) via (cos t_i, sin t_i). Exactly d, no boundary."""

    def gen(n, seed):
        rng = np.random.default_rng(seed)
        t = rng.random((n, d)) * 2 * math.pi
        return np.concatenate([np.cos(t), np.sin(t)], axis=1) / math.sqrt(d)

    return gen


def sphere(d):
    """Uniform on the unit d-sphere in R^(d+1). Exactly d, no boundary."""

    def gen(n, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, d + 1))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return gen


def gaussian(d):
    """i.i.d. standard normal in R^d. Exactly d (full support), non-uniform."""

    def gen(n, seed):
        return np.random.default_rng(seed).normal(size=(n, d))

    return gen


def uniform_ball(d):
    def gen(n, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x * rng.random((n, 1)) ** (1.0 / d)

    return gen


def thin_slab(eps, d=2):
    """[0,1] x [0,eps]^(d-1). Exactly d, but looks (d-1)-dimensional past
    radius ~eps. The estimator is asked to notice its own scale limit."""

    def gen(n, seed):
        x = np.random.default_rng(seed).random((n, d))
        x[:, 1:] *= eps
        return x

    return gen


def anisotropic_cube(eps):
    """[0,1] x [0,eps] x [0,eps] -- exactly 3, but a fibre at coarse scale."""

    def gen(n, seed):
        x = np.random.default_rng(seed).random((n, 3))
        x[:, 1:] *= eps
        return x

    return gen


def annulus(n, seed):
    rng = np.random.default_rng(seed)
    t = rng.random(n) * 2 * math.pi
    r = np.sqrt(0.25 + 0.75 * rng.random(n))
    return np.c_[r * np.cos(t), r * np.sin(t)]


def nonuniform_disc(n, seed):
    """Disc with density ~ 1/r (heavily concentrated at the centre). Still
    exactly 2-dimensional; the SAMPLING is what is adversarial."""
    rng = np.random.default_rng(seed)
    t = rng.random(n) * 2 * math.pi
    r = rng.random(n) ** 2
    return np.c_[r * np.cos(t), r * np.sin(t)]


def clustered_circle(n, seed):
    """Circle with density ~ exp(3 cos t): exactly 1-dimensional, wildly
    non-uniform along the curve."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        t = rng.random(4 * n) * 2 * math.pi
        keep = rng.random(4 * n) < np.exp(3 * np.cos(t) - 3)
        out.extend(t[keep].tolist())
    t = np.sort(np.asarray(out[:n]))
    return np.c_[np.cos(t), np.sin(t)]


def square_in_r10(n, seed):
    """A 2-D square isometrically embedded in R^10 by a random rotation.
    Exactly 2; tests whether ambient dimension leaks in."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    return rng.random((n, 2)) @ q[:2, :]


def square_with_outliers(frac):
    def gen(n, seed):
        rng = np.random.default_rng(seed)
        m = int(n * frac)
        core = rng.random((n - m, 2))
        out = rng.random((m, 2)) * 6.0 - 2.5
        return np.vstack([core, out])

    return gen


def noisy_circle(sigma):
    """Circle + isotropic Gaussian noise. NOT exact: dimension is 1 above the
    noise scale and 2 below it, so this is recorded, never scored."""

    def gen(n, seed):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.random(n)) * 2 * math.pi
        return np.c_[np.cos(t), np.sin(t)] + rng.normal(scale=sigma, size=(n, 2))

    return gen


def sierpinski_carpet(n, seed):
    return _ifs(_CARPET, 1.0 / 3.0, n, seed)


def vicsek(n, seed):
    return _ifs(_VICSEK, 1.0 / 3.0, n, seed)


def menger(n, seed):
    return _ifs(_MENGER, 1.0 / 3.0, n, seed)


def koch(n, seed):
    """Koch curve, D = log4/log3 = 1.2619. CONNECTED, unlike Cantor dust --
    the same truth value on a support the graph can actually traverse."""
    rng = np.random.default_rng(seed)
    depth = 12
    addr = rng.integers(0, 4, size=(n, depth))
    ang = np.array([0.0, math.pi / 3, -math.pi / 3, 0.0])
    off = np.array([[0.0, 0.0], [1 / 3, 0.0], [0.5, math.sqrt(3) / 6], [2 / 3, 0.0]])
    pts = np.zeros((n, 2))
    for d in range(depth - 1, -1, -1):
        a = ang[addr[:, d]]
        c, s = np.cos(a), np.sin(a)
        rot = np.c_[c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]] / 3.0
        pts = off[addr[:, d]] + rot
    return pts


def sierpinski_biased(n, seed):
    """Sierpinski gasket sampled with probabilities (0.8, 0.1, 0.1).

    The SUPPORT is unchanged, so D_0 = log3/log2 = 1.5850 still. But the
    measure is multifractal: D_1 = sum p log p / log 2 = 1.0664 and
    D_2 = -log(sum p^2)/log 2 = 0.6842. A finite i.i.d. sample sees the
    measure, so D_0 is arguably NOT the fair truth at finite n -- recorded
    with truth_kind MULTIFRACTAL and never counted as a violation.
    """
    return _ifs(_V3, 0.5, n, seed, probs=[0.8, 0.1, 0.1])


def cantor_1d(n, seed):
    """The 1-D middle-thirds Cantor set in R^1. D_0 = ln2/ln3 = 0.6309."""
    return _ifs(np.array([[0.0], [1.0]]), 1.0 / 3.0, n, seed)


def lorenz(n, seed):
    """Well-sampled Lorenz attractor. D_KY = 2.0622 is D_1, NOT D_0 -- v2
    section 4. Recorded, never scored."""
    del seed
    dt = 0.002
    x = np.array([1.0, 1.0, 1.0])
    out = []
    stride = max(1, int(60 / dt / n))
    for i in range(n * stride + 20000):
        s, r, b = 10.0, 28.0, 8.0 / 3.0
        dx = np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])
        x = x + dt * dx
        if i >= 20000 and (i - 20000) % stride == 0:
            out.append(x.copy())
    return np.asarray(out[:n]) / 30.0


def uniform_d(d):
    def gen(n, seed):
        return np.random.default_rng(seed).random((n, d))

    return gen


def circle_equispaced_noisy(jitter):
    """An 'almost' ring lattice: equispaced circle with tiny jitter. Attacks
    DEGENERATE-EXACT's zero-width interval."""

    def gen(n, seed):
        rng = np.random.default_rng(seed)
        t = np.arange(n) * 2 * math.pi / n + rng.normal(scale=jitter, size=n)
        return np.c_[np.cos(t), np.sin(t)]

    return gen


def grid_2d(n, seed):
    """An exact square lattice: n ~ m^2 points on a regular grid. Exactly 2,
    and the k-NN graph is a perfect circulant in TWO directions -- the
    structural sibling of the ring lattice that DEGENERATE-EXACT trusts."""
    del seed
    m = int(round(math.sqrt(n)))
    g = np.arange(m) / m
    return np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)


def grid_3d(n, seed):
    del seed
    m = int(round(n ** (1 / 3)))
    g = np.arange(m) / m
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)


def two_scales(n, seed):
    """A 2-D square with a 1-D whisker attached: connected, but no single
    dimension. Structural control -- truth None."""
    rng = np.random.default_rng(seed)
    a = rng.random((n // 2, 2))
    t = np.sort(rng.random(n - n // 2))
    b = np.c_[1.0 + 3.0 * t, 0.5 * np.ones_like(t)]
    return np.vstack([a, b])


@dataclass(frozen=True)
class T:
    name: str
    gen: Callable[[int, int], np.ndarray]
    truth: float | None
    kind: str
    why: str


LN = math.log
TARGETS: tuple[T, ...] = (
    # --- boundary-free manifolds: the boundary-cancellation of v2 s6 is gone
    T("torus-2d-in-3d", torus_2d, 2.0, "EXACT", "closed 2-manifold, no boundary"),
    T("sphere-2d-in-3d", sphere(2), 2.0, "EXACT", "closed 2-manifold, no boundary"),
    T("sphere-3d-in-4d", sphere(3), 3.0, "EXACT", "closed 3-manifold, no boundary"),
    T("flat-torus-3d-in-6d", flat_torus(3), 3.0, "EXACT", "no boundary at D=3"),
    T("flat-torus-4d-in-8d", flat_torus(4), 4.0, "EXACT", "the v2 s3.3 row, boundary-free"),
    T("sphere-4d-in-5d", sphere(4), 4.0, "EXACT", "closed 4-manifold"),
    # --- filled bodies
    T("uniform-cube-3d", uniform_d(3), 3.0, "EXACT", "the closest MEASURED row to a miss"),
    T("uniform-ball-3d", uniform_ball(3), 3.0, "EXACT", "3-ball, curved boundary"),
    T("uniform-4d", uniform_d(4), 4.0, "EXACT", "declared UNDECIDED by the build"),
    T("gaussian-2d", gaussian(2), 2.0, "EXACT", "full support, unbounded, non-uniform"),
    T("gaussian-3d", gaussian(3), 3.0, "EXACT", "full support, unbounded, non-uniform"),
    # --- exactly 2 or 1 but adversarially shaped / sampled
    T("thin-slab-2d-eps0.02", thin_slab(0.02), 2.0, "EXACT", "2-D, fibre-like past r~eps"),
    T("thin-slab-2d-eps0.005", thin_slab(0.005), 2.0, "EXACT", "2-D, fibre-like sooner"),
    T("aniso-cube-3d-eps0.05", anisotropic_cube(0.05), 3.0, "EXACT", "3-D, fibre-like"),
    T("annulus-2d", annulus, 2.0, "EXACT", "2-D with a hole"),
    T("nonuniform-disc-2d", nonuniform_disc, 2.0, "EXACT", "2-D, density ~ 1/r"),
    T("clustered-circle-1d", clustered_circle, 1.0, "EXACT", "1-D, density ~ exp(3cos t)"),
    T("square-in-r10", square_in_r10, 2.0, "EXACT", "2-D in ambient R^10"),
    T("square-outliers-5pct", square_with_outliers(0.05), 2.0, "EXACT", "2-D + 5% spread"),
    T("grid-2d", grid_2d, 2.0, "EXACT", "exact square lattice; DEGENERATE-EXACT sibling"),
    T("grid-3d", grid_3d, 3.0, "EXACT", "exact cubic lattice"),
    T("jittered-ring-1e-3", circle_equispaced_noisy(1e-3), 1.0, "EXACT", "near-ring lattice"),
    T("jittered-ring-1e-2", circle_equispaced_noisy(1e-2), 1.0, "EXACT", "near-ring lattice"),
    # --- CONNECTED fractals: no fragmentation escape hatch
    T("sierpinski-carpet", sierpinski_carpet, LN(8) / LN(3), "EXACT", "connected, D=1.8928"),
    T("vicsek-fractal", vicsek, LN(5) / LN(3), "EXACT", "connected, D=1.4650"),
    T("menger-sponge-3d", menger, LN(20) / LN(3), "EXACT", "connected, D=2.7268"),
    T("koch-curve", koch, LN(4) / LN(3), "EXACT", "connected curve, D=1.2619"),
    T("cantor-1d-in-r1", cantor_1d, LN(2) / LN(3), "EXACT", "D=0.6309, disconnected"),
    # --- structural / mis-typed controls: recorded, NEVER scored
    T("sierpinski-biased-p8", sierpinski_biased, None, "MULTIFRACTAL", "D_0 1.585 vs D_2 0.684"),
    T("noisy-circle-s0.02", noisy_circle(0.02), None, "SCALE-DEPENDENT", "1-D or 2-D by scale"),
    T("noisy-circle-s0.05", noisy_circle(0.05), None, "SCALE-DEPENDENT", "1-D or 2-D by scale"),
    T("lorenz-well-sampled", lorenz, None, "INDEPENDENT-D1", "D_KY=2.0622 is D_1, not D_0"),
    T("square-plus-whisker", two_scales, None, "NONE", "no single dimension"),
)

BY_NAME = {t.name: t for t in TARGETS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[1600])
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 7])
    ap.add_argument("--only", type=str, nargs="*", default=None)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    targets = [BY_NAME[x] for x in args.only] if args.only else list(TARGETS)
    payload: dict = {
        "contract_version": CIC_CONTRACT_VERSION,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "independent adversarial hunt for a CIC calibration violation",
        "sizes": args.sizes,
        "seeds": args.seeds,
        "rows": [],
        "summary": {},
    }
    out.write_text(json.dumps(payload, indent=2))

    violations = measured = undecided = 0
    hdr = f"{'target':<26}{'n':>6}{'seed':>8} {'verdict':<16}{'lo':>8}{'hi':>8}{'w':>7}{'truth':>8}"
    print(hdr)
    print("-" * len(hdr))
    for t in targets:
        for n in args.sizes:
            for seed in args.seeds:
                pts = t.gen(n, seed)
                t0 = time.perf_counter()
                res = certify(pts, label=f"{t.name}/n={n}/seed={seed}")
                el = time.perf_counter() - t0
                scored = t.kind == "EXACT" and t.truth is not None
                contains = res.contains(t.truth) if t.truth is not None else None
                viol = bool(res.verdict != Verdict.UNDECIDED and scored and contains is False)
                if res.verdict == Verdict.UNDECIDED:
                    undecided += 1
                else:
                    measured += 1
                violations += int(viol)
                payload["rows"].append(
                    {
                        "target": t.name,
                        "n": n,
                        "seed": seed,
                        "truth": t.truth,
                        "truth_kind": t.kind,
                        "why": t.why,
                        "scored": scored,
                        "contains_truth": contains,
                        "calibration_violation": viol,
                        "elapsed_s": el,
                        "result": res.to_dict(),
                    }
                )
                payload["summary"] = {
                    "rows": len(payload["rows"]),
                    "measured": measured,
                    "undecided": undecided,
                    "calibration_violations": violations,
                    "valid": violations == 0,
                }
                out.write_text(json.dumps(payload, indent=2))
                lo = "  --  " if res.d_lo is None else f"{res.d_lo:>8.4f}"
                hi = "  --  " if res.d_hi is None else f"{res.d_hi:>8.4f}"
                w = "  --  " if res.width is None else f"{res.width:>7.3f}"
                tr = "  --  " if t.truth is None else f"{t.truth:>8.4f}"
                sh = res.diagnostics.shell_readout
                gp = res.diagnostics.gp_readout
                print(
                    f"{t.name:<26}{n:>6}{seed:>8} {res.verdict:<16}{lo}{hi}{w}{tr}"
                    f"  sh={sh:.3f} gp={gp:.3f} k={res.certificate.settings.k}"
                    f" mr={res.certificate.settings.max_radius}"
                    f"  {','.join(res.signals) or '-'}"
                    f"{'   <<< CALIBRATION VIOLATION' if viol else ''}"
                )
    print(json.dumps(payload["summary"], indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
