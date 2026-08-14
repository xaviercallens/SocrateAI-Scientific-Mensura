"""Known-answer targets for the CIC core.

Every truth here is EXACT by construction -- no literature value, no
consensus range, no estimate produced by any method in the comparison panel
(v2 section 4's taxonomy rule). The Cantor construction and its ambient
dimension are stated explicitly, because v2 section 3.2 records that the draft
suite confused the 1-D middle-thirds set (ln2/ln3 = 0.6309) with the 2-D dust
(2*ln2/ln3 = 1.2619), which differ by a factor of two.

Generators are copied in shape from the recorded diagnostics
(`diagnostics/f4_multifractal_targets.py`, `diagnostics/e4_uniform_square_bias.py`)
so that a number produced here is comparable to a number already on the record.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_V4 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
_V3 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])


def _ifs(vertices: np.ndarray, ratio: float, n: int, seed: int) -> np.ndarray:
    """`n` i.i.d. points from the uniform (normalised Hausdorff) measure on the
    attractor of an equal-ratio iterated function system. Uniform probabilities,
    so D_0 = D_1 = D_2 exactly and there is no measure confound at all."""
    depth = int(math.ceil(60.0 / math.log2(1.0 / ratio)))
    rng = np.random.default_rng(seed)
    addr = rng.choice(len(vertices), size=(n, depth))
    weights = (1.0 - ratio) * ratio ** np.arange(depth)
    return np.einsum("ndc,d->nc", vertices[addr], weights)


def uniform_square(n: int, seed: int) -> np.ndarray:
    """i.i.d. uniform on the unit square. D_0 = D_1 = D_2 = 2 exactly."""
    return np.random.default_rng(seed).random((n, 2))


def uniform_cube(n: int, seed: int) -> np.ndarray:
    """i.i.d. uniform on the unit cube. Exactly 3."""
    return np.random.default_rng(seed).random((n, 3))


def uniform_tesseract(n: int, seed: int) -> np.ndarray:
    """i.i.d. uniform in R^4. Exactly 4. v2 section 3.3 predicts UNDECIDED."""
    return np.random.default_rng(seed).random((n, 4))


def uniform_6d(n: int, seed: int) -> np.ndarray:
    """i.i.d. uniform in R^6. Exactly 6.

    The row that found this build's one real defect: with the divergence
    threshold at its first setting, the instrument emitted MEASURED
    [2.61, 5.06] here. Kept as a standing negative control on the detector.
    """
    return np.random.default_rng(seed).random((n, 6))


def uniform_interval(n: int, seed: int) -> np.ndarray:
    """i.i.d. uniform on a segment in R^2. Exactly 1."""
    t = np.sort(np.random.default_rng(seed).random(n))
    return np.c_[t, np.zeros_like(t)]


def circle(n: int, seed: int) -> np.ndarray:
    """i.i.d. uniform ANGLE on the unit circle in R^2. Exactly 1.

    Randomly spaced, not equispaced: a smooth closed curve sampled at random
    is the honest version of this target. The equispaced variant below is kept
    separately because it is the one that makes the k-NN graph an EXACT
    circulant, which is the DEGENERATE-EXACT verdict's precondition.
    """
    t = np.sort(np.random.default_rng(seed).random(n)) * 2.0 * math.pi
    return np.c_[np.cos(t), np.sin(t)]


def circle_equispaced(n: int, seed: int) -> np.ndarray:
    """Equispaced points on the unit circle. Exactly 1, and the k-NN graph is an
    exact circulant ring lattice."""
    del seed
    t = np.arange(n) * 2.0 * math.pi / n
    return np.c_[np.cos(t), np.sin(t)]


def helix(n: int, seed: int) -> np.ndarray:
    """A 1-D curve embedded in R^3. Exactly 1, but not a closed orbit."""
    t = np.sort(np.random.default_rng(seed).random(n)) * 6.0 * math.pi
    return np.c_[np.cos(t), np.sin(t), 0.25 * t]


def swiss_roll(n: int, seed: int) -> np.ndarray:
    """A 2-D sheet embedded in R^3. Exactly 2 -- the manifold is 2-dimensional
    however it is curved, so this separates intrinsic from ambient dimension."""
    rng = np.random.default_rng(seed)
    t = 1.5 * math.pi * (1.0 + 2.0 * rng.random(n))
    h = 12.0 * rng.random(n)
    return np.c_[t * np.cos(t), h, t * np.sin(t)] / 12.0


def cantor_dust_2d(n: int, seed: int) -> np.ndarray:
    """2-D Cantor DUST C x C: 4 corner maps of the unit square, ratio 1/3.

    D_0 = D_1 = D_2 = log 4 / log 3 = 2 * ln2/ln3 = 1.2619 (NOT the 1-D
    middle-thirds set's 0.6309 -- v2 section 3.2). Totally disconnected: the
    k-NN graph fragments, and this is the suite's negative control for the
    instrument itself. The correct verdict here is UNDECIDED.
    """
    return _ifs(_V4, 1.0 / 3.0, n, seed)


def sierpinski_gasket(n: int, seed: int) -> np.ndarray:
    """Sierpinski gasket: 3 maps, ratio 1/2. D_0 = log3/log2 = 1.5850, and the
    attractor is CONNECTED -- the discriminating partner to Cantor dust."""
    return _ifs(_V3, 0.5, n, seed)


def two_clusters(n: int, seed: int) -> np.ndarray:
    """Two well-separated uniform squares. No single dimension is the truth --
    a structural negative control: the instrument must not answer."""
    rng = np.random.default_rng(seed)
    a = rng.random((n // 2, 2))
    b = rng.random((n - n // 2, 2)) + np.array([40.0, 0.0])
    return np.vstack([a, b])


def repeated_orbit(n: int, seed: int) -> np.ndarray:
    """A closed orbit traversed three times: finding F3's near-duplicate
    corruption, reproduced deliberately. Truth would be 1, but the cloud is
    corrupt and the instrument must say so."""
    del seed
    per = n // 3
    t = np.arange(per) * 2.0 * math.pi / per
    base = np.c_[np.cos(t), np.sin(t)]
    return np.vstack([base, base + 1e-9, base + 2e-9])[:n]


def undersampled_trajectory(n: int, seed: int) -> np.ndarray:
    """A Lorenz trajectory sampled too coarsely in space but finely in time.

    Round-3 (AR1) measured 100% of k-NN edges joining index-adjacent points at
    n=100 and 57% at n=400, over which range the shell estimator reported
    1.01-1.17 on a 2.05-dimensional attractor. This is the chain artifact, and
    the instrument must not report ~1 here.
    """
    del seed
    dt = 0.004
    x = np.array([1.0, 1.0, 1.0])
    out = []
    for i in range(n * 4 + 2000):
        s, r, b = 10.0, 28.0, 8.0 / 3.0
        dx = np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])
        x = x + dt * dx
        if i >= 2000 and (i - 2000) % 4 == 0:
            out.append(x.copy())
    return np.asarray(out[:n]) / 30.0


@dataclass(frozen=True)
class Target:
    name: str
    generator: Callable[[int, int], np.ndarray]
    truth: float | None
    truth_kind: str
    expect: str
    why: str


# `expect` is what the CIC core SHOULD do, declared before the run.
CORE_TARGETS: tuple[Target, ...] = (
    Target(
        "uniform-square-2d",
        uniform_square,
        2.0,
        "EXACT",
        "MEASURED",
        "i.i.d. uniform on the unit square; required known-answer case.",
    ),
    Target(
        "circle-1d",
        circle,
        1.0,
        "EXACT",
        "MEASURED",
        "smooth closed curve, randomly spaced; required known-answer case.",
    ),
    Target(
        "cantor-dust-2d",
        cantor_dust_2d,
        1.2619,
        "EXACT",
        "UNDECIDED",
        "totally disconnected support; the negative control for the instrument.",
    ),
)

EXTRA_TARGETS: tuple[Target, ...] = (
    Target(
        "circle-1d-equispaced",
        circle_equispaced,
        1.0,
        "EXACT",
        "MEASURED",
        "exact circulant ring lattice; the DEGENERATE-EXACT precondition.",
    ),
    Target(
        "interval-1d",
        uniform_interval,
        1.0,
        "EXACT",
        "MEASURED",
        "a straight segment: the simplest 1-D support there is.",
    ),
    Target(
        "helix-1d-in-3d",
        helix,
        1.0,
        "EXACT",
        "MEASURED",
        "1-D curve in R^3: intrinsic vs ambient dimension.",
    ),
    Target(
        "swiss-roll-2d-in-3d",
        swiss_roll,
        2.0,
        "EXACT",
        "MEASURED",
        "curved 2-D sheet in R^3: intrinsic vs ambient dimension.",
    ),
    Target(
        "sierpinski-gasket-2d",
        sierpinski_gasket,
        math.log(3) / math.log(2),
        "EXACT",
        "MEASURED",
        "connected fractal, D_0 = D_1 = D_2 = 1.5850; Cantor dust's partner.",
    ),
    Target(
        "uniform-cube-3d",
        uniform_cube,
        3.0,
        "EXACT",
        "MEASURED",
        "Round-3 measured the shell estimator reaching only ~2.47 here.",
    ),
    Target(
        "uniform-4d",
        uniform_tesseract,
        4.0,
        "EXACT",
        "UNDECIDED",
        "v2 section 3.3 predicts UNDECIDED at attainable n, and says so is correct.",
    ),
    Target(
        "uniform-6d",
        uniform_6d,
        6.0,
        "EXACT",
        "UNDECIDED",
        "found this build's one calibration violation; standing detector control.",
    ),
    Target(
        "two-clusters",
        two_clusters,
        None,
        "NONE",
        "UNDECIDED",
        "no single dimension is the truth; structural negative control.",
    ),
    Target(
        "repeated-orbit-f3",
        repeated_orbit,
        None,
        "CORRUPT",
        "UNDECIDED",
        "finding F3 near-duplicate corruption, reproduced deliberately.",
    ),
    Target(
        "undersampled-trajectory",
        undersampled_trajectory,
        2.06,
        "INDEPENDENT-D1",
        "UNDECIDED",
        "the AR1 chain artifact: shell reads ~1 on a 2.06-dimensional attractor.",
    ),
)

ALL_TARGETS: tuple[Target, ...] = CORE_TARGETS + EXTRA_TARGETS
