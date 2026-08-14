"""Exactly-known multifractal targets for the F4 dimension-type identification.

WHY THIS EXISTS
---------------
Reviewer finding F4 (dimension-type mismatch) claims the benchmark commits a
category error: Grassberger-Procaccia estimates the CORRELATION dimension D_2
by construction, while the shell-growth estimator counts NODES inside a graph
ball, which is a counting/volume quantity plausibly closer to the box dimension
D_0 (or to something else entirely). On Lorenz/Rossler the two are numerically
indistinguishable, so the benchmark could never have told them apart.

This module builds targets where they provably differ, by a lot.

THE CONSTRUCTION AND ITS CLOSED FORMS
-------------------------------------
Every target is a self-similar iterated function system (IFS) in the plane with
N maps of EQUAL contraction ratio r,

    f_i(x) = r * x + (1 - r) * v_i ,      i = 1..N,

whose fixed points v_i are chosen so the open set condition (OSC) holds. The
invariant measure mu is the self-similar measure with probability vector p:
mu = sum_i p_i * (mu o f_i^{-1}).

For an equal-ratio IFS satisfying the OSC the Renyi spectrum has closed form
(Cawley & Mauldin 1992; Falconer, "Techniques in Fractal Geometry", ch. 17).
Writing the partition function over level-m cylinders (each of diameter r^m and
measure a product of p's):

    sum_cylinders mu_j^q = (sum_i p_i^q)^m ,   eps = r^m ,
    sum_cylinders mu_j^q ~ eps^{tau(q)}   =>   tau(q) = log(sum_i p_i^q) / log r
    D_q = tau(q) / (q - 1)

which gives, with log(1/r) > 0:

    D_0 = log N            / log(1/r)          (box dimension of the SUPPORT)
    D_1 = -sum_i p_i log p_i / log(1/r)        (information dimension)
    D_2 = log(1 / sum_i p_i^2) / log(1/r)      (correlation dimension)

D_0 depends only on (N, r) -- the support. D_1 and D_2 depend only on p -- the
measure. That separation is the whole experiment: it lets the same support
carry measures with wildly different D_2, and lets different supports carry
measures with IDENTICAL D_2.

Three supports are used, giving three distinct D_0:

    SQ  4 maps, r = 1/2, corners of the unit square.  Attractor = the FULL
        UNIT SQUARE (the four half-squares tile it), so D_0 = 2 exactly and
        the support is an ordinary 2-dimensional set with interior. A skewed p
        turns it into a binomial/multiplicative cascade on the square: a
        measure whose support is trivially 2-dimensional but whose D_2 can be
        pushed below 1.
    GA  3 maps, r = 1/2, vertices of an equilateral triangle: the Sierpinski
        gasket, D_0 = log 3 / log 2 = 1.5850.
    CD  4 maps, r = 1/3, corners of the unit square: the Cantor dust
        C x C (C = middle-thirds Cantor set), D_0 = log 4 / log 3 = 1.2619.

SAMPLING
--------
`sample` draws points i.i.d. FROM THE INVARIANT MEASURE, not uniformly from the
support: each point gets an independent random address (i_1..i_L) with the
i_j drawn i.i.d. from p, and is placed at

    x = (1 - r) * sum_{j=1..L} r^{j-1} v_{i_j}   (+ r^L * x_0, dropped: r^L
                                                  is below machine epsilon)

This is exact i.i.d. sampling of mu, NOT a chaos-game orbit, so there is no
temporal correlation at all and no Theiler window is needed or meaningful --
which removes an entire confound the benchmark had to fight on Lorenz/Rossler.

Setting p uniform makes mu the normalised Hausdorff measure on the support,
i.e. "uniform on the support" -- the D_0-appropriate sampling. Every target
therefore has a matched uniform-p sibling on the SAME support, which is how the
"is the shell estimator sensitive to the sampling measure at all?" question is
asked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- supports -----------------------------------------------------------------

_SQ_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
_GA_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
_CD_VERTICES = _SQ_VERTICES  # same corners, ratio 1/3 -> Cantor dust
# Sierpinski tetrahedron: 4 maps, r=1/2, vertices of a regular tetrahedron in
# R^3. D_0 = log 4 / log 2 = 2.0000 EXACTLY, in a 3-dimensional embedding --
# the closest exactly-known analogue of the benchmark's Lorenz/Rossler
# geometry (3D ambient, dimension ~2), but with a D_2 that can be tuned.
_ST_VERTICES = 0.5 * np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
)

SUPPORTS: dict[str, tuple[np.ndarray, float, str]] = {
    "SQ": (_SQ_VERTICES, 0.5, "filled unit square (4 maps, r=1/2)"),
    "GA": (_GA_VERTICES, 0.5, "Sierpinski gasket (3 maps, r=1/2)"),
    "CD": (_CD_VERTICES, 1.0 / 3.0, "Cantor dust CxC (4 maps, r=1/3)"),
    "ST": (_ST_VERTICES, 0.5, "Sierpinski tetrahedron in R^3 (4 maps, r=1/2)"),
}


@dataclass(frozen=True)
class Target:
    """One exactly-known multifractal measure."""

    name: str
    support: str
    probabilities: tuple[float, ...]

    @property
    def vertices(self) -> np.ndarray:
        return SUPPORTS[self.support][0]

    @property
    def ratio(self) -> float:
        return SUPPORTS[self.support][1]

    @property
    def n_maps(self) -> int:
        return len(self.vertices)

    # --- closed forms (derivation in the module docstring) --------------------

    @property
    def D0(self) -> float:
        return math.log(self.n_maps) / math.log(1.0 / self.ratio)

    @property
    def D1(self) -> float:
        p = self.probabilities
        return -sum(pi * math.log(pi) for pi in p if pi > 0) / math.log(1.0 / self.ratio)

    @property
    def D2(self) -> float:
        p = self.probabilities
        return math.log(1.0 / sum(pi * pi for pi in p)) / math.log(1.0 / self.ratio)

    @property
    def spread(self) -> float:
        return self.D0 - self.D2

    def sample(self, n: int, seed: int, *, depth: int | None = None) -> np.ndarray:
        """`n` i.i.d. points from the invariant measure (see module docstring)."""
        v = self.vertices
        r = self.ratio
        if depth is None:
            # r^depth below machine epsilon, so the dropped seed-point term and
            # the truncated address tail are both numerically invisible.
            depth = int(math.ceil(60.0 / math.log2(1.0 / r)))
        rng = np.random.default_rng(seed)
        addr = rng.choice(len(v), size=(n, depth), p=np.asarray(self.probabilities))
        weights = (1.0 - r) * r ** np.arange(depth)  # (depth,)
        chosen = v[addr]  # (n, depth, 2)
        return np.einsum("ndc,d->nc", chosen, weights)

    def uniform_sibling(self) -> Target:
        """Same support, uniform probabilities: the normalised Hausdorff measure.

        For an equal-ratio OSC system this is exactly "uniform on the support",
        and it has D_0 = D_1 = D_2 = D_0 -- the monofractal control.
        """
        n = self.n_maps
        return Target(f"{self.support}-unif", self.support, tuple([1.0 / n] * n))


def _probs_for_target_D2(support: str, target_D2: float) -> tuple[float, ...]:
    """Probabilities (a, b, b, ..) on `support` realising an exact D_2.

    Solves sum p_i^2 = (1/r)^{-D_2} for the one-parameter family
    p = (1 - (N-1)b, b, .., b), which is monotone in b on [0, 1/N] -- so the
    root is unique there and a plain bisection is exact to machine precision.
    """
    vertices, r, _ = SUPPORTS[support]
    n = len(vertices)
    want = math.exp(-target_D2 * math.log(1.0 / r))  # = sum p_i^2

    def sum_sq(b: float) -> float:
        a = 1.0 - (n - 1) * b
        return a * a + (n - 1) * b * b

    lo, hi = 0.0, 1.0 / n  # sum_sq(0)=1 (max), sum_sq(1/n)=1/n (min)
    if not (sum_sq(hi) <= want <= sum_sq(lo)):
        raise ValueError(f"D_2={target_D2} unreachable on support {support}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sum_sq(mid) > want:
            lo = mid
        else:
            hi = mid
    b = 0.5 * (lo + hi)
    return (1.0 - (n - 1) * b,) + (b,) * (n - 1)


# --- the target set -----------------------------------------------------------
#
# ARM 1 (same support, different measure): D_0 held fixed, D_2 swept. If the
#       shell estimator tracks D_0 it must return the SAME value along a row.
# ARM 2 (different support, same measure-dimension): D_2 held fixed at 1.18442
#       across all three supports. If the shell estimator tracks D_2 it must
#       return the same value across the three "-matched" targets; if it tracks
#       D_0 it must order them 2.00 > 1.585 > 1.262.

_MATCHED_D2 = Target("GA-mild", "GA", (0.6, 0.2, 0.2)).D2  # 1.18442

TARGETS: tuple[Target, ...] = (
    # --- monofractal controls: D_0 = D_1 = D_2, both estimators must agree ---
    Target("SQ-unif", "SQ", (0.25, 0.25, 0.25, 0.25)),
    Target("GA-unif", "GA", (1 / 3, 1 / 3, 1 / 3)),
    Target("CD-unif", "CD", (0.25, 0.25, 0.25, 0.25)),
    # --- ARM 1: fixed support, increasing skew ---
    Target("SQ-mild", "SQ", (0.4, 0.2, 0.2, 0.2)),
    Target("SQ-strong", "SQ", (0.7, 0.1, 0.1, 0.1)),
    Target("GA-mild", "GA", (0.6, 0.2, 0.2)),
    Target("GA-strong", "GA", (0.8, 0.1, 0.1)),
    Target("CD-strong", "CD", (0.7, 0.1, 0.1, 0.1)),
    # --- ARM 2: three supports, identical D_2 ---
    Target("SQ-matched", "SQ", _probs_for_target_D2("SQ", _MATCHED_D2)),
    Target("CD-matched", "CD", _probs_for_target_D2("CD", _MATCHED_D2)),
)

# ARM 3, added after the first sweep: the 3D arm. Same question as ARM 1, but in
# the ambient dimension and at the D_0 the benchmark's own chaotic problems live
# at (Lorenz D_0 ~ 2.06, Rossler ~ 2.02), so the identification is made in the
# geometry the 08/09 verdicts are actually about rather than only in the plane.
TARGETS_3D: tuple[Target, ...] = (
    Target("ST-unif", "ST", (0.25, 0.25, 0.25, 0.25)),
    Target("ST-mild", "ST", (0.4, 0.2, 0.2, 0.2)),
    Target("ST-matched", "ST", _probs_for_target_D2("ST", _MATCHED_D2)),
    Target("ST-strong", "ST", (0.7, 0.1, 0.1, 0.1)),
)

ALL_TARGETS: tuple[Target, ...] = TARGETS + TARGETS_3D

BY_NAME = {t.name: t for t in ALL_TARGETS}


def box_count_dimensions(points: np.ndarray, *, ratio: float, levels: range) -> dict:
    """D_0, D_1, D_2 measured DIRECTLY off the point cloud by box counting.

    Independent of the closed forms above and of both estimators under test:
    grid the plane at eps = ratio^m, form the empirical box measures
    mu_j = (points in box j) / n, and read off

        D_0 : slope of  log #{occupied boxes}       vs log(1/eps)
        D_1 : slope of  -sum mu_j log mu_j          vs log(1/eps)
        D_2 : slope of  -log sum mu_j^2             vs log(1/eps)

    This is the known-answer validation for the SAMPLER and the ALGEBRA: if the
    cloud is drawn correctly and the closed forms are right, these three slopes
    must reproduce them. (D_0 is the one that saturates first at finite n --
    once a box holds ~1 point, counting occupied boxes counts the sample.)
    """
    n = len(points)
    lo = points.min(axis=0)
    span = float((points.max(axis=0) - lo).max())
    logs, n_occ, entropy, corr = [], [], [], []
    for m in levels:
        eps = ratio**m
        idx = np.floor((points - lo) / (span * eps)).astype(np.int64)
        _, counts = np.unique(idx, axis=0, return_counts=True)
        mu = counts / n
        logs.append(math.log(1.0 / eps))
        n_occ.append(math.log(len(counts)))
        entropy.append(float(-np.sum(mu * np.log(mu))))
        corr.append(-math.log(float(np.sum(mu * mu))))
    x = np.asarray(logs)
    fit = lambda y: float(np.polyfit(x, np.asarray(y), 1)[0])  # noqa: E731
    return {
        "D0_boxcount": fit(n_occ),
        "D1_boxcount": fit(entropy),
        "D2_boxcount": fit(corr),
        "levels": list(levels),
        "occupied": [int(round(math.exp(v))) for v in n_occ],
    }
