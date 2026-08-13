"""Round-3 (N8d): shared measurement harness for near-constant acceptance policies.

Computes each sampled node's `DimensionEstimate` ONCE per (cloud, n) and then
scores several accept/reject policies against the same estimates, so that
"pre-N8", "N8b as shipped" and any candidate graph-level rule are compared on
byte-identical inputs rather than on separate runs.

The policies are re-implementations for measurement purposes; every headline
number is additionally cross-checked against the real production path
(`comparison.poly_algebraic_minimum_points`) in n8d_verify.py.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CV,
    NEAR_CONSTANT_SLOPE_BOUND,
    DimensionEstimate,
    local_dimension,
)
from socrates.hypergraph.pointcloud import DuplicatePointsError, knn_hypergraph  # noqa: E402

R2_THRESHOLD = 0.9  # mean_dimension's own threshold


def fit_length(est: DimensionEstimate) -> int:
    """Number of radii actually used in the log-log fit for this estimate.

    Reconstructed from the stored volume sequence exactly as `local_dimension`
    builds it: shells from radius 1 upward, stopping at the first saturated
    (non-growing) shell.
    """
    volumes = (1,) + tuple(est.volumes)
    shells = [volumes[i] - volumes[i - 1] for i in range(1, len(volumes))]
    length = 0
    for shell in shells:
        if shell <= 0:
            break
        length += 1
    return length


def near_degenerate_ignoring_length(est: DimensionEstimate) -> bool:
    """What `near_degenerate` would be with the min-fit-length gate removed."""
    if est.degenerate or fit_length(est) < 2:
        return False
    return est.shell_cv <= NEAR_CONSTANT_CV and est.slope_bound <= NEAR_CONSTANT_SLOPE_BOUND


@dataclass(frozen=True)
class GraphMeasurement:
    """All sampled estimates for one (cloud, n) plus the derived fractions."""

    n: int
    estimates: tuple[DimensionEstimate, ...]

    @property
    def n_sampled(self) -> int:
        return len(self.estimates)

    @property
    def degenerate_fraction(self) -> float:
        return sum(1 for e in self.estimates if e.degenerate) / max(1, self.n_sampled)

    @property
    def near_degenerate_fraction(self) -> float:
        return sum(1 for e in self.estimates if e.near_degenerate) / max(1, self.n_sampled)

    @property
    def near_degenerate_fraction_nolength(self) -> float:
        return sum(
            1 for e in self.estimates if near_degenerate_ignoring_length(e)
        ) / max(1, self.n_sampled)

    @property
    def low_dynamic_range_fraction(self) -> float:
        """Degenerate OR near-degenerate: "is this whole graph low-dynamic-range?"."""
        return sum(
            1 for e in self.estimates if e.degenerate or e.near_degenerate
        ) / max(1, self.n_sampled)

    @property
    def r2_pass_fraction(self) -> float:
        return sum(
            1 for e in self.estimates if e.r_squared >= R2_THRESHOLD
        ) / max(1, self.n_sampled)

    def mean(self, *, policy: str, consensus_threshold: float = 0.0) -> float:
        kept = self.kept(policy=policy, consensus_threshold=consensus_threshold)
        if not kept:
            return float("nan")
        return sum(e.dimension for e in kept) / len(kept)

    def kept(
        self, *, policy: str, consensus_threshold: float = 0.0
    ) -> list[DimensionEstimate]:
        if policy == "pre_n8":
            return [e for e in self.estimates if e.r_squared >= R2_THRESHOLD]
        if policy == "n8b":  # currently shipped: node-local, length-gated
            return [
                e for e in self.estimates if e.r_squared >= R2_THRESHOLD or e.near_degenerate
            ]
        if policy == "consensus":  # length gate AND graph-level consensus
            fires = self.near_degenerate_fraction >= consensus_threshold
            return [
                e
                for e in self.estimates
                if e.r_squared >= R2_THRESHOLD or (e.near_degenerate and fires)
            ]
        if policy == "consensus_nolength":  # graph-level consensus INSTEAD of the length gate
            frac = self.near_degenerate_fraction_nolength
            fires = frac >= consensus_threshold
            return [
                e
                for e in self.estimates
                if e.r_squared >= R2_THRESHOLD
                or (near_degenerate_ignoring_length(e) and fires)
            ]
        raise ValueError(f"unknown policy {policy!r}")

    def admitted(
        self, *, policy: str, consensus_threshold: float = 0.0
    ) -> list[DimensionEstimate]:
        """Nodes this policy accepts that R^2 alone would have rejected."""
        return [
            e
            for e in self.kept(policy=policy, consensus_threshold=consensus_threshold)
            if e.r_squared < R2_THRESHOLD
        ]


def measure_graph(
    points, *, k: int, max_radius: int, samples: int = 40
) -> GraphMeasurement | None:
    """Build the k-NN hypergraph and estimate every sampled node's dimension.

    Sampling matches `mean_dimension` exactly (sorted nodes, strided to
    `samples`), so the estimates here are the same objects the production path
    would average.
    """
    try:
        hg = knn_hypergraph(points, k=k, dedupe=True)
    except DuplicatePointsError:
        return None
    nodes = sorted(hg.nodes)
    if samples is not None and samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    estimates = tuple(local_dimension(hg, node, max_radius=max_radius) for node in nodes)
    return GraphMeasurement(n=len(points), estimates=estimates)


def measure_grid(points, spec, *, samples: int = 40) -> list[GraphMeasurement]:
    """One GraphMeasurement per n in the problem's production n_grid."""
    out = []
    for n in spec.n_grid:
        if n > len(points) or spec.k >= n:
            break
        m = measure_graph(
            points[:n], k=spec.k, max_radius=spec.max_radius, samples=min(n, samples)
        )
        if m is None:
            continue
        out.append(GraphMeasurement(n=n, estimates=m.estimates))
    return out


def min_n(
    measurements: list[GraphMeasurement],
    true_dimension: float,
    tolerance: float,
    *,
    policy: str,
    consensus_threshold: float = 0.0,
) -> int | None:
    """`poly_algebraic_minimum_points`'s stable-convergence rule, re-implemented.

    Identical logic to comparison.poly_algebraic_minimum_points: the first n
    that is within tolerance AND stays within tolerance at every larger n.
    """
    dims = [
        (m.n, m.mean(policy=policy, consensus_threshold=consensus_threshold))
        for m in measurements
    ]
    for i, (n, dim) in enumerate(dims):
        if not math.isfinite(dim):
            continue
        if abs(dim - true_dimension) > tolerance:
            continue
        if all(math.isfinite(d) and abs(d - true_dimension) <= tolerance for _, d in dims[i:]):
            return n
    return None
