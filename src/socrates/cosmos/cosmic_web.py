"""Topology of the cosmic web.

Galaxies trace a network of clusters, filaments, sheets and voids. Persistent
homology gives that network a quantitative signature, and the correspondence
is direct:

    H_0  connected components  -> clusters and groups
    H_1  independent loops     -> filament circuits enclosing under-dense regions
    H_2  enclosed cavities     -> voids

This is the standard interpretation in the cosmological TDA literature (see
van de Weygaert et al. 2011; Sousbie 2011; Pranav et al. 2017).

Redshift is converted to comoving distance under a flat LambdaCDM cosmology.
Note that redshift-space distortions (peculiar velocities) elongate structures
along the line of sight -- the "Fingers of God" effect -- so H_1/H_2 features
are systematically distorted relative to true real-space topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tda.persistence import PersistenceDiagram, alpha_complex, persistence_entropy
from .fetch import GalaxySample


@dataclass(frozen=True)
class CosmicWebTopology:
    """Persistent homology of a galaxy point cloud, with cosmological labels."""

    diagram: PersistenceDiagram
    points: np.ndarray
    scale_mpc: float

    @property
    def n_galaxies(self) -> int:
        return int(self.points.shape[0])

    def structure_counts(self, threshold_mpc: float) -> dict[str, int]:
        """Counts of clusters / loops / voids resolved at a given smoothing scale."""
        betti = self.diagram.betti_numbers(threshold_mpc)
        labels = ("clusters", "filament_loops", "voids")
        return dict(zip(labels, betti + [0] * (3 - len(betti)), strict=False))

    def dominant_void_scale(self) -> float | None:
        """Filtration scale (Mpc) of the longest-lived H_2 class, if any.

        Physically: the characteristic radius at which the largest void in the
        sample is best resolved as an enclosed cavity.
        """
        if self.diagram.max_dimension < 2:
            return None
        top = self.diagram.most_persistent(2, 1)
        if top.size == 0:
            return None
        birth, death = top[0]
        return float(birth if not np.isfinite(death) else 0.5 * (birth + death))

    def summary(self) -> dict[str, object]:
        return {
            "n_galaxies": self.n_galaxies,
            "extent_mpc": self.scale_mpc,
            "structure_counts": self.structure_counts(0.05 * self.scale_mpc),
            "void_scale_mpc": self.dominant_void_scale(),
            "H1_entropy": persistence_entropy(self.diagram, 1),
            "total_persistence": {
                f"H{k}": self.diagram.total_persistence(k)
                for k in range(self.diagram.max_dimension + 1)
            },
        }


def to_comoving_cartesian(
    sample: GalaxySample,
    *,
    hubble_constant: float = 70.0,
    omega_matter: float = 0.3,
) -> np.ndarray:
    """Convert (ra, dec, z) to comoving Cartesian coordinates in Mpc.

    Uses a flat LambdaCDM cosmology; Omega_Lambda is fixed by flatness as
    1 - Omega_m. Returns an (n, 3) array suitable for persistent homology.
    """
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=hubble_constant, Om0=omega_matter)
    distance = cosmology.comoving_distance(sample.redshift).value  # Mpc

    coords = SkyCoord(ra=sample.ra, dec=sample.dec, unit="deg")
    unit_vectors = np.column_stack(
        [
            np.cos(coords.dec.radian) * np.cos(coords.ra.radian),
            np.cos(coords.dec.radian) * np.sin(coords.ra.radian),
            np.sin(coords.dec.radian),
        ]
    )
    return unit_vectors * distance[:, None]


def analyze_cosmic_web(
    points: np.ndarray,
    *,
    max_dimension: int = 2,
    max_points: int = 20_000,
    seed: int | None = None,
) -> CosmicWebTopology:
    """Compute the persistent homology of a comoving galaxy point cloud.

    Above `max_points` the cloud is randomly subsampled. Random subsampling is
    the right choice here: it is unbiased with respect to large-scale structure,
    whereas spatial cropping would truncate the very filaments being measured.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected an (n, 3) array of positions, got {pts.shape}")

    if pts.shape[0] > max_points:
        idx = np.random.default_rng(seed).choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    extent = float(np.ptp(pts, axis=0).max()) if pts.shape[0] else 0.0
    diagram = alpha_complex(pts, max_dimension=max_dimension)

    return CosmicWebTopology(diagram=diagram, points=pts, scale_mpc=extent)


def random_catalog(reference: np.ndarray, *, seed: int | None = None) -> np.ndarray:
    """A Poisson (structureless) cloud matching `reference` in count and extent.

    This is the null hypothesis for cosmic-web analysis. Real large-scale
    structure is only demonstrated by contrast: topology of the observed
    catalog must differ significantly from this random baseline.
    """
    rng = np.random.default_rng(seed)
    lo = reference.min(axis=0)
    hi = reference.max(axis=0)
    return rng.uniform(lo, hi, size=reference.shape)
