"""Retrieval and caching of SDSS spectroscopic galaxy samples.

Galaxy positions arrive as (ra, dec, redshift); converting them into a
comoving 3-D point cloud is what makes the cosmic web amenable to persistent
homology. Downloads are cached on disk because the SDSS SkyServer applies
rate limits and query rows are capped per request.

Data source
-----------
SDSS DR17 SpecObj table via `astroquery.sdss`. SDSS data is public;
please cite the survey (see docs/REFERENCES.md) in any derived work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "sdss"

# SDSS class/quality cuts: spectroscopic galaxies with reliable redshifts.
# zWarning = 0 is the survey's own flag for an unproblematic redshift fit.
_GALAXY_QUERY = """
SELECT TOP {limit}
    s.ra, s.dec, s.z, s.zErr
FROM SpecObj AS s
WHERE
    s.class = 'GALAXY'
    AND s.zWarning = 0
    AND s.z BETWEEN {zmin} AND {zmax}
    AND s.ra BETWEEN {ramin} AND {ramax}
    AND s.dec BETWEEN {decmin} AND {decmax}
"""


@dataclass(frozen=True)
class GalaxySample:
    """A sky-region galaxy sample: angular positions plus redshifts."""

    ra: np.ndarray
    dec: np.ndarray
    redshift: np.ndarray
    redshift_error: np.ndarray
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        n = len(self.ra)
        if not (len(self.dec) == len(self.redshift) == len(self.redshift_error) == n):
            raise ValueError("ra, dec, redshift and redshift_error must be equal length")

    def __len__(self) -> int:
        return len(self.ra)

    def filter_redshift(self, zmin: float, zmax: float) -> GalaxySample:
        mask = (self.redshift >= zmin) & (self.redshift <= zmax)
        return GalaxySample(
            ra=self.ra[mask],
            dec=self.dec[mask],
            redshift=self.redshift[mask],
            redshift_error=self.redshift_error[mask],
            provenance={**self.provenance, "redshift_cut": (zmin, zmax)},
        )

    def subsample(self, n: int, *, seed: int | None = None) -> GalaxySample:
        """Uniform random subsample -- preserves large-scale structure statistics."""
        if n >= len(self):
            return self
        idx = np.random.default_rng(seed).choice(len(self), size=n, replace=False)
        return GalaxySample(
            ra=self.ra[idx],
            dec=self.dec[idx],
            redshift=self.redshift[idx],
            redshift_error=self.redshift_error[idx],
            provenance={**self.provenance, "subsampled_to": n, "seed": seed},
        )


def _cache_key(**params: object) -> str:
    payload = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def fetch_galaxies(
    *,
    limit: int = 50_000,
    zmin: float = 0.01,
    zmax: float = 0.15,
    ra_range: tuple[float, float] = (120.0, 240.0),
    dec_range: tuple[float, float] = (0.0, 60.0),
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    data_release: int = 17,
) -> GalaxySample:
    """Fetch an SDSS galaxy sample, using a local cache when available.

    The default window is the SDSS Northern Galactic Cap at low redshift --
    the contiguous, well-sampled volume where cosmic-web filaments and voids
    are best resolved. Requires network access on a cache miss.
    """
    cache_dir = cache_dir or DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(
        limit=limit, zmin=zmin, zmax=zmax, ra=ra_range, dec=dec_range, dr=data_release
    )
    cache_file = cache_dir / f"sdss_galaxies_{key}.npz"

    if cache_file.exists() and not force_refresh:
        with np.load(cache_file, allow_pickle=True) as data:
            return GalaxySample(
                ra=data["ra"],
                dec=data["dec"],
                redshift=data["redshift"],
                redshift_error=data["redshift_error"],
                provenance=json.loads(str(data["provenance"])),
            )

    from astroquery.sdss import SDSS

    query = _GALAXY_QUERY.format(
        limit=limit,
        zmin=zmin,
        zmax=zmax,
        ramin=ra_range[0],
        ramax=ra_range[1],
        decmin=dec_range[0],
        decmax=dec_range[1],
    )
    table = SDSS.query_sql(query, data_release=data_release)
    if table is None or len(table) == 0:
        raise RuntimeError("SDSS query returned no rows; check the region and redshift cuts")

    provenance = {
        "source": f"SDSS DR{data_release} SpecObj",
        "query_limit": limit,
        "redshift_range": [zmin, zmax],
        "ra_range": list(ra_range),
        "dec_range": list(dec_range),
        "n_rows": len(table),
    }
    sample = GalaxySample(
        ra=np.asarray(table["ra"], dtype=float),
        dec=np.asarray(table["dec"], dtype=float),
        redshift=np.asarray(table["z"], dtype=float),
        redshift_error=np.asarray(table["zErr"], dtype=float),
        provenance=provenance,
    )

    np.savez_compressed(
        cache_file,
        ra=sample.ra,
        dec=sample.dec,
        redshift=sample.redshift,
        redshift_error=sample.redshift_error,
        provenance=json.dumps(provenance),
    )
    return sample
