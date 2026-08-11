"""Dual-scale T-dual geometry and regularized cascades (paper Sections 2-5)."""

from .geometry import (
    certify_geometry,
    effective_radius,
    effective_radius_exact,
    effective_wavenumber,
    maximal_wavenumber,
    minimal_scale,
)
from .shell import (
    ShellResult,
    compare_regularization,
    convergence_study,
    dyadic_wavenumbers,
    enstrophy_bound,
    simulate_shell_model,
)

__all__ = [
    "ShellResult",
    "certify_geometry",
    "compare_regularization",
    "convergence_study",
    "dyadic_wavenumbers",
    "effective_radius",
    "effective_radius_exact",
    "effective_wavenumber",
    "enstrophy_bound",
    "maximal_wavenumber",
    "minimal_scale",
    "simulate_shell_model",
]
