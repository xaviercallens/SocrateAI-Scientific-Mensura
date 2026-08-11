"""Exact algebraic geometry: polynomial ideals, Groebner bases, variety sampling."""

from .polynomial import (
    Polynomial,
    divide,
    groebner_basis,
    ideal_membership,
    is_groebner_basis,
    reduce_full,
    s_polynomial,
)
from .variety import (
    project_to_variety,
    sample_variety,
    sphere_polynomial,
    torus_polynomial,
)

__all__ = [
    "Polynomial",
    "divide",
    "groebner_basis",
    "ideal_membership",
    "is_groebner_basis",
    "project_to_variety",
    "reduce_full",
    "s_polynomial",
    "sample_variety",
    "sphere_polynomial",
    "torus_polynomial",
]
