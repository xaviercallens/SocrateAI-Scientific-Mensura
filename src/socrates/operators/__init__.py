"""Poly-algebraic calculus: recurrence/differential operators and Sym^2 locks."""

from .recurrence import (
    PolynomialRecurrence,
    RecurrenceOperator,
    apery_zeta2,
    apery_zeta3,
    closed_form_symmetric_square,
    symmetric_square,
    verify_symmetric_square,
)

__all__ = [
    "PolynomialRecurrence",
    "RecurrenceOperator",
    "apery_zeta2",
    "apery_zeta3",
    "closed_form_symmetric_square",
    "symmetric_square",
    "verify_symmetric_square",
]
