"""Linear recurrence operators and the symmetric-square lock.

Implements the discrete half of the L_3 = Sym^2(L_2) mechanism: given an
order-2 recurrence operator, construct the order-3 operator annihilating all
products of pairs of its solutions, and *certify* the construction by exact
rational arithmetic rather than asserting it.

The certificate discipline is deliberate (Tier B of the programmatic method):
every claim here is decided by exact `Fraction` arithmetic, never floating
point. Coefficient swell in these operators is severe enough that rounding
would silently produce a wrong operator, not merely an imprecise one.

References
----------
Clausen (1828), J. Reine Angew. Math. 3, 89-91.
Cooper (2012), "Sporadic sequences, modular forms and new series for 1/pi",
    Ramanujan J. 29, 163-183.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement

Rational = Fraction | int


@dataclass(frozen=True)
class RecurrenceOperator:
    """A constant-coefficient linear recurrence operator.

    Encodes u_{n+d} = c_{d-1} u_{n+d-1} + ... + c_0 u_n, storing
    `coefficients` as (c_0, ..., c_{d-1}) in ascending shift order.
    """

    coefficients: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("a recurrence operator needs at least one coefficient")
        object.__setattr__(
            self, "coefficients", tuple(Fraction(c) for c in self.coefficients)
        )

    @classmethod
    def of(cls, *coefficients: Rational) -> RecurrenceOperator:
        return cls(tuple(Fraction(c) for c in coefficients))

    @property
    def order(self) -> int:
        return len(self.coefficients)

    def step(self, window: Sequence[Rational]) -> Fraction:
        """Next term from the last `order` terms (ascending index)."""
        if len(window) != self.order:
            raise ValueError(f"expected {self.order} terms, got {len(window)}")
        return sum(
            (c * Fraction(u) for c, u in zip(self.coefficients, window, strict=True)),
            Fraction(0),
        )

    def iterate(self, initial: Sequence[Rational], n_terms: int) -> list[Fraction]:
        """Generate `n_terms` of the solution with the given initial window."""
        if len(initial) != self.order:
            raise ValueError(f"expected {self.order} initial terms, got {len(initial)}")
        seq = [Fraction(v) for v in initial]
        while len(seq) < n_terms:
            seq.append(self.step(seq[-self.order :]))
        return seq[:n_terms]

    def characteristic_polynomial(self) -> list[Fraction]:
        """Coefficients of x^d - c_{d-1} x^{d-1} - ... - c_0, ascending degree."""
        coeffs = [-c for c in self.coefficients]
        coeffs.append(Fraction(1))
        return coeffs

    def annihilates(self, sequence: Sequence[Rational]) -> bool:
        """Exact check that `sequence` satisfies this recurrence throughout."""
        seq = [Fraction(v) for v in sequence]
        if len(seq) <= self.order:
            raise ValueError("sequence is too short to test the recurrence")
        return all(
            self.step(seq[i : i + self.order]) == seq[i + self.order]
            for i in range(len(seq) - self.order)
        )


def symmetric_square(operator: RecurrenceOperator) -> RecurrenceOperator:
    """The order-3 operator L_3 = Sym^2(L_2) annihilating products of solutions.

    For u_{n+2} = a u_{n+1} + b u_n the result is

        v_{n+3} = (a^2 + b) v_{n+2} + b(a^2 + b) v_{n+1} - b^3 v_n

    derived here from the elementary symmetric functions of {L^2, LM, M^2}
    where L, M are the characteristic roots of L_2. Working through symmetric
    functions keeps the construction exact and root-free: no radicals, no
    numerical root-finding, valid even when the roots are irrational or
    complex.
    """
    if operator.order != 2:
        raise ValueError(f"symmetric square is defined for order 2, got {operator.order}")

    b, a = operator.coefficients  # u_{n+2} = a u_{n+1} + b u_n

    # Roots L, M of x^2 - a x - b satisfy L+M = a, LM = -b.
    e1_roots = a
    e2_roots = -b

    # Elementary symmetric functions of the three products {L^2, LM, M^2},
    # expressed in e1_roots and e2_roots via Newton-style identities.
    p1 = e1_roots**2 - e2_roots  # L^2 + LM + M^2
    p2 = e2_roots * (e1_roots**2 - e2_roots)  # sum of pairwise products
    p3 = e2_roots**3  # (LM)^3

    # Characteristic polynomial x^3 - p1 x^2 + p2 x - p3 gives the recurrence
    # v_{n+3} = p1 v_{n+2} - p2 v_{n+1} + p3 v_n.
    return RecurrenceOperator.of(p3, -p2, p1)


def verify_symmetric_square(
    operator: RecurrenceOperator,
    *,
    n_terms: int = 20,
    test_windows: Sequence[tuple[Rational, Rational]] | None = None,
) -> dict[str, object]:
    """Certify Sym^2 by exact arithmetic on independent solutions and products.

    Tests that squares of solutions, and products of *distinct* solutions,
    are all annihilated by the constructed order-3 operator. Products of
    distinct solutions matter: an operator could annihilate every square while
    failing on the cross term LM, which is a genuine element of the symmetric
    square's solution space.
    """
    if operator.order != 2:
        raise ValueError("verification applies to order-2 operators")

    windows = list(test_windows) if test_windows else [(0, 1), (1, 0), (1, 1), (2, -3)]
    l3 = symmetric_square(operator)

    solutions = [operator.iterate(w, n_terms) for w in windows]

    square_ok = all(l3.annihilates([u * u for u in sol]) for sol in solutions)
    cross_ok = all(
        l3.annihilates([x * y for x, y in zip(s1, s2, strict=True)])
        for s1, s2 in combinations_with_replacement(solutions, 2)
    )

    return {
        "operator": operator.coefficients,
        "symmetric_square": l3.coefficients,
        "squares_annihilated": square_ok,
        "cross_products_annihilated": cross_ok,
        "certified": square_ok and cross_ok,
        "n_terms": n_terms,
        "n_test_solutions": len(solutions),
        "arithmetic": "exact (Fraction)",
    }


def closed_form_symmetric_square(a: Rational, b: Rational) -> tuple[Fraction, ...]:
    """The paper's Theorem 3.1 coefficients, for cross-checking the construction.

    Returns (c_0, c_1, c_2) for v_{n+3} = c_2 v_{n+2} + c_1 v_{n+1} + c_0 v_n,
    namely (-b^3, b(a^2+b), a^2+b).
    """
    a, b = Fraction(a), Fraction(b)
    return (-(b**3), b * (a**2 + b), a**2 + b)


# -- Sporadic / Apery-type sequences ---------------------------------------


@dataclass(frozen=True)
class PolynomialRecurrence:
    """A recurrence with polynomial-in-n coefficients: sum_i p_i(n) u_{n+i} = 0.

    This is the shape that Apery-like and Cooper's sporadic sequences take, and
    the shape whose Picard-Fuchs operator is the object of interest. Each
    `coefficients[i]` lists the polynomial p_i in ascending powers of n.
    """

    coefficients: tuple[tuple[Fraction, ...], ...]
    name: str = ""

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def _evaluate(self, poly: Sequence[Fraction], n: int) -> Fraction:
        return sum(
            (c * Fraction(n) ** k for k, c in enumerate(poly)), Fraction(0)
        )

    def step(self, window: Sequence[Rational], n: int) -> Fraction:
        """Solve for u_{n+order} given the preceding terms, exactly.

        Raises if the leading coefficient vanishes at `n` -- the recurrence is
        genuinely undefined there and silently returning a value would be a
        division-by-zero fiction of exactly the kind that invalidated the
        paper's early Lean axiomatization.
        """
        leading = self._evaluate(self.coefficients[-1], n)
        if leading == 0:
            raise ZeroDivisionError(
                f"leading coefficient of {self.name or 'recurrence'} vanishes at n={n}"
            )
        total = sum(
            (
                self._evaluate(self.coefficients[i], n) * Fraction(window[i])
                for i in range(self.order)
            ),
            Fraction(0),
        )
        return -total / leading

    def iterate(
        self, initial: Sequence[Rational], n_terms: int, *, n_start: int = 0
    ) -> list[Fraction]:
        seq = [Fraction(v) for v in initial]
        n = n_start
        while len(seq) < n_terms:
            seq.append(self.step(seq[-self.order :], n))
            n += 1
        return seq[:n_terms]


def apery_zeta3() -> PolynomialRecurrence:
    """Apery's recurrence for the zeta(3) sequence.

        (n+1)^3 u_{n+1} - (34n^3 + 51n^2 + 27n + 5) u_n + n^3 u_{n-1} = 0

    Shifted to index from n so that u_{n+2} is the solved-for term. The
    associated Picard-Fuchs operator is the canonical order-3 example, and
    Cooper's sporadic sequences are its siblings.
    """
    # In shifted form with m = n-1: p_0(m) = m^3... expressed at index m,
    # terms are u_m, u_{m+1}, u_{m+2} with n = m+1.
    # p_2 = (m+2)^3 ; p_1 = -(34(m+1)^3 + 51(m+1)^2 + 27(m+1) + 5) ; p_0 = (m+1)^3
    return PolynomialRecurrence(
        coefficients=(
            (Fraction(1), Fraction(3), Fraction(3), Fraction(1)),  # (m+1)^3
            # -(34m^3 + 153m^2 + 231m + 117), ascending
            (Fraction(-117), Fraction(-231), Fraction(-153), Fraction(-34)),
            (Fraction(8), Fraction(12), Fraction(6), Fraction(1)),  # (m+2)^3
        ),
        name="apery-zeta3",
    )


def apery_zeta2() -> PolynomialRecurrence:
    """Apery's recurrence for the zeta(2) sequence.

        (n+1)^2 u_{n+1} - (11n^2 + 11n + 3) u_n - n^2 u_{n-1} = 0
    """
    # Shifted by m = n-1, n = m+1:
    # p_2 = (m+2)^2 ; p_1 = -(11(m+1)^2 + 11(m+1) + 3) ; p_0 = -(m+1)^2
    return PolynomialRecurrence(
        coefficients=(
            (Fraction(-1), Fraction(-2), Fraction(-1)),  # -(m+1)^2
            (Fraction(-25), Fraction(-33), Fraction(-11)),
            (Fraction(4), Fraction(4), Fraction(1)),  # (m+2)^2
        ),
        name="apery-zeta2",
    )
