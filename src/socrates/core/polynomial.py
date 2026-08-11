"""Multivariate polynomial arithmetic and Groebner basis computation over Q.

Implements Buchberger's algorithm with Buchberger's two criteria and the
standard normal-strategy pair selection, plus reduction to a reduced Groebner
basis. Polynomials are represented as dicts mapping exponent tuples to
Fraction coefficients, which keeps the arithmetic exact.

References
----------
Cox, Little, O'Shea, *Ideals, Varieties, and Algorithms*, 4th ed., Ch. 2.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering
from itertools import combinations
from typing import Literal

Monomial = tuple[int, ...]
MonomialOrder = Literal["lex", "grlex", "grevlex"]


def _order_key(order: MonomialOrder, exponent: Monomial):
    """Sort key placing larger monomials (w.r.t. `order`) later."""
    if order == "lex":
        return exponent
    if order == "grlex":
        return (sum(exponent), exponent)
    if order == "grevlex":
        # grevlex: compare total degree, then break ties by the *last* differing
        # exponent, with the smaller exponent ranking higher. Negating and
        # reversing turns that into a plain tuple comparison.
        return (sum(exponent), tuple(-e for e in reversed(exponent)))
    raise ValueError(f"unknown monomial order: {order!r}")


def _monomial_mul(a: Monomial, b: Monomial) -> Monomial:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _monomial_div(a: Monomial, b: Monomial) -> Monomial | None:
    """Return a/b if b divides a, else None."""
    quotient = []
    for x, y in zip(a, b, strict=True):
        if x < y:
            return None
        quotient.append(x - y)
    return tuple(quotient)


def _monomial_lcm(a: Monomial, b: Monomial) -> Monomial:
    return tuple(max(x, y) for x, y in zip(a, b, strict=True))


@total_ordering
@dataclass(frozen=True)
class Polynomial:
    """A multivariate polynomial over Q in `nvars` variables.

    Terms map exponent tuples to nonzero Fraction coefficients; zero
    coefficients are stripped at construction so equality is structural.
    """

    terms: Mapping[Monomial, Fraction]
    nvars: int
    order: MonomialOrder = "grevlex"

    def __post_init__(self) -> None:
        cleaned = {
            exp: Fraction(coeff)
            for exp, coeff in self.terms.items()
            if Fraction(coeff) != 0
        }
        for exp in cleaned:
            if len(exp) != self.nvars:
                raise ValueError(
                    f"exponent {exp} has arity {len(exp)}, expected {self.nvars}"
                )
        object.__setattr__(self, "terms", cleaned)

    # -- constructors ----------------------------------------------------

    @classmethod
    def zero(cls, nvars: int, order: MonomialOrder = "grevlex") -> Polynomial:
        return cls({}, nvars, order)

    @classmethod
    def constant(
        cls, value: Fraction | int, nvars: int, order: MonomialOrder = "grevlex"
    ) -> Polynomial:
        return cls({(0,) * nvars: Fraction(value)}, nvars, order)

    @classmethod
    def variable(
        cls, index: int, nvars: int, order: MonomialOrder = "grevlex"
    ) -> Polynomial:
        exp = [0] * nvars
        exp[index] = 1
        return cls({tuple(exp): Fraction(1)}, nvars, order)

    @classmethod
    def from_terms(
        cls,
        terms: Iterable[tuple[Monomial, Fraction | int]],
        nvars: int,
        order: MonomialOrder = "grevlex",
    ) -> Polynomial:
        acc: dict[Monomial, Fraction] = {}
        for exp, coeff in terms:
            acc[exp] = acc.get(exp, Fraction(0)) + Fraction(coeff)
        return cls(acc, nvars, order)

    # -- structure -------------------------------------------------------

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def total_degree(self) -> int:
        return max((sum(exp) for exp in self.terms), default=-1)

    def sorted_monomials(self) -> list[Monomial]:
        """Monomials in descending order w.r.t. this polynomial's order."""
        return sorted(self.terms, key=lambda e: _order_key(self.order, e), reverse=True)

    @property
    def leading_monomial(self) -> Monomial:
        if self.is_zero:
            raise ValueError("zero polynomial has no leading monomial")
        return max(self.terms, key=lambda e: _order_key(self.order, e))

    @property
    def leading_coefficient(self) -> Fraction:
        return self.terms[self.leading_monomial]

    @property
    def leading_term(self) -> Polynomial:
        lm = self.leading_monomial
        return Polynomial({lm: self.terms[lm]}, self.nvars, self.order)

    def with_order(self, order: MonomialOrder) -> Polynomial:
        return Polynomial(dict(self.terms), self.nvars, order)

    def monic(self) -> Polynomial:
        if self.is_zero:
            return self
        lc = self.leading_coefficient
        return Polynomial(
            {exp: coeff / lc for exp, coeff in self.terms.items()},
            self.nvars,
            self.order,
        )

    # -- arithmetic ------------------------------------------------------

    def __add__(self, other: Polynomial) -> Polynomial:
        self._check_compatible(other)
        acc = dict(self.terms)
        for exp, coeff in other.terms.items():
            acc[exp] = acc.get(exp, Fraction(0)) + coeff
        return Polynomial(acc, self.nvars, self.order)

    def __neg__(self) -> Polynomial:
        return Polynomial(
            {exp: -coeff for exp, coeff in self.terms.items()}, self.nvars, self.order
        )

    def __sub__(self, other: Polynomial) -> Polynomial:
        return self + (-other)

    def __mul__(self, other: Polynomial | Fraction | int) -> Polynomial:
        if isinstance(other, Fraction | int):
            scalar = Fraction(other)
            return Polynomial(
                {exp: coeff * scalar for exp, coeff in self.terms.items()},
                self.nvars,
                self.order,
            )
        self._check_compatible(other)
        acc: dict[Monomial, Fraction] = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in other.terms.items():
                exp = _monomial_mul(e1, e2)
                acc[exp] = acc.get(exp, Fraction(0)) + c1 * c2
        return Polynomial(acc, self.nvars, self.order)

    __rmul__ = __mul__

    def __pow__(self, exponent: int) -> Polynomial:
        if exponent < 0:
            raise ValueError("negative powers are not polynomials")
        result = Polynomial.constant(1, self.nvars, self.order)
        base = self
        while exponent:
            if exponent & 1:
                result = result * base
            base = base * base
            exponent >>= 1
        return result

    def _check_compatible(self, other: Polynomial) -> None:
        if self.nvars != other.nvars:
            raise ValueError(f"arity mismatch: {self.nvars} vs {other.nvars}")

    # -- evaluation ------------------------------------------------------

    def evaluate(self, point: Sequence[float]) -> float:
        if len(point) != self.nvars:
            raise ValueError(f"expected {self.nvars} coordinates, got {len(point)}")
        total = 0.0
        for exp, coeff in self.terms.items():
            term = float(coeff)
            for value, power in zip(point, exp, strict=True):
                if power:
                    term *= value**power
            total += term
        return total

    # -- comparison / display --------------------------------------------

    def __lt__(self, other: Polynomial) -> bool:
        """Order by leading monomial; zero is smallest. Used for stable output."""
        if self.is_zero:
            return not other.is_zero
        if other.is_zero:
            return False
        return _order_key(self.order, self.leading_monomial) < _order_key(
            other.order, other.leading_monomial
        )

    def __str__(self) -> str:
        if self.is_zero:
            return "0"
        parts = []
        for exp in self.sorted_monomials():
            coeff = self.terms[exp]
            factors = [
                f"x{i}" if p == 1 else f"x{i}^{p}"
                for i, p in enumerate(exp)
                if p
            ]
            if not factors:
                parts.append(str(coeff))
            elif coeff == 1:
                parts.append("*".join(factors))
            elif coeff == -1:
                parts.append("-" + "*".join(factors))
            else:
                parts.append(f"{coeff}*" + "*".join(factors))
        return " + ".join(parts).replace("+ -", "- ")

    def __repr__(self) -> str:
        return f"Polynomial({self}, nvars={self.nvars}, order={self.order!r})"


def divide(f: Polynomial, divisors: Sequence[Polynomial]) -> tuple[list[Polynomial], Polynomial]:
    """Multivariate division: return (quotients, remainder) with f = sum(q_i g_i) + r.

    Follows the deterministic division algorithm of Cox-Little-O'Shea 2.3: at
    each step the leading term of the intermediate dividend is tested against
    each divisor in order, and the first match is used.
    """
    if any(g.is_zero for g in divisors):
        raise ValueError("cannot divide by the zero polynomial")
    nvars, order = f.nvars, f.order
    quotients = [Polynomial.zero(nvars, order) for _ in divisors]
    remainder = Polynomial.zero(nvars, order)
    p = f

    while not p.is_zero:
        lm_p, lc_p = p.leading_monomial, p.leading_coefficient
        for i, g in enumerate(divisors):
            factor_exp = _monomial_div(lm_p, g.leading_monomial)
            if factor_exp is None:
                continue
            factor = Polynomial(
                {factor_exp: lc_p / g.leading_coefficient}, nvars, order
            )
            quotients[i] = quotients[i] + factor
            p = p - factor * g
            break
        else:
            remainder = remainder + p.leading_term
            p = p - p.leading_term

    return quotients, remainder


def reduce_full(f: Polynomial, basis: Sequence[Polynomial]) -> Polynomial:
    """Remainder of f on division by `basis` (the normal form when basis is Groebner)."""
    if not basis:
        return f
    return divide(f, basis)[1]


def s_polynomial(f: Polynomial, g: Polynomial) -> Polynomial:
    """S-polynomial: the combination engineered to cancel both leading terms."""
    lcm = _monomial_lcm(f.leading_monomial, g.leading_monomial)
    f_factor = Polynomial(
        {_monomial_div(lcm, f.leading_monomial): 1 / f.leading_coefficient},  # type: ignore[arg-type]
        f.nvars,
        f.order,
    )
    g_factor = Polynomial(
        {_monomial_div(lcm, g.leading_monomial): 1 / g.leading_coefficient},  # type: ignore[arg-type]
        g.nvars,
        g.order,
    )
    return f_factor * f - g_factor * g


def _coprime_leading_monomials(f: Polynomial, g: Polynomial) -> bool:
    """Buchberger's first criterion: disjoint leading monomials => S-pair reduces to 0."""
    return all(
        a == 0 or b == 0
        for a, b in zip(f.leading_monomial, g.leading_monomial, strict=True)
    )


def groebner_basis(
    generators: Sequence[Polynomial],
    order: MonomialOrder | None = None,
    *,
    reduced: bool = True,
) -> list[Polynomial]:
    """Compute a Groebner basis for the ideal generated by `generators`.

    Uses Buchberger's algorithm with the coprimality criterion and normal
    selection strategy (smallest LCM degree first), then optionally
    inter-reduces to the unique reduced Groebner basis.
    """
    nonzero = [g for g in generators if not g.is_zero]
    if not nonzero:
        return []

    nvars = nonzero[0].nvars
    if any(g.nvars != nvars for g in nonzero):
        raise ValueError("all generators must have the same arity")

    order = order or nonzero[0].order
    basis = [g.with_order(order).monic() for g in nonzero]

    pairs: set[tuple[int, int]] = set(combinations(range(len(basis)), 2))

    while pairs:
        # Normal strategy: pick the pair with the lowest-degree LCM.
        i, j = min(
            pairs,
            key=lambda p: (
                sum(
                    _monomial_lcm(
                        basis[p[0]].leading_monomial, basis[p[1]].leading_monomial
                    )
                ),
                p,
            ),
        )
        pairs.discard((i, j))

        if _coprime_leading_monomials(basis[i], basis[j]):
            continue

        remainder = reduce_full(s_polynomial(basis[i], basis[j]), basis)
        if remainder.is_zero:
            continue

        basis.append(remainder.monic())
        new_index = len(basis) - 1
        pairs.update((k, new_index) for k in range(new_index))

    return _reduce_basis(basis) if reduced else basis


def _reduce_basis(basis: Sequence[Polynomial]) -> list[Polynomial]:
    """Inter-reduce a Groebner basis to the unique reduced form."""
    # Minimalize: drop any generator whose leading monomial is divisible by
    # that of a generator we are keeping.
    minimal: list[Polynomial] = []
    for f in sorted(basis):
        if not any(
            _monomial_div(f.leading_monomial, g.leading_monomial) is not None
            for g in minimal
        ):
            minimal.append(f)

    # Fully reduce each generator against the others.
    reduced: list[Polynomial] = []
    for i, f in enumerate(minimal):
        others = minimal[:i] + minimal[i + 1 :]
        normal = reduce_full(f, others) if others else f
        if not normal.is_zero:
            reduced.append(normal.monic())

    return sorted(reduced, reverse=True)


def ideal_membership(f: Polynomial, generators: Sequence[Polynomial]) -> bool:
    """Decide whether f lies in the ideal generated by `generators`."""
    basis = groebner_basis(generators)
    if not basis:
        return f.is_zero
    return reduce_full(f.with_order(basis[0].order), basis).is_zero


def is_groebner_basis(basis: Sequence[Polynomial]) -> bool:
    """Verify the Buchberger criterion: all S-pairs reduce to zero modulo `basis`."""
    return all(
        reduce_full(s_polynomial(f, g), basis).is_zero
        for f, g in combinations([b for b in basis if not b.is_zero], 2)
    )


def iter_monomials(nvars: int, max_degree: int) -> Iterator[Monomial]:
    """All monomials in `nvars` variables of total degree <= max_degree."""

    def rec(remaining_vars: int, budget: int) -> Iterator[Monomial]:
        if remaining_vars == 1:
            yield from ((d,) for d in range(budget + 1))
            return
        for d in range(budget + 1):
            for rest in rec(remaining_vars - 1, budget - d):
                yield (d, *rest)

    yield from rec(nvars, max_degree)
