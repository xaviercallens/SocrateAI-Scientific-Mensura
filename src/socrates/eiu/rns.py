"""Residue Number System arithmetic: the Exact Inference Unit's datapath model.

Software model of the EIU described in `paper/eiu_rns_simulation.md`, with the
corrections recorded in `paper/REVIEW_haloalg.md`:

* iterative extended Euclid (the recursive form overflows the Python stack for
  cryptographic-size moduli);
* explicit dynamic-range discipline -- RNS arithmetic is exact only inside
  [0, M); a silent wraparound is precisely the kind of unflagged error the
  exactness programme exists to eliminate, so overflow raises here;
* honesty about closure: RNS closes ring operations (+, -, *). Division, sign
  detection and magnitude comparison require reconstruction -- the classical
  weakness of the representation. Rational arithmetic therefore rides on top
  as (numerator, denominator) pairs with multiplication-only kernels.

References
----------
Szabo & Tanaka, *Residue Arithmetic and its Applications* (1967).
Garner, "The residue number system", IRE Trans. (1959).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, prod


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Iterative extended Euclid: returns (g, x, y) with a*x + b*y = g."""
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y


def mod_inverse(a: int, modulus: int) -> int:
    g, x, _ = extended_gcd(a % modulus, modulus)
    if g != 1:
        raise ValueError(f"{a} has no inverse modulo {modulus}")
    return x % modulus


@dataclass(frozen=True)
class RNSCodec:
    """An RNS channel set over pairwise-coprime moduli.

    Encodes integers in [0, M) as residue vectors; add/sub/mul act channel-wise
    with no carries between channels (the EIU's iso-latency property). Decode
    reconstructs via the Chinese Remainder Theorem.
    """

    moduli: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.moduli) < 2:
            raise ValueError("need at least two moduli")
        for m in self.moduli:
            if m < 2:
                raise ValueError(f"modulus {m} must be >= 2")
        for i in range(len(self.moduli)):
            for j in range(i + 1, len(self.moduli)):
                if gcd(self.moduli[i], self.moduli[j]) != 1:
                    raise ValueError(
                        f"moduli {self.moduli[i]} and {self.moduli[j]} are not coprime"
                    )

    @property
    def dynamic_range(self) -> int:
        """M = product of moduli; arithmetic is exact on [0, M)."""
        return prod(self.moduli)

    def _crt_constants(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        big_m = self.dynamic_range
        m_i = tuple(big_m // m for m in self.moduli)
        inv = tuple(mod_inverse(mi, m) for mi, m in zip(m_i, self.moduli, strict=True))
        return m_i, inv

    def encode(self, value: int) -> tuple[int, ...]:
        if not 0 <= value < self.dynamic_range:
            raise OverflowError(
                f"{value} outside dynamic range [0, {self.dynamic_range})"
            )
        return tuple(value % m for m in self.moduli)

    def decode(self, residues: tuple[int, ...]) -> int:
        """CRT reconstruction (the EIU_RNS_CRT_Decoder of the spec)."""
        if len(residues) != len(self.moduli):
            raise ValueError("residue vector length mismatch")
        m_i, inv = self._crt_constants()
        big_m = self.dynamic_range
        total = 0
        for r, mi, iv in zip(residues, m_i, inv, strict=True):
            total = (total + r * mi * iv) % big_m
        return total

    # -- channel-wise (carry-free) ring operations ------------------------

    def add(self, x: tuple[int, ...], y: tuple[int, ...]) -> tuple[int, ...]:
        return tuple((a + b) % m for a, b, m in zip(x, y, self.moduli, strict=True))

    def sub(self, x: tuple[int, ...], y: tuple[int, ...]) -> tuple[int, ...]:
        return tuple((a - b) % m for a, b, m in zip(x, y, self.moduli, strict=True))

    def mul(self, x: tuple[int, ...], y: tuple[int, ...]) -> tuple[int, ...]:
        return tuple((a * b) % m for a, b, m in zip(x, y, self.moduli, strict=True))

    # -- checked high-level operations -------------------------------------

    def checked_mul(self, x: int, y: int) -> int:
        """Encode-multiply-decode with overflow discipline.

        Raises OverflowError when the true product leaves [0, M) rather than
        silently returning the wrapped value: unflagged wraparound is exactly
        the failure mode the exactness programme forbids.
        """
        if x * y >= self.dynamic_range:
            raise OverflowError(
                f"product {x}*{y} exceeds dynamic range {self.dynamic_range}"
            )
        return self.decode(self.mul(self.encode(x), self.encode(y)))

    def dot(self, xs: list[int], ys: list[int]) -> int:
        """Exact dot product in residue space -- the EIU's core inference kernel.

        All multiplications and the accumulation run channel-wise; only the
        final result is reconstructed. Overflow of the *true* dot product is
        checked against M explicitly.
        """
        if len(xs) != len(ys):
            raise ValueError("length mismatch")
        true_value = sum(a * b for a, b in zip(xs, ys, strict=True))
        if not 0 <= true_value < self.dynamic_range:
            raise OverflowError(
                f"dot product {true_value} outside dynamic range {self.dynamic_range}"
            )
        acc = self.encode(0)
        for a, b in zip(xs, ys, strict=True):
            acc = self.add(acc, self.mul(self.encode(a), self.encode(b)))
        return self.decode(acc)


def default_codec(n_channels: int = 8, bits_per_modulus: int = 31) -> RNSCodec:
    """A codec built from distinct primes near 2^bits, mimicking EIU register width."""
    from sympy import prevprime

    moduli: list[int] = []
    candidate = 2**bits_per_modulus
    while len(moduli) < n_channels:
        candidate = prevprime(candidate)
        moduli.append(int(candidate))
    return RNSCodec(tuple(moduli))


# -- Bit-width growth: the honest version of "Memory Boundedness" ------------


def denominator_bits(value: Fraction) -> int:
    """Bit-width of a rational in lowest terms (max of numerator/denominator)."""
    return max(value.numerator.bit_length(), value.denominator.bit_length())


def state_collapse(value: Fraction, max_denominator: int) -> tuple[Fraction, Fraction]:
    """The spec's K-step collapse, with its accuracy cost made explicit.

    Returns (collapsed, error). GCD reduction alone does NOT bound bit-width --
    a fraction in lowest terms can still be astronomically wide -- so a genuine
    bound requires this lossy rounding.

    The certified error bound is |error| < 1/(q * (N + 1 - q)) for N =
    `max_denominator` and q the collapsed denominator: x lies between Farey-N
    neighbours p/q and p'/q' whose gap is 1/(q*q') with q' > N - q. Note the
    HaloAlg spec's stated bound 1/(q * D_max) is FALSE in general (measured
    counterexample in `scripts/solver_ladder.py` history: error exceeded it by
    ~2x); use `collapse_error_bound` for the guaranteed one.
    """
    collapsed = value.limit_denominator(max_denominator)
    return collapsed, value - collapsed


def collapse_error_bound(collapsed: Fraction, max_denominator: int) -> Fraction:
    """Guaranteed bound on the state-collapse error (Farey-neighbour gap)."""
    q = collapsed.denominator
    if q > max_denominator:
        raise ValueError("collapsed value exceeds the stated max_denominator")
    return Fraction(1, q * (max_denominator + 1 - q))
