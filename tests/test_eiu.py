"""Tests for the EIU model: RNS arithmetic and rational state collapse."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from socrates.eiu import (
    RNSCodec,
    collapse_error_bound,
    default_codec,
    extended_gcd,
    mod_inverse,
    state_collapse,
)

CODEC = RNSCodec((97, 101, 103, 107))  # the spec's worked example set


def test_spec_worked_example() -> None:
    """The exact example from paper/eiu_rns_simulation.md must reproduce."""
    assert CODEC.dynamic_range == 107972737
    assert CODEC.encode(123456) == (72, 34, 62, 85)
    assert CODEC.encode(789) == (13, 82, 68, 40)
    product = CODEC.mul(CODEC.encode(123456), CODEC.encode(789))
    assert product == (63, 61, 96, 83)
    assert CODEC.decode(product) == 97406784 == (123456 * 789) % CODEC.dynamic_range


@given(st.integers(min_value=0, max_value=107972736))
@settings(max_examples=300, deadline=None)
def test_encode_decode_roundtrip(value: int) -> None:
    assert CODEC.decode(CODEC.encode(value)) == value


@given(
    st.integers(min_value=0, max_value=10_000),
    st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=200, deadline=None)
def test_ring_operations_match_integers(x: int, y: int) -> None:
    """Channel-wise add/sub/mul agree with integer arithmetic inside the range."""
    ex, ey = CODEC.encode(x), CODEC.encode(y)
    assert CODEC.decode(CODEC.add(ex, ey)) == x + y
    assert CODEC.decode(CODEC.mul(ex, ey)) == x * y
    assert CODEC.decode(CODEC.sub(ex, ey)) == (x - y) % CODEC.dynamic_range


def test_non_coprime_moduli_rejected() -> None:
    with pytest.raises(ValueError):
        RNSCodec((6, 10, 7))


def test_overflow_raises_instead_of_wrapping() -> None:
    """Silent wraparound is the unflagged error the exactness programme forbids."""
    with pytest.raises(OverflowError):
        CODEC.checked_mul(123456, 123456)  # true product exceeds M
    with pytest.raises(OverflowError):
        CODEC.encode(CODEC.dynamic_range)


def test_dot_product_exact() -> None:
    xs, ys = [3, 141, 5926, 53589], [2, 71, 828, 1828]
    assert CODEC.dot(xs, ys) == sum(a * b for a, b in zip(xs, ys, strict=True))


def test_extended_gcd_is_iterative_and_scales() -> None:
    """The spec's recursive form would exceed the stack near these sizes."""
    a, b = 2**4096 - 1, 2**4093 - 1
    g, x, y = extended_gcd(a, b)
    assert a * x + b * y == g


def test_default_codec_prime_channels() -> None:
    codec = default_codec(n_channels=4)
    value = 12345678901234567890 % codec.dynamic_range
    assert codec.decode(codec.encode(value)) == value


def test_mod_inverse_known_value() -> None:
    assert (mod_inverse(3, 7) * 3) % 7 == 1


# -- State collapse: the honest Memory Boundedness ---------------------------


@given(
    st.fractions(min_value=Fraction(-100), max_value=Fraction(100)),
    st.integers(min_value=10, max_value=10**6),
)
@settings(max_examples=300, deadline=None)
def test_collapse_error_respects_farey_bound(value: Fraction, max_den: int) -> None:
    """|error| < 1/(q(N+1-q)) -- the bound that is actually true.

    The HaloAlg spec's 1/(q*N) claim fails on real inputs (found empirically
    during the ladder experiment); this property test guards the corrected one.
    """
    collapsed, error = state_collapse(value, max_den)
    assert abs(error) <= collapse_error_bound(collapsed, max_den)


def test_collapse_is_identity_when_within_budget() -> None:
    value = Fraction(355, 113)
    collapsed, error = state_collapse(value, 1000)
    assert collapsed == value and error == 0
