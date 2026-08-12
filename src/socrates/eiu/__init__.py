"""Exact Inference Unit model: RNS parallel arithmetic and rational discipline."""

from .rns import (
    RNSCodec,
    collapse_error_bound,
    default_codec,
    denominator_bits,
    extended_gcd,
    mod_inverse,
    state_collapse,
)

__all__ = [
    "RNSCodec",
    "collapse_error_bound",
    "default_codec",
    "denominator_bits",
    "extended_gcd",
    "mod_inverse",
    "state_collapse",
]
