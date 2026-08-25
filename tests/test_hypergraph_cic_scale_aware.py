"""Regression pins for the SCALE-AWARE (local-Mahalanobis) CIC readouts.

WHAT THIS FILE EXISTS TO PREVENT. `docs/MENSURA_BENCH_V2.md` section 7.1
refuted the cic-1.0 instrument on a one-line transform: dimension is a
bi-Lipschitz invariant, but rescaling one coordinate of a passing uniform
square by 0.01 (metres -> kilometres) moved `certify` from MEASURED
[1.5859, 2.2822] to MEASURED [0.6374, 1.3626] -- two DISJOINT measured
intervals for the same set, with no abstention signal raised. One of them is
necessarily a calibration violation, and a textbook Takens delay embedding at
lag 1 reaches that regime, so it was reachable by ordinary use.

The tests below pin the four properties the fix was validated on (v2 sections
8.2, 10.2 and 10.4) plus the boundary it does NOT cross:

  * the refutation pair itself -- both MEASURED, both containing 2.0;
  * the isotropic control -- rescaling BOTH axes changes nothing but float
    noise in the continuous arm;
  * rotation equivariance -- the construction depends on the anisotropy
    spectrum and not on its orientation;
  * a MONOTONIC measured region: measured out to ~100:1, abstaining above,
    with no abstain/measure/abstain alternation, so the boundary can be
    honestly pre-registered;
  * the three named regression rows that produced (or nearly produced) the
    only calibration violations this project ever found -- all UNDECIDED.

CALIBRATION IS THE ONLY UNFORGIVABLE OUTCOME. Every assertion here is written
as "never confidently wrong", not "gets the right number": UNDECIDED is always
an acceptable answer and never a failure, because it never violates and never
scores. A test that demanded MEASURED where the instrument honestly abstains
would be pressure to loosen the gate, which is the anti-pattern in
`docs/LL.md` lessons 8-9.

COST. `certify` is deterministic, so the module-level `_certified` cache lets
the ladder, the pinned rows and the refutation pair share results instead of
re-running the O(n^2 d^2) construction per test. n is kept at the smallest
value the recorded evidence covers (1600 for the anisotropy family, whose
measured ceiling is n-dependent); the one n=3200 row is marked `slow`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from socrates.hypergraph.cic import (
    CONNECTIVITY_FRACTION,
    DUPLICATE_FRACTION_MAX,
    K_INSTABILITY_FACTOR,
    READOUT_DIVERGENCE_FACTOR,
    RHO_BIAS_FLOOR,
    RING_FRACTION,
    RING_GP_TOLERANCE,
    WELL_FIT_FRACTION_MIN,
    CICResult,
    Verdict,
    certify,
)
from socrates.hypergraph.pointcloud import (
    GLOBAL_RIDGE_FLOOR_FRAC,
    METRIC_RIDGE_FLOOR_FRAC,
    TOPOLOGY_RIDGE_FLOOR_FRAC,
)

TRUTH_SQUARE = 2.0

# --- targets ---------------------------------------------------------------


def base_square(n: int = 1600, seed: int = 11) -> np.ndarray:
    """The exact cloud v2 sections 7.1/8.1/9.1/10.4 were all measured on."""
    return np.random.default_rng(seed).random((n, 2))


def rescaled(a: np.ndarray, ratio: float) -> np.ndarray:
    """`a` with its second coordinate compressed by `ratio`:1."""
    return a * np.array([1.0, 1.0 / ratio])


def rotation_2d(degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def ellipse(n: int, seed: int, aspect: float) -> np.ndarray:
    t = np.sort(np.random.default_rng(seed).random(n)) * 2.0 * math.pi
    return np.c_[np.cos(t), aspect * np.sin(t)]


def helix(n: int, seed: int, turns: float = 3.0) -> np.ndarray:
    t = np.sort(np.random.default_rng(seed).random(n)) * 2.0 * math.pi * turns
    return np.c_[np.cos(t), np.sin(t), t / (2.0 * math.pi * turns)]


# --- shared, deterministic result cache ------------------------------------

_CACHE: dict[str, CICResult] = {}


def _certified(key: str, points: np.ndarray) -> CICResult:
    if key not in _CACHE:
        _CACHE[key] = certify(points, label=key)
    return _CACHE[key]


def _assert_no_violation(result: CICResult, truth: float, what: str) -> None:
    if result.verdict == Verdict.UNDECIDED:
        return  # never violates, never scores
    assert result.contains(truth), (
        f"CALIBRATION VIOLATION on {what}: {result.verdict} "
        f"[{result.d_lo}, {result.d_hi}] does not contain {truth}. "
        f"shell={result.diagnostics.shell_readout} gp={result.diagnostics.gp_readout}"
    )


# --- the refutation, which is the whole point ------------------------------


def test_refutation_pair_both_measured_and_contain_the_truth():
    """v2 section 7.1's refutation, inverted into an acceptance test.

    Under cic-1.0 these two rows returned DISJOINT measured intervals with no
    signal raised. They are the same set up to an invertible linear map, so a
    dimension estimator must either agree on them or abstain on one.
    """
    a = base_square()
    plain = _certified("A", a)
    thin = _certified("A@diag(1,0.01)", rescaled(a, 100.0))

    for name, result in (("A", plain), ("A @ diag(1, 0.01)", thin)):
        _assert_no_violation(result, TRUTH_SQUARE, name)
        assert result.verdict == Verdict.MEASURED, f"{name}: {result.verdict} {result.signals}"
        assert result.contains(TRUTH_SQUARE)

    # and the intervals must be mutually compatible, not merely both correct
    assert plain.d_lo <= thin.d_hi and thin.d_lo <= plain.d_hi


def test_isotropic_control_changes_nothing_but_float_noise():
    """Rescaling BOTH axes is a similarity, so nothing may move.

    The shell arm is a neighbour RANKING and is therefore exactly invariant.
    The correlation-sum arm is continuous arithmetic on coordinates that
    `* 100.0` does not scale exactly in binary floating point, so it carries a
    last-ULP difference -- documented in v2 section 10.4 and pinned here at a
    tolerance ~1e3 times tighter than anything that could change a verdict.
    """
    a = base_square()
    plain = _certified("A", a)
    scaled = _certified("A@diag(100,100)", a * 100.0)

    assert scaled.verdict == plain.verdict
    assert scaled.certificate.settings.k == plain.certificate.settings.k
    assert scaled.certificate.settings.admissible_k == plain.certificate.settings.admissible_k
    assert scaled.certificate.settings.metric_iterations == (
        plain.certificate.settings.metric_iterations
    )
    # exact: the shell arm reads a ranking, not a distance
    assert scaled.diagnostics.shell_readout == plain.diagnostics.shell_readout
    assert scaled.diagnostics.k_spread == plain.diagnostics.k_spread
    # ULP-level: the correlation-sum arm reads distances
    assert scaled.diagnostics.gp_readout == pytest.approx(
        plain.diagnostics.gp_readout, rel=0, abs=1e-12
    )
    assert scaled.d_lo == pytest.approx(plain.d_lo, rel=0, abs=1e-12)
    assert scaled.d_hi == pytest.approx(plain.d_hi, rel=0, abs=1e-12)


@pytest.mark.parametrize("ratio", [1.0, 100.0])
def test_rotation_equivariance(ratio):
    """A validated property (v2 section 8.2): the construction depends on the
    anisotropy SPECTRUM, not on the axes it happens to be expressed in --
    exactly what a covariance-based local metric should do, and the reason
    this is a metric fix rather than a per-axis normalisation.

    Rotating float64 coordinates is not an exact operation, so the continuous
    arm again carries a few ULP. The discrete half -- verdict, selected k,
    admissible rungs, and the shell readout -- must match exactly.
    """
    a = base_square()
    thin = _certified(f"rescale_{ratio:g}", rescaled(a, ratio))
    turned = _certified(
        f"rescale_{ratio:g}+rot37", rescaled(a, ratio) @ rotation_2d(37.0).T
    )

    assert turned.verdict == thin.verdict
    assert turned.certificate.settings.k == thin.certificate.settings.k
    assert turned.certificate.settings.admissible_k == thin.certificate.settings.admissible_k
    assert turned.diagnostics.shell_readout == thin.diagnostics.shell_readout
    assert turned.diagnostics.gp_readout == pytest.approx(
        thin.diagnostics.gp_readout, rel=0, abs=1e-12
    )
    _assert_no_violation(turned, TRUTH_SQUARE, f"rotated {ratio:g}:1")
    if thin.verdict == Verdict.MEASURED:
        assert turned.contains(TRUTH_SQUARE)


# --- the measured region, and its boundary ---------------------------------

# Ratios spanning the whole validated range: inside the measured region, the
# documented ~100:1 ceiling, and well past it. v2 section 10.4 records
# MEASURED 1:1..100:1 then UNDECIDED above, at n=1600.
LADDER_RATIOS = (1.0, 10.0, 100.0, 500.0, 1000.0)


def test_anisotropy_ladder_is_monotonic_and_never_violates():
    """The property that makes the boundary pre-registrable.

    v2 section 9.2 rejected an earlier composition precisely because its
    measured region ALTERNATED -- abstain at 1:1-20:1, measure at 50:1-100:1,
    abstain above -- which cannot be honestly stated as a boundary however
    good the individual rows look. So this asserts the SHAPE of the sequence,
    not just its rows: once the ladder starts abstaining as anisotropy grows,
    it must never measure again.

    Note the asymmetry, which is deliberate. Being MEASURED is required only
    where the recorded evidence says the instrument reaches (<= 100:1);
    everywhere else UNDECIDED is a pass. An abstention is never a failure.
    """
    a = base_square()
    results = [
        (ratio, _certified(f"rescale_{ratio:g}", rescaled(a, ratio)))
        for ratio in LADDER_RATIOS
    ]

    for ratio, result in results:
        _assert_no_violation(result, TRUTH_SQUARE, f"{ratio:g}:1 rescale")

    measured = [result.verdict != Verdict.UNDECIDED for _, result in results]
    first_abstention = next((i for i, m in enumerate(measured) if not m), len(measured))
    assert not any(measured[first_abstention:]), (
        "NON-MONOTONIC measured region -- the boundary cannot be pre-registered: "
        f"{[(r, m) for (r, _), m in zip(results, measured, strict=True)]}"
    )

    # inside the validated reach, abstention would be a regression of REACH
    # (not of validity), so it is still worth catching
    for ratio, result in results:
        if ratio <= 100.0:
            assert result.verdict == Verdict.MEASURED, (
                f"reach regression at {ratio:g}:1: {result.verdict} {result.signals}"
            )
            assert result.contains(TRUTH_SQUARE)

    # past the structural ceiling the gate must be the thing that stops it
    for ratio, result in results:
        if ratio >= 500.0:
            assert result.verdict == Verdict.UNDECIDED
            assert result.signals


# --- the named regression rows ---------------------------------------------

# Every calibration violation this project has ever found, as an input.
#   8.1  -- the prototype's own violation: bracket [0.749, 1.989], missed by 0.011
#   9.1a -- found by the final gate at 500:1, missed truth by 0.0044
#   9.1b -- found by PERTURBING n, not the rescale factor; missed by 0.090
# All three must abstain. If one ever comes back MEASURED, read its interval
# before touching anything: containing 2.0 would be a genuine reach extension,
# but missing it is the outcome this whole file exists to prevent.
PINNED_ROWS = [
    ("8.1_1000to1_n1600", 1600, 1000.0),
    ("9.1a_500to1_n1600", 1600, 500.0),
    ("9.1b_1000to1_n1440", 1440, 1000.0),
]


@pytest.mark.parametrize(("label", "n", "ratio"), PINNED_ROWS)
def test_pinned_violation_rows_stay_undecided(label, n, ratio):
    key = f"rescale_{ratio:g}" if n == 1600 else label
    result = _certified(key, rescaled(base_square(n), ratio))
    _assert_no_violation(result, TRUTH_SQUARE, label)
    assert result.verdict == Verdict.UNDECIDED, (
        f"{label} is no longer abstaining: {result.verdict} [{result.d_lo}, {result.d_hi}]"
    )
    # the shell arm has collapsed to the chain artifact; that is WHY it abstains
    assert result.signals


# --- the 1-D family, which the scale-aware selection is what fixes ---------


@pytest.mark.parametrize(
    ("name", "points"),
    [
        ("ellipse_20to1", ellipse(1600, 23, 0.05)),
        ("helix_3d", helix(1600, 23)),
    ],
)
def test_one_dimensional_family_measures_one(name, points):
    """v2 section 8.2: an elongated ellipse and a helix in 3-D are 1-D
    supports whose local covariance is near-degenerate by construction. The
    shell arm used to read `nan` on them -- k-starvation, not rank deficiency,
    confirmed in round 2 -- and the fix is that the ladder's k selection now
    runs on the scale-aware edge set, which reaches the k=25 rung these need.
    """
    result = certify(points, label=name)
    _assert_no_violation(result, 1.0, name)
    assert result.verdict in {Verdict.MEASURED, Verdict.DEGENERATE_EXACT}
    assert result.contains(1.0)


# --- the certificate must say what was done -------------------------------


def test_certificate_records_the_self_selected_metric():
    """v2 section 8.4 item 1: metric, floors and iteration count are recorded.

    Which points count as neighbours is now chosen from the data, so a
    certificate that named only `k` would understate what was selected -- and
    a replay could not tell whether it reproduced the measurement.
    """
    settings = _certified("A", base_square()).certificate.settings
    assert settings.metric == "local-mahalanobis"
    assert settings.covariance_window_k0 > 0
    assert 1 <= settings.metric_iterations <= 8
    assert settings.metric_converged is True
    assert settings.topology_ridge_floor == TOPOLOGY_RIDGE_FLOOR_FRAC
    assert settings.metric_ridge_floor == METRIC_RIDGE_FLOOR_FRAC
    assert settings.global_ridge_floor == GLOBAL_RIDGE_FLOOR_FRAC


def test_the_three_ridge_floors_stay_split():
    """They are three constants doing three different jobs, and merging them
    was MEASURED to break the construction: one shared 0.05 floor caps the
    correctable anisotropy at 20:1 everywhere -- including on the global
    bootstrap, whose only job is to avoid literal singularity -- so the
    refutation case's 10000:1 spectrum was only ever corrected 20:1 deep.
    """
    assert GLOBAL_RIDGE_FLOOR_FRAC == 1e-9
    assert TOPOLOGY_RIDGE_FLOOR_FRAC == 0.05
    assert METRIC_RIDGE_FLOOR_FRAC == 0.001
    assert len({GLOBAL_RIDGE_FLOOR_FRAC, TOPOLOGY_RIDGE_FLOOR_FRAC, METRIC_RIDGE_FLOOR_FRAC}) == 3


def test_the_abstention_stack_was_not_loosened():
    """The scale-aware work changed the READOUTS. It changed no gate.

    Pinned because this is the exact pressure point: every one of these
    constants sits between a plausible-looking number and a calibration
    violation, and `READOUT_DIVERGENCE_FACTOR` in particular was tightened
    3.0 -> 1.0 after the detector was caught emitting MEASURED [2.61, 5.06] on
    an i.i.d. uniform cloud in R^6 against a truth of exactly 6.0. Loosening a
    calibration gate so wins pass is `docs/LL.md` lessons 8-9.
    """
    assert RHO_BIAS_FLOOR == 0.18
    assert READOUT_DIVERGENCE_FACTOR == 1.0
    assert K_INSTABILITY_FACTOR == 2.0
    assert CONNECTIVITY_FRACTION == 0.90
    assert WELL_FIT_FRACTION_MIN == 0.50
    assert RING_FRACTION == 0.50
    assert RING_GP_TOLERANCE == 0.25
    assert DUPLICATE_FRACTION_MAX == 0.10


# --- n-dependence of the ceiling (the slow one) ---------------------------


@pytest.mark.slow
def test_measured_region_extends_with_n():
    """v2 section 10.4 records the ceiling moving with sample size: MEASURED
    to 100:1 at n=1600 and to 200:1 at n=3200. Pinned as a DIRECTION, not a
    number -- a row that abstains here is still a pass, because the reach is
    the thing being measured and abstention never violates.

    Marked slow: the construction is O(n^2 d^2) and this is the only n=3200
    row in the suite. Deselect with `-m "not slow"`.
    """
    a = base_square(3200)
    result = certify(rescaled(a, 200.0), label="n=3200/rescale_200to1")
    _assert_no_violation(result, TRUTH_SQUARE, "200:1 at n=3200")
    if result.verdict == Verdict.MEASURED:
        assert result.contains(TRUTH_SQUARE)
    # whatever it decides, the 1000:1 row at the same n must not be MEASURED
    far = certify(rescaled(a, 1000.0), label="n=3200/rescale_1000to1")
    _assert_no_violation(far, TRUTH_SQUARE, "1000:1 at n=3200")
    assert far.verdict == Verdict.UNDECIDED
