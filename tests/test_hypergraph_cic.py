"""Tests for the CIC core (`socrates.hypergraph.cic`).

The load-bearing ones are the calibration tests at the bottom: a `MEASURED`
interval that does not contain the truth is the one unforgivable outcome under
MENSURA-BENCH v2, so those are written as "never confidently wrong", not as
"gets the right number".

Point counts are kept at the small end of the useful range so the suite stays
fast; the full known-answer run lives in
`scripts/hypergraph_benchmark/v2/cic_known_answer.py`.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from socrates.hypergraph.cic import (
    K_LADDER,
    MIN_RADIUS,
    RADIUS_CAP,
    CICResult,
    Signal,
    Verdict,
    _as_array,
    _dedupe,
    _refit,
    build_interval,
    certify,
    select_settings,
)
from socrates.hypergraph.dimension import local_dimension
from socrates.hypergraph.pointcloud import knn_hypergraph

# --- targets, all with exactly-known truths -------------------------------


def square(n: int, seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 2))


def circle(n: int, seed: int = 7) -> np.ndarray:
    t = np.sort(np.random.default_rng(seed).random(n)) * 2.0 * math.pi
    return np.c_[np.cos(t), np.sin(t)]


def cantor_dust(n: int, seed: int = 20260814) -> np.ndarray:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ratio = 1.0 / 3.0
    depth = int(math.ceil(60.0 / math.log2(1.0 / ratio)))
    addr = np.random.default_rng(seed).choice(4, size=(n, depth))
    return np.einsum("ndc,d->nc", vertices[addr], (1.0 - ratio) * ratio ** np.arange(depth))


# --- (a) the output schema -------------------------------------------------


def test_result_schema_is_frozen_and_json_round_trips():
    result = certify(square(400), label="unit")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = "nope"  # frozen dataclass
    payload = result.to_dict()
    assert payload["verdict"] in {
        Verdict.MEASURED,
        Verdict.DEGENERATE_EXACT,
        Verdict.UNDECIDED,
    }
    # the certificate carries what a replay needs
    cert = payload["certificate"]
    assert cert["inputs_hash"].startswith("sha256:")
    assert cert["adapter"]
    assert cert["contract_version"]
    assert cert["replay_command"]
    assert cert["settings"]["k_ladder"] == list(K_LADDER)
    # the mandatory diagnostics are present and never used as gates
    diag = payload["diagnostics"]
    for field in (
        "n_points_used",
        "n_ball_expansions",
        "n_pair_counts",
        "wall_time_s",
        "k_ensemble",
        "component_counts",
        "well_fit_fraction",
        "shell_readout_half_sample",
    ):
        assert field in diag
    json.dumps(payload)  # must be serialisable for the incremental on-disk record


def test_interval_is_none_exactly_when_undecided():
    for points in (square(400), cantor_dust(400), circle(400)):
        result = certify(points)
        if result.verdict == Verdict.UNDECIDED:
            assert result.d_lo is None and result.d_hi is None
            assert result.signals, "UNDECIDED must name at least one signal"
        else:
            assert result.d_lo is not None and result.d_hi is not None
            assert result.d_lo <= result.d_hi
            assert not result.signals


def test_inputs_hash_is_content_addressed():
    a = square(300, seed=1)
    b = square(300, seed=2)
    assert certify(a).certificate.inputs_hash == certify(a.copy()).certificate.inputs_hash
    assert certify(a).certificate.inputs_hash != certify(b).certificate.inputs_hash


# --- ZERO KNOB -------------------------------------------------------------


def test_certify_accepts_no_estimator_knob():
    """The correctness requirement of v2 section 2.2: a caller must not be able
    to choose k, max_radius, min_radius, the sample size or the fit threshold."""
    import inspect

    params = set(inspect.signature(certify).parameters) - {"points"}
    assert params == {"label", "adapter", "replay_command", "time_indices"}


def test_settings_are_selected_and_recorded():
    result = certify(square(800))
    settings = result.certificate.settings
    assert settings.min_radius == MIN_RADIUS
    if result.verdict != Verdict.UNDECIDED:
        assert settings.k in K_LADDER
        assert 3 <= settings.max_radius <= RADIUS_CAP
        assert settings.k == min(settings.admissible_k)
    assert settings.selection_rule  # the rule travels with the number


def test_selection_differs_across_inputs():
    """A rule that picks the same thing on every input has not selected anything."""
    picked = {
        name: select_settings(points)
        for name, points in (
            ("square", square(1600)),
            ("circle", circle(1600)),
            ("cantor", cantor_dust(1600)),
            ("cube", np.random.default_rng(5).random((1600, 3))),
        )
    }
    assert picked["cantor"].k == 0  # no admissible rung at all
    assert picked["square"].k != picked["circle"].k
    assert len({(s.k, s.max_radius) for s in picked.values()}) >= 3


def test_select_settings_matches_what_certify_recorded():
    points = square(800)
    assert select_settings(points) == certify(points).certificate.settings


# --- the refit shortcut must not be a different estimator ------------------


def test_refit_reproduces_local_dimension_exactly():
    """`_refit` reads already-computed ball volumes instead of paying for a
    second BFS. If it ever diverges from `local_dimension` the CIC core is
    silently a different estimator from the one the record was taken with."""
    hg = knn_hypergraph([tuple(p) for p in square(600)], k=8)
    nodes = sorted(hg.nodes)[:40]
    for node in nodes:
        at_cap = local_dimension(hg, node, max_radius=RADIUS_CAP, min_radius=MIN_RADIUS)
        for max_radius in range(3, RADIUS_CAP + 1):
            direct = local_dimension(hg, node, max_radius=max_radius, min_radius=MIN_RADIUS)
            shortcut = _refit(at_cap, max_radius)
            assert shortcut.dimension == pytest.approx(direct.dimension, abs=1e-12)
            assert shortcut.r_squared == pytest.approx(direct.r_squared, abs=1e-12)
            assert shortcut.degenerate == direct.degenerate
            assert shortcut.near_degenerate == direct.near_degenerate
            assert shortcut.underdetermined == direct.underdetermined
            assert shortcut.volumes == direct.volumes


# --- (b) the interval builder ----------------------------------------------


def test_bracket_straddles_when_readouts_straddle():
    lo, hi, hull_lo, hull_hi, allowance, _ = build_interval(2.15, 1.93, k_spread=0.0)
    assert (hull_lo, hull_hi) == (1.93, 2.15)
    assert lo < 2.0 < hi
    assert allowance > 0


def test_allowance_carries_the_measured_bias_floor():
    """The hull alone is not safe: v2 section 6 measured the shell readout
    0.053 BELOW truth at n=800, so both arms can be low at once."""
    lo, hi, hull_lo, hull_hi, _, _ = build_interval(1.947, 1.927, k_spread=0.0)
    assert hull_hi < 2.0, "premise of the test: the hull misses the truth"
    assert lo <= 2.0 <= hi, "the allowance must recover it"


def test_interval_widens_when_the_readouts_disagree():
    narrow = build_interval(2.0, 2.0, k_spread=0.0)
    wide = build_interval(2.4, 1.6, k_spread=0.0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_interval_widens_with_residual_k_instability():
    calm = build_interval(2.0, 1.95, k_spread=0.0)
    jittery = build_interval(2.0, 1.95, k_spread=1.2)
    assert (jittery[1] - jittery[0]) > (calm[1] - calm[0])


def test_interval_never_goes_negative():
    lo, _, _, _, _, _ = build_interval(0.2, 0.1, k_spread=0.0)
    assert lo >= 0.0


def test_slope_bound_branch_joins_the_hull():
    without = build_interval(1.0, 1.0, k_spread=0.0)
    with_bound = build_interval(1.0, 1.0, k_spread=0.0, ring_bound=0.2)
    assert with_bound[2] <= 0.8 and with_bound[3] >= 1.2
    assert (with_bound[1] - with_bound[0]) > (without[1] - without[0])
    assert "slope-bound" in with_bound[5]


# --- (d) the abstention detector -------------------------------------------


def test_cantor_dust_abstains():
    """v2 section 3.1: this is the suite's negative control. The shell
    estimator is known to produce confident NEGATIVE dimensions here, and
    emitting MEASURED would invalidate the round."""
    result = certify(cantor_dust(1600))
    assert result.verdict == Verdict.UNDECIDED
    assert result.d_lo is None and result.d_hi is None
    assert Signal.FRAGMENTATION in result.signals


def test_disconnected_clusters_abstain():
    rng = np.random.default_rng(3)
    points = np.vstack([rng.random((400, 2)), rng.random((400, 2)) + np.array([50.0, 0.0])])
    result = certify(points)
    assert result.verdict == Verdict.UNDECIDED
    assert Signal.FRAGMENTATION in result.signals


def test_near_duplicate_cloud_abstains():
    """Finding F3: a closed orbit sampled over three periods."""
    t = np.arange(400) * 2.0 * math.pi / 400
    base = np.c_[np.cos(t), np.sin(t)]
    points = np.vstack([base, base + 1e-9, base + 2e-9])
    result = certify(points)
    assert result.verdict == Verdict.UNDECIDED
    assert Signal.NEAR_DUPLICATE_POINTS in result.signals


def test_too_few_points_abstains():
    assert certify(square(20)).verdict == Verdict.UNDECIDED


def test_every_undecided_names_a_signal_that_exists():
    known = {v for k, v in vars(Signal).items() if not k.startswith("_")}
    for points in (square(20), cantor_dust(400), square(600), circle(600)):
        result = certify(points)
        assert set(result.signals) <= known


# --- calibration: the one unforgivable outcome -----------------------------


def _assert_no_violation(result: CICResult, truth: float, what: str) -> None:
    if result.verdict == Verdict.UNDECIDED:
        return  # never violates, never scores
    assert result.contains(truth), (
        f"CALIBRATION VIOLATION on {what}: {result.verdict} "
        f"[{result.d_lo}, {result.d_hi}] does not contain {truth}. "
        f"shell={result.diagnostics.shell_readout} gp={result.diagnostics.gp_readout}"
    )


@pytest.mark.parametrize("seed", [11, 20260814, 7])
def test_uniform_square_interval_contains_two(seed):
    result = certify(square(1600, seed=seed))
    assert result.verdict == Verdict.MEASURED
    _assert_no_violation(result, 2.0, f"uniform square seed={seed}")


@pytest.mark.parametrize("seed", [11, 20260814, 7])
def test_circle_interval_contains_one(seed):
    result = certify(circle(1600, seed=seed))
    assert result.verdict in {Verdict.MEASURED, Verdict.DEGENERATE_EXACT}
    _assert_no_violation(result, 1.0, f"circle seed={seed}")


def test_equispaced_circle_is_degenerate_exact():
    """An equispaced circle makes the k-NN graph an EXACT circulant ring
    lattice, whose growth dimension is 1 by structural identity rather than by
    fit -- which is what the third verdict is for."""
    t = np.arange(1200) * 2.0 * math.pi / 1200
    result = certify(np.c_[np.cos(t), np.sin(t)])
    assert result.verdict == Verdict.DEGENERATE_EXACT
    assert (result.d_lo, result.d_hi) == (1.0, 1.0)
    _assert_no_violation(result, 1.0, "equispaced circle")


@pytest.mark.parametrize("dimension", [4, 5, 6])
def test_high_dimension_abstains_rather_than_underestimating(dimension):
    """The one real defect this module found in itself.

    The shell arm collapses badly as dimension rises (Round-3 measured ~2.47 on
    a 3-D cube). At R^6 it reads ~3.3 while the correlation sum reads ~4.4, and
    an earlier `READOUT_DIVERGENCE_FACTOR` of 3.0 let that through as MEASURED
    [2.61, 5.06] against a truth of exactly 6.0 -- a calibration violation. The
    guard is that the two arms must agree to within one measured bias floor.
    v2 section 3.3 predicts UNDECIDED at D >= 4 and says that is the criterion
    working, not a regression.
    """
    points = np.random.default_rng(11).random((1600, dimension))
    result = certify(points)
    _assert_no_violation(result, float(dimension), f"uniform {dimension}-D")
    assert result.verdict == Verdict.UNDECIDED
    assert Signal.READOUT_DIVERGENCE in result.signals


def test_no_calibration_violation_across_exact_targets():
    """The whole criterion, swept. Abstaining is a pass; being confidently
    wrong is the only failure."""
    rng = np.random.default_rng(19)
    cases = [
        ("square", square(1200), 2.0),
        ("circle", circle(1200), 1.0),
        ("cube", rng.random((1200, 3)), 3.0),
        ("segment", np.c_[np.sort(rng.random(1200)), np.zeros(1200)], 1.0),
        ("cantor", cantor_dust(1200), 2.0 * math.log(2) / math.log(3)),
    ]
    for name, points, truth in cases:
        _assert_no_violation(certify(points), truth, name)


# --- housekeeping ----------------------------------------------------------


def test_dedupe_reports_what_it_dropped():
    base = square(200)
    points = np.vstack([base, base + 1e-12])
    kept, dropped, fraction = _dedupe(_as_array(points))
    assert dropped == 200
    assert len(kept) == 200
    assert fraction == pytest.approx(1.0)


def test_certify_is_deterministic():
    points = square(600)
    first, second = certify(points), certify(points)
    assert first.verdict == second.verdict
    assert first.d_lo == second.d_lo and first.d_hi == second.d_hi
    assert first.signals == second.signals
