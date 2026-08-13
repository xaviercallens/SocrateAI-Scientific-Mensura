---
name: poly-algebraic-calculus
description: Use the hypergraph rewriting / dimension-estimation toolkit (src/socrates/hypergraph/) rigorously — building point clouds, running convergence sweeps, comparing against the traditional baseline, and avoiding the specific failure modes a 10-problem physics benchmark already found. Use when asked to apply, extend, or benchmark the "Poly-Algebraic Calculus" module, not for the T-dual/W1 dual-scale work (a separate initiative — see the decisive-experiment skill for that).
---

# Poly-Algebraic Calculus: how to use it without repeating found bugs

`src/socrates/hypergraph/` is Tier C physics infrastructure with Tier B
code (see `docs/HYPERGRAPH_NOTES.md`): hypergraph rewriting, a shell-growth
dimension estimator, and a rule-search solver, built to test the Wolfram
Physics Project hypothesis computationally. `docs/POLY_ALGEBRAIC_BENCHMARK.md`
applied it to 10 known physics problems and found real bugs, real
limitations, and — after fixes — a real comparator against the traditional
approach. `docs/LL.md` distills the general lessons. **Read both before
extending this module** — several of the failure modes below cost real
time to diagnose the first time and are cheap to avoid the second.

## The module map

| Piece | File | What it does |
|---|---|---|
| Hypergraph core | `core.py` | Immutable hyperedge multiset; `ball()`/`adjacency()` (cached — see below) |
| Rewriting | `rewriting.py` | Pattern matching, single-history and multiway (branching) evolution |
| Dimension | `dimension.py` | Shell-growth volume estimator; `DimensionEstimate.degenerate` distinguishes a real fit from a trivial one |
| Point-cloud bridge | `pointcloud.py` | `knn_hypergraph()` — turns a trajectory into a graph; raises on near-duplicate points by default |
| Rule search | `polyalgebra.py` | `solve_for_rule()` — the literal "solve for an unknown bond" operation |
| **Traditional baseline** | `baseline.py` | The actual classical Grassberger-Procaccia correlation-sum method — NOT part of the toolkit, the thing to compare against |
| Comparison harness | `comparison.py` | Apples-to-apples: minimum points each method needs for stable convergence |

## Five failure modes the benchmark already found — check for these first

1. **Degenerate "perfect" fits (finding F1).** A k-NN graph on any smooth
   closed curve is an exact circulant ring lattice — `local_dimension`
   *cannot* return anything but dimension 1 with `r_squared=1.0` on it,
   regardless of whether anything was measured well. Before citing an
   `r_squared` value as evidence of fit quality, check
   `DimensionEstimate.degenerate` (or call `.is_genuinely_well_fit()`
   instead of `.is_well_fit()`). `degenerate_fraction()` reports what share
   of a sample hit this regime — a high value means the r² column you are
   looking at is mostly sentinel.

2. **Unconverged fractal estimates mistaken for measurements (F4/F4b).**
   A non-integer dimension estimate (chaotic attractors, anything without a
   closed-form target) needs a `dim vs n` sweep showing the estimate has
   plateaued, not a single point count. Both `baseline.minimum_points_for_target_accuracy`
   and `comparison.poly_algebraic_minimum_points` already enforce this
   (require the estimate to stay within tolerance at *every* larger n
   tested, not cross into tolerance once) — use them instead of a bare
   `mean_dimension()` call when the target is non-trivial.

3. **Duplicate-point corruption from repeated sampling (F3).** Sampling
   multiple periods of a closed orbit (or any repeated traversal) produces
   near-exact duplicate point clusters that corrupt the k-NN graph.
   `knn_hypergraph` raises `DuplicatePointsError` by default — do not
   reflexively pass `dedupe=True` to silence it without checking whether
   your sampling is genuinely producing repeats you should instead avoid at
   the source (single-period sampling, when the system is periodic).

4. **A detection threshold corrupted by the thing it detects.** If you
   build a new anomaly/scale detector on this data, do not calibrate it
   against a statistic the anomaly itself moves (the first
   duplicate-detector draft used local nearest-neighbour distance, which a
   duplicate *is*, and silently never fired). Calibrate against something
   structurally independent (`pointcloud.py` uses the bounding-box
   diagonal).

5. **"Better than traditional" needs the real traditional method.**
   `baseline.py` is a real, tested implementation of the classical
   approach, not a strawman — use `comparison.compare()` for any claim
   of this shape, and note that `ComparisonResult.compute_savings_fraction`
   is `None` (not a misleading number) whenever either method fails to
   converge. A "win" requires `poly_algebraic_wins=True`, which already
   encodes "both methods actually converged."

## Performance note

`Hypergraph.adjacency()` is cached (`functools.lru_cache`, safe because
`Hypergraph` is immutable/hashable) and `local_dimension` does one
incremental BFS rather than calling `ball()` per radius. A 3200-point
convergence sweep that took 136s before this fix takes ~5s after. If you
are about to write a loop that calls `ball()` or `local_dimension` many
times on the same hypergraph and it feels slow, check you are not on a
version predating this fix (`git log -- src/socrates/hypergraph/core.py`)
before assuming a new bottleneck.

## Running a benchmark or comparison rigorously

Follow `.claude/skills/decisive-experiment/SKILL.md`'s six-phase shape for
anything whose result will be cited (a new problem added to the benchmark, a
claim that a change improves accuracy or efficiency): measure → audit →
(targeted retry if needed) → re-audit → independent skeptic verification of
the audit itself. That skill's phase 6 ("verify the auditor") is not
optional here — the original benchmark's own audit needed correcting by a
second independent pass, three separate times across the W1 and hypergraph
work in this repository. Do not skip straight from "an agent measured it"
to "it is true."

When defining what counts as a "win" against the traditional baseline,
be explicit about which of the two honest metrics you mean:
- **Compute savings**: fewer points needed for equivalent, stably-converged
  accuracy (`comparison.compare()`, `compute_savings_fraction`).
- **Density-robustness / singularity-avoidance**: a local method (shell-growth
  on a k-NN graph) is structurally less sensitive to non-uniform sampling
  density than a global method (correlation-sum pools all pairwise
  distances) — testable directly on systems with a genuine close-approach
  or slow/fast turning point (CR3BP near a primary, Kepler near perihelion,
  a pendulum near its turning point), by comparing each method's error on
  the natural time-uniform (non-uniformly-dense) sample. Report both
  methods' actual errors side by side, not an assertion.

Neither metric is "the tool is right" — both are narrow, checkable claims
about one algorithm's behavior on one input, exactly as Tier B requires.
