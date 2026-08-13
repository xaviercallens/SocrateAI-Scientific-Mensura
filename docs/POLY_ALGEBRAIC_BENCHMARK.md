# Poly-Algebraic Calculus dimension estimator — physics benchmark ledger

Audit of a 10-problem benchmark run in which the shell-growth dimension
estimator (`socrates.hypergraph.dimension.local_dimension` / `mean_dimension`)
was applied, via the k-NN point-cloud bridge
(`socrates.hypergraph.pointcloud.knn_hypergraph`), to physics trajectories
whose dimension is known independently.

Ten problems were each run by a separate agent. This document is the **audit**,
not the self-report: every number below was re-derived or re-executed by the
auditor before being written down. Where an agent's self-report and the
re-run disagree, the re-run wins and the discrepancy is recorded.

**This ledger was itself independently verified after the audit pass, and
the verification found real errors in the audit — not in the 10 underlying
self-reports.** All 10 self-reported numbers and every `passed` verdict were
confirmed correct; the errors were in the auditor's own added analysis: an
over-generalized claim about a shared-module bug (later shown to be
value-specific, not universal), a wrong speculative reattribution of one
agent's `nan` diagnosis, and a factually incorrect claim about which
problems disclosed an unfiltered comparison — which had propped up one of
the document's two "genuine success" verdicts without the same scrutiny
applied elsewhere. **Every correction below is marked inline** with a
"**Correction (post-audit independent verification)**" note at the point it
applies, rather than silently edited in place — this is deliberate, matching
this repository's standing rule that failed gates (here, an audit's own
errors) are findings, not things to smooth over.

**Tier conventions** (inherited from `docs/HYPERGRAPH_NOTES.md` and
`docs/FINDINGS.md`): every measured number in this document is **Tier B** — a
statistic of a specific finite point-cloud graph, computed by code that does
what it says. Nothing here is Tier A (nothing is kernel-checked). Any
statement about what these results mean for hypergraph physics, emergent
spacetime, or "Poly-Algebraic Calculus" as a research programme is **Tier C**
and is confined to the explicitly-labelled section at the end.

---

## 0. Headline, stated honestly

**10 of 10 problems passed their stated tolerance, and all 10 pass/fail
arithmetic checks are correct. That headline is true and it is also
misleading.** Six of the ten passes come from a degenerate regime in which the
estimator is structurally incapable of returning any answer other than exactly
1.0 (finding **F1**), and **both** hardest cases — Lorenz and Rössler — are not
converged in point count, so their agreement with the literature values is
substantially fortuitous (findings **F4**, **F4b**).

The defensible summary is: **the pipeline is correct; two measurements
(Lissajous torus, Brownian motion) are the least-degenerate results and pass
under every variant checked, but were not put through the same
convergence/sampling-sensitivity scrutiny as the chaotic cases until an
independent verification pass did so post-hoc (§5, corrected); two (Lorenz,
Rössler) are unconverged and should not be cited as recovering a literature
correlation dimension; and six are plumbing tests that could not have
returned any answer other than the right one.** *(This paragraph was
corrected after an independent verification pass found the original draft's
"genuine successes" framing had not been checked to the same standard as the
rest of the document — see the correction note at the top of §5.)*

---

## 1. What the auditor actually did

| Check | Coverage | Result |
|---|---|---|
| Re-derived `abs(measured − known)` and the `≤ tolerance` verdict for all 10 | 10/10 | **0 mismatches** — no `passed` boolean was wrong |
| Re-ran the scratch script end-to-end and compared to the self-report | 6/10 (03, 04, 06, 07, 08, 09) | **6/6 reproduced to the reported digits** |
| Read the script for the remaining problems and checked stated constants | 4/10 (01, 02, 05, 10) | constants match the self-reports (minor prose slips, §4 F6) |
| Confirmed no agent modified shared modules | `git diff HEAD -- src/socrates/hypergraph src/socrates/solvers` | **empty** — all 10 complied with standing rule 6 |
| Independent reproduction of the R² artifact reported by problem 01 | plain 400-pt circle | **confirmed and found to be more general** (F2) |
| Independent test of problem 07's stated caveat | 1-period vs 3-period point cloud | **agent's explanation refuted** (F3) |
| Point-count convergence sweep on the fractal cases | Lorenz, Rössler | **neither is converged** (F4, F4b) |

The sweeps in F3, F4 and F4b are audit work that no individual agent performed;
they are the reason this ledger does not simply endorse the self-reports.

---

## 2. Summary table

Error = `|measured − known|`. r² is the mean over the well-fit subset
(`r_squared ≥ 0.9`, the threshold `mean_dimension` itself applies).
**Read the r² column together with F1** — a value of exactly `1.0000` is a
sentinel, not a fit-quality measurement. **Tolerance added in post-audit
correction** — the original table omitted it despite every `Passed` verdict
depending on it, and the four tolerances are not uniform (0.15 for the
cleanest exact cases, rising to 0.5 for the two unconverged chaotic cases) —
without this column a reader cannot verify the pass column, and cannot see
that the two chaotic passes ride on a tolerance more than 3× looser than the
`exact_periodic` cases.

| # | Problem | Category | Known dim | Src | Measured | Error | Tol | r² | Passed | Verification note (auditor) |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | Simple harmonic oscillator | exact_periodic | 1 | [E1] | 1.0000 | 0.0000 | ±0.15 | 1.0000† | ✅ | Real: max\|x−cos t\|=2.6e−6, energy drift 2.5e−7, closure at all 10 period boundaries. Also found F2. |
| 02 | Nonlinear pendulum (θ₀=2.0) | exact_periodic | 1 | [E1] | 1.0000 | 0.0000 | ±0.15 | 1.0000† | ✅ | Real: period vs exact elliptic integral 4K(sin²(θ₀/2)), rel err 2.0e−8; state closure 2.3e−4. |
| 03 | Kepler two-body (e=0.6) | exact_periodic | 1 | [E1] | 1.0000 | 0.0000 | ±0.15 | 1.0000† | ✅ | Real, re-run: E drift 1.9e−6, L drift 2.0e−14, period rel err 2.7e−6. All three gates. |
| 04 | Mars vs JPL Horizons (real data) | exact_periodic | 1 | [E2] | 1.0000 | 0.0000 | ±0.2 | 1.0000† | ✅ | Real, re-run: two-body vs real ephemeris rel RMS 3.60e−5 (gate 5e−3); raw-data smoothness checked. See F1a. |
| 05 | Lissajous 2-torus, ω₂/ω₁=√2 | exact_quasi_periodic | 2 | [E3] | 1.9433 | 0.0567 | ±0.3 | 0.9650 | ✅ | Real: closed form checked against DOP853 (3e−10); non-closure proved algebraically **and** numerically. Unfiltered mean 1.7521, r² 0.7204 — see §5 correction. |
| 06 | Planar Brownian motion | exact_stochastic | 2 | [T53] | 1.9428 | 0.0572 | ±0.3 | 0.9631 | ✅ | Real, re-run: MSD log-log slope 1.0514, r²=0.9996 (diffusive scaling confirmed). Unfiltered mean 1.7821, r² 0.8429, 55% of nodes discarded — see §5 correction. |
| 07 | CR3BP Lyapunov orbit near L1 | exact_periodic | 1 | [E1] | 1.0000 | 0.0000 | ±0.2 | 1.0000† | ✅ | **Strongest in suite**, re-run: closure \|Δpos\|=2.7e−13, \|Δvel\|=3.1e−13 vs 1e−8 gate; Jacobi drift 5.6e−16. Caveat text wrong — see F3. |
| 08 | Lorenz attractor | chaotic_literature | 2.05 | [GP83] | 1.9863 | 0.0637 | ±0.5 | 0.9682 | ⚠️ ✅ | Real, re-run: both lobes visited, 48 lobe switches, 46.8/53.2 balance, no settling. **But not converged in n — see F4.** |
| 09 | Rössler attractor | chaotic_literature | 2.01 | [GP83] | 1.9965 | 0.0135 | ±0.5 | 0.9563 | ⚠️ ✅ | Real, re-run: band/fold structure quantified, positive finite-time divergence rate ~0.037. **But not converged in n, and ~0.05 subsample-sensitive — see F4b (corrected).** Reporting slip, F6. |
| 10 | Driven damped pendulum, mode-locked | exact_periodic | 1 | [E4] | 1.0000 | 0.0000 | ±0.2 | 1.0000† | ✅ | Real: stroboscopic section converged to a point (spread 2.6e−10); closure 9.8e−10. Found the degeneracy behind F3. |

† = sentinel value produced by the `ss_tot == 0` branch of `_log_log_fit`, **not**
a measured goodness of fit. See **F1**.
⚠️ = passes its stated gate but the auditor does not regard the agreement as
load-bearing evidence. See **F4** / **F4b**.

**Sources (used exactly as given; none rounded or substituted):**

- **[E1]** Exact. A bound periodic orbit of a Hamiltonian/dissipative system is a
  smooth closed curve in phase space — topological/correlation dimension exactly 1.
- **[E2]** Task brief: a 182-day arc of the real perturbed Mars orbit is a smooth
  curve segment in 3D, dimension exactly 1 regardless of Jovian perturbation.
- **[E3]** Standard result: `x=cos(ω₁t)`, `y=cos(ω₂t)` with `ω₂/ω₁` irrational
  densely fills a 2-torus (Strogatz, *Nonlinear Dynamics and Chaos*).
- **[E4]** Standard result: a period-1 mode-locked limit cycle is by definition a
  smooth closed curve, dimension exactly 1 (Strogatz).
- **[T53]** S. J. Taylor (1953), *The Hausdorff α-dimensional measure of Brownian
  paths in n-space*, Proc. Camb. Phil. Soc. **49** — planar Brownian path has
  Hausdorff dimension almost surely exactly 2 (a theorem, not an estimate).
- **[GP83]** Grassberger & Procaccia (1983), *Measuring the strangeness of strange
  attractors*, Physica D **9**, 189–208 — D₂ = 2.05 ± 0.01 (Lorenz, σ=10, ρ=28,
  β=8/3); Rössler D₂ ≈ 2.01–2.02 at a=b=0.2, c=5.7.

---

## 3. Pass/fail and error, broken out by category

**These four categories differ enormously in difficulty and must not be
averaged into a single number without saying so.** An `exact_periodic` problem
asks the estimator to distinguish a curve from a non-curve; a
`chaotic_literature` problem asks it to resolve a non-integer value to two
decimal places. Reporting one MAE across both would be meaningless.

| Category | n | Passed | MAE | Max error | Mean r² | What it actually tests |
|---|---|---|---|---|---|---|
| `exact_periodic` | 6 | 6/6 | 0.0000 | 0.0000 | 1.0000† | Degenerate — see F1. Confirms the pipeline, not its accuracy. |
| `exact_quasi_periodic` | 1 | 1/1 | 0.0567 | 0.0567 | 0.9650 | Genuine 2D fill measurement |
| `exact_stochastic` | 1 | 1/1 | 0.0572 | 0.0572 | 0.9631 | Genuine, against a rigorous theorem |
| `chaotic_literature` | 2 | 2/2 | 0.0386 | 0.0637 | 0.9623 | Genuine fractal target; **weakest evidence — both unconverged** (F4, F4b) |
| **All** | **10** | **10/10** | **0.0191** | 0.0637 | — | — |

Two derived splits that are more informative than the overall MAE:

- **Non-chaotic MAE 0.0142 vs chaotic MAE 0.0386** — but the non-chaotic figure
  is dominated by six identically-zero errors from the degenerate regime, so
  this comparison flatters the estimator.
- **Non-degenerate cases only (05, 06, 08, 09): MAE 0.0478, n = 4.** This is
  the honest accuracy figure for this benchmark run — and even it is optimistic,
  because two of those four (08, 09) are unconverged in point count and their
  small errors are not reproducible under a change of n or of subsample stride
  (F4, F4b).

### Signed error — a downward bias at the point counts tested, not necessarily beyond them

| Problem | Signed error (measured − known) |
|---|---|
| 05 Lissajous torus | **−0.0567** |
| 06 Brownian motion | **−0.0572** |
| 08 Lorenz | **−0.0637** |
| 09 Rössler | **−0.0135** |

**Four out of four non-degenerate cases underestimate at the point counts
this run used.** Under a null hypothesis of unbiased error and no prior
directional expectation, this is a 1-in-8 outcome (two-tailed — the
probability all four agree in *either* direction), not 1-in-16 (which is the
one-tailed probability of this *specific* direction, wrongly stated in the
first draft of this section). Treating the direction as pre-specified is
defensible here, since boundary bias predicts underestimation a priori and
is already documented independently — but that assumption should be stated,
not left implicit.

**This framing also sits in tension with F4/F4b's own data and should not be
read as settled.** At the *largest* point counts tested, Lorenz (n=1600:
+0.0426 above the literature value) and Rössler (n=2400: +0.0621 above)
**both overestimate**, having crossed the literature value from below. So
"consistent downward bias" is a true description only of the specific,
under-converged n this benchmark could afford — not of the estimator's
limiting behavior, which the data in hand cannot characterize because
neither chaotic sweep reached a plateau.

F4/F4b establish that for the two chaotic cases the bias **at the tested n**
is **mostly finite-sample density, not an intrinsic property of the
method**: raising the point count drives both estimates up through their
literature values and past them. The bias is a resolution artifact, and it
is fixable by N5 rather than by
changing the estimator's mathematics.

---

## 4. Audit findings

### F1 — Six of the ten "passes" come from a degenerate regime (major, structural)

Problems 01, 02, 03, 04, 07 and 10 all report dimension **exactly** 1.0000 with
r² **exactly** 1.0000. That is not six independent confirmations; it is one
structural fact appearing six times.

Mechanism, verified directly on a plain 400-point circle rather than inferred:

```
node 0: vols=[1, 7, 13, 19, 25, 31, 37]   shells=[6, 6, 6, 6, 6, 6]
```

A k-NN graph built on a densely-sampled curve is a **circulant ring lattice**:
with k=6 every node links to 3 neighbours on each side, so the shell size is the
exact integer constant 6 at every radius. `_log_log_fit` then computes a slope of
exactly 0.0, and `dimension = slope + 1` returns **exactly** 1.0 — by
construction, for any input whatsoever that produces a constant shell sequence.

Worse for interpretation: with a perfectly constant shell sequence `ss_tot == 0`,
so r² is not computed at all. It comes from the fallback branch of

```python
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
```

**So on six of ten problems, `r_squared = 1.0` is a sentinel meaning "the shell
sequence was perfectly flat", not evidence that anything was fitted well.**
Standing rule 4 asked for fit quality to be reported rather than just the point
estimate; on these six problems the reported fit quality is vacuous, and none of
the six agents noticed this.

This does not make the six results *wrong* — a closed curve is 1-dimensional and
the estimator says 1. It makes them **weak**: they verify that
`knn_hypergraph` + `local_dimension` correctly turns a curve into a ring lattice
and reads 1 off it. **Correction, precision (post-audit independent
verification):** "the estimator has no freedom to be inaccurate" overstates
the mechanism. On problems 02, 03 and 07 a substantial fraction of nodes
were *not* well-fit (30/400, 5/40, 17/40 respectively), meaning their shell
sequences were not constant, and their **unfiltered** means genuinely
departed from 1.0 (0.9786 for 07, 1.0003 for 03) before the `r²≥0.9` filter
pulled the reported number back to exactly 1.0. The more precise statement:
it is the well-fit filter that selects specifically the degenerate,
constant-shell nodes, which then average to exactly 1.0 — the estimator
itself, run unfiltered, does show some spread on these six problems, it is
just discarded by the same mechanism that (correctly) discards genuinely
poor fits elsewhere. The substance of F1 is unchanged by this correction;
the mechanism description is now accurate.

**F1a — a concrete consequence at problem 04.** The Mars benchmark measured the
real JPL Horizons ephemeris *and* a simulated two-body leapfrog arc of the same
system, and got 1.0000 / r² 1.0000 / 144-of-162 well-fit for **both**, identical
to four decimals, despite the two point clouds being physically distinct at
3.6e−5 relative RMS (Jupiter's perturbation). The agent reported this honestly
and explained it correctly. The audit's addition: this is F1 in action — the
comparison had **zero discriminating power by construction**, so it should not be
cited as evidence that the estimator agrees between real and simulated data. It
would have returned the identical answer for any smooth curve.

### F2 — R² catastrophic cancellation fires at the module's *default* `max_radius` (major, actionable defect)

Problem 01 reported this and worked around it. The audit confirms it and finds
it is **more general and more serious** than that report implies.

For a perfectly constant shell sequence, `ss_tot` should be exactly 0. At some
fit lengths, floating-point rounding in `mean_y = sum(log_y) / n` leaves it at
~1e−31 instead, so the `ss_tot > 0` branch is taken and R² collapses to ~0 by
pure noise — while the fitted dimension remains correct to ~1e−32.

Reproduced across fit lengths for a constant shell of 6:

| fit points | ss_tot | r² |
|---|---|---|
| 2 | 0.0 | 1.0 |
| **3** | **1.48e−31** | **0.0** |
| 4 | 0.0 | 1.0 |
| 5 | 0.0 | 1.0 |
| **6** | **2.96e−31** | **0.0** |
| 7 | 0.0 | 1.0 |
| 8 | 0.0 | 1.0 |

Six fit points is exactly what `local_dimension`'s **default `max_radius=6`**
produces. Confirmed end-to-end on a plain circle:

```
max_radius=5 -> dim=1.0  r2=1.0
max_radius=6 -> dim=1.0  r2=0.0   <-- DEFAULT
```

**Therefore `mean_dimension(hg)` called at its own default `max_radius=6`
returns `nan` for *some* clean 1-manifold point clouds** — specifically, any
whose (near-)constant shell value happens to land on one of the fit lengths
and shell values where the cancellation fires. **Correction (post-audit
independent verification):** the first draft of this finding claimed this
fires for *any* clean 1-manifold point cloud. That is false. Sweeping shell
values 2–24 at fit-length 6 finds only shells 6, 17 and 18 trigger the
collapse; k=4 and k=8 on the same circle return `dimension=1.0, r²=1.0`
cleanly at the default `max_radius`. The defect is real and it does bite
this benchmark specifically (problems 01–04 and 07 all used k=6, landing
exactly on the bad value), but it is a value-specific floating-point
accident, not a universal failure of the estimator at its defaults.

**The claim that this is "very likely the true cause of problem 10's
`mean_dimension -> nan`" is also wrong and is retracted.** Problem 10 used
k=10 with `max_radius=5` — a constant shell of 10 at fit-length 6 does *not*
trigger the cancellation (verified directly), and problem 10's `nan` arose
from an irregular, non-constant shell sequence in its first (duplicate-
stacked) point cloud, where this mechanism cannot apply by construction.
Problem 10's own diagnosis (duplicate stacking, §F3 below) stands
uncontested; this ledger should not have second-guessed a correct,
carefully-verified agent finding with an untested guess.

Not fixed during the parallel run — standing rule 6 forbade modifying
`src/socrates/hypergraph/` while agents were reading it concurrently. **Fixed
as N1 immediately after this ledger's post-audit correction pass**: `_log_log_fit`
now compares `ss_tot` against a relative-scale tolerance
(`ss_tot <= 1e-24 * max(1, sum(y*y))`) instead of exact zero, returning
`r_squared=1.0` for a sequence that is constant to within floating-point
noise. Verified against every case in F2's table above (shells 6, 17, 18 at
fit-lengths 3 and 6, plus shells 2/4/5/7/8/10 as negative controls — all now
return `r_squared=1.0` where they previously collapsed to `0.0`), and against
the concrete end-to-end case (a 400-point circle at k=6, problem 01's exact
configuration), now well-fit at `max_radius=6` where it previously was not.
Regression tests added:
`test_log_log_fit_r_squared_does_not_collapse_on_constant_shells` and
`test_circle_at_k6_is_well_fit_at_the_default_max_radius` in
`tests/test_hypergraph.py`. Full suite (29 tests across the hypergraph
module) green after the fix.

### F3 — Problem 07's caveat gives the wrong mechanism (major; verdict unaffected)

Problem 07 reported 23/40 well-fit nodes on what is a perfectly smooth closed
curve, and attributed the shortfall to "the orbit's curvature extrema … a known
k-NN-graph-on-curved-manifold effect".

The auditor tested this by re-running problem 07's own pipeline with the point
cloud drawn from **one** period instead of three, changing nothing else:

| Point cloud | Well-fit | Mean dim (well-fit) | Unfiltered dim | Unfiltered r² |
|---|---|---|---|---|
| 1 period, 400 pts | **37/40** | 1.0000 | 0.9994 | 0.9270 |
| 3 periods, 400 pts | **23/40** | 1.0000 | 0.9786 | 0.7715 |

Curvature is identical in both runs. The well-fit rate is not. **The real cause
is multi-period duplicate stacking** — sampling N periods of a closed orbit
produces N near-exact copies of the same curve, so `knn_hypergraph` wires each
point to its own duplicates instead of its along-curve neighbours. This is
precisely the mechanism problem 10 diagnosed independently and problem 01 hit in
its aliasing form.

Problem 07's headline number and its pass are unaffected (1.0000 either way, and
its solver verification is the most rigorous in the suite). Its *explanation* is
wrong and is corrected here rather than left in the record.

**Problems 02 (3.2 periods, 370/400 well-fit) and 03 (3 periods, 35/40) sampled
multiple periods too** and their sub-100% well-fit rates are most plausibly the
same artifact. Neither identified it. Problems 01 and 10 both hit it, diagnosed
it correctly, and worked around it — the two best pieces of methodological work
in the run.

### F4 — Neither chaotic result is converged in point count (major; the weakest results in the suite)

This is the finding that most changes how the benchmark should be read.
Standing checklist item 5 asks whether n is adequate for a fractal estimate. For
Lorenz at n=800 the answer is **no**, and F4b below shows the same for Rössler
at n=1516. Auditor sweep on Lorenz, holding k=15,
`max_radius=5`, and the 60-node sample fixed, varying only point count:

| n_points | well-fit | dim (well-fit) | r² (well-fit) | dim (unfiltered) |
|---|---|---|---|---|
| 200 | 7/60 | 1.5084 | 0.9277 | 1.4389 |
| 400 | 31/60 | 1.7474 | 0.9549 | 1.7086 |
| **800** *(reported)* | 46/60 | **1.9863** | 0.9682 | 1.9071 |
| 1200 | 46/60 | 1.9935 | 0.9692 | 1.9129 |
| 1600 | 45/60 | 2.0926 | 0.9759 | 1.9868 |

The estimate rises monotonically and **crosses the literature D₂ = 2.05 between
n=1200 and n=1600**. It is still climbing at the largest n tested. The reported
"absolute error 0.0637" therefore describes where a still-moving quantity
happened to be at one point count, not the estimator's accuracy: **point count
alone moves the answer by ~0.58 across the tested range and by ~0.11 between the
reported n and n=1600.**

Problem 08 flagged the concern qualitatively and honestly ("closer to a local
box-counting estimate over a narrow radius range than a true asymptotic
correlation-dimension measurement"), which is to its credit — but it did not run
the sweep that would have quantified it, and its self-reported error bar is
consequently far too tight. **The agreement with 2.05 at n=800 is substantially
fortuitous, and this ledger does not treat it as evidence that the shell-growth
method recovers the Lorenz correlation dimension.**

**F4b — Rössler is not converged either, and shows subsample sensitivity on top.**
The same sweep on problem 09's pipeline (k=10, `max_radius=5`, 40-node sample):

| n_points | well-fit | dim (well-fit) | r² (well-fit) | dim (unfiltered) |
|---|---|---|---|---|
| 400 | 10/40 | 1.8076 | 0.9369 | 1.7497 |
| 800 | 22/40 | 1.9653 | 0.9543 | 1.8859 |
| **1516** *(reported n)* | 36/40 | **2.0396** | 0.9649 | 2.0176 |
| 2400 | 30/40 | 2.0721 | 0.9591 | 2.0129 |

Same monotone climb, crossing the literature D₂ = 2.01 between n=800 and
n=1516 and continuing past it — so problem 09's headline error of 0.0135, the
smallest in the entire benchmark, is likewise an artifact of where a rising
curve was sampled.

There is a second effect visible here, though the first draft's comparison
mixed two different node sets and overstated its precision.
**Correction (post-audit independent verification):** the auditor's
stride-65 value (2.0396) was compared directly against "the benchmark
script's subsample (stride 66) gives 1.9965" — but 1.9965 is the
`mean_dimension` wrapper's value over 40 nodes, while the script's own
manual 41-node list (the thing actually comparable to the auditor's manual
sweep) gives 1.9885. Apples-to-apples the spread is **~0.05, not 0.043** —
this makes the finding slightly *stronger*, not weaker, but the original
comparison committed exactly the node-set-mixing error that F6 separately
flags as a pattern in this benchmark. Matching stride-65 to the script's
exact n=1516 also requires truncating a 1,539-point candidate set, which
shortens the covered time window by ~1.5% — a reproduction of the truncated
run gives 2.0379 (37/41 well-fit), consistent with but not identical to the
originally reported 2.0396 (36/40). **A subsample spread of roughly 0.05,**
on a reported error of 0.0135, is the corrected finding. Neither chaotic
problem reported an uncertainty that accounts for this.

Taken together, F4 and F4b mean **neither chaotic result should be cited as a
successful recovery of a literature correlation dimension.** Both are consistent
with the literature values in the loose sense that a rising, unconverged
estimate passes through them; neither demonstrates that the shell-growth method
converges to them.


### F5 — Systematic downward bias on every non-degenerate case (moderate)

See the signed-error table in §3: 4/4 underestimates. F4 shows finite-sample
density is a major contributor, and the direction matches the boundary-bias
effect the module's own test suite already pins. This is consistent with, not
an addition to, `HYPERGRAPH_NOTES.md`'s known limitations — but this run is the
first time it has been observed on continuous physical trajectories rather than
on synthetic lattices and cubes.

### F6 — Minor reporting inconsistencies (none change any verdict)

Recorded for completeness, because standing rule 5 treats unreported deviations
as the problem rather than the deviations themselves.

- **09 Rössler:** the reported dimension 1.9965 comes from the `mean_dimension`
  wrapper (40 nodes); the reported r² 0.9563 comes from the script's own 41-node
  manual list, whose well-fit mean dimension is 1.9885. **The reported dimension
  and the reported r² are computed over slightly different node sets.** Effect on
  the verdict: none (both are ~0.02 from 2.01 against a 0.5 tolerance).
- **09 Rössler:** the scope note says "41 nodes actually sampled by the
  stride-based sampling in `mean_dimension`". `mean_dimension` truncates to 40
  (`nodes[::step][:samples]`); the 41 is the script's own list.
- **06 Brownian:** self-report describes 31 sampled nodes and "14 of the 31"
  well-fit, but the headline number came from `mean_dimension(samples=30)`. Both
  routes return 1.9428, so this is cosmetic.
- **05 Lissajous:** the self-report gives the parameter sweep as `T ∈ {200…2000}`;
  the script comment says `T=600…2000`. The sweep is also, unavoidably, a
  selection step — mitigated by the fact that the reported range 1.77–2.16 lies
  entirely inside the ±0.3 tolerance, so no configuration choice could have
  manufactured the pass.
- **02 Pendulum:** justifies `min_radius=2` by "well-fit fraction from 8/60 to
  370/400" — different denominators (60-node vs 400-node samples), an
  apples-to-oranges comparison. The underlying diagnosis (an r=1 transient shell
  of 8 before the flat 6,6,6,6 tail) is separately sound and was verified from
  printed shell sequences. Note also that `min_radius=2` with `max_radius=6`
  leaves 5 fit points, which is why problem 02 silently dodged **F2**.
- **03 Kepler:** `solver_description` says 6 orbital periods integrated; the
  point cloud is drawn from a 3-period window. Imprecise, not incorrect.
- **03 Kepler:** measured `(x,y)` position only, not the 4D phase space. This is
  disclosed and well-argued in the self-report, and is the right choice for the
  stated claim.

---

## 5. Honest assessment: where the estimator did well and where it struggled

### Where it did well

**Space-filling 2D targets, with a caveat this ledger did not originally
apply consistently.** The two least-degenerate measurements are the
Lissajous 2-torus (1.9433, r² 0.9650) and planar Brownian motion (1.9428, r²
0.9631) — both within 0.06 of a value that is *exact* rather than empirical,
and neither is fractal-with-fine-structure, which is plausibly why they
behaved better than Lorenz and Rössler. **Correction (post-audit independent
verification):** the first draft called these two "genuine successes" and
called the Brownian result "the most impressive in the suite" without
running the same convergence/sensitivity checks it demanded of the chaotic
cases. Checked post-hoc: 06's unfiltered mean is 1.7821 (r² 0.8429, 55% of
nodes discarded by the filter — see the correction above), and 05 is
measurably sampling-sensitive at a comparable magnitude to the Rössler
stride effect this ledger treats as disqualifying (§F4b): resampling the
same closed-form trajectory at spacing 3.6072 instead of 3.6000 — a 0.2%
change — moves the estimate from 1.9433 to 1.9883 and drops the well-fit
count from 24/40 to 17/40. Both problems still pass their stated tolerance
under every variant checked. But **"genuine success" is too strong given
neither received the convergence/stride sweep this ledger insists is
necessary before trusting 08 or 09** — the honest label is that 05 and 06
are the two *least degenerate* results, not that they are confirmed.

**Robustness to input provenance.** Problem 04 fed the estimator real
downloaded JPL Horizons ephemeris data, and problem 06 fed it a stochastic path
with no smooth generator at all. Neither broke the pipeline. (Per F1a the Mars
result carries little information, but the plumbing held.)

**The `is_well_fit` filter behaves as designed where it was actually checked
— which is fewer problems than this ledger first claimed.** **Correction
(post-audit independent verification):** the first draft asserted that 03,
05, 07, 08 and 09 all reported both filtered and unfiltered means, and that
every agent using the filter disclosed the unfiltered number. That is
factually wrong: only **02, 03, 07 and 08** compute an unfiltered mean in
their scripts. **04, 05, 06 and 09 filtered without ever reporting the
unfiltered comparison**, and 02 — which did report it — was omitted from the
original list.

Where the unfiltered comparison actually exists (02, 03, 07, 08), it
supports the intended reading: the filter moves the estimate toward the true
value and the unfiltered mean stays within tolerance anyway, so the filter is
doing real work rather than manufacturing the pass. **For 05, 06 and 09 this
was never checked by the agent, and the first draft of this ledger did not
check it either.** Computed post-hoc for this correction:

- **06 Brownian motion: unfiltered mean 1.7821 (r² 0.8429)** against the
  reported filtered 1.9428 — **17 of 31 nodes (55%) were discarded**, the
  most aggressive filtering in the entire suite. The result still passes
  (error 0.218 against a 0.3 tolerance either way), but §"Where it did well"
  below called this problem "the most impressive in the suite" without ever
  computing the number that would test that claim. That framing is corrected
  below.
- **05 Lissajous torus: unfiltered mean 1.7521 (r² 0.7204)** on the script's
  own graph (24/40 well-fit) — a similarly large gap between filtered and
  unfiltered that was not previously disclosed.

Both remain passes against their stated tolerance either way. The point is
narrower: this ledger's own claim to have re-derived every number did not,
in its first draft, extend the same scrutiny to 05/06 that it correctly
insisted on for 08/09, and one of the two problems it called "genuine
successes" turns out to lean on its filter at least as heavily as the
chaotic cases it treats as unconverged. **N3 is extended below to cover 05
and 06, not only 08 and 09.**

### Where it struggled

**Fractal targets, because of sampling density (F4, F4b).** The headline
weakness, and it applies to *both* chaotic problems, not just the one the
agents flagged. On a genuinely non-integer target the estimate is a strong
monotone function of point count over exactly the range that
`knn_hypergraph`'s O(n²) construction makes affordable, and it is additionally
sensitive at the ~0.04 level to which points a stride happens to select.
`HYPERGRAPH_NOTES.md` already warns that the estimator "is validated on regular
lattices only"; this run supplies the quantitative version of that warning for
fractal sets. The radius window is the binding constraint: the
attractor's transverse thickness means shells leave the local scaling regime
after only 2–5 hops, so the fit spans well under one decade of scale, whereas
Grassberger & Procaccia's original correlation-integral method spans several.

**Multi-period and aliased sampling (F3).** The k-NN bridge has a sharp failure
mode nothing in the module documents: feeding it repeated traversals of a
periodic orbit produces near-duplicate points at ~1e−9 separation, and the graph
wires each point to its own duplicates rather than along the curve. Observed
severity ranges from mild (07: well-fit rate 37/40 → 23/40, headline unaffected)
to total (10: `mean_dimension → nan`, and a spurious ~0.86 reading that the agent
correctly identified as a false near-pass; 01: dimension corrupted down to
0.03–0.34 under integer-divisor aliasing). This is squarely an instance of the
**"graph distance ≈ geodesic distance" approximation failing**, which
`pointcloud.py`'s own docstring names as its central caveat — duplicate points
make Euclidean nearest-neighbour ranking say nothing about position along the
manifold.

**Boundary bias, exactly as already documented.** Problem 04 had to trim 10
points from each end of an open arc; problem 05 explicitly attributes its
lowest-r² nodes (0.17–0.64) to Lissajous turning points; problem 08's poorer fits
cluster the same way. This is the same effect as
`test_boundary_points_of_a_cube_underestimate_dimension` and needs no new
explanation — but it is now confirmed on curves and attractors, not only cubes.

**Its own fit-quality diagnostic (F1, F2).** The estimator cannot currently tell
you that a perfect answer is perfect: on flat shell sequences r² is a sentinel
(F1), and for specific shell values (6, 17, 18 confirmed at the default
`max_radius=6` fit length — not universally) it inverts to 0.0 and rejects the
node instead (F2). Standing rule 4 says "a dimension estimate with poor fit
quality is not evidence of anything" — the corollary this run exposes is that
**an r² of 1.0000 from this estimator is not evidence of anything either.**

---

## 6. Next steps — specific to what this run revealed

Ordered by the strength of the evidence behind them. None are generic advice;
each traces to a numbered finding above.

**N1 — ✅ DONE. Fix the R² computation for near-constant shell sequences (from
F2).** Implemented immediately after this ledger's correction pass: `_log_log_fit`
now compares `ss_tot` against a relative tolerance
(`ss_tot <= 1e-24 * max(1, sum(y*y))`) instead of exact zero, returning
`r_squared=1.0` for a sequence that is constant to within floating-point
noise. Verified against F2's table (shells 6, 17, 18) and against negative
controls that were already fine (2, 4, 5, 7, 8, 10). Two regression tests
added in `tests/test_hypergraph.py`; full 29-test hypergraph suite green.
See the updated F2 section above for the exact verification.

**N2 — Distinguish "perfect fit" from "degenerate fit" in the return type (from
F1).** `DimensionEstimate` should expose whether the fit was over a constant
shell sequence — e.g. a `degenerate: bool` field, or an r² of `None` rather than
1.0. Six of ten benchmark problems reported a fit-quality number that means
nothing, and no agent caught it, which is strong evidence the current API invites
the mistake.

**N3 — Run the point-count convergence sweep as part of any non-degenerate
claim, not only fractal ones (from F4, F4b, and the §5 correction).** Each
sweep took well under an hour and completely changed the interpretation of
both chaotic results. Any future non-integer dimension measurement from this
estimator should report a `dim vs n` curve **and** a subsample/sampling-grid
sensitivity, not a single number. If the curve has not flattened, the honest
output is a bound, not an estimate. Concretely: repeat Lorenz at n =
2000–4000 and Rössler at n = 3000–6000 to find where they plateau, which
requires N5. **Extended scope (from the §5 correction): the same sweep is
owed to problems 05 and 06**, both of which showed a comparable-magnitude
shift under a small change to their sampling grid but were never checked
against varying point count the way 08/09 were.

**N4 — ✅ DONE. Detect and reject near-duplicate points in `knn_hypergraph`
(from F3).** Implemented as `DuplicatePointsError` plus a `dedupe=True`
option: duplicate *clusters* (not just pairs — a 3-period orbit produces
clusters of 3, which a pairwise-only check can miss) are found via
`scipy.spatial.cKDTree.query_pairs` + connected components, thresholded
against the point cloud's **bounding-box diagonal**, not the local
nearest-neighbour spacing. That distinction mattered: an initial
implementation using median nearest-neighbour distance as the scale
reference failed silently, because a duplicated point's "nearest neighbour"
*is* its own duplicate — the corruption you are trying to detect corrupts
the very statistic used to calibrate the detector. Verified on the exact
reproduction case (300 points, each period-tripled at ~1e-9 offset): raises
naming all 300 size-3 clusters; `dedupe=True` recovers exactly 300 nodes and
dimension 1.0. Verified for no false positives on a 5000-point densely
(but genuinely) sampled circle. Four new tests in
`tests/test_hypergraph_pointcloud.py`.

**N5 — ✅ DONE. Replace the O(n²) neighbour search (enabling N3).** Rebuilt
`knn_hypergraph` on `scipy.spatial.cKDTree` (O(n log n) construction).
Verified directly: 15,000 points in ~1.2s, a scale the O(n²) brute-force
construction made impractical (the concrete cost F4 measured). This is the
prerequisite for N3's convergence sweeps to reach the point counts (Lorenz
2000–4000, Rössler 3000–6000) needed to find where the chaotic estimates
actually plateau, rather than being capped by the neighbour-search cost
itself.

**N6 — Widen the fit window before trusting fractional slopes (from F4/§5).**
Every fractal fit here spanned radii 1–5, well under a decade of scale. Investigate
whether a denser graph (larger n, larger k) extends the usable radius range
before shell saturation, and report the achieved decades of scale alongside any
fractional dimension.

**N7 — Re-run problems 02, 03 and 07 with single-period point clouds (from F3).**
Cheap, and would confirm whether their sub-100% well-fit rates are entirely the
duplicate-stacking artifact. Expected outcome from the F3 experiment: well-fit
rates rise substantially, headline dimensions unchanged at 1.0000.

**N8 — ✅ DONE (round 3, §11.1). Fix near-constant (not just exactly-constant)
shell sequences collapsing R² (from R2-F4, §9.5/§10.3).** N1 fixed the *exact*-constant case
(`ss_tot` within float noise of zero). The improvement loop found a distinct,
still-unfixed defect: shell sequences that are *nearly* but not exactly
constant produce an unreliable R² that does not trigger N1's relative-tolerance
branch, so a genuinely well-fit node can be discarded as poorly fit (or vice
versa). Anchored to a real, reproduced case: problem 01 at n=200, k=6,
`max_radius`=6 has shells `(6,5,5,6,6,6)`, giving dimension 1.0333 (error
0.033 — well inside a reasonable tolerance) but discarded at R²=0.0550. A
synthetic hypergraph built to have exactly those shells reproduces
`dimension=1.0333, R²=0.0550` to the digit (§9.5), and §10.3 reproduced all
six of §9.5's table rows independently. This is the one finding in both
improvement-loop rounds that is a real, actionable defect in shipped code
(`dimension.py`'s `_log_log_fit`) rather than a defect in an experiment's
methodology — fix by gating on relative shell spread in addition to R²,
with known-answer regression tests before any re-run of problem 01 (§10.8).

**Closed in round 3, after two refuted attempts (§11.1):** a per-node
CV+slope-bound gate looked correct but had a real, measured collateral-
damage defect; a follow-up minimum-fit-length gate fixed that case but
silently changed other problems' recorded numbers without disclosing it.
The fix that finally held is a graph-level consensus rule
(`near_degenerate_fraction`) backed by a proof that the pooled estimate is
structurally confined to `[0.75, 1.25]` whenever it fires — independently
verified twice. See §11.1 for the full three-round account.

---

## 7. Reproducibility

Activate the venv first: `source /home/xavkal/socrates-project/.venv/bin/activate`

| # | Problem | Script |
|---|---|---|
| 01 | Harmonic oscillator | `scripts/hypergraph_benchmark/01_harmonic_oscillator.py` |
| 02 | Nonlinear pendulum | `scripts/hypergraph_benchmark/02_nonlinear_pendulum.py` |
| 03 | Kepler orbit | `scripts/hypergraph_benchmark/03_kepler_orbit.py` |
| 04 | Mars vs JPL Horizons | `scripts/hypergraph_benchmark/04_mars_horizons.py` |
| 05 | Quasi-periodic torus | `scripts/hypergraph_benchmark/05_quasiperiodic_torus.py` |
| 06 | Brownian motion | `scripts/hypergraph_benchmark/06_brownian_motion.py` |
| 07 | CR3BP Lyapunov orbit | `scripts/hypergraph_benchmark/07_restricted_three_body.py` |
| 08 | Lorenz attractor | `scripts/hypergraph_benchmark/08_lorenz_attractor.py` |
| 09 | Rössler attractor | `scripts/hypergraph_benchmark/09_rossler_attractor.py` |
| 10 | Driven damped pendulum | `scripts/hypergraph_benchmark/10_driven_pendulum_periodic.py` |

Determinism: 06 is seeded (`default_rng(42)`, single realization — no multi-seed
robustness check was run, which the self-report discloses). All others are
deterministic given their initial conditions. Problem 04 reads the cached
ephemeris `data/horizons_mars_2025.npz` and does not refetch.

The audit's own sweeps (F3, F4, F4b) were run as throwaway scripts in the
session scratchpad and are not committed; all are fully specified by the tables
above — F3 re-runs `07_restricted_three_body.py`'s pipeline with
`n_periods=1.0`; F4 re-runs `08_lorenz_attractor.py`'s pipeline (k=15,
`max_radius=5`, 60-node sample) varying only the point count; F4b does the same
for `09_rossler_attractor.py` (k=10, `max_radius=5`, 40-node sample). Note that
F4b's n=1516 row does **not** reproduce problem 09's headline number, because
the auditor's stride (65) differs from the script's (66) — that discrepancy is
itself the subsample-sensitivity finding, not a reproduction failure.

Shared-module integrity: `git diff HEAD -- src/socrates/hypergraph/
src/socrates/solvers/` is **empty**. No agent modified shared code, and the only
untracked addition is `scripts/hypergraph_benchmark/`.

---

## 8. Tier C — interpretation, explicitly not load-bearing

Everything above is Tier B. The following is **Tier C** and nothing in this
repository should depend on it:

- This run does **not** validate "Poly-Algebraic Calculus" as a physical theory,
  and does not bear on the Wolfram Physics Project's hypothesis. It measures a
  graph statistic on point clouds sampled from ODE solutions. The concept-level
  triage in `HYPERGRAPH_NOTES.md` §"Triage of the four brainstormed concepts"
  stands unchanged.
- The most that can be said is the Tier C sentence: *a shell-growth dimension
  estimator calibrated on discrete lattices returns plausible values when applied
  to k-NN graphs built from continuous physical trajectories, with a consistent
  downward bias and a strong sensitivity to sampling density on fractal sets.*
  That is a statement about an algorithm, not about space, time, or emergence.
- In particular, the six exact-1.0000 results must not be cited as evidence that
  the estimator is accurate. Per F1 they are evidence that it is correctly
  plumbed.
- No claim is made that this estimator reproduces Grassberger–Procaccia
  correlation dimension. F4 and F4b show it does not, in the asymptotic sense
  G&P intended, at the point counts affordable here — both chaotic estimates
  were still climbing at the largest n tested.
- The two chaotic "passes" are passes against a generous ±0.5 tolerance, not
  confirmations of D₂. Anyone citing this ledger for a claim about fractal
  dimension estimation should cite F4/F4b, not the summary table.

---

# Round 2, phase 1 — audit of the "beat the traditional method" run (2026-08-13)

Ten agents re-ran the benchmark against a stated goal: demonstrate, on **at
least 8 of 10** problems, that the shell-growth ("Poly-Algebraic") estimator
beats the classical Grassberger–Procaccia baseline
(`socrates.hypergraph.baseline`) by either

- **(a)** >10% fewer points for equivalent *stably converged* accuracy
  (`ComparisonResult.compute_savings_fraction > 0.10` with
  `poly_algebraic_wins` True), or
- **(b)** measurably smaller error (>30%) on a naturally density-varying
  time-uniform sample — the checkable form of "singularity avoidance".

This section is the audit of that run. As in the round-1 ledger, every number
was re-derived or re-executed before being written down. All measured numbers
below are **Tier B**.

## 9.0 Headline

**Reported: 5 of 10 wins. Verified: 2 of 10.** The 8-of-10 goal was not met,
and was not close to being met.

The two verified wins (problems 02 and 10) are genuine, parameter-robust
criterion-(a) compute-savings wins. **All four criterion-(b)
density-robustness wins fail a control the round-1 discipline should have
required, and none of them survives.** One criterion-(a) win (problem 01)
is contingent on a non-default hyperparameter and disappears at the module
default.

| # | Problem | Reported | Verified | Basis |
|---|---|---|---|---|
| 01 | Harmonic oscillator | **win** (a) | **contested** | win only at `max_radius`≤5; tie at the default 6 (R2-F3) |
| 02 | Nonlinear pendulum | **win** (a)+(b) | **WIN — (a) only** | (a) holds in 18/20 (k, max_radius) combos; (b) fails control (R2-F1) |
| 03 | Kepler orbit | **win** (b) | **no** | (b) fails control; no compute savings (tie at n=100) |
| 04 | Mars / Horizons | no | **no** ✓ | tie at n=100; correctly reported |
| 05 | Quasiperiodic torus | no | **no** ✓ | traditional never converges; correctly *not* claimed (R2-F5) |
| 06 | Brownian motion | no | **no** ✓ | traditional never converges; correctly *not* claimed (R2-F5) |
| 07 | CR3BP (close lunar approach) | **win** (b) | **no** | (b) fails control decisively, sign inverted (R2-F1) |
| 08 | Lorenz | no | **no** ✓ | poly needs 4× *more* points; honest loss |
| 09 | Rössler | no | **no** ✓ | poly needs 2× *more* points; honest loss |
| 10 | Driven pendulum | **win** (a)+(b) | **WIN — (a) only** | (a) holds in 20/20 combos; (b) fails control (R2-F1) |

**Arithmetic audit: all 10 self-reports are internally consistent.** Every
`poly_algebraic_wins` boolean matches its two `min_n` values, every
`compute_savings_fraction` equals `1 − poly_n/trad_n` (including the negative
values −3.0 and −1.0 for problems 08/09 and the `None`s for 05/06), and every
claimed ">=30% error reduction" is arithmetically correct on the two error
numbers reported. **No agent fabricated or miscomputed a number.** The failures
found below are failures of experimental design and of one hyperparameter
choice, not of honesty or arithmetic.

## 9.1 What the auditor did

- Re-ran problems **01, 02, 07, 10** end-to-end from their committed scripts.
  All four reproduced their reported numbers exactly (`poly_min_n`,
  `trad_min_n`, savings fraction, and both density error figures).
- Independently re-derived the nonlinear pendulum's single-period trajectory
  from scratch (leapfrog, exact period 4·K(m) = 8.3497529269, closure 2.7e−10)
  rather than importing the agent's cloud.
- Built the **control experiment** the density-robustness claims lacked
  (§9.2), for problems 02, 07 and 10.
- Swept `max_radius` ∈ {2,3,4,5,6} on problem 01 and (k, `max_radius`) ∈
  {4,6,8,10} × {3,4,5,6,8} on problems 02 and 10.
- Re-checked all 10 for the three round-1 self-deception patterns plus the
  new fourth pattern (winning because the *traditional* method failed).

Scripts: `/tmp/.../scratchpad/audit_controls{,2,3,4}.py` (scratch, not
committed — they import from `src/` and the round-2 scripts, and modify
nothing).

## 9.2 R2-F1 — All four density-robustness wins fail a control (major; four verdicts overturned)

Problems 02, 03, 07 and 10 each claimed criterion (b) by computing, on the
natural time-uniform sample, `poly |error|` vs `traditional |error|` and
observing poly's error was 100% smaller (poly returned exactly 1.0000 in every
case). That comparison **cannot distinguish the hypothesis from two competing
explanations**, because it has no control arm:

1. Is poly's zero error *robustness to density*, or is poly pinned to exactly
   1.0000 on **any** closed curve regardless of density (the F1 ring-lattice
   sentinel)?
2. Is the traditional method's error *caused by the density variation*, or is
   it ordinary finite-n Grassberger–Procaccia bias that would be present on a
   perfectly uniform sample too?

The control resolves both: build a **density-uniform** version of the *same
curve* at the *same n* (resampled evenly in arc length), and run **both**
methods on **both** samples. The claim requires
`trad_err(natural) >> trad_err(uniform)` and `poly_err` unchanged.

Results (k=6, `max_radius`=6, true dimension 1.0; "ratio" is the p90/p10 local
arc-length step ratio, 1.00 = uniform):

**Problem 02 — nonlinear pendulum**

| n | ratio | poly nat | trad nat | **err nat** | ratio | poly uni | trad uni | **err uni** |
|---|---|---|---|---|---|---|---|---|
| 500 | 1.80 | 1.0000 | 1.0667 | 0.0667 | 1.00 | 1.0000 | 1.0633 | 0.0633 |
| 1000 | 1.79 | 1.0000 | 1.0319 | 0.0319 | 1.00 | 1.0000 | 1.0356 | 0.0356 |
| 2000 | 1.79 | 1.0000 | 1.0163 | 0.0163 | 1.00 | 1.0000 | 1.0217 | 0.0217 |
| 4000 | 1.79 | 1.0000 | 1.0087 | 0.0087 | 1.00 | 1.0000 | 1.0123 | 0.0123 |

Traditional error is worse on the density-varying sample in **1 of 4** n;
median error ratio natural/uniform = **0.84×** — i.e. the traditional method is
*slightly more accurate* with the density variation present.

**Problem 07 — CR3BP, 5.5× density ratio, closest approach 0.0135 (~3 lunar radii)**

| n | ratio | poly nat | trad nat | **err nat** | ratio | poly uni | trad uni | **err uni** |
|---|---|---|---|---|---|---|---|---|
| 1000 | 5.49 | 1.0000 | 1.0171 | 0.0171 | 1.00 | 1.0000 | 1.0192 | 0.0192 |
| 2000 | 5.50 | 1.0000 | 1.0063 | 0.0063 | 1.00 | 1.0000 | 1.0183 | 0.0183 |
| 5000 | 5.51 | 1.0000 | 1.0001 | 0.0001 | 1.00 | 1.0000 | 1.0127 | 0.0127 |
| 10000 | 5.51 | 1.0000 | 0.9982 | 0.0018 | 1.00 | 1.0000 | 1.0090 | 0.0090 |
| 20000 | 5.51 | 1.0000 | 0.9972 | 0.0028 | 1.00 | 1.0000 | 1.0071 | 0.0071 |

Traditional error is worse on the density-varying sample in **0 of 5** n;
median ratio = **0.22×**. **The sign of the effect is inverted**: on the
sharpest close-approach test in the whole suite, the traditional method is
**4.5× more accurate** *with* the density variation than without it.
Reproduced on a 5×-finer base trajectory (200k steps) with identical results,
so this is not an interpolation artifact of the control resampling.

**Correction (post-audit independent verification, §10.10):** the "4.5×"
headline does not follow from the table printed directly above it. The five
per-row ratios (err nat / err uni) are 0.0171/0.0192=0.89, 0.0063/0.0183=0.34,
0.0001/0.0127=0.008, 0.0018/0.0090=0.20, 0.0028/0.0071=0.39 — **median 0.34,
i.e. ~2.9×**, not 4.5×. An independent CR3BP re-derivation (own shooting
solution, 21296-point single period, arc-length control implemented
independently) reproduces the natural-arm column to 4 decimals and gives
median 0.35. **The direction of the finding (0 of 5, sign inverted) is
correct; the magnitude was overstated by roughly 50%.** Every later repetition
of "4.5×" in this document (§9.6, §10.2, §10.6) should be read as **~2.9×**.

**Problem 10 — driven pendulum**

Median error ratio = **0.99×**, worse in 2 of 4 n. The traditional method's
error is *unchanged* by removing the density variation; its n-dependence
(0.069 → 0.035 → 0.018 → 0.011 as n doubles) is textbook 1/n finite-sample
bias.

**Problem 03 — Kepler.** The Kepler agent ran this control itself, found the
natural sample worse in only 2 of 6 n, and reported plainly that the
traditional method's residual error "is not clearly, causally attributable to
the density variation". That agent was right, and its recommendation to weight
the problem cautiously should have been applied to problems 02, 07 and 10 as
well.

**In all four problems, poly returned exactly 1.0000 on the natural sample and
exactly 1.0000 on the uniform control, at every n, with degenerate-fit fraction
1.000.** Poly's score is a structural constant of the ring-lattice regime; it
carries no information about density whatsoever.

**Verdict: 0 of 4 density-robustness wins are supported. Problems 03 and 07
rested on criterion (b) alone and therefore have no verified win at all.**

This is not a marginal call. Two independent things had to be true for the
claim and *neither* is: the traditional method is not measurably biased by
these density variations, and the poly method's perfect score is not evidence
of anything.

## 9.3 R2-F2 — The density criterion as operationalized is unfalsifiable (major, methodological)

R2-F1's root cause is a design flaw, and it is **the fourth kind of
self-deception** this benchmark has now caught — the round-2 brief predicted a
fourth kind would appear if the agents were not careful, and it did.

On a smooth, densely sampled closed 1D curve, the k-NN graph is a circulant
ring lattice, so `local_dimension` returns the exactly-constant-shell sentinel
and `mean_dimension` returns exactly 1.0000 — **necessarily, on every such
curve, at every n, under every sampling density**. A test scored as
"poly error vs traditional error on a curve of known dimension 1" therefore
awards poly a perfect score *by construction*, and reduces to asking only
whether the traditional method's error is nonzero. It always is, at finite n.

So criterion (b), applied to any of this suite's closed-orbit problems, is a
test the poly method cannot fail and the traditional method cannot pass.
It measures the F1 degeneracy, relabelled.

The clearest evidence that this was invisible from inside a single problem:
**problem 07's own script scored `won_on_density_robustness = True` on its
n=1000 window, whose measured density ratio was 0.985 — i.e. no density
variation at all.** The script printed that ratio, correctly noted the window
"does NOT exercise the close-approach density variation", and still returned a
win for it, because poly scored 1.0000 and traditional scored 0.9894. That
self-refuting control was run, printed, and not recognized.

**Correction (post-audit independent verification, §10.10): "run, printed,
and not recognized" overstates the failure.** The raw script report shows the
agent *did* recognize the n=1000 window was null — it explicitly labels it
the "PARTIAL window ... does not exercise the effect" and rests the
`won_on_density_robustness = True` verdict on the full single period instead
(n=21723, measured density ratio 5.518:1), not on the null window. The real
defect is narrower and still stands: **the claim lacked a mandatory
uniform-resample control arm**, which is what let a structurally-pinned
metric (§9.3 above) pass on the full-period data regardless. The agent did
not fail to notice its own null window; the criterion it was applying had no
control that could have caught the structural pinning either way.

**Consequence for the criterion, not just this run:** any future
density-robustness claim must report the uniform-resample control arm and show
`trad_err(natural)/trad_err(uniform)` > 1 by a stated margin. A "poly error vs
traditional error" comparison on the natural sample alone is not evidence and
should not be accepted again.

## 9.4 R2-F3 — Problem 01's win is contingent on a non-default `max_radius`, and its stated justification is wrong (major; one verdict contested)

Problem 01 reported 87.5% compute savings (poly n=50 vs traditional n=400)
using `max_radius=3` instead of `comparison.py`'s default of 6 — the only
round-2 problem to deviate from the default. Sweeping it:

| `max_radius` | poly min_n | trad min_n | savings | win |
|---|---|---|---|---|
| 2 | 50 | 400 | 0.875 | yes |
| 3 (**used**) | 50 | 400 | 0.875 | yes |
| 4 | 50 | 400 | 0.875 | yes |
| 5 | 50 | 400 | 0.875 | yes |
| **6 (default)** | **400** | **400** | **0.000** | **no** |

At the default, problem 01 is a **tie**, not an 87.5% win. The whole win rests
on this one parameter.

The agent disclosed the deviation prominently and argued it was "a fair,
honestly-disclosed tuning choice, not a hyperparameter search for the answer
that wins". The disclosure is real and creditable. **But the stated mechanism
is factually wrong**, which is what makes the choice unverified rather than
merely contingent. The script claims `max_radius=6` "saturates the WHOLE k-NN
graph at n=100/200". Direct inspection of the failing case (n=200, k=6) shows
volumes `(7, 12, 17, 23, 29, 35)` out of 200 nodes — **nowhere near
saturation**. The real mechanism is R2-F4 below, and it is the opposite of
benign: shortening `max_radius` shortens the shell sequence, which makes it
more likely to be *exactly* constant, which routes it into the degenerate
sentinel branch where `r_squared = 1.0` and the estimate passes
`mean_dimension`'s well-fit filter. Measured at n=200: `max_radius=3` gives 20
well-fit nodes of which **20 are degenerate sentinels**; `max_radius=6` gives 0
well-fit and 0 degenerate.

**The tuning that produces problem 01's win works by maximizing the fraction of
F1 degenerate sentinels.** That is precisely the regime round 1 established
must not be cited as accuracy. Verdict: **not verified**; re-run at the default,
or fix R2-F4 first.

## 9.5 R2-F4 — New estimator defect: near-constant shell sequences collapse R² and are silently discarded (major, actionable; distinct from F2/N1)

Round 1's F2 found that *exactly* constant shell sequences collapsed R² to 0
through floating-point cancellation, and N1 fixed it with a relative-scale
tolerance. **A distinct and unfixed failure sits immediately adjacent**: a
shell sequence that is *nearly* but not exactly constant has genuinely tiny
`ss_tot`, so N1's tolerance does not fire, `degenerate=False`, and R² — which
measures *fraction of variance explained* — collapses on its own merits
because there is almost no variance to explain. The dimension estimate is
excellent; the quality flag says it is worthless:

| shells (fit length 6) | dimension | R² | degenerate | kept by `is_well_fit(0.9)` |
|---|---|---|---|---|
| (6,6,6,6,6,6) | 1.0000 | 1.0000 | True | **yes** |
| (6,6,6,6,6,7) | 1.0488 | 0.2642 | False | **no** |
| (6,5,6,6,6,6) | 1.0335 | 0.0889 | False | **no** |
| (5,6,6,6,6,6) | 1.0911 | 0.6572 | False | **no** |
| (7,5,5,6,6,6) | 0.9563 | 0.0505 | False | **no** |
| (4,8,4,8,4,8) | 1.1836 | 0.1027 | False | **no** |

A sequence off-by-one at a single radius yields dimension 1.0488 — a better
estimate than most of this benchmark's non-degenerate results — and is thrown
away with R²=0.26. Note the last row: a wildly oscillating garbage sequence
scores R²=0.1027, statistically indistinguishable from the near-perfect
(6,5,6,6,6,6) at R²=0.0889. **In the near-constant regime R² is not merely
conservative, it is uninformative** — it cannot rank a good fit above a bad one.

Consequences, both observed in this run:
- When *all* sampled nodes land in this band, `mean_dimension` returns `nan`
  (problem 01 at n=200, `max_radius`=6) even though every discarded per-node
  dimension was within 0.08 of the truth. That `nan` then breaks
  `poly_algebraic_minimum_points`'s stable-convergence chain and moves
  `poly_min_n` from 50 to 400 — the entire mechanism of R2-F3.
- It creates a perverse incentive to shrink `max_radius` until shells go
  exactly constant, converting real (if noisy) fits into F1 sentinels.

**Suggested fix** (not applied — round-2 scripts are being read concurrently):
gate on an absolute residual criterion in the near-constant regime rather than
on R² alone, e.g. accept a fit when the shell sequence's relative spread is
below a threshold (treat it as constant-like and report the slope with a
`near_degenerate` flag), reserving R² for sequences with real dynamic range.
This would subsume N1's exact-constant special case rather than sitting beside
it. Needs its own known-answer tests before use.

## 9.6 R2-F5 — The predicted fourth failure pattern was correctly avoided (positive)

The brief warned specifically against claiming a compute-savings win because
the *traditional* method failed to converge. **Checked explicitly on all three
criterion-(a) claims and on both non-convergence cases; no agent committed
this error.**

- Problems 05 (torus) and 06 (Brownian) both had `traditional_min_n = None`
  and `poly_algebraic_min_n` finite (400 in both). Both agents reported
  `overall_win = False` and declined to convert the traditional method's
  failure into a savings number, explicitly citing the rule. Problem 05 went
  further and verified the non-convergence was structural (a second,
  independently-ordered IID-time construction up to n=16000 gave the same
  1.59–1.63 plateau at R²>0.999) rather than a too-small n_grid.
- For problems 02 and 10, the traditional method genuinely converges: dim
  1.9358/1.9890 → 1.51 → 1.51/1.44 → **1.138 (in tolerance at n=256)** →
  1.067 → 1.034 → 1.018 → 1.010, monotone with R² rising 0.93 → 1.000.
  This is real convergence at n=256, not a lucky crossing, and the poly method
  beat it from a genuinely lower n.

Problem 05's agent also flagged, as an unscored diagnostic, that
`baseline.correlation_dimension`'s fixed `r_min_frac=0.01`/`r_max_frac=0.2`
window (not exposed through `compare()`) may bias the traditional estimator on
non-uniform invariant measures generally. That is worth pursuing: it is a
plausible partial explanation for the *inverted* CR3BP result in R2-F1, where
changing the point distribution moved the traditional estimate by ~4.5× in
error while the fitting window stayed fixed.

## 9.7 R2-F6 — The two verified wins, and what they actually show

Problems **02** (nonlinear pendulum) and **10** (driven pendulum) are genuine
criterion-(a) wins and survive parameter sweeps: poly beats traditional by
>10% savings in **18 of 20** (problem 02) and **20 of 20** (problem 10)
combinations of k ∈ {4,6,8,10} and `max_radius` ∈ {3,4,5,6,8}. Both used the
module default `max_radius=6`; neither was tuned. Reported savings 0.75 at
(k=6, `max_radius`=6) sits mid-range of the sweep (0.50–0.875), not at a
favourable edge.

Both agents' point-cloud construction is sound and worth reusing: a single
period only (removing F3 duplicate corruption at the source rather than via
`dedupe`), reordered by a **bit-reversal permutation** so that `points[:n]` is
a genuine time-uniform sample of the *whole* orbit at every power-of-two n
rather than a growing arc. Problem 02 verified the bit-reversal prefix property
programmatically instead of assuming the textbook result. Problem 01's
golden-ratio Weyl sequence achieves the same nesting property and is equally
sound.

**What the win means, stated narrowly:** on a smooth closed 1D orbit, a local
shell-growth estimator identifies the ring topology as soon as points are
well-separated (n≈64), while a global pairwise correlation sum needs enough
points to populate a clean multi-radius log-log scaling region (n≈256). That
is a real and reproducible algorithmic difference in sample efficiency. Both
agents disclosed, correctly, that poly's estimate in this regime is the F1
degenerate sentinel — so this is a statement about *how few points each method
needs to identify a ring*, **not** a statement that the shell-growth estimator
is more accurate. On the two problems where accuracy was genuinely tested
against a non-trivial answer (Lorenz, Rössler), poly needed **4× and 2× more**
points than the traditional method, not fewer.

## 9.8 Where round 2 stands against its goal

- **Goal: ≥8 of 10 wins. Achieved: 2 of 10** (3 if problem 01's non-default
  `max_radius` is accepted, which R2-F3 argues it should not be).
- The 5-of-10 self-reported figure was not dishonest — every number in it is
  correct and every agent disclosed its degeneracy caveats, several in
  considerable detail. It was **under-controlled**: four of the five wins
  rested on a comparison with no control arm (R2-F1/R2-F2) or on a
  hyperparameter whose stated justification did not survive inspection
  (R2-F3).
- The round-1 discipline held where it had been encoded in code
  (`poly_algebraic_wins`, `compute_savings_fraction`, the stable-convergence
  requirement, `DuplicatePointsError`) — every problem passed those checks
  correctly, and R2-F5 shows the newest trap was avoided everywhere. **The
  failures were exclusively in the one area that had no code-level guard: the
  new, hand-rolled density-robustness test.** That is the lesson to carry
  forward — a criterion that lives only in prose gets applied inconsistently
  across ten agents, and it did.

## 9.9 What phase 2 should do

1. **Do not re-attempt criterion (b) on closed 1D orbits.** It is unfalsifiable
   there (R2-F2). If singularity-avoidance is to be tested, it needs a system
   where poly's answer is *not* structurally pinned — i.e. a non-degenerate
   regime (dimension ≠ 1, or a curve with genuine self-intersection/branching),
   with the uniform-resample control arm mandatory and reported.
2. **Encode the control in code**, as a `compare_density_robustness()` in
   `comparison.py` that takes both samples and refuses to return a win unless
   `trad_err(natural)/trad_err(uniform)` exceeds a stated margin. Prose
   criteria did not survive ten parallel agents; the coded ones did.
3. **Fix R2-F4** (near-constant R² collapse) with known-answer tests, then
   re-run problem 01 at the default `max_radius` to settle R2-F3 honestly.
4. **Retract the (b) sub-claims of problems 02 and 10** in any summary; their
   (a) claims stand and are the two solid results of this round.
5. The honest current headline is: *on smooth closed 1D orbits the local
   shell-growth estimator needs ~4× fewer points than Grassberger–Procaccia to
   identify the ring, on chaotic attractors it needs 2–4× more, and no
   robustness-to-sampling-density advantage has been demonstrated.*

## 9.10 Tier C — interpretation, explicitly not load-bearing

Nothing in round 2 changes the round-1 Tier C position. The two verified wins
are a statement about sample efficiency of a graph statistic on a ring lattice,
not about geometry, emergence, or physics. In particular, the phrase
"singularity/stiffness avoidance" has **no supporting evidence** after this
round: the four measurements that were offered for it are, on control, either
null (problems 02, 03, 10) or inverted (problem 07). It should not be repeated
in the README, the paper, or any summary until a test that can fail produces a
result that does not.

---

# Round 2, phase 2 — FINAL AUDIT and closing verdict (2026-08-13)

This is the closing audit of the round-2 programme. It re-derives every
verdict independently of both the ten problem agents and the phase-1 audit
(§9), spot-checks four problems end-to-end from scratch, and runs two
controls §9 did not run. It supersedes nothing in §9 — it confirms §9's
headline and adds three findings, one of which *strengthens* the surviving
wins and one of which is a criticism of the scoreboard itself.

## 10.0 Final verdict

**Goal: ≥8 of 10 wins. Final verified count: 2 of 10. The goal was not
reached.** Phase-1's count is confirmed by fully independent re-derivation.

The two wins (problems **02** and **10**) are criterion-(a) compute-savings
wins, both at module-default hyperparameters. **Criterion (b),
density-robustness — the "singularity avoidance" axis — produced zero verified
wins from four claims**, and an independent replication here (§10.3) shows why.

**Correction (post-audit independent verification, §10.10): this "2 of 10"
count is an audit of what was run, not of what the benchmark shows under this
same audit's own audited-best methodology.** A third-level skeptic applying
§10.4/R2-F8's endorsed bit-reversal/Weyl whole-orbit prefix construction
(which §10.1–§10.7 use for problems 01, 02 and 10, but never applied to 03 or
04) found problems 03 (Kepler) and 04 (Mars) each cross to a genuine
criterion-(a) win — poly `min_n`=64 vs traditional `min_n`=256, savings 0.75 —
once measured on a whole-orbit-covering prefix instead of the time-ordered
arc §10.1/§10.2 actually used. **Corrected final verified count: 4 of 10**
(02, 03, 04, 10), still far short of the 8/10 goal, and all four wins share
the identical underlying mechanism (see §10.10). The goal was still not
reached; §10.2's per-problem reasoning for 03 and 04 is superseded by §10.10,
not by this line.

## 10.1 What this audit did, independently of §9

Nothing below reuses the round-2 scripts' point clouds or the phase-1 audit's
scratch files. Trajectories were re-integrated with a *different* integrator
(`scipy.solve_ivp`/RK45 at rtol=atol=1e-12, where the repo scripts use
hand-rolled leapfrog/RK4), and the two estimators were re-implemented from
scratch in numpy (brute-force O(n²) k-NN adjacency, hand-written BFS shell
counting and log-log least squares; explicit O(n²) pair-count
Grassberger–Procaccia) and checked against the library before any library
output was trusted.

| check | result |
|---|---|
| Problem 02 re-derived from scratch (RK45, closure 5.3e−12 over the exact period 4K(m)=8.3497529269) | poly `min_n`=64, trad `min_n`=256, savings **0.75** — identical to the report |
| My brute-force estimators vs the library, every n ≤ 1024 | agree to 4 decimal places on **both** methods at **every** n |
| Bit-reversal prefix property (n=32…256) | verified: prefix indices evenly strided over the whole period, not assumed |
| Problem 10 re-derived from scratch (stroboscopic spread 4.7e−13 over 50 drive periods, closure 2.4e−14) | poly 64, trad 256, savings **0.75** — identical to the report |
| Problem 01 re-derived from the **exact** solution (cos t, −sin t; Weyl index sequence, 3200 distinct indices) | tie at the default `max_radius`, confirming R2-F3 |
| Problem 08 (Lorenz) re-derived (RK45, 18000 points) | poly 800, trad 200, savings **−3.0** — identical to the report, per-n values matching to 3 decimals |
| Arithmetic audit of all 10 self-reports | every `poly_algebraic_wins` and `compute_savings_fraction` consistent with its two `min_n` values, ✓ (confirms §9) |

**Correction (post-audit independent verification, §10.10):** the Lorenz
row's "matching to 3 decimals" is inconsistent with §10.5's own table, which
lists Lorenz at n=12800 as poly 1.9932 / trad 1.9300 — neither the round-2
agent's reported 2.038/1.943 nor this row's own independent RK4 re-derivation
(dt=0.005, t=100, transient 10, 18001 points, k=10: poly 2.0375, trad 1.9429,
`compare()` savings −3.0, confirmed) matches those two decimals. §10.5's row
comes from a different, coarser cloud (its scratch script samples 4096 points
via `solve_ivp`) that was never labelled as such, unlike the analogous
Rössler discrepancy in §10.5 which *is* labelled. The direction and the
savings figure (−3.0) are unaffected; the mismatch actually **understates**
poly's accuracy advantage (real |err| 0.0125 vs traditional's 0.1071, not the
0.0568/0.1200 §10.5 prints), so R2-F9's argument in §10.5 is unaffected.

Scripts: `scratchpad/final_audit_{1,2,3,4}.py` (scratch, uncommitted; they
import from `src/` and modify nothing). `src/socrates/hypergraph/` and all
round-1 and round-2 scripts were left untouched — confirmed by `git status`:
the only modified file in the repository is this document.

## 10.2 Final scoreboard

| # | Problem | Criterion (a) | Criterion (b) | **Verified** | Why |
|---|---|---|---|---|---|
| 01 | Harmonic oscillator | tie at default | n/a | **no** | 87.5% savings only at `max_radius`≤5; **tie (0.000) at the default 6** — re-derived here from the exact solution (R2-F3) |
| 02 | Nonlinear pendulum | **0.75** (64 vs 256) | fails control | **WIN (a)** | holds in 20/20 (k, `max_radius`) combos and **19/19 baseline fitting windows** (§10.4) |
| 03 | Kepler orbit | tie at n=100 | fails control | **no** | both converge at the smallest n tested; (b) null on control (R2-F1, replicated §10.3) |
| 04 | Mars / Horizons | tie at n=100 | n/a | **no** | tie; finer grid favours *traditional* (converges at n≈15) |
| 05 | Quasiperiodic torus | trad never converges | n/a | **no** | rule 2 forbids scoring the baseline's failure as a win — **but see §10.5** |
| 06 | Planar Brownian motion | trad never converges | n/a | **no** | same as 05 — **but see §10.5** |
| 07 | CR3BP, close lunar approach | tie at n=100 | fails control, **inverted** | **no** | the sharpest close-approach test in the suite; traditional was 4.5× *more* accurate with the density variation (R2-F1) |
| 08 | Lorenz | **−3.0** (poly needs 4× more) | n/a | **no** | honest loss, re-derived here independently |
| 09 | Rössler | **−1.0** (poly needs 2× more) | n/a | **no** | honest loss |
| 10 | Driven pendulum | **0.75** (64 vs 256) | fails control | **WIN (a)** | holds in 20/20 (k, `max_radius`) combos and 19/19 fitting windows |

**2 / 10 verified. Both are criterion (a). Criterion (b): 0 verified from 4
claimed.**

**Correction (post-audit independent verification, §10.10):**

- **Row 02's "20/20"** is a copy-paste error: the phase-2 scratch script that
  produced it (`final_audit_3.py`) ran the (k, `max_radius`) sweep on
  problem 10's point cloud only and its 20/20 result was written into
  problem 02's row too, silently overwriting §9.7's correct 18/20 for
  problem 02 in the same document. Re-measured independently on both an
  RK45 re-derivation and the agent's own committed cloud
  (`02_nonlinear_pendulum.generate_single_period_cloud`): **17/20** (k=10
  with `max_radius` in {5, 6, 8} ties at 256=256, savings 0.0). The win
  still holds in 17 of 20 combinations — the win itself is not in question,
  only this one cell's number.
- **Rows 03 and 04 ("no")** used a time-ordered prefix of the trajectory
  (Kepler: n=100 of 12798 points, under 1% of the orbit) rather than the
  bit-reversal/Weyl whole-orbit construction §10.4/R2-F8 explicitly
  endorses and problems 01/02/10 actually used. Under that same
  construction (single period, 8192 time-uniform points, k=6,
  `max_radius`=6): Kepler (e=0.6, tolerance 0.15) gives poly 64 vs
  traditional 256 → **savings 0.75**, robust across k (0.875, 0.75, 0.5,
  0.5 at k=4/6/8/10); a Mars-like ellipse (e=0.0934, tolerance 0.2) gives
  the same 64 vs 256 → **0.75**. Under the time-ordered construction both
  instead give poly 64 vs trad 32 (savings −1.0) — the reported tie was a
  prefix artifact, not a property of the orbit. **Corrected: 03 and 04 are
  criterion-(a) WINS.** CR3BP (07) re-measured the same way stays a genuine
  tie (128 vs 128), so this construction does not mechanically manufacture
  wins everywhere it is applied — 07's "no" stands.

**Corrected count: 4 / 10 (02, 03, 04, 10), all criterion (a).** See §10.10
for the full derivation and its limits.

## 10.3 R2-F1 replicated on a fresh system — criterion (b) is null, independently

§9's control was rebuilt here from scratch on a **different** problem (Kepler,
e=0.6, integrated at rtol 1e-12, energy drift 2.4e−12, r ∈ [0.400, 1.600]
exactly as theory requires), with arc-length resampling implemented
independently. Density ratio on the natural time-uniform sample is a genuine
**3.2×**:

| n | ratio nat | poly nat | trad nat | \|err\| nat | ratio uni | poly uni | trad uni | \|err\| uni | nat/uni |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 3.20 | 1.0000 | 1.0572 | 0.0572 | 1.00 | 1.0000 | 1.0652 | 0.0652 | 0.88 |
| 1000 | 3.21 | 1.0000 | 1.0279 | 0.0279 | 1.00 | 1.0000 | 1.0310 | 0.0310 | 0.90 |
| 2000 | 3.22 | 1.0000 | 1.0146 | 0.0146 | 1.00 | 1.0000 | 1.0168 | 0.0168 | 0.87 |
| 4000 | 3.22 | 1.0000 | 1.0080 | 0.0080 | 1.00 | 1.0000 | 1.0107 | 0.0107 | 0.75 |
| 8000 | 3.27 | 1.0000 | 1.0047 | 0.0047 | 1.00 | 1.0000 | 1.0069 | 0.0069 | 0.68 |

**The traditional method is worse on the density-varying sample in 0 of 5 n**
(median error ratio 0.87×) — it is slightly *more* accurate with the density
variation present, the same sign inversion §9 found on CR3BP. Poly returns
exactly 1.0000 on both samples at every n, degenerate fraction **1.000**.

R2-F1 and R2-F2 are therefore confirmed on an independent implementation and a
system §9 did not itself re-derive. **No evidence for singularity/stiffness
avoidance exists in this benchmark.**

R2-F4 was likewise reproduced exactly — all six rows of §9.5's table — and
anchored to the real failing case: problem 01 at n=200, k=6, `max_radius`=6 has
node volumes `(7,12,17,23,29,35)`, i.e. shells `(6,5,5,6,6,6)`, giving
**dimension 1.0333** (error 0.033) discarded at **R²=0.0550**. A synthetic
graph built to have exactly those shells reproduces `dimension=1.0333,
R²=0.0550` to the digit. The volumes reach 35 of 200 nodes, so §9.4's rebuttal
of the "saturation" justification is confirmed: nothing is saturated.

## 10.4 R2-F7 and R2-F8 (NEW, both positive) — the two surviving wins survive both controls

### R2-F7 — the wins are not an artifact of the baseline's fitting window

§9.6 left a phase-2 lead open: `baseline.correlation_dimension` hard-codes
`r_min_frac=0.01`, `r_max_frac=0.2` and does not expose them through
`compare()`, so "traditional needs n=256" might be a property of that window
rather than of the algorithm. If so, the two surviving wins would be
contingent on a baseline hyperparameter — exactly the objection R2-F3 raised
against problem 01's `max_radius`, and fairness requires asking it in both
directions.

**Closed, in poly's favour.** Sweeping 19 defensible windows
(`r_min_frac` ∈ {0.001, 0.003, 0.01, 0.03} × `r_max_frac` ∈ {0.05, 0.1, 0.2,
0.3, 0.5}) on both problems:

| | best trad `min_n` over all windows | default window | worst | poly `min_n` | savings range |
|---|---|---|---|---|---|
| 02 nonlinear pendulum | 128 (at `r_min_frac`=0.03) | 256 | 2048 | 64 | **0.50 – 0.97** |
| 10 driven pendulum | 128 (at `r_min_frac`=0.03) | 256 | 2048 | 64 | **0.50 – 0.97** |

The traditional method never converges before n=128 under **any** window
tested, while poly converges at n=64. Savings exceeds the 10% bar in **19 of
19** windows on both problems. The default window is mid-range, not a
handicap. Combined with the 20/20 (k, `max_radius`) sweeps, **problems 02 and
10 are robust wins on every hyperparameter either method has.**

**Correction (post-audit independent verification, §10.10):** the grid is
`r_min_frac` (4 values) × `r_max_frac` (5 values) = **20** combinations, not
19 — a trivial miscount. Re-run in full on problem 02: all **20 of 20**
windows clear the 10% bar (best traditional `min_n`=128 at `r_min_frac`=0.03,
default 256, worst 2048, savings range 0.50–0.97), reproducing the table
above exactly apart from the denominator. Combined with the 20/20 and 17/20
(k, `max_radius`) sweeps (previous correction), problems 02 and 10 remain
robust wins on every hyperparameter either method has — only the count of
windows was wrong, not the result.

### R2-F8 — criterion (a) is *not* unfalsifiable the way criterion (b) is

R2-F2 showed poly is structurally pinned to 1.0000 on any
smooth closed curve, which destroyed criterion (b). The obvious next question,
which §9 did not ask, is whether that same pin also hollows out criterion (a):
if poly reported ≈1.0 at n=64 for *every* point cloud, then "poly converged at
n=64" on a dimension-1 problem would be a prior, not a measurement. Tested
directly — `mean_dimension`, k=6, `max_radius`=6, on clouds of known dimension
at the same small n:

| cloud (true dim) | n=32 | n=64 | n=128 | n=256 | n=512 | n=1024 |
|---|---|---|---|---|---|---|
| circle (1) | nan | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| uniform square (2) | 1.3206 | nan | 1.6420 | 1.9437 | 1.8488 | 1.9977 |
| uniform disk (2) | 1.5730 | **1.5642** | 1.5709 | 1.8835 | 1.9620 | 2.0003 |
| uniform cube (3) | 1.6227 | nan | 1.7839 | 2.1302 | 2.3233 | 2.4705 |
| Lorenz, whole attractor (2.05) | nan | nan | 1.2270 | 1.7564 | 1.8991 | 1.9129 |

Poly **never returns ≈1.0 for a 2- or 3-dimensional cloud** at any n ≥ 32
(degenerate fraction is 0.000 for all four non-circle clouds at every n, vs
0.45→0.98 for the circle). So poly's 1.0000 at n=64 on the pendulums does
carry dimensional information, and the criterion-(a) wins are measurements,
not tautologies. This is the one place where a control *supports* the claimed
mechanism.

Two honest qualifications. First, poly is biased **downward** at small n on
higher-dimensional clouds (2D square needs n≈256 to reach 2.0), so its early
convergence is specific to the dimension-1 case. Second, an artifact worth
recording: a *time-ordered prefix* of a chaotic trajectory is a short smooth
arc, and poly correctly calls it 1-dimensional — Lorenz at n=100 with
consecutive dt=0.005 samples gives exactly 1.0000 (degenerate fraction 0.35).
That is right about the sample and wrong about the attractor. It is an argument
for the whole-orbit-covering prefix constructions (bit-reversal, Weyl) that
problems 01, 02 and 10 used, and against naive time-ordered prefixes.

## 10.5 R2-F9 (NEW, methodological) — the scoreboard hides poly's only non-degenerate advantage

The two surviving wins are on problems where poly's estimate is the F1
degenerate sentinel (degenerate fraction 1.000). The two problems where poly is
**most clearly and non-degenerately better than the baseline** score as
non-wins. Verified independently here at max n:

| problem | true | poly | \|err\| | degen. frac | trad | \|err\| | trad R² | scored |
|---|---|---|---|---|---|---|---|---|
| 05 quasiperiodic torus (n=12800) | 2.00 | **2.0500** | **0.0500** | **0.00** | 1.6357 | 0.3643 | 0.9994 | **no win** |
| 06 Brownian motion (n=25600) | 2.00 | **2.0270** | **0.0270** | **0.00** | 1.3943 | 0.6057 | 0.9980 | **no win** |
| 09 Rössler (n=12800) | 2.01 | 1.9011 | 0.1089 | 0.00 | 1.7036 | 0.3064 | 0.9997 | **no win (poly needs 2× more points)** |
| 08 Lorenz (n=12800) | 2.05 | 1.9932 | 0.0568 | 0.00 | 1.9300 | 0.1200 | 0.9989 | **no win (poly needs 4× more points)** |

(05 and 06 reproduce the agents' figures to the digit; my Rössler differs from
the agent's cloud in transient/stride and lands at 1.90 rather than 2.04 — the
ordering is the same, the exact value is not, and I report mine as the
independent number.)

On problems 05 and 06 the shell-growth estimator is within 0.03–0.05 of the
truth **with zero degenerate fits**, while Grassberger–Procaccia is off by
0.36–0.61 with R²>0.998 — a confident, converged, and wrong fit. Rule 2
correctly refuses to score this as a win (the baseline never converging is not
poly's achievement, and scoring it would have been the fourth self-deception
the brief warned about). **But the consequence should be stated plainly: the
programme's win condition measures points-to-converge only, so it is
structurally incapable of rewarding the one thing this benchmark shows the
shell-growth estimator genuinely does better — asymptotic accuracy on
non-degenerate measures.** The 2-of-10 scoreboard is correct under the stated
criteria and is *not* the most informative summary of the data.

This is a criticism of the criterion, not a licence to re-score. The count
stays 2.

**Correction (post-audit independent verification, §10.10):** "the count
stays 2" was correct given what §10.1–§10.6 actually measured, but §10.1/§10.2
measured 03 and 04 with a methodology this same section (§10.4) had already
found unsound for problems 01/02/10. Applying the endorsed methodology
uniformly moves the count to **4**. This section's criticism of the *criterion*
(that it is blind to 05/06) is untouched by that correction — it is a
separate finding.

## 10.6 Which win type carried the total, and which problems remain unwon

**Win type.** Criterion (a) carried **100%** of the verified total (2 of 2 —
**corrected to 4 of 4**, §10.10). Criterion (b) carried **0%**, from 4 claims
— and after §10.3's independent replication, the honest position is stronger
than "unproven": on every system tested (pendulum, driven pendulum, Kepler,
CR3BP) the traditional method is *equally or more* accurate on the
density-varying sample, so the effect criterion (b) was designed to detect
**does not exist at these densities and point counts**, in the direction
hypothesized.

Both (a) wins are the *same* win, twice: a small-amplitude pendulum orbit and a
mode-locked pendulum limit cycle are both smooth closed 1D curves. The suite
contains no second, structurally different (a) win.

**Correction (post-audit independent verification, §10.10):** with 03 and 04
corrected into wins, this is the *same win, four times*, not twice — Kepler
and Mars are also smooth closed (03) or effectively closed (04) 1D orbits,
identical in kind to 02/10. The qualitative conclusion is unchanged and
somewhat reinforced: every verified win in this benchmark, without exception,
is one mechanism (ring-lattice degeneracy giving the shell-growth estimator
an early, exact answer) applied to structurally identical inputs.

**Unwon problems, each a finding rather than a gap** (standing rule 5):

- **01 harmonic oscillator** — a genuine tie (400 vs 400) at the default. The
  reported win required `max_radius`≤5, whose effect is to shorten shell
  sequences until they are *exactly* constant and so pass the well-fit filter
  as F1 sentinels. Blocked by the R2-F4 estimator defect, not by physics: the
  discarded fits at the default were accurate to 0.03.
- **03 Kepler, 04 Mars, 07 CR3BP** — ties: both methods converge at the
  smallest n tested (100). These problems cannot produce a savings number at
  all without an n_grid extended *below* 100, and on problem 04 the finer grid
  favours the traditional method (converges at n≈15). Their (b) claims are
  null (03, replicated here) or inverted (07).

  **Correction (post-audit independent verification, §10.10): this stated
  cause is wrong for 03 and 04.** The tie was a prefix artifact, not a
  property of the physics or the grid floor — both problems fed `compare()`
  a time-ordered arc rather than the whole-orbit bit-reversal/Weyl
  construction §10.4 itself endorses. Under that construction, 03 and 04
  each cross to a genuine (a) win (savings 0.75). 07 (CR3BP) re-measured the
  same way stays a genuine tie, so this correction applies to 03/04 only;
  07's diagnosis is unaffected.
- **05 torus, 06 Brownian** — the baseline never stably converges (verified to
  n=16000 and n=25600 respectively, R²>0.998, drifting *away* from the truth).
  Correctly not banked. See §10.5: this is where poly's real advantage lives.
- **08 Lorenz, 09 Rössler** — honest losses on the stated criterion; poly needs
  4× and 2× *more* points to stabilize. Both are non-degenerate, both survived
  parameter sweeps, and 08 was re-derived from scratch here. Grassberger–
  Procaccia was designed and literature-calibrated for exactly this quantity on
  exactly these attractors; it is not a strawman losing artificially, it is the
  right tool for its own home problem.

## 10.7 The honest one-paragraph summary

On smooth closed 1D orbits the local shell-growth estimator identifies the ring
topology from ~4× fewer points than the classical Grassberger–Procaccia
correlation sum (verified on two problems, robust across every k,
`max_radius`, and correlation-fit window tested). On chaotic attractors it
needs 2–4× more points to stabilize, though it lands closer to the true value
once it does. On two non-degenerate 2-dimensional measures (quasiperiodic
torus, Brownian path) it is accurate to 0.03–0.05 where the classical estimator
is confidently wrong by 0.36–0.61 — the strongest real result in the suite, and
one the win criterion cannot score. **No robustness-to-sampling-density
advantage has been demonstrated; four attempts to show one were null or
inverted under control.** The programme's stated goal of 8 of 10 was not met:
the final count is **2 of 10** (**corrected to 4 of 10 — see §10.10**).

## 10.8 Carry-forward

1. **Fix R2-F4 before any re-run of problem 01** (gate on relative shell
   spread, not R² alone; needs known-answer tests). It is the only finding in
   two rounds that is an actionable defect in shipped code rather than a
   defect in an experiment.
2. **Retire criterion (b) in its current form.** §9.9's recommendation stands
   and is now backed by an independent replication on a fresh system. If it is
   ever retried, the uniform-resample control arm must be *in code*, and the
   system must be one where poly is not structurally pinned.
3. **The next round's headline question should be accuracy on non-degenerate
   measures, not points-to-converge** (§10.5). Problems 05 and 06 are the
   template: a known non-integer or 2-dimensional answer, no closed-curve
   degeneracy, both methods run to large n. That is a test poly can fail and
   did not.
4. Do not repeat "singularity/stiffness avoidance" in the README, the paper, or
   any summary. After two audits and an independent replication it has no
   supporting measurement.

## 10.9 Tier C — interpretation, explicitly not load-bearing

Unchanged from §9.10. The two (now four — §10.10) verified wins are
statements about the sample efficiency of a graph statistic on a ring
lattice. The 05/06 accuracy results in §10.5 are the first measurements in
this benchmark that plausibly say something about the estimator itself
rather than about ring lattices, and they are Tier B statistics of two
specific point clouds — not evidence about geometry, emergence, or physics.

## 10.10 Third-level verification — corrected final count and consolidated summary

A skeptic tasked with checking §10 itself (per the decisive-experiment
skill's phase 6, "verify the auditor, not just the workers" — already
confirmed twice earlier in this document) found that §10, despite being an
independent re-derivation of §9, repeated the same class of error it exists
to catch: it endorsed a methodology fix (§10.4/R2-F8's whole-orbit
bit-reversal/Weyl prefix) without checking whether §10.1's own measurements
actually used it everywhere the fix applied. They did not — problems 03 and
04 were measured with the older, unsound time-ordered-prefix construction
that §9/§10 had already diagnosed as invalid for exactly this kind of orbit.
This is a third independent confirmation, within this single document, that
a reviewer's favorable/confirming claims receive less scrutiny from itself
than the claims it is busy criticizing.

**Six findings, all independently re-derived by the skeptic before being
recorded here** (full reasoning and numbers are inline at each correction
site above; this section is the consolidated index, not a new claim):

1. **(major)** Problems 03 (Kepler) and 04 (Mars) are misclassified ties in
   §10.1/§10.2 — see the correction blocks at §10.0, §10.2, §10.6. Corrected:
   both are genuine criterion-(a) wins, savings 0.75.
2. **(moderate)** §10.2's problem 02 row states "20/20"; the real,
   independently re-measured figure is **17/20** (§10.2 correction) — a
   copy-paste of problem 10's number, not a new measurement error. The win
   itself still holds.
3. **(moderate)** The CR3BP "4.5× more accurate" headline (§9.2, repeated in
   §9.6/§10.2/§10.6) does not follow from its own printed table; the correct
   median is **~2.9×** (§9.2 correction). Direction unaffected, magnitude
   overstated ~50%.
4. **(minor)** §10.1's Lorenz row ("matching to 3 decimals") is internally
   inconsistent with §10.5's own table, which uses an unlabelled, coarser
   point cloud (§10.1 correction). Direction and −3.0 savings figure
   unaffected; if anything the mismatch understates poly's accuracy.
5. **(trivial)** §10.4's "19 defensible windows" / "19 of 19" is a miscount:
   the grid is 4×5=**20** combinations, and re-running all 20 confirms **20
   of 20** clear the bar (§10.4 correction).
6. **(minor)** §9.3's characterization of problem 07's script as having
   printed a self-refuting control "and not recognized" it is unfair to the
   agent: the raw report explicitly labels the n=1000 window as not
   exercising the effect and rests its verdict on the full period instead
   (§9.3 correction). The underlying criticism — no mandatory uniform-resample
   control arm — is correct and stands.

**Corrected final scoreboard: 4 of 10 verified (02, 03, 04, 10), against the
stated goal of 8 of 10 — the goal was still not reached.** All four wins are
criterion (a) (compute savings); criterion (b) (density-robustness) remains
at 0 of 4, confirmed null or inverted under control, unaffected by any of the
six corrections above. All four wins are the *same* win, four times over: a
smooth closed (or near-closed) 1D orbit puts the k-NN graph into an exact
ring-lattice regime where the shell-growth estimator's fit is degenerate
(F1) but correct, and converges from a much smaller n than a global
correlation-sum needs to escape its own fitting-window bias. §10.5/R2-F9's
point stands unchanged by this correction: the scoreboard's points-to-converge
criterion is structurally blind to the one place (problems 05/06) this
benchmark shows the estimator doing something a non-degenerate control
cannot — that is a critique of the criterion, not a reason to re-score 05/06,
and is not folded into the 4/10 count.

**What is unchanged by this correction:** the goal (8/10) was not met; no
density-robustness win survives; R2-F4 (near-constant-shell R² collapse) is
still an open, unfixed defect blocking problem 01; criterion (b) still needs
retirement or a redesign with a mandatory control arm; §10.8's carry-forward
list stands as written. **What changed:** the win count (2→4) and the
generalization claim (two independent smooth-orbit demonstrations become
four, strengthening rather than weakening the "one mechanism, repeated"
reading) — not the qualitative verdict.

Scripts: skeptic's re-derivations were scratch, uncommitted, read `src/` and
modified nothing (confirmed by `git status` at the time) — the only file this
correction pass modifies is this document.

---

## 11.0 Round 3 headline

**Goal (revised downward from round 2's 8/10, per programme owner direction):
6 of 10, via compute savings (criterion a) or a newly-formalized accuracy
criterion (criterion c) — criterion b stays retired. Final independently
verified count: 6 of 10 (01, 02, 03, 04, 06, 10). The goal was met**, after
a genuinely difficult path: three repair rounds to close a real estimator
defect (N8), a literature-grounded fairness fix to the traditional baseline
that turned out to change more verdicts than expected, an AutoResearch-style
rapid-iteration loop, and a four-level review chain (self-report → audit →
two independent refute-stage skeptics → a dedicated close-out pass) that
twice found the *reviewers* repeating the exact "endorsed but not applied
everywhere" failure mode this ledger's own lesson 8 exists to catch.

Design/repair/adjudication throughout this round used a higher-tier model
per explicit programme-owner direction. Three literature sources grounded
the round's hypotheses: Theiler (1986) on autocorrelation bias in
correlation-sum estimators; Levina & Bickel (2004) and Farahmand, Szepesvári
& Audibert (2007) on alternative/adaptive local-dimension estimation; and
Karpathy (2026, "AutoResearch") for the propose-measure-keep-or-rollback
mechanism used in the H2 loop below.

## 11.1 N8: three repair rounds to close a real, previously-open defect

Round 2 left N8/R2-F4 (near-constant shell sequences collapsing R² and
being wrongly rejected) as an open defect blocking a fair measurement of
problem 01. Closing it took three attempts, each one independently
verified, and each verification found something real:

- **Round 1** (per-node gate: coefficient-of-variation ≤0.15 AND a
  `slope_bound` ≤0.25 on the shell sequence): correctly fixed the
  documented anchor case, passed 56 threshold-robustness pairs and 10 new
  adversarial families, but the skeptic found the central safety claim was
  false — `slope_bound` bounds `|dimension − 1|`, not `|dimension − truth|`.
  On 2D Brownian motion (true dimension 2) at `k=6, max_radius=3`, the gate
  newly admitted 13/200 nodes with error up to 1.18, moving the sampled
  mean *away* from the truth (1.8321 → 1.7381). **Refuted.**
- **Round 2** (added a minimum-fit-length ≥5 gate, tested and proposed by
  round 1's own skeptic): fixed the Brownian collateral case cleanly and
  generalized correctly to 10 new families and problem 05. But the second
  skeptic found the repair's own reasoning for declaring "no round-2
  verdict moves" was backwards — it argued every other round-2 script's
  fit length was ≥5 as a reason *not* to check them, when fit length ≥5 is
  exactly the regime where the branch is *active*. Measured directly: the
  fix silently changed problem 06's recorded `poly_algebraic_min_n` (400→800)
  and — undetected by either prior round — had also silently *inflated*
  problems 02 and 10's recorded savings (0.75→0.875) via newly-admitted
  short-window nodes. The residue was a class (14/141 swept configs across
  Brownian/torus families), not the single case round 1 fixed. **Refuted.**
- **Round 3** (graph-level consensus, human-selected after the programme
  owner was shown the round-2 skeptic's two candidate fixes and their
  tradeoffs): replaces the per-node decision with a graph-level one —
  `near_degenerate_fraction` must clear a majority-consensus bar before the
  branch fires at all, calibrated against a broadened set (problem 01's
  ring plus problems 02/03/04/10's rings, not just one example as round 2's
  candidate was). Backed by a proof, not just measurement: when the
  confident branch decides, the pooled estimate is provably confined to
  `[0.75, 1.25]` (a convex combination of confident-branch and near-constant
  values, each independently bounded), so **every 2D/3D problem in the
  suite is structurally immune to this failure mode**, not merely
  empirically clean on the cases tried. Independently reproduced from
  scratch (own leapfrog/RK4 integrators, own AGM elliptic-integral code, own
  numpy re-derivation of every statistic) — anchor case preserved, the
  round-2 Brownian regression correctly rejected, problem 05 unchanged, and
  the broadened calibration set matches to the digit. Also confirmed the
  fix restores problems 02/10's correct, documented savings (silently
  inflated by round 2, now fixed). **Confirmed.**

Eight non-blocking caveats travel with the final gate (documented in the
round-3 workflow journal): the fix is one-sided (it protects targets whose
true dimension is far from 1, not targets near 1, though none in this suite
are exposed); the fallback branch's lower edge is anchored on a still-small
number of must-fire examples; four of ten problems were not independently
re-measured by the round-3 skeptic; and the node-local layer's constants
(0.15/0.25/5) remain calibrated by looking at the documented cases, same as
before — threshold circularity reduced, not eliminated.

## 11.2 H1 (Theiler-window baseline fix) and H3 (criterion c)

**H1** adds an opt-in Theiler-window exclusion to the traditional
correlation-sum baseline (`baseline.py`), keyed correctly on each point's
*original trajectory time index* rather than its position in a
bit-reversal/Weyl-reordered array — verified against a from-scratch brute
force under six index regimes including an actual bit-reversal permutation,
`max|Δ|=0` at every radius. Getting this wrong (keying on array position)
would have silently corrupted every reordered-cloud measurement in the
benchmark; the skeptic confirmed it was implemented correctly and that
default behaviour (`window=0`) reproduces prior results bit-for-bit.

**H3** formalizes criterion (c), "asymptotic accuracy on a non-degenerate
target," as `compare_accuracy_at_max_n()` in `comparison.py` — the
programme's answer to R2-F9's complaint that the benchmark's only
points-to-converge criterion structurally cannot credit the estimator's
real advantage on problems 05/06. It runs both methods to a fixed large n,
compares each to the known truth, and explicitly refuses to call a win
where poly sits in the F1 degenerate-sentinel regime (so a real accuracy
advantage cannot be confused with the ring-lattice artifact). Both changes
passed independent verification on the first attempt.

## 11.3 H2: an AutoResearch-style loop, and what it did and did not survive

Following Karpathy's AutoResearch mechanism (propose one change, measure it
immediately, keep only on a real measured improvement, roll back otherwise,
no predefined search space), four iterations targeted Lorenz and Rössler —
the two problems where round 2 recorded poly needing 2–4× *more* points
than the traditional method:

1. **Kept.** Apply the Theiler exclusion to the k-NN graph construction
   itself, not only the correlation sum (informed by the same Theiler 1986
   mechanism as H1, applied to the estimator under test rather than the
   baseline).
2. **Kept.** A finite-size saturation guard (`max_ball_fraction`) on the
   shell-growth fit window, informed by the manifold-adaptive-neighborhood
   literature (Farahmand et al.) — truncates a radius's shell from the fit
   once its ball has swallowed too large a fraction of the graph.
3. **Rejected.** A residual-degrees-of-freedom floor on the R² acceptance
   branch — net zero effect on Lorenz, a measured regression on Rössler
   (100→200).
4. **Rejected.** Metric (Euclidean-weighted geodesic) balls in place of
   hop-count balls — no help on Lorenz across a 5-point fraction sweep, and
   a severe regression on Rössler (savings 0.875→0.111).

Both kept changes are genuine, useful additions and remain in the codebase.
But **the loop's own headline verdict on Rössler did not survive the
benchmark's separate, slower verification discipline**: the two kept
changes appeared to flip Rössler from a recorded loss to a 16×-fewer-points
win (1600→100), but the Full Remeasure agent for problem 09 — applying the
same "does this point cloud actually cover the attractor" check the ledger
already required for problems 03/04 — found the winning n=100 cell is a
short arc that has not yet completed one loop of the attractor, the same
time-ordered-prefix artifact §10.10 diagnosed for Kepler and Mars.
Re-measured under an attractor-covering (bit-reversal) construction,
problem 09 is at best a tie. **This is the AutoResearch loop working as
designed and being correctly checked**: a fast, cheap propose-measure-keep
cycle found real local code improvements, but its own convergence
*verdict* needed the slower, separate adversarial discipline to catch a
flaw the cheap in-loop checks were not built to see. Treat the two kept
code changes as real; treat the in-loop win claim they produced as retracted.

## 11.4 Full remeasure, adjudication, and a new, larger fairness problem

Full remeasure self-reported 7/10 wins (problem 09's own agent, per §11.3,
disqualified its own AutoResearch-era win — a second, independent instance
of the lesson-8 discipline working exactly as intended, inside the very
same pipeline stage this time).

The adjudicator went further than any prior round: it discovered that the
traditional baseline's hard-coded correlation-sum radius window
(`[0.01, 0.2]×diag`, never exposed as a parameter) was itself materially
misconfigured on non-uniform-density targets — the round's answer to
R2-F7's still-open lead. Giving the baseline a fair, non-default window
(via two standards: an answer-aware "best of 20" oracle, and an
answer-blind "plateau" selector validated on known-answer clouds) **flipped
problem 07 (CR3BP) from a documented tie to an outright poly loss**, and
left problem 04 "contested" (a credible but knife-edge tie). Firm count:
4/10 (01, 02, 03, 10). A further, real scope-narrowing finding: the whole
ring-lattice win family requires *deterministic* sampling — on an
i.i.d.-uniform-angle-sampled circle (same manifold, same true dimension as
problem 01), poly returns `nan` at every n across 10 seeds while the
traditional method converges cleanly at every n across the same 10 seeds, a
complete inversion. "On a smooth closed 1D orbit" was always the wrong
scope; "on a *regularly, deterministically* sampled smooth closed 1D orbit"
is the honest one.

## 11.5 Refute: the lesson-8 pattern, caught a fourth time

Two independent refute-stage skeptics reviewed the adjudication:

- The first confirmed 4/10, but argued the answer-blind plateau standard
  (validated on known answers, not answer-aware) is more principled than
  the oracle standard the adjudicator called firm — under it, 04 becomes a
  clean win and the count is 5/10.
- The second, instructed specifically to check whether round 3 repeated
  the exact failure `docs/LL.md` lesson 8 names, found that it had: **the
  adjudicator's own baseline-fairness fix was applied to the six problems
  where doing so removes wins (04, 07) but never applied to problems 05 and
  06**, where `traditional_min_n = None` ("baseline never converges") had
  been accepted as a flat fact rather than checked under the same fair
  window. Run once, quickly, both flip to wins under the plateau standard
  — taking the count to 6/10, meeting the goal — but the skeptic explicitly
  flagged this as *unconfirmed*: 05's candidate crossing looked like it
  might drift back out of tolerance at larger n, and 06's tie looked
  knife-edge. It also found H1 had been swept and disclosed for every
  problem except 01 (the round's largest claimed win, 0.875), an
  undisclosed-robustness gap rather than a refutation.

Rather than bank a number off an admittedly-unconfirmed quick check, both
threads were sent to one more, tightly-scoped closing pass.

## 11.6 Close-out: a symmetric standard, applied with full rigor, settles it

Two more independently-verified agents (one resolving, one checking the
resolution from scratch) closed both threads using the benchmark's own
full stable-convergence rule (in tolerance at the first n **and every
larger n tested**, not a single crossing point) rather than the quick
checks the refute stage used:

- **Problem 05 is confirmed a non-win.** Its apparent in-band crossing
  under the plateau baseline was a transient drift, not convergence:
  extending the grid to n=102400 (8× the production maximum) shows the
  baseline drifts back out of tolerance under 27 of 30 swept baseline
  settings. The direction of the original refute-stage instinct was right;
  its specific evidence (which stopped at the production grid) was not yet
  strong enough to show it.
- **Problem 06 is confirmed a genuine, stable win** — savings 0.9688,
  unanimous across all 30 swept baseline settings, sentinel fraction
  **0.00 at every n**. This is the only non-degenerate win in either round
  of this benchmark: not "fewer points to identify a ring," but a real
  accuracy/sample-efficiency advantage on a target with no ring-lattice
  shortcut available.
- **Problem 01's Theiler-window robustness gap is closed, favourably.**
  The win survives up to W=8 (dies at W=16), which is *more* robust in
  absolute terms than problems 02/04/10 (die at W=2) and problem 03 (dies
  at W=4) — the gap was one of disclosure, not of fragility. All five
  ring-lattice wins die at roughly the same ≈2-sample-spacing Theiler
  window once normalized by each cloud's own sampling density — a real,
  shared fragility that should travel with the family, not a defect
  specific to 01.
- **The deeper finding: the 4/10 and 5/10 readings were both artifacts of
  a one-sided fairness fix, not of a stricter standard.** The
  adjudicator's oracle baseline let the traditional method choose its best
  radius window (answer-aware) while poly stayed pinned at module
  defaults, per standing rule R2-F3. Measured symmetrically — either both
  methods answer-blind/default, or both given a matched oracle sweep — the
  verdict is the *same* on every contested problem (04, 05, 06, 07), and
  matches the plateau-standard count. **6 of 10 is not the generous
  reading; it is what a fairly-applied standard gives on every axis
  checked.**

Independently re-verified from scratch by a second agent (own point
clouds, own plateau selector, own brute-force Theiler-corrected baseline,
imported nothing from the resolving agent's scripts): confirmed, to four
decimals on every disputed cell. Four minor, non-count-changing residual
caveats were logged (05's non-win depends on an 8×-extended grid applied
only to 05/06 — under the same production-grid protocol used for every
other problem, 05 would be a 7th win at 0.9375, a strictly conservative
choice since 6/10 already meets the goal either way; 06's win magnitude is
selector-dependent, 0.500–0.992, though never a non-win; a grid-choice
inconsistency in one peer-comparison table that does not change any
conclusion; the (c) refusals and 08/09 caveats were not independently
re-verified this pass).

## 11.7 Final scoreboard

| # | Problem | Mechanism | poly `min_n` | trad `min_n` (fair) | Savings | Verdict |
|---|---|---|---|---|---|---|
| 01 | Harmonic oscillator | F1 ring-lattice | 50 | 100 | 0.500 | **WIN** |
| 02 | Nonlinear pendulum | F1 ring-lattice | 64 | 128 | 0.500 | **WIN** |
| 03 | Kepler orbit | F1 ring-lattice | 64 | 128 | 0.500 | **WIN** |
| 04 | Mars / Horizons | F1 ring-lattice | 64 | 128 | 0.500 | **WIN** |
| 05 | Quasiperiodic torus | n/a | 200 | none (drifts out at n=102400) | — | non-win (7th under an alternate, less strict, but internally consistent protocol — not banked) |
| 06 | Planar Brownian motion | non-degenerate | 200 | 6400 | **0.969** | **WIN** |
| 07 | CR3BP | F1 ring-lattice | 64 | 64 | — | tie → non-win |
| 08 | Lorenz attractor | n/a | 100–200 | 200 | — | tie → non-win |
| 09 | Rössler attractor | n/a | — | — | — | non-win (AutoResearch-era "win" retracted, §11.3) |
| 10 | Driven pendulum | F1 ring-lattice | 64 | 128 | 0.500 | **WIN** |

**6 of 10 verified: 01, 02, 03, 04, 06, 10. Goal (≥6/10) MET.**
Criterion (c): 0 of 10 (every candidate — 05, 06, 08, 09 — refused once the
baseline received the same fair-window treatment). Criterion (b) stays
retired, correctly unused throughout.

## 11.8 Mandatory caveats that travel with the six

1. **Five of six wins are the same mechanism, one more time.** 01, 02, 03,
   04, and 10 all read sentinel (degenerate + near-degenerate) fraction
   ≈1.00 — the win means "fewer points to identify a ring," not a more
   accurate fit. This is now four rounds' worth of the identical finding.
2. **Every headline savings figure is smaller than previously recorded,
   once the baseline is fairly tuned.** 01 falls from its round-3
   self-reported 0.875 to **0.500**; 02/03/04/10 fall from 0.75 to
   **0.500**. The round's largest genuine win is now **06 at 0.969**, not
   01's inflated 0.875 — and 06 is also the round's only non-degenerate
   result. This same fairness gap (an un-tuned, hard-coded baseline
   window) plausibly affects round 1 and round 2's recorded savings
   figures too; those are not retroactively corrected here, but should not
   be read as the final word on the traditional method's real performance.
3. **All five ring-lattice wins die at a Theiler window of roughly 2
   sample spacings**, poly's side first. Quote this alongside any of the
   five; it is a shared fragility, not a per-problem one.
4. **The ring-lattice mechanism requires deterministic sampling.** On the
   same manifold with i.i.d.-random instead of regular sampling, the
   result inverts completely (§11.4). "Smooth closed 1D orbit" was never
   sufficient scope on its own.
5. **Problem 05 remains the sharpest open lead, not a settled failure.**
   Its shell-growth estimate holds within 0.04 of the truth from n=200 to
   n=102400 while every swept baseline radius window drifts monotonically
   downward and eventually exits tolerance — R2-F9's point, restated with
   far stronger evidence than either prior round had, and still
   unscoreable by either win criterion as currently defined.

## 11.9 Carry-forward

1. **Audit round 1 and round 2's recorded savings figures against a fairly
   -tuned baseline before citing them going forward.** §11.8's point 2
   applies generally, not only to round 3's own numbers.
2. **Give `comparison.py`'s convergence functions a mandatory baseline
   -window arm.** `baseline.correlation_dimension` already exposes
   `r_min_frac`/`r_max_frac`; `minimum_points_for_target_accuracy` and
   `poly_algebraic_minimum_points` do not, which is what let an unfair
   default window go unnoticed for two full rounds. This is the single
   highest-value fix for round 4.
3. **Formalize the "matched fairness" check as code, not a one-off
   audit.** §11.6's finding (measure both methods under the same
   standard — both answer-blind or both oracle-tuned — never one of each)
   should become a reusable assertion in the comparison harness, not
   something a skeptic has to rediscover by hand each round.
4. **Problem 05/06-style non-degenerate targets are the next round's
   real opportunity**, not the ring-lattice family. 06 is the strongest
   result either round has produced; 05 is the sharpest unresolved lead.
   Six repetitions of the same F1 mechanism have now been measured char-
   acterized in more forensic detail than the underlying phenomenon
   probably warrants.
5. **N8's fallback-branch threshold and the node-local layer's constants
   (0.15/0.25/5) still need a broader calibration set** before the next
   round trusts them without question — real progress was made (round 3
   broadened from 1 to 5 must-fire examples) but threshold circularity is
   reduced, not eliminated.
6. Criterion (c) as currently defined has never produced a single verified
   win across two full attempts (round 2 and round 3). Before extending
   it further, ask whether the definition itself, not just its
   application, needs revisiting.

## 11.10 Tier C — interpretation, explicitly not load-bearing

The six verified wins are statements about the sample efficiency (five
cases) or the accuracy (one case, problem 06) of a graph statistic on
specific finite point clouds — Tier B, same convention as every prior
round. The programme's stated goal was reached, but the path there
(a fairness bug in the *comparator*, not the *estimator*, that took a
four-level review chain and an explicit re-litigation of "what counts as
fair" to surface) is arguably the more durable result of this round: it is
now demonstrated, not merely argued, that "verify the auditor" has to
include auditing the fairness of the instrument the auditor built to do
the auditing, and that a review which corrects an asymmetry in one
direction and not the other can look more rigorous than it is.
