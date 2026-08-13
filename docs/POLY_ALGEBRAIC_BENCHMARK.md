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
