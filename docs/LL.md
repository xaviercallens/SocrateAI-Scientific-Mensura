# Lessons learned: building and benchmarking the MENSURA

Distilled from building `src/socrates/hypergraph/` and running it against 10
known physics problems (`docs/MENSURA_BENCHMARK.md`). Written so the
*next* module in this repository — or the next round of this one — starts
from these lessons instead of rediscovering them. Each entry states what
happened, why it matters beyond this one module, and what to do about it.

---

## 1. A green test suite tests what you thought to test, not what is true

**What happened.** `local_dimension`'s known-answer tests (a path measuring
dimension 1, a grid measuring dimension 2) passed cleanly from the start.
The estimator still had two real, independent defects that those tests
never exercised: an R² catastrophic-cancellation bug that made
`mean_dimension` return `nan` at its own documented default `max_radius`,
and an O(n²)-compounding performance bug that made a realistic convergence
sweep take over two minutes. Neither showed up until the module was run on
*ten different real problems* rather than the two or three synthetic
lattices its own unit tests used.

**Why it generalizes.** This is the same lesson `docs/FINDINGS.md` recorded
for the W1 stepper (`src/socrates/dualscale/shell_sym2.py`): a test suite
confined to the configurations its author thought to check is not evidence
about configurations nobody thought to check. Unit tests on synthetic,
hand-constructed inputs (a perfect circle, a perfect grid) are necessary but
not sufficient — they establish that the *algorithm* is correct on the
shapes you built to exercise it, not that the *implementation* is robust
across the shapes reality will actually hand it.

**What to do about it.** Before trusting a numerical module, run it on
inputs you did not design to be easy — real data, adversarially-chosen
parameters, larger scale than your unit tests use. A benchmark against real
problems is not a nice-to-have on top of unit tests; it is the only thing
that finds the defects unit tests are structurally unable to find.

---

## 2. The bug is often in the fit-quality diagnostic, not the fit

**What happened.** Both real bugs found post-benchmark were in code
*adjacent* to the core computation, not the computation itself: the R²
formula's `ss_tot > 0` exact-zero comparison (not the slope computation),
and `Hypergraph.adjacency()`'s lack of caching (not the BFS logic). The
slope/dimension values were correct throughout both bugs.

**Why it generalizes.** Diagnostic and auxiliary code (fit quality, error
bars, performance) gets less scrutiny than the "real" computation, precisely
because it feels secondary. But a caller who trusts a wrong diagnostic
trusts a wrong thing with more confidence than one who trusts no diagnostic
at all — `r_squared = 1.0` on a fit that should have failed is *worse* than
no `r_squared` field, because it actively vouches for a bad answer. Finding
F1 of the benchmark generalizes this further: even a *correct* diagnostic
(`r_squared == 1.0`, honestly computed) can be a **sentinel of "nothing to
fit" rather than "fit well"** — 6 of 10 benchmark "passes" were exactly
this, a k-NN graph on a smooth closed curve producing a perfectly constant
shell sequence that no method could possibly report as anything but a
perfect fit.

**What to do about it.** Give a diagnostic field its own known-answer tests,
separately from the value it diagnoses. And when a diagnostic can be
trivially maximized by a degenerate input (zero variance, in this case),
expose that degeneracy as its own explicit flag (`DimensionEstimate.degenerate`,
added as N2) rather than letting a caller infer it from context they may not
have.

---

## 3. A statistic corrupted by the thing it is meant to detect fails silently

**What happened, twice.** (a) The original attempt to calibrate a
duplicate-point detection threshold used the point cloud's own median
nearest-neighbour distance as the scale reference — but a duplicated
point's "nearest neighbour" *is* its own duplicate, so the very presence of
duplicates corrupts the statistic meant to detect them. The detector
silently never fired. (b) The exact same shape of bug appeared in the
original benchmark itself: round 1's agents used `is_well_fit()` (r² ≥
0.9) as their signal for whether a measurement was informative, but the
degenerate constant-shell regime *always* reports r²=1.0 — the diagnostic
that was supposed to flag "this measurement is uninformative" instead
actively certified the uninformative case as the best possible one.

**Why it generalizes.** Any "detect anomaly X using statistic S" design
should ask: does X's presence change S in a way that could mask X? A scale
reference, a quality score, a confidence interval — all are vulnerable to
this if computed *from the same data the anomaly lives in*, rather than
from an independent reference (here: the point cloud's bounding-box extent,
which duplicates do not measurably affect).

**What to do about it.** Prefer calibrating a detection threshold against a
scale that is structurally independent of the thing being detected. When
that is not possible, test the detector specifically on cases engineered to
corrupt its calibration statistic, not only on cases where the anomaly is
mild.

---

## 4. "It converged" needs a convergence sweep, not one number

**What happened.** Both chaotic-attractor benchmark problems (Lorenz,
Rössler) reported a dimension estimate within tolerance of the literature
value at their tested point count. An auditor's convergence sweep — varying
only n — showed both estimates rising monotonically *through* the
literature value and past it, still climbing at the largest n tested. The
"passing" numbers were not measurements; they were a snapshot of a still-moving
quantity that happened to be sampled near the target.

**Why it generalizes.** This is the single most expensive lesson in the
whole exercise, and it recurred: the *same* pattern (bigger sweep, refuted
by a denser sweep, three rounds running in `docs/FINDINGS.md`'s W1 work)
showed that when a validation strategy is inherently sample-limited — finite
seed sweeps, finite point counts, finite radius windows — reporting a single
configuration's result as "the" answer is a category error, no matter how
close that one number lands to the expected value. Convergence has to be
demonstrated by showing the answer stop moving as the limiting resource
(points, seeds, radius) increases, not inferred from one sample landing
close to a target.

**What to do about it.** Any claim expressed as a single number derived
from a finite, controllable resource (point count, seed count, sweep
range) needs that resource varied and the result shown to plateau before
the number is trusted. This is now built into the tooling directly:
`baseline.minimum_points_for_target_accuracy` and
`comparison.poly_algebraic_minimum_points` both require the estimate to
stay within tolerance at *every larger* n tested, not just cross into
tolerance once.

---

## 5. Verify the auditor, not only the workers

**What happened.** A 10-agent benchmark run was reviewed by an Opus
auditor, who correctly caught real structural problems the original agents
missed. A second, independent skeptic was then set loose specifically on
the *auditor's* ledger — and found three real errors in the audit's own
added analysis: an over-generalized bug claim ("fires for any clean
1-manifold" when it fired for specific shell values only), a wrong
speculative reattribution that overturned a *correct* original finding
without testing the counter-claim, and a factually wrong claim about which
sub-results had been checked, which happened to prop up one of the ledger's
two headline "success" verdicts.

**Why it generalizes.** A reviewer/auditor/adjudicator step is not exempt
from the discipline it applies to others. It is, if anything, *more*
exposed to a specific failure mode: over-generalizing a real, narrow finding
into a sweeping claim, because a narrow claim reads as less satisfying to
write up. And a reviewer's own favorable verdicts get less scrutiny from the
reviewer than the verdicts it is busy criticizing — exactly because writing
the criticism *feels* like the rigorous part of the job.

**What to do about it.** In any workflow with a review/adjudicate stage,
add an explicit stage that checks the adjudicator, not just the original
work — with its own instruction to default to "not accurate" and to
spot-check the reviewer's *positive* verdicts as hard as its negative ones.
This is now a standing pattern
(`.claude/skills/decisive-experiment/SKILL.md`, updated alongside this
document) rather than something to reintroduce ad hoc each time.

---

## 6. "Traditional" needs an actual implementation, not a strawman

**What happened.** The first framing of "better than traditional" had no
concrete referent — the module estimates dimension, but nothing in the
repository *did* dimension estimation the traditional way to compare
against. Before any comparison could mean anything,
`src/socrates/hypergraph/baseline.py` had to implement the actual classical
method (Grassberger-Procaccia correlation-sum), verified on the same
known-answer cases, built on the same underlying primitive
(`scipy.spatial.cKDTree`) so a compute-cost comparison isolates the
*algorithm* difference rather than an implementation-quality difference
between a careful new module and a strawman old one.

**Why it generalizes.** "Better than the traditional approach" is not a
claim about a vibe; it is a claim about two specific, runnable procedures
producing two specific numbers on the same input. Skipping straight to
"our method wins" without a real baseline implementation is not a shortcut,
it is begging the question.

**What to do about it.** Build the comparator you are claiming superiority
over, to the same standard of correctness (known-answer tests, its own
convergence discipline) as the thing being compared. If building an honest
baseline is more work than expected, that itself is informative about how
seriously the comparison should be taken until it exists.

---

## 7. Fix what the benchmark finds, immediately and traceably

**What happened.** Both real defects found by the benchmark (the R²
cancellation, the adjacency performance bug) were fixed the same session
they were found, each with a regression test that reproduces the *exact*
case the benchmark surfaced (not a generic "does it work" test), and each
fix is referenced from the finding that motivated it
(`docs/MENSURA_BENCHMARK.md`'s N1/N5 entries updated to "✅ DONE" in
place, not left as permanently-open TODOs once actually resolved).

**Why it generalizes.** A benchmark that finds real bugs and doesn't lead
to fixes is a report nobody acts on. Tracing each fix back to the specific
finding that motivated it (not just "fixed some bugs") keeps the benchmark
document itself honest as a live record, and the regression test pins the
*exact* previously-broken case rather than a generic sanity check that
would not have caught the original defect.

**What to do about it.** Treat "next steps" sections (the N1–N8 pattern
here) as a literal backlog with a status, not a list of suggestions — update
each entry to done/still-open as work lands, in the same document, so the
document stays the single source of truth for what is actually fixed versus
merely diagnosed.

---

## 8. A methodology fix a review endorses is not the same as a methodology fix a review applied

**What happened.** The improvement loop's own §10 audit *diagnosed and
endorsed* the correct fix for a real methodology flaw: measuring a periodic
orbit's convergence on a time-ordered prefix (a growing smooth arc covering
under 1% of the orbit at small n) instead of a whole-orbit-covering prefix
(bit-reversal or Weyl-sequence reordering). It applied that endorsed fix to
re-verify problems 01, 02 and 10 — and then, in the very same document,
scored problems 03 and 04 as ties using the old, already-diagnosed-as-unsound
time-ordered construction, attributing the tie to "physics" and "grid floor"
reasons that were never tested against the alternative. A third-level
skeptic applying the audit's own endorsed methodology uniformly moved the
verified win count from 2/10 to 4/10 — not by finding a new bug, but by
finding the review had not applied its own conclusion consistently.

**Why it generalizes.** This is a variant of lesson 5 (verify the auditor)
one level sharper: it is not enough for a review to *state* the correct
method and use it somewhere. A methodology correction has to be swept back
over every measurement it logically applies to, not just the ones the
reviewer happened to be looking at when the fix was found — otherwise the
document ends up internally inconsistent, citing its own best practice in
one section and violating it three sections later, and nothing about a
green top-line "audit accurate" checkmark will surface that on its own.

**What to do about it.** When a review identifies that a technique X is the
correct way to measure a class of inputs, treat that as a checklist item to
re-run across *every* input in that class before finalizing the scoreboard —
explicitly ask "does this fix apply anywhere I did not just apply it?" as
its own review step, not just "is this fix correct where I used it?" A
skeptic verifying a review should always check for this pattern specifically:
a correct, stated principle applied selectively.

---

## 9. A fairness fix applied to only one side of a comparison looks more rigorous than it is

**What happened.** Round 3's adjudicator made a genuine, valuable discovery:
the traditional baseline's correlation-sum radius window was hard-coded and
materially misconfigured, so every prior round's "traditional_min_n" numbers
had been measured against a quietly handicapped comparator. Correcting it
was the right call — but the adjudicator applied the correction only to the
six problems where a fairer baseline removes a poly win (flipping one tie
to a loss, one win to contested), and left two problems ("baseline never
converges") unexamined, where the identical correction would have added
wins. The result read as a careful, skeptical, conservative 4-of-10 — and
was in fact an asymmetric application of a real fix, not a stricter
standard. A dedicated closing pass later found that applying the *same*
fairness correction to both sides, in every direction, moved the honest
count to 6 of 10 — the round's actual, harder-won result.

**Why it generalizes.** This is lesson 8's failure mode wearing a more
convincing disguise. Lesson 8 was "a stated correction not applied
everywhere it should have been" — a coverage gap, easy to frame as an
oversight. This is the same gap, but it happens to move the answer in the
direction a skeptical reviewer's instincts already favour (toward "fewer
wins, more scrutiny"), which makes it *look* like diligence rather than an
error. A reviewer auditing another party's claim for fairness is exactly as
capable of building an unfair *instrument* to do the auditing — a
comparator tuned generously for one side and left alone for the other —
and the direction of the unfairness being "toward skepticism" provides no
protection, because the goal was never skepticism for its own sake; it was
an accurate number.

**What to do about it.** Any fix billed as making a comparison "fair" must
be checked for whether it was applied to literally every problem the
comparison covers, not only the ones where the reviewer happened to be
looking when the unfairness was discovered — and specifically checked in
*both* directions: does applying it anywhere create a win as readily as it
removes one? If a fairness correction has only ever been observed to
remove wins, that asymmetry is itself the signal to check harder, not
evidence the correction is conservative. Prefer making the corrected
comparison a mandatory, symmetric code path (as round 3's carry-forward
item 2 now requires) over a one-off audit that has to be re-litigated by
hand each round.
