# MENSURA-BENCH v2 — adopted design and critical path

**Status:** adopted 2026-08-14. Supersedes the three-axis scoring of the
Stage-2 roadmap. The design (universality principle, Certified Interval
Criterion, truth taxonomy, tiered suite, zero-knob operation) is the owner's;
this document records it as adopted and adds the implementation analysis that
the F3/F4 diagnostic data forces on it.

**Naming:** the instrument is **MENSURA**. The name "Poly-Algebraic" is
reserved for the PIVP solver (Engine C) at programme level — see the Stream-0
spec, `SocrateAI-Mathesis/docs/SPEC-STREAM0-DRAFT-received.md` §6.3. Rename is
staged: human-facing names now, the twelve `poly_algebraic_*` API identifiers
during the Phase-A rewrite of `comparison.py` (deferring avoids ~60 files of
churn with a live risk of silently perturbing recorded numbers).

---

## 1. What v2 changes, in one paragraph

Scoring collapses from three axes to one: **a method succeeds on a problem iff
it emits an interval that contains the declared truth and is no wider than that
family's pre-registered sharpness cap.** Being confidently wrong (a `MEASURED`
interval that misses the truth) is a **calibration violation** and invalidates
the round for that method. `UNDECIDED` never violates and never scores.
Everything else — points-to-converge, compute savings, robustness sweeps —
survives as a mandatory diagnostic on the certificate but wins nothing.

Rationale, restated because it is the load-bearing argument: every dispute in
three rounds of this benchmark was a dispute about *comparative* scoring. The
one thing nobody gamed and nobody disputed was whether an estimate was near the
truth. v2 keeps what survived adversarial review and demotes what did not.

---

## 2. Critical path: v2 is not runnable against today's code

This is the main finding of the adoption review. **Two capabilities do not
exist yet, and CIC cannot be evaluated at all without them.** Both were
optional upgrades under the old roadmap; they are now prerequisites.

### 2.1 Interval output (blocking)

`local_dimension`/`mean_dimension` return a point estimate plus `r_squared`.
CIC requires `[d_lo, d_hi]`. Nothing in the module produces an interval.

**Recommended interval-builder, and it is data-backed rather than assumed.**
The F4 sweep shows the two readouts fail in *opposite directions* on a 2-D
uniform support at n=12800: shell reads **2.08–2.20** (biased high), GP reads
**1.927** (biased low), truth 2.000. An interval spanned by two readouts that
bracket the truth from opposite sides is honest by construction and widens
exactly when the readouts disagree — which is the behaviour B1's dual
shell/spectral readout was proposed to give. Build B1 first and let the
interval fall out of it, rather than bolting an error bar onto a point
estimate.

### 2.2 Zero-knob self-configuration (blocking, and load-bearing for validity)

`k` and `max_radius` are caller-supplied today. Zero-knob operation is not a
fairness convenience here — it is what stands between the instrument and a
calibration violation. Evidence, same sweep, Cantor dust at n=12800, one
target, varying only `k`:

| k | shell estimate |
|---|---|
| 8 | −0.13 |
| 10 | −0.39 |
| 15 | **+0.99** |
| 25 | +0.40 |

A caller choosing `k` chooses the answer, across a range of 1.4 in dimension.
Self-selection from the data, recorded in the certificate, is therefore a
correctness requirement, not a tidiness requirement.

---

## 3. Core-10 corrections forced by the diagnostic data

### 3.1 Cantor dust is the highest-risk row in the suite

The row is well-motivated (D₀ = D₂ there, so it sidesteps the F4 type
mismatch). But measured behaviour is a round-killer:

| | value |
|---|---|
| truth (2-D dust) | **1.2619** |
| GP | 1.2653 (error 0.003) |
| shell, across k | **−0.39 … +0.99**, frequently **negative** |
| well-fit nodes | as low as 10/80 |

A **negative dimension estimate** is not a near miss; it is the estimator
operating outside its domain, on a totally disconnected support where the k-NN
graph fragments. Under CIC, emitting `MEASURED` here is a calibration
violation that invalidates MENSURA's entire round.

**Consequences, both mandatory:**

1. **MENSURA must be able to abstain.** An explicit out-of-domain detector is
   now a Phase-B deliverable, not a nice-to-have. Two detectable signals are
   already visible in the data: a non-positive dimension estimate, and
   instability of the estimate across the method's own internal `k` selection.
   Graph-fragmentation (component count of the k-NN graph) is the obvious
   structural third.
2. **Keep the row.** It is the suite's only genuine negative control for the
   instrument itself. A benchmark on which the instrument cannot fail is the
   Goodhart trap in a new costume; this row is where MENSURA is *supposed* to
   say `UNDECIDED`, and saying so should count as a pass of the validity
   condition.

### 3.2 Declared truth value needs disambiguating

The draft lists Cantor dust as `EXACT ln2/ln3` ≈ 0.6309, which is the **1-D**
middle-thirds Cantor set. The 2-D Cantor *dust* (the product set, which is what
the F4 targets used) has D₀ = **2·ln2/ln3 ≈ 1.2619**. Pre-registration must
state which construction, in which ambient dimension, since the two differ by a
factor of two.

### 3.3 The 4-torus row will likely be UNDECIDED, and that is acceptable

Shell already carries a **+5% to +10% positive bias at D = 2** (2.08–2.20
against truth 2.000) and Round-3 data showed the bias growing with dimension
(a 3-D cube reached only ~2.47 at n=1024). At D = 4 either `W_4` must be wide
enough to be nearly uninformative, or the honest verdict is `UNDECIDED` at
attainable n. Setting expectations now: **Core-10 will probably score low on
its first run, and that is the criterion working**, not a regression.

---

## 4. Preserved from the diagnostic work

- **F3 is structurally dead, not merely fixed.** The taxonomy rule — no target
  may be produced by any method in the comparison panel — makes circular
  targets impossible rather than caught-by-review. The in-house Kaplan–Yorke
  values computed for this purpose are `D_KY(Lorenz) = 2.062152 ± 0.000024` and
  `D_KY(Rössler) = 2.013242 ± 0.000059`, from a Benettin solver validated
  against an 8-case known-answer battery (Hénon λ₁ vs published, err 3.7e−04;
  Hénon Σλ = ln 0.3, err 1.6e−14) with the analytic divergence identity holding
  to ~6e−06 across 96 runs.
- **F3's fix must not re-import F4's error.** `D_KY` is the *information*
  dimension D₁ under the Kaplan–Yorke conjecture — not D₀, not D₂. For Lorenz
  the ordering is D₂ ≈ 2.05 ≤ D₁ = 2.0622 ≤ D₀. Using KY as the target for a
  D₀-estimator would replace a circular target with a mis-typed one. It is the
  independent anchor **for D₁ only**; the `INDEPENDENT` class must carry the
  dimension type alongside the value.
- **F4 as a standing diagnostic.** Shell tracks D₀, GP tracks D₂; on targets
  built so D₀ − D₂ ≈ 1.0 each estimator goes to its own quantity. Core-10
  avoids the problem by choosing D₀ = D₂ targets, which is correct for
  *scoring* — but the multifractal targets belong in Extended-30 as the
  evidence that the two instruments measure different things.

---

## 5. Open items for pre-registration

Deferred to `MENSURA_PREREGISTRATION.md`, which must be committed before any
v2 run:

- `W_family` sharpness caps — **do not invent these**. Derive them once, from
  the median interval width the three-baseline panel produces on the `EXACT`
  uniform rows at the reference n, rounded up; then freeze. This is answer-blind
  and defensible, where hand-chosen caps are neither. **But see §6 first: the
  achievable floor is now measured, and it is not small.**
- Which Cantor construction (§3.2), and the ambient dimension of every target.
- The dimension *type* attached to each `INDEPENDENT` truth (§4).
- Noise ladder rows and σ values.
- The frozen default configuration for every panel method, and the abstention
  rule that decides `UNDECIDED` (§3.1).
- LL-6 retrieval pass on: Kaplan–Yorke published values, Myrheim–Meyer, TwoNN,
  and every `CONSENSUS` range cited.

---

## 6. E4 measured: the uniform-square bias, and what it costs `W_family`

Decision E4 asked whether the shell estimator's +0.02…+0.17 error on a plain
uniform square is boundary effect, finite-size effect, or variance, since that
determines whether B3 (boundary correction) fixes it before the caps freeze.
Measured (`scripts/hypergraph_benchmark/diagnostics/e4_uniform_square_bias.py`,
truth exactly 2.0 by construction):

**It is none of the three, and the roadmap's B3 assumption is refuted.**

| hypothesis | prediction | measured |
|---|---|---|
| finite-size | bias shrinks with n | bias **grows**: −0.053 (n=800) → +0.107 (n=12800); interior-only flat at +0.14…+0.16 |
| boundary | interior restriction removes it | interior restriction makes it **worse**, median \|err\| +0.046 |
| variance | sign unstable across seeds | sign stable 6/6, mean +0.085, sd 0.025 → **bias** |

The mechanism is the reverse of the assumed one: **boundary nodes are biased
*down*, interior nodes biased *up*, and the all-node figure is a partial
cancellation of two opposite biases.** Applying a boundary correction would
remove the cancellation and *expose* the full interior bias of +0.14…+0.21.
**B3 must not be applied to this estimator without re-deriving its
justification** — as specified it would make the flagship EXACT rows worse.

A follow-up tested whether the bias lives in the small-radius regime, where
ball growth is driven by `k` rather than geometry (the bias scales with k:
+0.004 at k=6/mr=4 up to +0.21 at k=15/mr=4). Starting the fit window at
radius 2 or 3 instead of 1 does **not** give a clean fix either: at k=6 it
makes things markedly worse (+0.05 → +0.20 → +0.24), at k=10 it is roughly
flat, and only at k=15 does it help (+0.076 → +0.029). No single window rule
improves all k.

**Consequence for pre-registration, which is the decision-relevant part.**
On a target whose dimension is known *exactly*, with the estimator free to
pick any (k, max_radius, min_radius) in the ranges tested, the achieved error
spans roughly **+0.03 to +0.24**. Under zero-knob operation MENSURA must
commit to one configuration in advance, so the honest expectation is an error
of order **0.1–0.2 on EXACT 2-dimensional rows**. Therefore:

- A `W_family` cap tighter than about **0.25** on EXACT 2-D rows would fail
  MENSURA on every one of them, regardless of interval construction.
- Reducing that floor is a genuine research task (the bias is systematic,
  k-dependent, and not explained by boundary or finite-size effects), not a
  correction to be slipped into Phase A.
- The caps should still be derived from the baseline panel as planned — but
  the panel's spread is now known to be the *optimistic* side of the story,
  and the gap between the two is itself the first honest thing v2 will report.
