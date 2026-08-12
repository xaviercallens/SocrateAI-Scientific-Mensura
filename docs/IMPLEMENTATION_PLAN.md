# Implementation plan: model-tiered workflows for the HaloAlg/SOCRATES programme

How to run this programme with Claude Code using the right model tier for each
job. Principle: **route by verifiability, not difficulty.** Work whose
correctness a test can decide goes to the cheapest tier that passes the gate;
work whose correctness requires judgment goes up-tier. Every workflow ends in a
machine-checkable gate, so a wrong cheap attempt costs one failed test run, not
a wrong conclusion.

## Tier routing table

| Tier | Use for | Never use for |
|---|---|---|
| **Haiku** | Mechanical, high-volume, test-guarded work: writing parametrized tests from a spec table, data-fetch plumbing, cache/checksum code, docstring and lint fixes, porting a verified algorithm between languages, running experiment sweeps and tabulating results | Choosing gates/tolerances, interpreting anomalies, anything whose failure mode is *silently wrong* |
| **Sonnet** | Module implementation against a written contract: new solvers matching a reference, RNS/operator extensions with property tests, refactors under a green suite, first-draft experiment scripts, figure generation | Novel mathematical claims, correcting the paper, designing controls |
| **Opus** | Design and adjudication: choosing numerical controls (the dt-independence/energy-oracle pattern), deriving and checking bounds (the Farey-bound correction, the Theorem 2.5 counterexample), reviewing Sonnet's modules for silent-failure modes, writing FINDINGS entries, deciding when a result is real | Bulk mechanical edits (wasteful) |

Escalation rule: any tier may **flag up**, never **guess down**. A Haiku task
that meets an unexpected result (a gate that fails, a bound that's violated)
stops and reports; it does not loosen the gate. Loosening gates is an
Opus-only action and must be justified in writing in `docs/FINDINGS.md`.

---

## Workflow W1 — Sym²-constrained shell model (the decisive experiment)

**Goal G1:** measure whether the Sym² lock changes the Hypothesis-U exponent
from −2/3. This is the programme's sharpest open question; either outcome is
publishable.

| Step | Tier | Task | Gate |
|---|---|---|---|
| 1 | Opus | Derive the Sym²-constrained coupling: shell amplitudes as symmetric squares of a half-ladder, spectrum restricted to {λ², λμ, μ²}; write the model contract in `docs/` | Written derivation reviewed against Thm 3.1 certificates |
| 2 | Sonnet | Implement `dualscale/shell_sym2.py` to the contract, reusing the adaptive-RK4/energy-oracle machinery from `shell.py` | Energy oracle: inviscid drift < 1e-6; suite green |
| 3 | Haiku | Sweep α' ∈ 10⁻²…10⁻¹⁰, tabulate peak enstrophy, fit exponent (copy `experiment_stage1.py` pattern) | All runs terminate `t_max`; drift < 1e-6 each |
| 4 | Opus | Adjudicate: exponent ≈ 0 (supports Conjecture 5.2), ≈ −2/3 (lock inert), or other (new physics/bug); write FINDINGS entry | dt-refinement control at two cfl values agrees |

**Loop L1** (repeat until stable): steps 3–4 with doubled shells and halved
cfl until the fitted exponent moves < 0.02 between iterations.

## Workflow W2 — Solver ladder maintenance and extension

**Goal G2:** every solver change re-climbs the ladder; new physics enters as a
new rung, never by editing an old rung's gate.

| Step | Tier | Task |
|---|---|---|
| 1 | Haiku | Run `scripts/solver_ladder.py` on every PR touching `solvers/`; paste the table into the PR |
| 2 | Sonnet | Add rungs by pattern: restricted three-body (oracle: Jacobi constant), Mercury perihelion with 1PN term (oracle: 43″/century), Horizons targets beyond Mars |
| 3 | Opus | Set each new rung's gate from the physics (what residual is *expected*), not from what the first run produced |

**Loop L2** (per new rung): Sonnet implements → Haiku runs ladder →
if FAIL, Opus decides "bug vs gate mis-set" — Sonnet never both implements
and re-gates its own rung.

## Workflow W3 — EIU / exact-arithmetic layer

**Goal G3:** an exact-mode path for every solver where forces are rational,
with bit-growth economics measured, en route to the RNS-backed dot-product
kernel.

| Step | Tier | Task | Gate |
|---|---|---|---|
| 1 | Haiku | Benchmark `RNSCodec.dot` vs int dot vs float dot across sizes; tabulate | pytest-benchmark artifacts committed |
| 2 | Sonnet | Fixed-point ↔ RNS bridge for solver states (scaled-integer encoding); collapse scheduling (the K-step policy) with per-collapse certified error via `collapse_error_bound` | Property tests: cumulative certified error bounds the true error |
| 3 | Sonnet | Exact-mode pendulum via rational Taylor sin (the spec's Constructive-Softmax pattern, order N = ⌈log D_max⌉) | Agreement with float mode to certified truncation bound |
| 4 | Opus | Review: where does exactness actually pay? Write the honest scoping note (gravity's 1/r² is outside exact mode — square roots) | FINDINGS entry |

**Loop L3:** benchmark → optimize hot channel (Sonnet may propose Rust port;
Haiku executes the port against the existing PyO3 pattern in
`rust/socrates_numerics`) → re-benchmark; stop when RNS dot is within 10× of
native int (document the constant, don't chase it).

## Workflow W4 — Paper corrections and Lean alignment

**Goal G4:** the paper, the Lean development, and this repo state the same
theorems.

**Status:** `lean/CallensDualScale.lean` received and checked (Lean 4.32.2 +
Mathlib, builds clean, axiom footprint verified — see `docs/FINDINGS.md` §2b).
It contains 3 theorems: `genesis_no_singularity`, `Reff_ge_sqrt`,
`sym2_recurrence`. Step 1's checklist below is therefore partially done —
these three are checked; the *rest* of the paper's LEAN-tagged theorems
(notably anything addressing Theorem 2.5's invariance form, and Theorems
2.3–2.4) have not been located and remain open. The full original paper
source (LaTeX, with LEAN tags) has not been supplied to this repo, so step 1
cannot be completed exhaustively until it is.

| Step | Tier | Task | Status |
|---|---|---|---|
| 1 | Haiku | Extract every LEAN-tagged theorem from the paper into a checklist table | Partial — 3 theorems checked ad hoc; full paper source needed for the rest |
| 2 | Opus | For each: does the Python certificate check the *same statement*? (Theorem 2.5 already caught; the spec's `sym2_recurrence`-proves-bit-width overload already caught) | Done for the 3 received theorems — see FINDINGS §2b cross-validation |
| 3 | Sonnet | Write the Python-side certificate for any statement lacking one | Not started |
| 4 | Opus | Produce the errata list for the next paper draft | Not started |

**Loop L4:** rerun after every paper revision; the checklist diff is the
review.

## Standing rules (all workflows)

1. **Gates are code.** A claim without a pytest/certificate is Tier C by
   definition and may not appear in FINDINGS as a result.
2. **Controls before contrasts.** Any classical-vs-regularized comparison ships
   with its dt-independence and conservation-oracle controls, or it doesn't ship.
3. **Exact where algebraic, float where dynamical, never silently mixed.**
   Fraction arithmetic decides algebraic identities; float+oracle decides
   dynamics; each module states which regime it is in.
4. **Network at the edges.** Real-data fetches are cached, checksummed, and
   `network`-marked; CI runs offline.
5. **Failed gates are findings.** The Farey-bound violation and the Theorem 2.5
   counterexample both came from gates failing; record them, don't tune them away.
