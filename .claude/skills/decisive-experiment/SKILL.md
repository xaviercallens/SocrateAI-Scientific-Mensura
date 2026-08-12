---
name: decisive-experiment
description: Launch a tier-routed, adversarially-verified Workflow for a decisive SOCRATES experiment (W1-W4 from docs/IMPLEMENTATION_PLAN.md, or any measurement the programme will act on). Use when a result needs to be trustworthy enough to change what Conjecture 5.2 / Hypothesis U claims, not for routine implementation or mechanical sweeps.
---

# Decisive-experiment workflow pattern

This project (SOCRATES: dual-scale T-dual regularization for Navier-Stokes
global regularity) has a validated shape for Workflow-tool orchestration on
any experiment whose outcome the programme will act on. It was arrived at by
running it once (W1 round 1) and watching a naive version fail silently —
see `docs/FINDINGS.md` §6 and the `socrates-workflow-orchestration` memory.

**Do not use this for routine work.** A new solver rung, a lint fix, a figure
regeneration — just do it directly or with a single Agent call. This pattern
is for the sharp, falsifiable measurements: W1 (Sym² lock vs. Hypothesis U),
and by extension W2-W4 in `docs/IMPLEMENTATION_PLAN.md` once they reach their
own decisive-measurement stage.

## The six-phase shape

1. **Repair/Design** (Opus, high or xhigh effort). Whatever the previous
   round's FINDINGS entry identified as the blocker — a numerical defect, an
   underspecified protocol, a missing control. Must produce *proof* against
   the real configuration (full window, real N, real cfl range), not a
   narrower one that happens to look clean. Tell the agent explicitly: a
   false "fixed" here poisons every downstream measurement, so an honest
   `fixed=false` is more valuable than a manufactured green.

2. **Verify the repair** (Opus, matching effort). An independent skeptic
   agent, defaulting to `refuted=true` unless the fix survives real attacks:
   read the actual diff for loosened gates/tolerances, reproduce the
   convergence claim from scratch (not the first agent's script), test
   generalization at configs the repairer didn't try, confirm nothing
   upstream broke (e.g. an invariant like exact energy conservation).
   **This is a hard gate** — if refuted, the workflow should halt and return,
   not continue with unverified data.

3. **Protocol/decision** (Opus, high effort). Any remaining open modelling
   choice the plan left unresolved (stopping criterion, seed convention,
   what a guard is actually for) — settle it empirically with measurements,
   not by picking the more convenient option.

4. **Measure** (Sonnet, parallel across arms). The actual sweep(s), run with
   identical seeds/settings across every arm so comparisons are like-for-like.
   Require agents to name the gate precisely (e.g. `t_max` reached AND drift
   below threshold) and to report scope reductions explicitly rather than
   silently trimming compute-expensive configs.

5. **Adjudicate** (Opus, xhigh effort). Validity before interpretation:
   did the measurement actually meet its gates? Only then compute the
   comparison the experiment exists to make (e.g. Δp between locked and
   control arms — never the single-arm number alone, if a dimension-matched
   control exists). **Name the most likely spurious-positive trap explicitly
   in the prompt** — for W1 that's the Galerkin-artifact trap (control arm
   also arresting means dimensionality did it, not the mechanism). Append the
   dated FINDINGS.md entry from this step, in the file's existing Tier B/C
   style.

6. **Refute the verdict** (Opus ×2+, parallel, genuinely different lenses).
   Not redundant copies — e.g. a numerics lens (is the measurement itself
   sound?) and a mechanism lens (does the conclusion actually follow, even if
   the numbers are right?). Tell every skeptic explicitly that a second
   honest "inconclusive" is a completely acceptable outcome, so there's no
   implicit pressure to rescue a clean answer.

## Tier routing (from docs/IMPLEMENTATION_PLAN.md, with one addition)

Use the plan's table as the default (Haiku: mechanical/sweep, Sonnet:
implementation-against-contract, Opus: design/adjudication) — but **escalate
tier on any step the rest of the pipeline is gated behind**, even past the
plan's default. Xavier confirmed this explicitly for W1 round 2 ("get more
intelligence" on the repair/verify/adjudicate/refute steps specifically).
Routine per-arm sweeps stay at Sonnet or Haiku; the steps that decide whether
those sweeps mean anything go to Opus.

## After the workflow completes

Regardless of verdict:

1. Read the actual journal/result before summarizing — do not narrate a
   result you haven't seen (see the harness's own guidance on this).
2. Run the full offline test suite (`pytest -m "not network"`) and `ruff
   check` before committing anything the workflow wrote.
3. Commit with a message that states the verdict honestly, including
   "inconclusive" or "anomaly" verdicts — round 1's commit message is the
   template (`287731f`, "sweep inconclusive, gates caught it").
4. Update the `socrates-dualscale-project` and `socrates-workflow-orchestration`
   memory files with what changed: new Tier A/B results, new standing
   methodology gotchas, current W-phase status. Keep the project memory's
   "status" paragraph current rather than letting it drift stale across
   sessions — a future session should be able to read it and know exactly
   where the decisive experiment stands without re-reading every FINDINGS
   entry.
5. If a genuinely new failure mode was discovered (like round 1's "green
   test at the wrong window" trap), add it as a bullet to this skill's
   phase descriptions above, not just to memory — the skill is what shapes
   the *next* workflow's prompts.
