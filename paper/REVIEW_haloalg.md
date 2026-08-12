# Tier triage of the HaloAlg specification

The spec's own epistemic rule — Tier A established, Tier B checkable, Tier C
conjecture, "never load-bearing" — applied to the spec itself. This is the
review the document would receive under the programme's stated discipline.

## Tier A/B — sound, and now implemented here

| Claim | Status |
|---|---|
| RNS add/sub/mul in parallel channels, CRT reconstruction | Textbook-correct (Garner, Szabó–Tanaka). Implemented and tested in `socrates/eiu/rns.py`, with corrections below. |
| T-dual metric `max(R, α'/R)`, bounce, minimal scale | Certified exactly (see `docs/FINDINGS.md`); note the Theorem 2.5 correction carried over from the first paper. |
| Sym² spectral restriction {λ², λμ, μ²} | Certified exactly for constant-coefficient order-2 operators. |
| Exact rationals distinguish real singularities from float artifacts | Correct in principle and demonstrated concretely by the Stage 1 controls. |

Corrections applied in the production RNS implementation:

1. **Recursive `extended_gcd` overflows the Python stack** for cryptographic-size
   moduli (recursion depth ~ number of Euclid steps). Replaced with the
   iterative form.
2. **RNS has no cheap division or comparison.** The spec presents RNS as if it
   closed the rational field; it closes the *ring* operations only. Division,
   sign detection, and magnitude comparison all require (partial) reconstruction
   — this is the classical weakness of RNS and any honest EIU design must budget
   for it. Stated explicitly in the module docstring.
3. **Overflow discipline**: products must stay inside the dynamic range M or
   they wrap silently. The implementation tracks range and raises.

## Tier C — conjecture presented with Tier A tone; must be relabeled

- **"Hallucination and Navier–Stokes blow-up share a common root."** An analogy,
  not a theorem. No mechanism connects transformer semantic drift to the field ℝ;
  transformers already run on discrete float lattices.
- **Celestial holography / w_{1+∞} as the *mechanism* of fluid regularity.**
  The cited soft-theorem structure applies to asymptotically flat gravity; no
  derivation is given (or known) mapping the Navier–Stokes enstrophy flux onto
  that algebra. Stage 3 of the validation table treats this as method — it is a
  hope.
- **"Memory Boundedness Theorem."** As stated it is not a theorem: periodic GCD
  reduction does *not* bound bit-width, because a fraction in lowest terms can
  still have arbitrarily large numerator and denominator. Bounding bit-width
  requires *rounding* (e.g. `limit_denominator`), which breaks exactness — the
  trade-off is real and quantified in `scripts/solver_ladder.py` (exact-mode
  denominators grow without bound; collapse bounds them at a measured accuracy
  cost). The theorem as stated should be withdrawn or restated as: *bit-width
  is bounded iff a lossy collapse is admitted, with error ≤ 1/(q·D_max)*.
- **"Great Dismantling" (LayerNorm etc. removable over ℚ).** Untested. LayerNorm
  does optimization-dynamics work (conditioning), not only numeric stabilization;
  removing it is an empirical question, not a corollary of exactness.
- **`sym2_recurrence` "proves bit-width growth is bounded"** — the Lean theorem
  of that name (verified in the first paper) proves a recurrence identity about
  squares of solutions. It says nothing about bit-width. The spec overloads a
  kernel-checked name onto an unproven claim; this is exactly the failure mode
  the tier system exists to prevent.

## Improvements adopted into the codebase

1. **Smooth T-dual metric** `R + α'/R` alongside the max-form, with the no-go
   lemma explaining the trade-off (see `dualscale/geometry.py`): exact duality +
   exact inertial invisibility *forces* the non-smooth max-form; smoothness
   costs exact invisibility (deviation α'/R, decaying above the cutoff).
2. **Smooth effective wavenumber** `k/(1 + α'k²)` — exactly T-dual-invariant,
   peaks at 1/(2√α'), C^∞; the frequency-space regularizer a PDE mollifier
   actually wants.
3. **RNS module** with the three corrections above.
4. **Solver ladder** (`scripts/solver_ladder.py`): the "known easy physical
   problem, increasing complexity" programme — harmonic oscillator → pendulum →
   Kepler → real JPL Horizons ephemerides → dyadic cascade — each level gated
   on measured error before the next unlocks, per the spec's staged-validation
   philosophy.
