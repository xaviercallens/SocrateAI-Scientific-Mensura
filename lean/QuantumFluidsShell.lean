/- CROSS-STREAM IMPORT (2026): from SocrateAI-Scientific-QuantumFluids
   lean_src/QuantumFluidsShell.lean, where it is kernel-checked on
   leanprover/lean4:v4.33.0-rc2 + Mathlib 6d605ae1. This copy builds against
   THIS repo's pin (v4.32.2) -- if the build below ever diverges from the
   origin stream's, the origin is authoritative. Incoming material: subject
   to this repo's own adversarial verification before anything cites it. -/

/-
Formalisation of the conjugated complexification's energy conservation.

Closes audit ruling O2 of docs/designs/M2_W4_DISPERSIVE_SHELL.md, which
accepted the complexification as a labelled deformation "carrying no Tier A
backing until it has its own Lean development".

Deliberately mirrors MechanicaFluidorum's real-model development
(lean_src/DyadicShell_Statements.lean: `shellB`, `sum_mul_shellB`,
`shellB_energy_conservation`) so the two can be compared line for line, and
against the SAME pinned Mathlib revision, so a theorem here is checked
against the same library every result in that stream's LEDGER was verified
against.

WHAT IS AND IS NOT PROVED HERE. This file proves the algebraic identity that
the complexified nonlinearity conserves the energy pairing exactly. That is
the single load-bearing claim under W4's model -- memo section 2b, and the
property that makes the complexification usable at all. It says nothing about
boundedness, blow-up, or the dispersive regulator's effect on any observable;
those are Tier B or open.
-/

import Mathlib

namespace QuantumFluids.ShellComplex

/-- The complexified Katz-Pavlović shell nonlinearity, in the CONJUGATED form
selected by memo section 2b:

  `B_0(v)     = -(k_0 * conj(v_0) * v_1)`
  `B_{m+1}(v) = k_m * v_m^2 - k_{m+1} * conj(v_{m+1}) * v_{m+2}`

Squares are written as explicit products rather than `^2` to keep every
`Complex.re` expansion below a product of at most three factors, which `simp`
plus `ring` handles directly.

The single conjugation on the OUTGOING term is what makes the energy
cancellation work; the unconjugated version does not conserve (measured
numerically at ~1e+3 in exploration/verify_complexification.py). It is not
the unique conserving choice -- see memo section 2c(iii); it is chosen for
the invariant-subspace property proved as `shellBc_real` below. -/
noncomputable def shellBc (k : ℕ → ℝ) : ℕ → (ℕ → ℂ) → ℂ
  | 0,     v => -((k 0 : ℂ) * (starRingEnd ℂ) (v 0) * v 1)
  | m + 1, v => (k m : ℂ) * (v m * v m)
                  - (k (m + 1) : ℂ) * (starRingEnd ℂ) (v (m + 1)) * v (m + 2)

/-- The outflux through the seam above shell `m`, as a real number.
The energy pairing telescopes into differences of this quantity. -/
noncomputable def out (k : ℕ → ℝ) (v : ℕ → ℂ) (m : ℕ) : ℝ :=
  k m * ((starRingEnd ℂ) (v m) * (starRingEnd ℂ) (v m) * v (m + 1)).re

/-- **The identity that makes the cancellation work.**
`Re(conj(a)^2 * b) = Re(conj(b) * a^2)`.

Both sides are the real part of a pair of complex conjugates: applying
`starRingEnd` to `conj(a) * conj(a) * b` gives `a * a * conj b`, and
conjugation preserves the real part. This is the formal content of the
hand-derivation in memo section 2b. -/
theorem re_conj_sq_mul (a b : ℂ) :
    ((starRingEnd ℂ) a * (starRingEnd ℂ) a * b).re
      = ((starRingEnd ℂ) b * a * a).re := by
  simp [Complex.mul_re, Complex.mul_im]
  ring

/-- **Telescoping** (mirrors `sum_mul_shellB`): the energy pairing summed over
shells `0..N` collapses to minus the outflux at the top shell. -/
theorem sum_re_conj_mul_shellBc (k : ℕ → ℝ) (v : ℕ → ℂ) :
    ∀ N : ℕ, ∑ n ∈ Finset.range (N + 1),
        ((starRingEnd ℂ) (v n) * shellBc k n v).re = -(out k v N)
  | 0 => by
      rw [Finset.sum_range_one]
      show ((starRingEnd ℂ) (v 0) * -((k 0 : ℂ) * (starRingEnd ℂ) (v 0) * v 1)).re = _
      simp [out, Complex.mul_re, Complex.mul_im]
      ring
  | N + 1 => by
      rw [Finset.sum_range_succ, sum_re_conj_mul_shellBc k v N]
      have hkey := re_conj_sq_mul (v N) (v (N + 1))
      -- Expand every complex operation into real/imaginary components, so both
      -- the goal and the key identity become polynomial identities over ℝ.
      simp only [shellBc, out, Complex.sub_re, Complex.sub_im, Complex.mul_re,
                 Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im,
                 Complex.conj_re, Complex.conj_im] at hkey ⊢
      -- The cancellation is exactly `k N` times the key identity: the influx
      -- into shell N+1 is the outflux from shell N. `linarith` cannot scale a
      -- hypothesis by the VARIABLE `k N`, so `linear_combination` is required.
      linear_combination (k N) * hkey

/-- **Exact energy conservation for the complexified model (Tier A).**

Under the truncation boundary condition `v_{N+1} = 0`, the conjugated
complexified nonlinearity conserves the energy pairing exactly:
`Σ_{n≤N} Re(conj(v_n) · B_n(v)) = 0`, i.e. `d/dt (½ Σ |v_n|²) = 0` along the
nonlinear flow.

This is the complex counterpart of MechanicaFluidorum's
`shellB_energy_conservation`, and the formal backing audit ruling O2 required
before W4's model could carry a Tier A claim. -/
theorem shellBc_energy_conservation (k : ℕ → ℝ) (v : ℕ → ℂ) (N : ℕ)
    (hbc : v (N + 1) = 0) :
    ∑ n ∈ Finset.range (N + 1), ((starRingEnd ℂ) (v n) * shellBc k n v).re = 0 := by
  rw [sum_re_conj_mul_shellBc, out, hbc]
  simp

/-- **The reals are an invariant subspace.**

On real data the conjugation is inert, so `shellBc` returns a real number.
This is what makes MechanicaFluidorum's real Katz-Pavlović solutions -- including
the Katz-Pavlović finite-time blow-up solutions of the inviscid infinite system --
solutions of this model too, which is why the memo's O5 falsification trap survives
the change of model (memo section 2c(i)). Verified numerically to exactly 0.0;
here it is exact by construction. -/
theorem shellBc_real (k : ℕ → ℝ) (v : ℕ → ℂ) (hv : ∀ n, (v n).im = 0) (n : ℕ) :
    (shellBc k n v).im = 0 := by
  cases n <;>
    simp [shellBc, Complex.mul_im, Complex.mul_re, Complex.sub_im, hv]


/-! ## Liouville: the complexified flow is volume-preserving (shell-local part)

The review of 2026-08-14 (M2_REPORT §6b) established numerically that the realified
flow of the conjugated complexification has phase-space divergence ~4e-9 (round-off),
while the REAL Katz-Pavlović model has divergence `-Σ k_n a_{n+1} ≠ 0`. This section
formalises the algebraic core of that result.

SCOPE, stated precisely. `dv_n/dt` depends on `v_n` itself only through
(a) the outgoing term `-k_n · conj(v_n) · v_{n+1}` — an ℝ-linear map in `v_n` for
fixed `v_{n+1}` — and (b) the dispersive rotation `-(i D k_n²) v_n`. The phase-space
divergence is the sum over shells of the ℝ-trace of these diagonal blocks. The
theorems below prove each such trace is ZERO. What is NOT formalised here is the
differentiation-level statement that these maps ARE the diagonal blocks of the flow's
derivative — that step is by inspection of `shellBc` and is verified numerically
(finite-difference Jacobian trace, exploration record in M2_REPORT §6b). Tier A for
the trace identities; Tier B for the assembled model statement. -/

/-- The trace, over ℝ, of an ℝ-linear endomorphism of ℂ, read off in the basis
`{1, I}`: the coordinate of `f 1` along `1` plus the coordinate of `f I` along `I`. -/
noncomputable def rtrace (f : ℂ →ₗ[ℝ] ℂ) : ℝ := (f 1).re + (f Complex.I).im

theorem rtrace_add (f g : ℂ →ₗ[ℝ] ℂ) : rtrace (f + g) = rtrace f + rtrace g := by
  simp [rtrace]; ring

/-- `rtrace` is the trace: sanity anchor for the definition. Trace of multiplication
by `c` on ℂ ≅ ℝ² is `2·Re c`. -/
theorem rtrace_mulRight (c : ℂ) : rtrace (LinearMap.mulRight ℝ c) = 2 * c.re := by
  simp [rtrace, Complex.mul_im]
  ring

/-- **Dispersive block has zero trace**: multiplication by a purely imaginary constant
`d·I` (the quantum-pressure term has `d = -D k²`) preserves phase-space volume. -/
theorem rtrace_mul_I (d : ℝ) :
    rtrace (LinearMap.mulRight ℝ ((d : ℂ) * Complex.I)) = 0 := by
  rw [rtrace_mulRight]
  simp

/-- **Outgoing block has zero trace**: for every fixed `w`, the ℝ-linear map
`v ↦ conj(v) · w` has zero trace. This is why the conjugated nonlinearity —
unlike the real Katz-Pavlović form, whose divergence is `-Σ k_n a_{n+1}` —
contributes nothing to the phase-space divergence. -/
theorem rtrace_conj_mul (w : ℂ) :
    rtrace ((LinearMap.mulRight ℝ w).comp Complex.conjAe.toLinearMap) = 0 := by
  simp [rtrace, Complex.mul_im]

/-- **Shell-diagonal divergence vanishes.** The full diagonal block at shell `n` —
outgoing conjugated term (with `w = -k_n·v_{n+1}` folded into `w`) plus dispersive
rotation — has zero ℝ-trace. Summed over shells this is the Liouville property of
the complexified model. -/
theorem shell_divergence_zero (d : ℝ) (w : ℂ) :
    rtrace ((LinearMap.mulRight ℝ w).comp Complex.conjAe.toLinearMap
      + LinearMap.mulRight ℝ ((d : ℂ) * Complex.I)) = 0 := by
  rw [rtrace_add, rtrace_conj_mul, rtrace_mul_I]
  norm_num


/-! ## The conserving-seam theorem (both directions)

CLAIM-010 established, and `shellBc_energy_conservation` formalised, the
`⇐` half for the truncation seam. The telescoping identity
`sum_re_conj_mul_shellBc` in fact characterises EVERY conserving seam, and the
characterisation is the theorem that says what a T-dual "bounce" can be.

The clean way to state it avoids `Function.update` entirely: the telescoping
identity already gives the pairing as `-(out k v N)`, and `out k v N` is
`k N * Re(conj(v N)^2 * v (N+1))`. So "the seam value at N+1 conserves" is
literally "that real part vanishes", for the field `v` in which `v (N+1)` IS
the seam value. No update needed. -/

/-- **Seam characterisation.** For `k N ≠ 0`, the energy pairing over shells
`0..N` vanishes iff `Re(conj(v_N)² · v_{N+1}) = 0` -- i.e. iff the boundary
value `v_{N+1}` is orthogonal to `v_N²` under `Re(conj x · y)`. -/
theorem seam_conserves_iff (k : ℕ → ℝ) (v : ℕ → ℂ) (N : ℕ) (hk : k N ≠ 0) :
    (∑ n ∈ Finset.range (N + 1), ((starRingEnd ℂ) (v n) * shellBc k n v).re = 0)
      ↔ ((starRingEnd ℂ) (v N) * (starRingEnd ℂ) (v N) * v (N + 1)).re = 0 := by
  rw [sum_re_conj_mul_shellBc, out]
  constructor
  · intro h
    have h' : k N * ((starRingEnd ℂ) (v N) * (starRingEnd ℂ) (v N) * v (N + 1)).re = 0 := by
      linarith
    exact (mul_eq_zero.mp h').resolve_left hk
  · intro h
    rw [h]; ring

/-- Truncation `v_{N+1} = 0` conserves -- recovers `shellBc_energy_conservation`
from the characterisation. -/
theorem seam_zero_conserves (k : ℕ → ℝ) (v : ℕ → ℂ) (N : ℕ) (hk : k N ≠ 0)
    (hbc : v (N + 1) = 0) :
    ∑ n ∈ Finset.range (N + 1), ((starRingEnd ℂ) (v n) * shellBc k n v).re = 0 := by
  rw [seam_conserves_iff k v N hk, hbc]; simp

/-- **The GPE-like family conserves.** If `v_{N+1} = I·μ·v_N²` then the pairing
vanishes: `conj(v_N)² · (I μ v_N²) = I μ |v_N|⁴` is purely imaginary. -/
theorem seam_gpe_conserves (k : ℕ → ℝ) (v : ℕ → ℂ) (N : ℕ) (μ : ℝ) (hk : k N ≠ 0)
    (hbc : v (N + 1) = Complex.I * (μ : ℂ) * (v N * v N)) :
    ∑ n ∈ Finset.range (N + 1), ((starRingEnd ℂ) (v n) * shellBc k n v).re = 0 := by
  rw [seam_conserves_iff k v N hk, hbc]
  -- Expand into real and imaginary parts: the expression is I·μ·|v_N|⁴, whose
  -- real part is a polynomial identity that vanishes.
  simp only [Complex.mul_re, Complex.mul_im, Complex.conj_re, Complex.conj_im,
             Complex.I_re, Complex.I_im, Complex.ofReal_re, Complex.ofReal_im]
  ring

/-! ## Audit certificates (Gate 2 / L4.1)

Expected on every line: `[propext, Classical.choice, Quot.sound]` and nothing
else -- in particular no `sorryAx`, which would mean an unproved hole. -/

#print axioms re_conj_sq_mul
#print axioms sum_re_conj_mul_shellBc
#print axioms shellBc_energy_conservation
#print axioms shellBc_real
#print axioms rtrace_mulRight
#print axioms rtrace_mul_I
#print axioms rtrace_conj_mul
#print axioms shell_divergence_zero
#print axioms seam_conserves_iff
#print axioms seam_zero_conserves
#print axioms seam_gpe_conserves

end QuantumFluids.ShellComplex
