# Computational findings — Stage 1 dyadic laboratory

Results from `scripts/experiment_stage1.py`, reproducible with
`python scripts/experiment_stage1.py`. Every claim below is Tier B
(decided by computation) unless marked otherwise.

---

## 1. Correction to Theorem 2.5 (T-duality)

**The draft's stated form is false.** The paper writes

$$\alpha' / R_{\text{eff}}(\alpha', \alpha'/R) = R_{\text{eff}}(\alpha', R)$$

Counterexample: $\alpha' = 1/4$, $R = 1000$.
- $R_{\text{eff}}(\alpha', R) = \max(1000, 1/4000) = 1000$
- $\alpha'/R = 1/4000$, so $R_{\text{eff}}(\alpha', \alpha'/R) = \max(1/4000, 1000) = 1000$
- Left side $= (1/4)/1000 = 1/4000 \neq 1000$

The statement the `max` form actually satisfies is plain **invariance**:

$$R_{\text{eff}}(\alpha', \alpha'/R) = R_{\text{eff}}(\alpha', R)$$

since $\max(\alpha'/R,\, R) = \max(R,\, \alpha'/R)$ identically. This is also the
statement the physics requires — "effective geometry cannot distinguish a radius
from its dual" is an invariance claim, not a reciprocal one.

**Action required — still open.** The Lean file received and checked in §6 below
(`lean/CallensDualScale.lean`) does **not** contain a `Reff_tdual` theorem or
any statement of the reciprocal/invariance claim — it proves positivity
(`genesis_no_singularity`) and a lower bound (`Reff_ge_sqrt`) instead. Theorem
2.5's Lean audit therefore remains unresolved: if a Lean statement of the
invariance form exists elsewhere, it still needs to be located and checked
against the corrected form below.

Pinned as a regression test: `test_draft_theorem_2_5_as_written_is_false`.

Theorems 2.2 (bound), 2.3 (bounce) and 2.4 (inertial invisibility) are
**confirmed** by exact rational arithmetic over sampled radii spanning both
sides of the cutoff.

---

## 2. Theorem 3.1 (Sym² lock) — confirmed

Verified in exact `Fraction` arithmetic. The construction here derives $L_3$
from the elementary symmetric functions of $\{\lambda^2, \lambda\mu, \mu^2\}$,
which is root-free and therefore valid for irrational and complex roots:

$$e_1 = a^2+b, \qquad e_2 = -b(a^2+b), \qquad e_3 = -b^3$$

reproducing the paper's

$$v_{n+3} = (a^2+b)v_{n+2} + b(a^2+b)v_{n+1} - b^3 v_n$$

Certified on squares **and cross-products** of independent solutions, including
the degenerate cases $a=0$ and $b=0$. Cross-products matter: an operator could
annihilate every square while failing on $\lambda\mu$, which is a genuine
element of the symmetric square's solution space.

Apéry sequences reproduce known values exactly with integrality preserved —
$\zeta(3)$: 1, 5, 73, 1445, 33001, 819005; $\zeta(2)$: 1, 3, 19, 147, 1251, 11253.

---

## 2b. Lean 4 kernel verification — three theorems now Tier A

`lean/CallensDualScale.lean` (Lean 4.32.2 + Mathlib, built against a full
Mathlib cache in ~6–20s per target, reproducible via `cd lean && lake build
CallensDualScale`) contains three theorems. All three now **kernel-check
successfully** — this required a fix, documented below.

| Theorem | Statement | Status |
|---|---|---|
| `genesis_no_singularity` | $R>0 \Rightarrow R_{\text{eff}}(\alpha',R) > 0$ | Compiled as received |
| `Reff_ge_sqrt` | $R>0 \Rightarrow \sqrt{\alpha'} \le R_{\text{eff}}(\alpha',R)$ | **Compiled only after a fix — see below** |
| `sym2_recurrence` | $u_{n+2}=au_{n+1}+bu_n \Rightarrow v_n=u_n^2$ satisfies $v_{n+3}=(a^2+b)v_{n+2}+b(a^2+b)v_{n+1}-b^3v_n$ | Compiled as received |

**`Reff_ge_sqrt` as submitted did not compile** — four errors (type mismatches
from a `sq`/`*`-form confusion around `Real.sq_sqrt`, a misapplied
`mul_div_assoc` direction, and a `¬R < √α'` vs `√α' ≤ R` mismatch in the second
branch). This is exactly the failure mode the project's tier discipline exists
to catch: a proof script that *looks* complete is not the same as one the
kernel accepts. The proof was rewritten (same statement, `le_div_iff₀` plus a
direct `mul_le_mul_of_nonneg_left` calc step) and now compiles clean.

**Axiom footprint** (`lake env lean AxiomCheck.lean`, i.e. `#print axioms` on
each): `genesis_no_singularity` and `Reff_ge_sqrt` depend on
`[propext, CallensDualScale.alpha_prime_pos, Classical.choice, Quot.sound]` —
only the one declared axiom (α' > 0) plus Lean/Mathlib's standard foundational
axioms. `sym2_recurrence` depends on `[propext, Classical.choice, Quot.sound]`
— it doesn't even need the α' axiom, as expected for a pure recurrence-algebra
result. No `sorry`, no smuggled axioms in any of the three.

**Cross-validation with the Python Tier B certificate.** `sym2_recurrence`'s
conclusion — $v_{n+3}=(a^2+b)v_{n+2}+b(a^2+b)v_{n+1}-b^3v_n$ — is *exactly* the
coefficient triple `(-b**3, b*(a**2+b), a**2+b)` that
`closed_form_symmetric_square()` in `src/socrates/operators/recurrence.py`
returns, and that `verify_symmetric_square()` checks by exact `Fraction`
arithmetic (Theorem 3.1, §2 above). Two independent methods — a Lean kernel
proof and exact-rational computation — now agree on the same closed form. This
promotes the squares-case of Theorem 3.1 from Tier B to Tier A; the
cross-products case (also certified in §2) has no Lean counterpart yet.

Note the scope: `t_dual_radius` in the Lean file is definitionally the same
function as `effective_radius` in `dualscale/geometry.py` (the `if R <
√α′ then α′/R else R` form equals `max(R, α′/R)` identically, since the two
branches agree at the boundary). `genesis_no_singularity` and `Reff_ge_sqrt`
are therefore Tier A confirmations of (part of) Theorem 2.2 — but note neither
proves T-duality invariance or inertial-range invisibility (Theorems 2.3–2.4),
which remain Tier B only.

---

## 3. Methodological warning: the blow-up must be shown dt-independent

A fixed-timestep explicit integration of the dyadic model **cannot** support the
paper's Stage 1 claim. With $k_n = 2^n$ over 35 shells, $k_{\max} \approx 1.7\times10^{10}$;
an explicit step of $10^{-5}$ violates the stability condition $\Delta t \cdot k|u| \ll 1$
by five orders of magnitude. The classical run then diverges **numerically**,
which is visually indistinguishable from physical blow-up.

This biases *in favour of* the thesis: T-dual regularization caps
$k_{\text{eff}} \le 1/\sqrt{\alpha'}$, which incidentally makes the regularized run
numerically stable. The apparent contrast would be an artifact of timestep, not topology.

**Controls used here instead:**
- Adaptive RK4 with $\Delta t = \mathrm{cfl}/\max_n(k_n|u_n|)$
- Energy conservation as an independent oracle (the inviscid nonlinearity telescopes, so $dE/dt=0$)
- Timestep-refinement study

Energy drift falls $2.4\times10^{-3} \to 1.3\times10^{-7}$ as cfl goes $0.4 \to 0.05$,
confirming the integrator converges. Regularized runs complete to `t_max`;
classical runs do not, under identical settings.

---

## 4. Main result: the shell-model analogue of Hypothesis U **fails**

Peak enstrophy against $\alpha'$, 30 shells, nine values over eight decades, all
runs converged to $t=12$ with energy drift $\approx 1.3\times10^{-7}$:

| $\alpha'$ | $k_{\text{eff}}^{\max}$ | peak enstrophy | ceiling $2E/\alpha'$ |
|---|---|---|---|
| $10^{-2}$ | 8 | 19.3 | $10^{2}$ |
| $10^{-4}$ | 78.1 | 463 | $10^{4}$ |
| $10^{-6}$ | 977 | $1.14\times10^{4}$ | $10^{6}$ |
| $10^{-8}$ | 8192 | $2.23\times10^{5}$ | $10^{8}$ |
| $10^{-10}$ | $7.63\times10^{4}$ | $4.75\times10^{6}$ | $10^{10}$ |

Fitted power law:

$$\Omega_{\text{peak}} \sim \alpha'^{-0.672}$$

**Hypothesis U requires exponent 0** (bounded as $\alpha' \to 0$). It is not 0.
In the plain dyadic model, the regularized enstrophy diverges.

### The exponent is Kolmogorov's

$-0.672$ is $-2/3$ to within 0.7%. This is exactly K41 with a cutoff at
$k_{\max} = 1/\sqrt{\alpha'}$:

$$E(k) \sim k^{-5/3} \;\Longrightarrow\; \Omega = \int^{k_{\max}} k^2 E(k)\,dk \sim k_{\max}^{4/3} = \alpha'^{-2/3}$$

Two consequences, and they pull in opposite directions:

**Positive.** The regularized model reproduces Kolmogorov scaling — strong
evidence that the T-dual cutoff behaves like a physical dissipation scale rather
than an arbitrary truncation. Theorem 2.4 (inertial invisibility) is doing real
work: the inertial range is undisturbed.

**Negative.** Theorem 4.2's bound $\Omega \le 2E/\alpha'$ is **not tight** — the
true rate is $\alpha'^{-2/3}$, not $\alpha'^{-1}$ — but it is still a divergence.
Conjecture 5.2 does not hold for the bare dyadic cascade.

### Why this does not refute the programme

The plain Katz–Pavlović model has **no $\mathrm{Sym}^2$ structure**. The paper's
own Conjecture 5.2 attributes the hoped-for uniform bound specifically to the
symmetric-square lock constraining triadic interactions — precisely the rigidity
a generic shell model lacks. This measurement is therefore the **control arm**:
it quantifies what happens *without* the proposed mechanism, and it establishes
$-2/3$ as the number the mechanism must beat.

**This makes the next experiment sharp and falsifiable:** impose the
$\mathrm{Sym}^2$ constraint on shell couplings and re-measure the exponent.

- exponent → 0: strong computational support for Conjecture 5.2
- exponent stays $-2/3$: the lock does not control enstrophy flux, and the
  conjecture needs a different mechanism

Either outcome is publishable, which is the property Stage 1 was designed to have.

---

## 5. Recommended revisions to the paper

1. **Correct Theorem 2.5** to the invariance form; re-audit the Lean statement.
2. **State Theorem 4.2's bound with its energy factor.** Per-mode,
   $k_{\text{eff}}^2 \le 1/\alpha'$; summed, $\Omega \le 2E/\alpha'$. Plotting a bare
   $1/\alpha'$ line conflates the two.
3. **Add the numerical-controls paragraph** to Stage 1. Given the history of the
   problem, a reviewer will ask whether the classical blow-up was an integrator
   artifact. Answering pre-emptively with the energy oracle and a refinement
   study is cheap and decisive.
4. **Report the $-2/3$ exponent as a finding, not a setback.** It is the first
   quantitative content Stage 1 has produced, it matches K41 independently, and
   it converts Conjecture 5.2 into a measurable target.
5. **Consider softening "100 % prouvée"** in accompanying material. The Lean
   kernel certifies the scale geometry; it does not certify that the regularized
   dynamics converge to Navier–Stokes. The gap is Hypothesis U, and §4 above is
   evidence it is a real gap.
