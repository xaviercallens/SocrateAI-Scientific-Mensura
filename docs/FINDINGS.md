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

---

## 6. W1 Loop L1, round 1 (2026-08-12) — the Sym²-locked sweep does not yet measure an exponent

Adjudication of the first `lock="sym2"` sweep (`scripts/sweep_sym2_round.py`,
$N=24$, $t_{\max}=12$, $\nu=0$, $\mathrm{cfl}=0.05$, `closure="galerkin"`,
default `seed_profile`, nine $\alpha'$ over eight decades). Against the
contract in `docs/W1_sym2_contract.md`.

**Outcome: no exponent. The sweep is censored, not converged.** One run of nine
($\alpha'=10^{-2}$) reached $t_{\max}$; the other eight stopped at the
`rho_ceiling` guard ($\rho_3 > 1$). The fit is taken only over `t_max`
terminations, so with one point it is undefined. Energy drift was
$1.3\times10^{-5}$–$2.2\times10^{-5}$ for $\alpha' \ge 10^{-6}$ and
$1.1\times10^{-2}$–$1.8\times10^{-2}$ for $\alpha' \le 10^{-7}$: **0/9 runs meet
the workflow gate** (`t_max` *and* drift $<10^{-6}$). Loop L1 is therefore
halted after one round rather than iterated; there is nothing yet to check for
stability.

### 6.1 The rho_ceiling stop is real, not a timestep artifact (Tier B)

dt-refinement control at $\alpha'=10^{-6}$, $N=24$, re-run directly through
`simulate_sym2_shell_model`:

| cfl | terminated | $t_{\text{stop}}$ | $\Omega(t_{\text{stop}})$ | $\rho_3$ at stop | energy drift |
|---|---|---|---|---|---|
| 0.05 | `rho_ceiling` | 1.50787 | 6480.9 | 1.00901 | $1.27\times10^{-5}$ |
| 0.025 | `rho_ceiling` | 1.50711 | 6403.8 | 1.00348 | $6.77\times10^{-6}$ |
| 0.0125 | `rho_ceiling` | 1.50761 | 6407.3 | 1.00043 | $2.42\times10^{-5}$ |

Stopping time reproduces to $5\times10^{-4}$ relative and the reported enstrophy
to 1.2 % across a 4× refinement. The lock parameters genuinely drift to
$\rho_3 = 1$ — the profile stops decaying in $n$ — and that happens at
$t \approx 1.5$ almost independently of $\alpha'$ (measured: 1.65, 1.51, 1.58,
1.55 for $\alpha' = 10^{-4}, 10^{-6}, 10^{-8}, 10^{-10}$).

**Consequence: `peak_enstrophy` in this sweep is $\Omega$ at a stopping time, not
a peak.** $\Omega$ is still rising monotonically when the guard fires (last four
samples at $\alpha'=10^{-6}$, cfl 0.05: 4290, 4918, 5655, 6365, 6481). Censoring a
monotone quantity at a nearly $\alpha'$-independent time and regressing the
result yields a slope with no bearing on the $\alpha' \to 0$ limit. For the
record, that as-if regression over all nine censored points is
$-0.793 \pm 0.035$; it is reported here only so that nobody re-derives it and
mistakes it for the measurement.

### 6.2 The energy oracle is not currently an integrator oracle (Tier B)

Drift on a *fixed* window ($t_{\max}=0.8$, $\alpha'=10^{-6}$, $N=24$, so no
stopping-event confound), as cfl is swept over a factor of 200:

| cfl | 0.2 | 0.1 | 0.05 | 0.025 | 0.01 | 0.005 | 0.002 | 0.001 |
|---|---|---|---|---|---|---|---|---|
| drift | $1.02\text{e-}5$ | $9.33\text{e-}6$ | $1.27\text{e-}5$ | $6.77\text{e-}6$ | $6.57\text{e-}6$ | $6.27\text{e-}6$ | $1.10\text{e-}5$ | $2.43\text{e-}6$ |

No convergence, and not even monotone. RK4 would give ~$10^{9}$ over that range.
This is bug signature **B2** of the contract, at every $\alpha'$.

Localised: the whole drift is one step. At cfl 0.05 the relative energy error is
$5.5\times10^{-10}$ at $t=0.05$ and $-7.5\times10^{-7}$ at $t=0.20$, then jumps
to $-1.27\times10^{-5}$ across the single step $t \in [0.2086, 0.2249]$ and stays
flat to the end of the run. In that step $\rho_3$ collapses $0.65 \to 0.10$ and
$\sigma_{\min}(J)$ dips to $1.2\times10^{-4}$. At cfl 0.0125 the pre-event drift
is $1.2\times10^{-11}$ — the integrator *is* converging away from the event — and
the same passage then contributes $-2.4\times10^{-5}$.

The projector algebra is intact, so this is not the failure §4.6 warns about
("a Galerkin run whose drift does not shrink with cfl has a bug in the
projector"): measured directly on sampled states,
$\langle u, J\dot z\rangle \le 7\times10^{-16}$ and the Euler residual
$\|Pu-u\|/\|u\| \le 2\times10^{-14}$ throughout. Lemma 4.4 holds pointwise. What
fails is the timestep rule (4.8), which tracks $\max_n k_n|u_n|$ and the
parameter rates but **not** the conditioning of $J$, so it does not shorten the
step through an ill-conditioned passage of the constraint manifold. This is the
same transient the Step 2 suite already documents when it confines gate T5 to
$t_{\max}=0.05$; the sweep is the first place it bites at full window length.
Until it is fixed, drift on a locked run is not a clean measure of integration
error and the workflow's $<10^{-6}$ gate cannot be met.

### 6.3 Shell count is not converged either (Tier B)

$N=24$ vs $N=48$, cfl 0.05, same seed:

| $\alpha'$ | $\Omega$ at stop, $N=24$ | $N=48$ | terminated ($N=24 \to 48$) |
|---|---|---|---|
| $10^{-2}$ | 22.594 | 22.594 | `t_max` → `t_max` |
| $10^{-4}$ | 673.2 | 1222.6 | `rho_ceiling` → **`t_max`** |
| $10^{-6}$ | 6480.9 | 8123.2 | `rho_ceiling` → `rho_ceiling` |
| $10^{-8}$ | $1.102\times10^{6}$ | $1.278\times10^{6}$ | `rho_ceiling` → `rho_ceiling` |
| $10^{-10}$ | $4.610\times10^{7}$ | $5.221\times10^{7}$ | `rho_ceiling` → `rho_ceiling` |

Control C2 fails: the reported numbers move by 13–82 %, and at $\alpha'=10^{-4}$
even the termination reason changes. $N=24$ is also below the $N=30$ the
contract's §6.1 protocol specifies.

The cause is **not** tail truncation: the last three shells carry
$\le 1.7\times10^{-3}$ of $\Omega$ at $N=24$ and $\le 10^{-17}$ at $N=48$.
*Implementation note (Tier C):* `top_shell_enstrophy_fraction` sorts the weights
and sums the three **largest anywhere** (0.80–0.87 in these runs), which measures
spectral concentration, not the truncation edge guard V3 was written for. As
implemented, V3 cannot detect the condition it names, and its 0.84 here should
not be read as "N too small".

### 6.4 What the round nevertheless indicates (Tier C — provisional, not a result)

- The enstrophy-carrying shell tracks the T-dual cutoff at every $\alpha'$:
  measured $\arg\max_n k_{\text{eff},n}^2u_n^2$ = 3, 7, 10, 13, 17 against
  $n_\star$ = 3.32, 6.64, 9.97, 13.29, 16.61. Criterion **A4 fails**, so on the
  evidence available there is no arrest and D1 is not in play.
- The two $\rho_3$ estimators of §5.3 disagree, as the contract anticipated:
  empirical `profile_decay_rate` $= 0.7788$ (stable to $8\times10^{-4}$ across
  the three cfl values) gives $p = -0.639$ by (5.3), while the algebraic
  $\rho_3 = 1.009$ at the stop gives $-1.013$. Per §5.3 the empirical estimator
  is the one to use when they disagree, and $-0.639$ sits near the bare model's
  $-0.672$ — i.e. what little signal there is points at **D2 (lock inert)**, not
  at Conjecture 5.2. This is stated as an indication, not a measurement: it comes
  from censored runs that failed the energy gate.
- Vacuity guards that *do* pass: lock parameters are live ($\operatorname{std}_t a = 0.40$,
  $\operatorname{std}_t b = 0.22$, V1), and $\|PF\|/\|F\| \ge 0.66$ with tangency
  defect $\le 0.75$ (V2), so D5 "the model is frozen" is excluded.
- **V4 fails:** `fit_lock_state` on the default seed leaves a relative residual
  of $5.9\times10^{-7}$, against the $<10^{-10}$ that V4/T12 require. The run
  starts marginally off the constraint set.

### 6.5 Verdict and what round 2 needs

**Verdict: anomaly — measurement invalid (bug/protocol), not physics.** Neither
"exponent $\approx 0$" nor "exponent $\approx -2/3$" can be claimed: there is no
exponent. Per standing rule 5 the failed gates are the finding, and the gates
are not to be loosened to produce a number. Specifically, before this sweep can
be re-run and adjudicated:

1. **Fix the step control through the ill-conditioned passage** (§6.2) — add
   $\sigma_{\min}/\sigma_{\max}$ or $|\dot\rho_3|/\rho_3$ to the rate in (4.8),
   or detect the event — and re-check drift falls $\approx16\times$ per cfl
   halving over the *full* window, not only over $t \le 0.05$.
2. **Decide what `rho_ceiling` is for.** As set (1.0), the guard defines the
   measurement: it fires before $\Omega$ peaks, in 8/9 runs. Either let the runs
   continue past $\rho_3 = 1$ with $N$ large enough to keep the tail negligible
   and report a genuine $\Omega_{\text{peak}}$, or treat $\rho_3 \to 1$ as the
   contract's D4 declaration in its own right. What must not happen is reporting
   $\Omega(t_{\text{stop}})$ as $\Omega_{\text{peak}}$.
3. **$N=30$ and $N=60$** per §6.1/C2, not 24.
4. **Run the `"order3"` control (C5) and the same-seed `"none"` control (C4).**
   Until both exist there is no $\Delta p$, and §6.5 of the contract is explicit
   that $\Delta p$ — not the sym2 exponent alone — is the test of Conjecture 5.2.
   The published $-0.672$ used the $u=e_0$ seed and is not a like-for-like
   comparator for these runs.

Nothing here counts for or against Conjecture 5.2. The control arm's $-2/3$ of
§4 remains the number to beat, and it has not yet been contested.
