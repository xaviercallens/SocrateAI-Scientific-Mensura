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

---

## 7. W1 round 2, phase 1 repair — partially confirmed, headline claim refuted

Commit `465955e` (tagged `release-1`, since corrected — see below) claimed the
§6.2 energy-oracle defect was fixed, evidenced by full-window 4th-order
convergence across 8 configurations. An independent Opus skeptic (round 2
phase 2, per the workflow shape in
[`.claude/skills/decisive-experiment/SKILL.md`](../.claude/skills/decisive-experiment/SKILL.md))
was tasked with refuting that claim, defaulting to refuted unless it
survived real attacks — re-derivation from scratch, dense parameter scans,
independent controls, applying the new test's own pass criterion elsewhere.
**Verdict: refuted, on completeness, not on fabrication.** Every reported
number reproduced digit-for-digit; nothing was fabricated. What fails is
that the defect is fixed.

### 7.1 What survives, confirmed independently (Tier B)

- **The branch-point diagnosis is correct.** At $a=0$ the legacy Jacobian
  has rank 4 and $\sigma_{\min}/\sigma_{\max}=0$ exactly; in the repaired
  $A=a^2$ chart the same point has rank 5 and $\sigma_{\min}/\sigma_{\max} =
  6.70\times10^{-2}$. $\partial\Phi/\partial a = 2a\,\partial\Phi/\partial(a^2)$
  is visible by inspection.
- **The physics did not move.** The reduced field $\dot u = PF(u)$ agrees
  between the two charts to $10^{-14}$–$10^{-16}$ at three states including
  one near the fold; matched-initial-condition trajectories agree to
  $3.1\times10^{-9}$ before the fold.
- **Lemma 4.4 holds**, recomputed independently over the full window:
  $\max|\langle u, J\dot z\rangle|/(\|u\|\|J\dot z\|) = 9.54\times10^{-15}$,
  $\max\|Pu-u\|/\|u\| = 4.63\times10^{-15}$.
- **Gate T1 is bit-for-bit** ($\max|\Delta E| = \max|\Delta\Omega| = 0$,
  `lock="none"` vs `shell.py`, both inviscid and viscous).
- **The `fit_lock_state` bug and its fix are genuine**: residual
  $5.80\times10^{-7} \to 3.38\times10^{-17}$, recovering $A=0.5625$,
  $b=-0.125$ exactly.
- Suite green (69 tests), lint clean, no tolerance/window/assertion
  weakened — confirmed by reading the diff directly, not just the summary.

### 7.2 What is refuted: the step-rule defect is not fixed

**A.** Round 1 attributed the failure to the timestep rule *and* the chart;
the repair fixed the chart but re-derived the timestep rule (eq. 4.8′) and
never re-isolated it. A fixed-*dt* control — same repaired field, same
initial state, eq. 4.8′ removed entirely — gives a clean, monotone,
zero-sign-flip compensated-error curve (spread $2.71\times$ over 20 step
counts). The *shipped adaptive* rule, same field, same range: spread
$379.7\times$ with **~10 sign flips**. The non-smoothness is in eq. 4.8′
itself — the exact component the repair was assigned to fix.

**B.** *"Round 2: monotone over 8 decades of drift"* is false on the
report's own printed table. cfl $0.1\to0.05$: drift $+5.65\times10^{-8} \to
-6.87\times10^{-8}$ — refining cfl makes it $1.2\times$ **worse** and flips
sign, and both rows are in the table captioned "monotone." A worse exact
halving was also found: cfl $0.039685\to0.0198425$ gives
$+1.65\times10^{-10} \to -3.29\times10^{-9}$, a $20\times$ degradation —
contract bug-signature B2, verbatim, in the repaired code, at the flagship
configuration.

**C.** The report's explanation for fluctuating ratios ("energy_drift is
signed and crosses zero") does not apply to its own headline table: all six
tabulated drift values are negative, with no crossing among them. The drift
*does* flip sign at intermediate cfl not shown in that table — a faster
oscillation, which is a worse phenomenon than a single smooth crossing, not
an explanation for one.

**D.** The single-step error signature that diagnosed round 1 (§6.2) has
not gone away, only shrunk: at cfl $0.05/0.0125/0.00625$ the single largest
step still contributes $105.3\%/114.7\%/121.5\%$ of total drift. "Not
relocated, removed" does not match the trace.

**E.** Gate T5 ($\ge 8\times$ drift reduction per halving) fails at
configurations the fix should handle: the middle halving at every window
length tested ($t_{\max}=0.2,0.4,1.2,1.6$); the report's own headline
halving ($4.10\times$); 3 of 6 halvings at a contract-mandated seed (C7,
roots $(0.3,-0.4)$, effective order 3.23 where fixed-dt gives a textbook
16.9–26.6$\times$); and — critically — at the **actual sweep window**
($t_{\max}=12$, $N=30$, $\alpha'=10^{-2}$, the single run in the whole
programme that currently reaches the workflow's gate) one halving gives
only $5.26\times$.

**F. The new regression test is placed where it passes, not where the claim
lives.** `test_full_window_energy_drift_converges_at_fourth_order` pins
$N=6$. Applying its own pass criterion elsewhere: **passes** at $N=6, 8$;
**fails** at $N=7,10,12,16,20,24,30$ and at the T-dual ladder,
$\alpha'=10^{-6}$, for both $N=24$ and $N=30$ — i.e. at the exact
configuration the report's tables are about (first halving ratio $0.82$
against the test's required $\ge 8.0$). This is structurally the same
failure mode §6.2 caught in round 1: a green gate confined to a regime that
does not exercise the claim.

**One disclosed-but-understated item.** The module docstring and the
contract erratum both say "the constraint set is unchanged, only its
parametrisation is." True only when $M$ is read over $\mathbb{C}$ (as
Proposition A is stated). The contract's own real half-ladder condition
(eq. 1.5, $A\ge0$) is violated for **12.0%** of the flagship trajectory's
window ($t\in[0.1404,0.2347]$, $\min A = -0.2414$) — a region the legacy
chart could not represent at all. Disclosed as a trade-off, but "unchanged"
overstates what the contract text supports.

### 7.3 Bottom line and what round 2 still needs

This was good, honest work that found and fixed two real bugs the round-1
adjudication missed (the chart branch point; the `fit_lock_state` spurious
minimum) — both are kept, confirmed independently. But **the specific claim
under test — that the conditioning-blind step control is repaired — does
not survive.** The step rule still makes drift an erratic, sometimes-worse,
sign-flipping function of cfl, a fixed-dt control on the identical field is
clean, and the regression test meant to pin the claim fails at the
configuration the claim is about.

**Before round 2 can re-attempt the measurement:**

1. Re-isolate and repair the step rule itself (eq. 4.8′), independently of
   the (now-confirmed-good) chart fix — the two are separable, and item A
   above shows the fixed-dt control is clean on the *same* repaired field.
2. Re-pin the convergence regression test at $N\ge24$ on the T-dual ladder,
   where it currently fails — not at $N=6$, which items A/F show is a
   different, non-representative regime.
3. Re-verify gate T5 holds at the actual sweep window ($t_{\max}=12$,
   $N=30$), not only at a short diagnostic window.
4. Decide whether the $A<0$ excursion (12% of window, item 7.2 above) is
   in-scope for the model as specified, or needs its own guard.

**Commit `465955e` / tag `release-1` is retained as the historical record of
what was believed at the time** (§7.1's confirmed portion is real and
valuable), but the tag has been moved forward to the commit that includes
this correction, with an updated message — it was never pushed with the
original, overclaiming message. Per standing rule 5, this section is the
finding; the $3.948$ fitted order is not to be cited as an established
result until the above is resolved.

---

## 8. W1 round 2, phase 3 — the step rule, repaired (eq. 4.8″)

Answers §7.3 items 1–3. Item 4 (the $A<0$ excursion) is still open and is
*not* addressed here. Every adaptive-rule number below was produced by the
code as shipped in this commit: all scans were re-run from scratch after the
last code change (the estimator noise floor of §8.2), not carried over. The
fixed-$\Delta t$ control arm of §8.5 and §8.6.3 calls no part of the step
rule and is unaffected by it; the (4.8′) arm is `git show HEAD:…` run
verbatim.

The chart fix and the `fit_lock_state` fix of §7.1 are untouched. What
changed is only *how $\Delta t$ is chosen*.

### 8.1 Diagnosis: (4.8′) is not blind everywhere, it is blind in one place

§7.2.A established that the residual non-smoothness lives in the adaptive
rule itself. Tracing the flagship trajectory ($N=24$, $\alpha'=10^{-6}$,
$t_{\max}=0.8$) step by step at cfl $=0.05$ locates it exactly:

- (4.8′)'s rate has a **local minimum of 2.115 at $t=0.182$**, and the
  step-doubling error density measured at that state is $C^{1/5}=4.05$.
  Accuracy binds there and stability does not, so (4.8′) steps
  $\Delta t = 2.4\times10^{-2}$ straight through.
- Under (4.8′) at that cfl the whole run's drift is $6.868\times10^{-8}$ and
  **the single sampled interval $t\in[0.1739,0.2093]$ carries $105.3\%$ of
  it** — §6.2's signature, and §7.2.D's, unchanged. Under (4.8″) the run's
  drift is $1.931\times10^{-10}$ and the worst single step carries $17.3\%$.
- **The anti-correlation is local, not global**, and the earlier draft of the
  contract erratum overstated it. Over a uniform-time sample of the window
  the Pearson correlation between $\text{rate}_{4.8'}$ and $C^{1/5}$ is
  $+0.297$; the median $\text{rate}_{4.8''}/\text{rate}_{4.8'}$ over a whole
  run is $1.000$ and its maximum is $2.728$ (at $t=0.182$). (4.8″) shortens a
  handful of steps and changes nothing else. The contract's Erratum 2 has
  been rewritten to these measured numbers; the figures it previously quoted
  ($C^{1/5}\approx24$, rate bottoming at $1.41$, $\Delta t=3.5\times10^{-2}$,
  fixed-$\Delta t$ needing $6.7\times10^{-4}$) could not be reproduced and
  are withdrawn.

**Why an error *estimator* succeeds where two closed-form rates failed.**
The Richardson gap is not a closed-form density evaluated at a state; it is a
measurement of what a step of *that length* actually does. At the flagship
seed, $C^{1/5}$ measured at $\Delta t = 5\!\times\!10^{-3},2\!\times\!10^{-3},
10^{-3},5\!\times\!10^{-4},2\!\times\!10^{-4},10^{-4},5\!\times\!10^{-5},
2\!\times\!10^{-5},10^{-5}$ is $20.8,38.2,55.1,69.4,83.6,90.3,94.1,96.7,97.5$,
with the gap's local slope in $\Delta t$ rising $1.68\to4.93$ — i.e. the
estimator reports a *large* $C$ exactly when the step is too long to resolve
the state, and settles to the asymptotic density only once it is not. That
feedback is the property (4.8) and (4.8′) could not have: they evaluate a
formula at a point and cannot know how long a step that point can take. The
gap is essentially $N$-independent — $N=12$ and $N=24$ agree to 6 significant
figures at every $\Delta t$ in that list, $N=8$ to 3 — so this is a property
of the dynamics, not an artefact of truncation depth or of noise amplified
through the reconstruction recurrence.

### 8.2 What changed

$$\Delta t = \frac{\mathrm{cfl}}{\bigl(\text{rate}_{4.8'}^{\,5}+C\bigr)^{1/5}},
\qquad C=\frac{16}{15}\frac{\|\Phi(z_{\text{full}})-\Phi(z_{\text{half}})\|}
{\|\Phi(z_{\text{half}})\|\,\Delta t^{5}}$$

with the **two-half-step solution propagated** and the full step kept only as
the estimator; $C$ is carried to the next step, and a one-off probe at $t=0$
means no step of the run is uncontrolled. Contract §4.7, Erratum 2.

Two supporting changes: `_locked_zdot` factors the bare field out of
`locked_rhs` so the RK stages stop paying for the per-step diagnostics; and
`error_density` returns $0$ below a **noise floor of 64 ulps** of $\|u\|$.
The floor is not cosmetic — see §8.6.

### 8.3 Dense-scan evidence at the flagship configuration

$N=24$, $\alpha'=10^{-6}$, $t_{\max}=0.8$, `lock="sym2"`, Galerkin, inviscid,
default seed. 20 log-spaced cfl, all terminating `t_max`. The gate is that
$|{\rm drift}|/\mathrm{cfl}^4$ is **flat**, not that a fit returns $\approx4$.

| cfl | 0.1 | 0.0729 | 0.0532 | 0.0388 | 0.0283 | 0.0207 | 0.0151 | 0.0110 | 0.00802 | 0.005 |
|---|---|---|---|---|---|---|---|---|---|---|
| drift | $+3.56$e-9 | $+8.89$e-10 | $+2.55$e-10 | $+7.13$e-11 | $+2.01$e-11 | $+5.43$e-12 | $+1.50$e-12 | $+4.09$e-13 | $+1.04$e-13 | $+1.53$e-14 |
| $/\mathrm{cfl}^4$ | 3.56e-5 | 3.14e-5 | 3.18e-5 | 3.14e-5 | 3.12e-5 | 2.98e-5 | 2.90e-5 | 2.79e-5 | 2.51e-5 | 2.45e-5 |

(alternate rows shown; all 20 are in the same band). Over the full 20 points:
**compensated spread $1.45\times$, zero sign flips** (every drift positive),
worst adjacent pair equivalent to $13.8\times$ per halving, least-squares
order 4.102, across 5.4 decades of drift.

A second, wider scan (cfl $0.2\to0.01$, 20 points) gives compensated spread
$1.64\times$, zero sign flips, LSQ order 4.010 — see §8.6 for its coarse end.

Exact cfl halvings at the same configuration:

| cfl | 0.16 | 0.08 | 0.04 | 0.02 | 0.01 | 0.005 |
|---|---|---|---|---|---|---|
| drift | $+1.989$e-8 | $+1.423$e-9 | $+8.160$e-11 | $+4.720$e-12 | $+2.771$e-13 | $+2.243$e-14 |
| ratio | — | 13.98 | 17.44 | 17.29 | 17.03 | 12.36 |

**Minimum halving ratio 12.36 against gate T5's $\ge 8$**, compensated spread
$1.295\times$, zero sign flips. At $N=30$ on the same ladder: minimum 13.98,
spread $1.340\times$, zero sign flips.

### 8.4 Dense-scan evidence at the ACTUAL sweep window

$N=30$, $t_{\max}=12$, 20 log-spaced cfl from 0.2 to 0.0125, all `t_max`.
This is §7.2.E's configuration — the one the previous repair failed at.

| | $\alpha'=10^{-2}$ | $\alpha'=3\times10^{-3}$ |
|---|---|---|
| drift at cfl 0.2 | $-8.223\times10^{-7}$ | $-1.180\times10^{-6}$ |
| drift at cfl 0.0125 | $-1.267\times10^{-11}$ | $-1.894\times10^{-11}$ |
| compensated spread over 20 pts | **1.03×** | **1.06×** |
| sign flips | **0** | **0** |
| worst adjacent pair | 15.0× / halving | 14.5× / halving |
| LSQ order | 3.993 | 3.983 |
| min **exact** halving ratio | **15.76** | **15.68** |

Compensated $|{\rm drift}|/\mathrm{cfl}^4$ at $\alpha'=10^{-2}$ runs
$5.139,5.034,5.105,5.112,5.156,5.115,5.103,5.142,5.147,5.133,5.161,5.161,
5.162,5.167,5.175,5.173,5.186,5.191,5.194,5.190$ ($\times10^{-4}$) over the
whole scan. That is the flatness the gate asks for, over 4.8 decades of
drift, at the window the measurement actually uses.

Only $\alpha'=10^{-2}$ and $3\times10^{-3}$ reach $t_{\max}$ at $N=30$;
$10^{-3}$ and $10^{-4}$ stop at `rho_ceiling`, so a drift-refinement study
there would be confounded by a moving stopping time. That is a *disclosed
restriction on where this check can be run*, not a choice of where it passes:
§7.3 item 2's `rho_ceiling` question is still open and is what limits it.

### 8.5 Controls and contrast

**Control (fixed $\Delta t$, plain RK4, identical field, no adaptive rule).**
Compensated $|{\rm drift}|\cdot n_{\text{steps}}^4$ at the flagship window:
$2.59{\rm e}1, 2.22{\rm e}2, \dots$ rising to a plateau of
$4.36,4.48,4.43,4.27,4.09,3.92\ (\times10^3)$ for $n_{\text{steps}} =
2206\ldots9894$ — flat to $1.14\times$ once $n_{\text{steps}}\gtrsim2000$,
with one sign flip at $n_{\text{steps}}=200$. So the fixed-$\Delta t$ control
has a pre-asymptotic regime of its own below $\sim2000$ steps; §7.2.A's
"clean, monotone, 2.71× spread" holds only above it. (4.8″) is already
compensated-flat by $\approx250$ accepted steps (cfl $\approx0.11$).

**Contrast (the shipped eq. 4.8′, `git show HEAD:…/shell_sym2.py`, run
verbatim on the same machine).** Same 20-point scan, same field, same seed:

| | eq. 4.8′ (HEAD) | eq. 4.8″ |
|---|---|---|
| flagship, compensated spread | **876×** | **1.45×** |
| flagship, sign flips | 3 | 0 |
| flagship, worst adjacent pair | 167× **worse** under refinement | 13.8× better/halving |
| sweep window $\alpha'=10^{-2}$, spread | 3.94× | 1.03× |
| sweep window, worst adjacent pair | 0.81×/halving (i.e. worse) | 15.0×/halving |

This reproduces §7.2 independently, and is the direct before/after.

**Cost**, at matched cfl $=0.05$: flagship $6.10\,$s $\to10.26\,$s
($1.68\times$) for $355.7\times$ less drift; sweep window $14.56\,$s
$\to24.64\,$s ($1.69\times$) for $31.7\times$ less drift.

### 8.6 Where (4.8″) is *not* clean — disclosed limits

1. **Pre-asymptotic above cfl $\approx0.15$ at the flagship window.** In the
   cfl $0.2\to0.01$ scan the first adjacent pair ($0.2\to0.171$) reduces
   drift only $1.148\times$ (local order 0.88). The compensated band still
   holds across that scan ($1.64\times$), and because a compensated spread
   $S$ bounds every exact halving inside the scanned range below by $16/S$,
   the scan certifies $\ge9.8\times$ per halving — consistent with the
   measured $13.98\times$ at $0.16\to0.08$. But a *single* 17 % cfl step at
   the coarse end is not a convergence test, and should not be quoted as one.
2. **The energy oracle has a roundoff floor at $|{\rm drift}|\sim10^{-14}$.**
   At the flagship window: 1248 ulps of $E(0)$ at cfl $0.01$, 101 at $0.005$,
   26 at $0.0025$. Below $\approx10^{-13}$ the drift stops being a
   measurement of the integrator. Every gate above is stated on data at or
   above that floor except the last point of the cfl $\to0.005$ ladder, which
   is flagged here rather than dropped.
3. **The solution norm is *not* fourth order.** Against a fixed-$\Delta t$
   64000-step reference, $\|u(t_{\max})-u_{\text{ref}}\|$ under (4.8″) has
   local order $2.89, 3.40, 3.66, 3.70, 3.50$ over cfl $0.08\to0.0025$ and
   its $\mathrm{cfl}^4$-compensated value is still rising, where the
   fixed-$\Delta t$ control reaches $3.90$–$4.09$ at comparable step counts.
   (4.8″) is therefore "the rule under which the **energy oracle** converges
   at fourth order with flat compensation", **not** "a fourth-order method"
   in the solution norm. It is nonetheless far more accurate per unit work:
   $1.398\times10^{-12}$ in 5570 accepted steps ($=16{,}710$ RK4 steps, since
   each accepted step costs three) against $1.280\times10^{-10}$ for
   fixed-$\Delta t$ in 8000 RK4 steps — $91\times$ the accuracy for
   $2.1\times$ the field evaluations. Why the energy functional is cleaner than the
   solution it is computed from is **not explained here** and is a live
   question, not a settled one.
4. **The criterion fails on short windows — and so does a fixed-$\Delta t$
   control, which is the point.** Applying the new regression test's own
   criterion at other window lengths ($N=24$, $\alpha'=10^{-6}$, exact
   halvings cfl $0.08/0.04/0.02$):

   | $t_{\max}$ | min halving | sign flips | comp spread | verdict |
   |---|---|---|---|---|
   | 0.2 | 40.75× | 0 | **106.6×** | fails (floor) |
   | 0.4 | 10.84× | **1** | **4.75×** | fails (zero crossing) |
   | 0.8 | 17.29× | 0 | 1.178× | passes |
   | 1.2 | 16.89× | 0 | 1.145× | passes |
   | 1.6 | 16.51× | 0 | 1.106× | passes (`rho_ceiling`) |

   §7.2.E listed $t_{\max}=0.2,0.4,1.2,1.6$ as failing under (4.8′). Two of
   the four now pass cleanly; the other two fail for a reason that is **not
   the step rule**. The leading $O(\mathrm{cfl}^4)$ coefficient of the signed
   drift is $\approx5\times$ smaller at $t_{\max}=0.4$ than at $0.8$, so the
   drift reaches the §8.6.2 roundoff floor by cfl $\approx0.02$ ($1.1\times
   10^{-14}$, 50 ulps, at $t_{\max}=0.2$) and crosses zero. **Control:** a
   fixed-$\Delta t$ plain RK4 with no adaptive rule whatsoever, same field,
   2000–16000 steps, flips sign at $t_{\max}=0.2, 0.3, 0.4, 0.5$ and not at
   $0.6, 0.8$ — the same windows, the same way. This is a property of the
   energy functional on short windows, not of (4.8″), and short windows are
   therefore *not discriminating configurations* for this gate. Stated here
   rather than omitted, because "apply the test's own criterion elsewhere" is
   precisely the attack that refuted §7.
5. **Other seeds and locks** ($t_{\max}=0.8$ unless noted, same halvings).
   Contract seed C7, roots $(0.3,-0.4)$ — where §7.2.E measured effective
   order 3.23 and 3 of 6 failing halvings under (4.8′): now **16.58×** min
   halving, spread $1.075\times$, zero flips; at $t_{\max}=1.6$, **16.45×**,
   $1.065\times$, zero flips. `lock="veronese"`: 12.47×, spread $1.283\times$,
   zero flips. `lock="order3"`: 11.67× and zero flips, but spread $27.7\times$
   because its drift is $3.4\times10^{-12}\to6.7\times10^{-16}$ — 3 ulps, i.e.
   entirely inside the §8.6.2 floor, so nothing is measured there. Both
   non-sym2 locks terminate `rho_ceiling` on this window.
6. **A latent collapse, found and fixed.** Without the 64-ulp noise floor the
   estimator feeds back on its own roundoff: $C\sim\varepsilon/\Delta t^5$
   diverges as $\Delta t\to0$, the fixed-point map has gain
   $\approx1333\,\mathrm{cfl}$, and below cfl $\approx7.5\times10^{-4}$ the
   step spirals to zero. Measured on the unfloored code: clean at
   $5\times10^{-4}$, `dt_collapse` after 23 steps at $2\times10^{-4}$ and
   after 10 at $10^{-4}$. With the floor, cfl $=10^{-4}$ runs to $t_{\max}$
   in 11611 steps with drift $1.4\times10^{-14}$. The floor is placed where
   the signal dies, not where it helps: at the flagship seed the gap reaches
   clean $\Delta t^5$ scaling (slope 4.93) at $\Delta t=10^{-5}$, where it is
   4–7 ulps.

### 8.7 Gates re-verified under the new rule

- **T1 bit-for-bit.** `lock="none"` vs `shell.py`, $\nu\in\{0,10^{-3}\}$,
  $N\in\{12,24\}$: $\max|\Delta E| = \max|\Delta\Omega| = \max|\Delta t| =
  \max|\Delta u_{\text{final}}| = 0$ exactly, same termination reason.
- **Lemma 4.4.** Over the full window: $\max\|Pu-u\|/\|u\| =
  8.71\times10^{-15}$ (flagship), $1.16\times10^{-14}$ ($\alpha'=10^{-2}$
  sweep), $1.24\times10^{-14}$ ($3\times10^{-3}$ sweep); $\operatorname{cond}
  J \le 5.30\times10^{3}$ throughout. On 400 random states,
  $\max|\langle u,J\dot z\rangle|/(\|u\|\|J\dot z\|) = 2.71\times10^{-12}$
  and $\max\|Pu-u\|/\|u\| = 2.45\times10^{-13}$.
- **T5** at $N=24$ *and* $N=30$, flagship and sweep window: minimum measured
  exact-halving ratios 12.36 / 13.98 / 15.76 / 15.68, all $\ge8$.

### 8.8 The regression test, re-pinned honestly

`test_full_window_energy_drift_converges_at_fourth_order` no longer pins
$N=6$ on the dyadic ladder (§7.2.F: a different regime, and a placement where
the test passed rather than where the claim lived). It is now parametrised
over **$N=24$ and $N=30$ on the T-dual ladder at $\alpha'=10^{-6}$**, and it
asserts compensated flatness (no sign flip; every halving $\ge8\times$;
$|{\rm drift}|/\mathrm{cfl}^4$ spread $\le2$) instead of a fitted order —
because §7.2.A–B is a worked example of a scan that fits to $\approx3.95$
while individual halvings get worse and flip sign.

Applying that same criterion across $N = 6, 8, 12, 16, 20, 24, 30$ on the
T-dual ladder at $\alpha'=10^{-6}$ before shipping it: **passes at every one**
(halving ratios 16.11–17.92, compensated spread 1.102–1.179, zero sign flips
everywhere) — against §7.2.F, where the old rule's criterion failed at all of
$N=7,10,12,16,20,24,30$.

Two tests were added: `test_sweep_window_energy_drift_converges_at_fourth_order`
(the $t_{\max}=12$, $N=30$, $\alpha'=10^{-2}$ gate — §7.3 item 3), and
`test_step_doubling_estimator_does_not_feed_back_on_its_own_roundoff`
(§8.6.6). `test_repaired_timestep_rule_is_never_looser_than_contract_eq_4_8`
now exercises the shipped `stability_rate`/`step_rate` over 41 states and
both viscosities instead of re-deriving the formula inline, so it cannot pass
while the integrator uses something else.

### 8.9 Status

**The §7.2 defect is closed, subject to the same adversarial verification
that closed §7.** The step rule now makes full-window energy drift a strictly
monotone, sign-stable, compensated-flat function of cfl at the flagship
configuration *and* at the actual sweep window, at both $N=24$ and $N=30$,
with every exact halving clearing gate T5. $|{\rm drift}|$ is strictly
decreasing at all 20 points of all four dense scans.

**This section has not yet been through the independent-skeptic stage.** §7
is the record of a report that reproduced digit-for-digit and was still wrong
on its central claim, so nothing here should be promoted to the README, the
paper, or a release tag until someone re-derives it from scratch. The three
places to attack first, in order: (i) §8.6.3 — the solution norm converges at
$\approx3.7$, not 4, and the *reason* the energy functional is cleaner is not
established, so a demonstration that the energy flatness is coincidental
would sink §8.9; (ii) §8.4's restriction to two $\alpha'$, which is forced by
`rho_ceiling` and means the sweep-window check covers the mildest two of nine
sweep points; (iii) the $\le2\times$ band in the new regression test, which
has $1.7\times$ headroom at the configurations measured and would need
re-pinning if that headroom is an artefact of the seed.

Still open, and not touched by this section: §7.3 item 4 (the $A<0$
excursion), the `rho_ceiling` question of §6.5 item 2 — which is what
restricts §8.4 to two $\alpha'$ — the $N$-convergence failure of §6.3, and
the missing `order3`/`none` controls of §6.5 item 4. **Nothing here is a
result about Conjecture 5.2**; it removes the instrument defect that made the
measurement uninterpretable, and no more.

---

## 9. §8 adversarially checked — the estimator is real, but seed-dependent

Independent Opus skeptic, same discipline as §7: default to refuted unless
the claim survives real re-derivation. **Verdict: refuted, on
generalization** — narrower than §7's refutation, and on a point §8.9 named
in advance as attack surface (iii).

### 9.1 What reproduces (most of it)

The flagship and sweep-window dense scans reproduce essentially exactly
(19/20 flagship drift values match; all 20 sweep-window compensated values
match). The five new/changed regression tests pass at $N=24$ and $N=30$; full
suite green (72 passed); `ruff check src tests` clean; no gate, tolerance,
window, or assertion weakened — the $N=6\to\{24,30\}$ re-pin is genuinely
harder, not relocated to where it's easy. The cfl triple was checked against
9 alternatives, none cherry-picked; the 64-ulp noise floor holds at 4–1024
ulps; the estimator is load-bearing (forcing it off reproduces §7.2's
$324.7\times$ spread with a sign flip). The chart fix, `fit_lock_state`,
Lemma 4.4, and gate T1 are all independently reconfirmed intact. §8.6.3's
disclosed "solution norm is only $\approx3.7$ order, not 4" is actually
**pessimistic**: a Richardson-extrapolated reference gives local orders
climbing $2.89\to3.80$ — pre-asymptotic approach to 4th order, not a deficit.

### 9.2 What is refuted: seed-dependence not yet explained

At two additional, **well-conditioned** seeds the repair agent did not test
— roots $(0.2,-0.7)$ and $(0.1,-0.6)$, $\max\operatorname{cond}(J) = 68$ and
$387$ respectively, i.e. nowhere near the branch-point pathology of §7 — a
fixed-$\Delta t$ control on the identical field is clean (compensated flat to
$1.03$–$1.05\times$, textbook halving ratios $\approx15.6$–$15.9\times$), but
the **shipped eq. (4.8″)** is not: compensated spread $3.65\times$ and
$24.7\times$ respectively, with the second failing the new regression test's
own $\le2\times$-band criterion outright. This is exactly the seed-dependence
§8.9 flagged in advance as the most likely thing to sink the claim, now
confirmed present at 2 of 13 seeds tested — and *not* explained by the known
$A<0$ excursion (§7.3 item 4): the failing seeds' excursions are smaller than
the flagship's, which is clean.

Two *other* failing seeds are correctly attributed elsewhere, not to (4.8″):
$(0.6,0.1)$ has a genuinely singular field ($\operatorname{cond}J\to10^8$,
terminates `lock_singular`; the fixed-$\Delta t$ control is far worse there
too), and $(0.45,0.45)$ sits on the Veronese double-root boundary. Both are
the known conditioning question (§7.3 item 4), not a new defect.

A minor, separate finding: one cell of §8.3's headline table
($+1.53\times10^{-14}$ at $\mathrm{cfl}=0.005$) does not reproduce against
the deterministic value ($+2.2427\times10^{-14}$) and is inconsistent with
that section's own exact-halvings table and ulp count. The section's
$\le2\times$ gate conclusion survives on the corrected value (worst adjacent
pair $3.70\times$, not the reported $13.8\times$), so this does not change
§8's bottom line — but it is the same *shape* of error that started §7: a
reported number in a "confirmed" table that a from-scratch rerun does not
reproduce.

### 9.3 Bottom line

The step-doubling estimator (eq. 4.8″) is real, substantial progress and is
kept: it measurably outperforms both the original rule and eq. (4.8′) at
every configuration tested, including ones where it fails outright. But the
claim **"(4.8″) makes full-window drift compensated-flat" is a property of
the two seeds it was measured at, not of the rule** — at $\ge2$ of 13 tried,
otherwise-clean seeds, it is not. Per standing rule 5, this is the finding:
the estimator needs a seed sweep as part of its own validation (not a single
seed at two shell counts, which §9's skeptic notes are "near-duplicates" —
$N=24$ and $N=30$ agree to four significant figures because the drift is
essentially $N$-independent above $N=12$, so testing both was never a
generalization check). **Before this closes:**

1. Diagnose the seed-dependence directly — the failing seeds are
   well-conditioned, so this is a *different* mechanism from the §7 branch
   point, not a recurrence of it.
2. Re-pin the regression test's $\le2\times$ band against a seed sweep
   (start from the 13 seeds already tried across both this section and §8),
   not the single default seed.
3. Everything still open from §7.3/§8.9 remains open: the $\rho_{\text{ceiling}}$
   question, $N$-convergence, the `order3`/`none` controls, the $A<0$
   excursion.

As in §7: still nothing here counts for or against Conjecture 5.2.

---

## 10. W1 round 2c — the seed-dependence, diagnosed and repaired (eq. 4.8‴)

Answers §9.3 items 1 and 2. Item 3 (the `rho_ceiling` question, $N$-convergence,
the `order3`/`none` controls, the $A<0$ excursion) is **not** addressed here and
remains open. Every number below was produced by the code as shipped in this
commit; the "before" column is §8's eq. (4.8″) loop re-run verbatim on the same
machine in the same session, not carried over from §8/§9.

### 10.1 Diagnosis: (4.8′) is blind at every turning point of a lock parameter

§9.2 reproduces exactly. At $N=24$, $\alpha'=10^{-6}$, $t_{\max}=0.8$, exact
halvings cfl $0.08\to0.005$: roots $(0.2,-0.7)$ gives compensated spread
$3.654\times$ with halving ratios $10.94/9.32/11.58/15.20$; roots $(0.1,-0.6)$
gives $24.745\times$ with $152.92/41.43/15.01/7.24$; the flagship gives
$1.295\times$ with $17.44/17.29/17.03/12.36$ — §8.3's corrected
$+2.2427\times10^{-14}$ included.

**The Richardson estimator is not the blind component.** Candidate (b) — "the
estimator has its own local minima, like the rate it replaced" — was tested
first and is **refuted**. Freezing the state at each failing passage and
sweeping the estimator's own step length gives $C^{1/5}$:

| roots | $\Delta t$ taken | $C^{1/5}$ over $\Delta t \times (2,1,\tfrac12,\tfrac14,\tfrac18)$ | under-report at the taken $\Delta t$ |
|---|---|---|---|
| $(0.2,-0.7)$, cfl 0.02 | $3.78\times10^{-3}$ | 5.452, 5.225, 5.088, 5.017, 4.981 | $1.04\times$ in rate |
| $(0.1,-0.6)$, cfl 0.02 | $2.47\times10^{-3}$ | 8.044, 7.800, 7.562, 7.418, 7.341 | $1.03\times$ |

A $3$–$4\%$ rate error cannot produce a $24.7\times$ spread. The estimator is
doing its job.

**What actually happens.** Every lock-parameter term of the contract's (4.8) and
of (4.8′) is a multiple of $|\dot\theta|$. It therefore vanishes **identically**
at every turning point of a lock parameter — and a turning point is exactly
where RK4's truncation error is *largest*, since that error is driven by the
high derivatives of the trajectory and not by its velocity. Tracing the shipped
integrator step by step:

- roots $(0.1,-0.6)$, cfl $=0.02$: $A(t)$ rises to a maximum $0.0792$ at
  $t=0.1545$. Across the eight steps astride that turn the (4.8′) rate falls
  $20.21\to20.19\to19.99\to19.60\to18.99\to18.11\to16.90\to15.23\to12.91\to9.54
  \to\mathbf{4.49}$ and $\Delta t$ **grows** $9.90\times10^{-4}\to
  2.47\times10^{-3}$. The three steps at $t=0.1525,0.1545,0.1570$ carry
  $+125.2\%$, $+102.0\%$, $-102.7\%$ of the whole run's energy drift.
- roots $(0.2,-0.7)$, cfl $=0.02$: same shape at $t=0.2658$ ($A$ turns over) and
  $t=0.2719$ ($b$ turns over): rate $8.61\to\mathbf{2.81}$, $\Delta t$ up
  $1.6\times$, one step carrying $-93.2\%$ of the drift.

This is §6.2's single-step signature and §8.1's local minimum — **the same
defect, one level deeper**: §8.1 found the local minimum but read it as an
accident of the flagship trajectory, so it was patched with an error estimator
rather than removed. It is not an accident; it is a structural property of a
first-order rate.

Independently, on a trajectory generated by **plain fixed-$\Delta t$ RK4 with no
adaptive rule at all** (so the measurement cannot be contaminated by the rule it
is about), locating every sign change of $\dot A$ or $\dot b$ and evaluating both
rates there:

| roots | turn at | $\mathrm{rate}_{4.8'}$ | dip over $\pm30$ steps | $\mathrm{rate}_{4.8'''}$ | lift |
|---|---|---|---|---|---|
| $(0.1,-0.6)$ | $t=0.1553$ | **3.294** | $16.8\times$ | 68.21 | $20.7\times$ |
| $(0.2,-0.7)$ | $t=0.2629$ | **2.672** | $10.3\times$ | 31.88 | $11.9\times$ |
| $(0.5,0.25)$ | $t=0.1742$ | **1.500** | $29.7\times$ | 42.95 | $28.6\times$ |

The third row is the flagship, and $t=0.1742$ is §8.1's "local minimum of 2.115
at $t=0.182$". **The defect was always present at the flagship too.**

### 10.2 Why it is seed-dependent

The step rule is *not* inhomogeneous in cfl — that hypothesis was tested and
rejected. Comparing $g(t)=\Delta t/\mathrm{cfl}$ across cfl on a common time
grid, the median deviation from the cfl $=0.01$ curve is $0.00$–$0.08\%$ at all
three seeds. What breaks $\mathrm{cfl}^4$ is that $g$ has a **local maximum at
the passage** (because the rate has a local minimum there), so the run's longest
steps sit exactly where the error is generated, and *those* steps are
pre-asymptotic: at $(0.2,-0.7)$, cfl $=0.08$, the passage step is
$1.52\times10^{-2}$ and its measured $C^{1/5}$ is $6.181$ against an asymptotic
$4.894$ — a $26\%$ rate excess, i.e. the local error there is $\approx3\times$
what the $\Delta t^5$ law predicts.

Splitting the compensated drift by time, $D(t)/\mathrm{cfl}^4$ ($\times10^5$),
under the shipped (4.8″):

| roots | $t=0.10$ | $0.15$ | $0.20$ | $0.30$ | $0.80$ |
|---|---|---|---|---|---|
| $(0.1,-0.6)$, cfl 0.08 | $-0.004$ | $-0.974$ | $\mathbf{-21.24}$ | $-21.30$ | $-19.77$ |
| $(0.1,-0.6)$, cfl 0.04 | $-0.004$ | $-0.883$ | $\mathbf{-3.74}$ | $-3.81$ | $-2.07$ |
| $(0.1,-0.6)$, cfl 0.02 | $-0.004$ | $-0.872$ | $\mathbf{-2.62}$ | $-2.76$ | $-0.80$ |
| $(0.5,0.25)$, cfl 0.08 | $+0.997$ | $+0.997$ | $+0.739$ | $+0.738$ | $+3.474$ |
| $(0.5,0.25)$, cfl 0.02 | $+0.878$ | $+0.880$ | $+0.007$ | $+0.002$ | $+2.950$ |

The curves are **identical up to $t=0.15$ and separate inside one passage**, and
then run parallel to $t_{\max}$. The flagship's curves separate in the *same*
passage — but that passage contributes $\approx26\%$ of its total drift, against
$\approx100\%$ at the two failing seeds. **That fraction is the entire
seed-dependence.** The flagship is protected by an accident: its flow drives
$A\to0$, where the $\varepsilon$-softened denominator of the *same* term spikes
to $\mathrm{rate}'=6.05\times10^{3}$ and over-resolves the neighbourhood. The
failing seeds reach only $331$ and $630$ — a turning point without a
near-zero gets the dip without the spike.

### 10.3 The repair, and the seed sweep

Contract Erratum 3: $\mathrm{rate}_{4.8'''}^2 = \mathrm{rate}_{4.8'}^2 +
\sum_\theta|\ddot\theta|/\sqrt{\theta^2+\varepsilon^2}$, with $\ddot z$ from one
directional finite difference at a probe length $\eta/\mathrm{rate}_{4.8'}$ that
is set by the state and never by cfl. (4.8″) is then applied to this rate. One
extra field evaluation per step out of thirteen.

**26 root pairs**, $N=24$, $\alpha'=10^{-6}$, $t_{\max}=0.8$, exact halvings cfl
$0.08/0.04/0.02$. "min ulps" is the smallest $|{\rm drift}|$ in the scan measured
in ulps of $E(0)$ — below $\approx50$ the energy oracle is measuring its own
roundoff (§8.6.2). The fixed-$\Delta t$ column is a plain RK4 control with **no
adaptive rule whatsoever** on the identical field, $2000/4000/8000$ steps.

| roots | $A_0$ | $b_0$ | max cond$J$ | term | 4.8″ spread | **4.8‴ spread** | flips | min halving | min ulps | fixed-$\Delta t$ spread |
|---|---|---|---|---|---|---|---|---|---|---|
| (0.5, 0.25) | 0.5625 | -0.125 | 7.34e+03 | t_max | 1.18 | **1.014** | 0 | 16.06 | 7820 | 1.1 |
| (0.2, -0.7) | 0.25 | 0.14 | 68 | t_max | 2.51 | **1.922** | 0 | 19.24 | 431 | 1.4 |
| (0.1, -0.6) | 0.25 | 0.06 | 387 | t_max | 24.74 | **1.178** | 0 | 14.02 | 1995 | 1.7 |
| (0.3, -0.4) | 0.01 | 0.12 | 1.53e+03 | t_max | 1.07 | **1.054** | 0 | 15.51 | 3124 | 1.2 |
| (0.6, 0.1) | 0.49 | -0.06 | 3.71e+09 | lock\_singular/t\_max | 82.54 | **81.381** | 0 | 1.48 | 11689005226 | 38.0 |
| (0.45, 0.45) | 0.81 | -0.2025 | 3.28e+07 | rho\_ceiling | 29.05 | **34.860** | 0 | 2.34 | 2945913 | 64.1 |
| (0.05, -0.55) | 0.25 | 0.0275 | 2.91e+05 | t_max | 40.06 | **4.750** | 0 | 21.93 | 1397103 | 182.9 |
| (0.15, -0.65) | 0.25 | 0.0975 | 136 | t_max | 12.67 | **1.360** | 0 | 13.12 | 1165 | 1.2 |
| (0.25, -0.75) | 0.25 | 0.1875 | 50.7 | t_max | 1.13 | **1.446** | 0 | 18.01 | 2527 | 1.4 |
| (0.35, -0.85) | 0.25 | 0.2975 | 25.3 | t_max | 15.19 | **3.472** | 1 | 4.61 | 26 | 404.6 |
| (0.7, -0.2) | 0.25 | 0.14 | 68 | t_max | 2.51 | **1.974** | 0 | 19.72 | 420 | 1.4 |
| (0.5, -0.5) | 0 | 0.25 | 114 | t_max | 1.23 | **1.067** | 0 | 15.33 | 10748 | 3.3 |
| (0.45, -0.35) | 0.01 | 0.1575 | 593 | t_max | 1.09 | **1.079** | 0 | 14.98 | 5087 | 1.1 |
| (0.55, -0.45) | 0.01 | 0.2475 | 134 | t_max | 1.67 | **1.066** | 0 | 15.32 | 9718 | 2.6 |
| (0.3, 0.1) | 0.16 | -0.03 | 6.23e+05 | t_max | 1.08 | **1.062** | 0 | 15.47 | 828 | 9.4 |
| (0.4, 0.2) | 0.36 | -0.08 | 2.6e+04 | t_max | 1.09 | **1.074** | 0 | 15.22 | 2673 | 1.3 |
| (0.6, 0.3) | 0.81 | -0.18 | 2.84e+03 | t_max | 1.41 | **1.056** | 0 | 15.52 | 14523 | 1.2 |
| (0.65, 0.15) | 0.64 | -0.0975 | 1.38e+09 | t_max | 55.50 | **50.652** | 0 | 1.52 | 43668514249 | 191.9 |
| (-0.3, 0.6) | 0.09 | 0.18 | 500 | rho\_ceiling | 2.84 | **1.458** | 0 | 11.49 | 5158 | 1.0 |
| (0.45, -0.15) | 0.09 | 0.0675 | 5.07e+08 | lock\_singular/t\_max | 83.20 | **30.378** | 0 | 2.08 | 221537619 | 252.1 |
| (0.12, -0.42) | 0.09 | 0.0504 | 1.27e+08 | lock\_singular | 85.10 | **41.079** | 0 | 2.16 | 136805091 | 198.5 |
| (0.22, 0.33) | 0.3025 | -0.0726 | 7.85e+04 | t_max | 1.06 | **1.189** | 0 | 13.46 | 1523 | 13.3 |
| (-0.2, 0.75) | 0.3025 | 0.15 | 51.1 | t_max | 1.27 | **1.476** | 0 | 18.05 | 3231 | 5.5 |
| (0.05, 0.85) | 0.81 | -0.0425 | 798 | t_max | 1.07 | **1.078** | 0 | 15.30 | 340 | 1.6 |
| (0.8, -0.1) | 0.49 | 0.08 | 156 | t_max | 1.03 | **1.031** | 0 | 16.19 | 5590 | 1.2 |
| (0.35, 0.35) | 0.49 | -0.1225 | 7.03e+06 | t_max | 72.12 | **206.287** | 0 | 1.08 | 3318812763 | 227.0 |

Reading it. **Both §9.2 failures are repaired** ($3.65\to1.92$ on the five-point
ladder; $24.74\to1.18$), and so is a **third that nobody had tried**,
$(0.15,-0.65)$, which fails at $12.67\times$ *with a sign flip* under (4.8″) —
found by this sweep, not reported to it. (4.8‴) improves the spread at **20 of
26** seeds; the six where it does not are $(0.45,0.45)$, $(0.25,-0.75)$,
$(0.22,0.33)$, $(-0.2,0.75)$, $(0.05,0.85)$ and $(0.35,0.35)$, of which four are
already flat to $\le1.5\times$ and the other two are the ill-conditioned pair
discussed next.

Note $(0.7,-0.2)$ and $(0.2,-0.7)$ give the same $(A_0,b_0)=(0.25,0.14)$, hence
the same seed profile and the same trajectory; their independently-computed
drifts agree to $2.7\%$ at the smallest value and better elsewhere, which is a
free reproducibility check and **not** a second seed.

### 10.4 Controls: every remaining failure fails a fixed-$\Delta t$ control too

Standing rule 3. At **every one of the eight seeds where (4.8‴) still fails the
$\le2\times$ band**, a fixed-$\Delta t$ plain RK4 control on the identical
field, with no adaptive rule at all and at $2000$–$8000$ steps (more resolved
than the adaptive runs' $1080$–$4320$ RK4 steps), fails the same criterion — in
seven of the eight by a much wider margin:

| roots | (4.8‴) spread | fixed-$\Delta t$ spread | fixed-$\Delta t$ min ratio per halving | why |
|---|---|---|---|---|
| $(0.35,0.35)$ | 206.3 | **227.0** | 1.05 | cond $J=7.0\times10^{6}$ |
| $(0.6,0.1)$ | 81.4 | 38.0 | 1.21 | cond $J=3.7\times10^{9}$, `lock_singular` |
| $(0.65,0.15)$ | 50.7 | **191.9** | 0.08 | cond $J=1.4\times10^{9}$ |
| $(0.12,-0.42)$ | 41.1 | **198.5** | 0.68 | cond $J=1.3\times10^{8}$, `lock_singular` |
| $(0.45,0.45)$ | 34.9 | **64.1** | 2.00 | Veronese double root, `rho_ceiling` |
| $(0.45,-0.15)$ | 30.4 | **252.1** | 0.39 | cond $J=5.1\times10^{8}$, `lock_singular` |
| $(0.05,-0.55)$ | 4.75 | **182.9** | 1.17 | cond $J=2.9\times10^{5}$ |
| $(0.35,-0.85)$ | 3.47 | **404.6** | 0.35 | drift $\le26$ ulps — roundoff floor |

The fixed-$\Delta t$ drifts at $(0.05,-0.55)$ are $-8.40,-7.05,-6.01\times10^{-6}$
for a $4\times$ refinement — essentially $\Delta t$-**insensitive**, which is the
signature of a closure/conditioning error rather than of integration error
(contract §4.6). (4.8‴) gets $-3.77\times10^{-7}\to-3.10\times10^{-10}$ there,
i.e. $2\times10^{4}$ times more accurate, and still cannot make it flat, because
what is left is not integration error. At $(0.35,-0.85)$ the control's drifts are
$+3.4,-1.9,+5.4\times10^{-15}$ — pure roundoff, sign-flipping, spread $405\times$;
(4.8‴)'s drift there is $+4.4\times10^{-13}\to-5.7\times10^{-15}$, the *smallest*
in the whole sweep, and it falls through the floor inside the scan.

So the classification is clean: **failures split into the open conditioning
question (§7.3 item 4) and the disclosed roundoff floor (§8.6.2), and at none of
them is the step rule the binding constraint.** This is the §8.6.4 argument
applied seed-wise instead of window-wise.

Conversely, (4.8‴) now *beats* the fixed-$\Delta t$ control at five
well-conditioned seeds where the control is the worse of the two —
$(0.3,0.1)$ $1.06$ vs $9.37$, $(0.22,0.33)$ $1.19$ vs $13.31$, $(-0.2,0.75)$
$1.48$ vs $5.52$, $(0.5,-0.5)$ $1.07$ vs $3.28$, $(0.55,-0.45)$ $1.07$ vs
$2.58$. §7.2.A's original refutation was "a fixed-$\Delta t$ control is clean
and the adaptive rule is not"; that has now reversed.

### 10.5 What is *not* fixed — disclosed limits

1. **The $\le2\times$ band has $4\%$ headroom at its worst seed.** Over the 13
   seeds that clear the evaluability preconditions (all three cfl terminate
   `t_max`, max cond $J<10^4$, drift above the floor), the worst compensated
   spread is $\mathbf{1.922}$ at $(0.2,-0.7)$ — against a band of $2.0$. That
   is the honest state of the gate. §8.9 flagged in advance that the band's
   headroom might be an artefact of the seed; it partly was. The band is
   **not** widened here (standing rule 2), and this is the first thing a
   skeptic should attack. Worse: the duplicate seed $(0.7,-0.2)$ — the *same*
   trajectory — computes $1.974$, not $1.922$, because its finest drift point
   ($420$ vs $431$ ulps) differs at roundoff level. So the worst-case number
   itself carries $\approx3\%$ of scatter from the floor, and the true margin
   on this ladder is $1$–$4\%$, not $4\%$. A gate with $1\%$ margin is not a
   gate that has been demonstrated to hold.
2. **The residual at $(0.2,-0.7)$ is coarse-end pre-asymptotic, not floor.**
   Its compensated values are $1.149,0.720,0.598\ (\times10^{-6})$ at cfl
   $0.08/0.04/0.02$ — falling monotonically, halvings $25.6$ and $19.2$, i.e.
   *super*-convergent, so cfl $=0.08$ (358 accepted steps) is not yet
   asymptotic at this seed. Shifting the ladder to $0.06/0.03/0.015$ gives
   spread $1.643$; that is reported here rather than substituted for the
   pinned ladder.
3. **The repair shrinks the measurable window.** Because the drift is now
   $3$–$84\times$ smaller at matched cfl, more configurations reach the
   $\sim10^{-14}$ roundoff floor of the energy sum inside a $4\times$ cfl
   scan. At $(0.2,-0.7)$, $N=30$, $\alpha'=10^{-2}$, $t_{\max}=12$, the drift
   runs $-1.607\times10^{-8}$, $-4.953\times10^{-11}$, $+5.33\times10^{-13}$,
   $+1.88\times10^{-13}$, $+6.41\times10^{-14}$ over cfl $0.2\to0.0125$ — a
   sign flip and a compensated spread of $117.7\times$, entirely because the
   last three points are $2401$, $846$ and $289$ ulps. The fixed-$\Delta t$
   control needs $80{,}000$ steps to reach $-1.04\times10^{-13}$, which the
   adaptive rule reaches at cfl $=0.05$. Between "pre-asymptotic above cfl
   $\approx0.15$" (§8.6.1) and "floor below $\approx10^{-13}$" this seed has
   essentially **no** evaluable window at that configuration. A more accurate
   integrator making the oracle unmeasurable is a real limitation and is
   stated as one.
4. **Two repairs that were tried and rejected**, both real experiments:
   (a) making the controller *self-consistent* — accept a step only if the
   error density measured on the step actually taken would itself have
   permitted a step that long, retrying otherwise — cuts $24.7\to5.4$ and
   $3.65\to2.43$ but no further, and adds nothing once the curvature term is
   present ($1.71$ vs $1.70$ at $(0.1,-0.6)$); (b) measuring $C$ from a
   cfl-independent Richardson probe of length $\theta/\mathrm{rate}'$ fixes
   $(0.2,-0.7)$ to $1.39$–$1.41$ but **breaks the flagship** ($13.6\times$
   with a sign flip at $\theta=0.05$), because where $\mathrm{rate}'$ is small
   the probe step is long enough to be meaningless. Neither is shipped.
5. **A wider curvature term changes little.** Extending the curvature sum to
   the amplitude coordinates $(u_0,u_1,u_2)$, softened by $\|u\|$, moves the
   worst seed $1.922\to1.724$ and every other seed by $<1\%$, at $0.5\%$ more
   steps. It is **not** shipped: the contract treats amplitudes through the
   KP CFL term, not through a parameter-drift term, and the defect diagnosed
   here is specifically in the lock-parameter terms. This is recorded because
   it is the obvious next lever if item 1's headroom proves too thin.
6. Everything open from §7.3/§8.9 is still open: `rho_ceiling` (which is why
   §8.4's sweep-window check still covers only the mildest two of nine
   $\alpha'$), $N$-convergence, the `order3`/`none` controls, and the $A<0$
   excursion.

### 10.6 Confirmed-good behaviour is not regressed

- **Flagship, $N=24$, $t_{\max}=0.8$:** compensated spread $1.178\to
  \mathbf{1.014}$, min halving $17.29\to16.06$, zero sign flips; drift at cfl
  $0.08$ improves $1.423\times10^{-9}\to4.505\times10^{-10}$ ($3.2\times$).
- **Sweep window, $N=30$, $t_{\max}=12$, $\alpha'=10^{-2}$** (§8.4's column):
  spread $1.03\times\to\mathbf{1.006}$, min exact halving $15.76\to15.99$,
  zero flips.
- **Sweep window, $\alpha'=3\times10^{-3}$:** spread $1.06\times\to
  \mathbf{1.018}$, drifts $-6.294\times10^{-8}$, $-4.001\times10^{-9}$,
  $-2.503\times10^{-10}$, zero flips.
- **Sweep window across 12 seeds** ($N=30$, $\alpha'=10^{-2}$, cfl
  $0.1/0.05/0.025$): 11 of 12 compensated-flat to $\le1.77\times$ with zero
  sign flips and min exact halvings $14.49$–$18.91$. The twelfth is
  $(0.2,-0.7)$, the floor case of §10.5 item 3.
- **Cost.** $+8\%$ field evaluations per step and $16$–$37\%$ more accepted
  steps. At cfl $=0.08$ the drift falls $3.2\times$ (flagship), $40\times$
  ($(0.2,-0.7)$) and $84\times$ ($(0.1,-0.6)$) — against the $1.8$, $3.2$ and
  $3.5\times$ that the extra steps alone would buy, i.e. **$1.8$, $12$ and
  $24\times$ beyond a uniform tightening.** That ratio is the signature of a
  targeted repair; a rule that merely shortened every step would show $1.0$.
  Measured wall-clock at matched cfl $=0.05$, same session, same machine:
  flagship $4.09\,$s $\to4.79\,$s ($1.17\times$, $559\to648$ steps),
  $(0.1,-0.6)$ $2.65\,$s $\to3.51\,$s ($1.33\times$, $350\to478$ steps). The
  (4.8″) arm of that comparison reproduces §8.1's flagship number at cfl
  $=0.05$ exactly ($+1.9308\times10^{-10}$ against §8.1's
  $1.931\times10^{-10}$), which is what licenses using it as the "before"
  column throughout §10.

### 10.7 Gates re-verified under eq. 4.8‴

- **T1 bit-for-bit.** `lock="none"` vs `shell.py`, $\nu\in\{0,10^{-3}\}$,
  $N\in\{12,24\}$: $\max|\Delta E| = \max|\Delta\Omega| = \max|\Delta t| =
  \max|\Delta u_{\text{final}}| = 0$ exactly, same termination reason. The
  curvature term is unreachable with no lock parameters, and
  `parameter_curvature` returns zeros there — asserted in the test as well as
  measured.
- **Lemma 4.4.** Over full windows, $\max\|Pu-u\|/\|u\| =
  8.29\times10^{-15}$ (flagship), $8.04\times10^{-15}$ ($(0.2,-0.7)$),
  $8.51\times10^{-15}$ ($(0.1,-0.6)$), $8.58\times10^{-15}$
  ($\alpha'=10^{-2}$ sweep window), $1.23\times10^{-14}$
  ($3\times10^{-3}$); $\operatorname{cond}J\le6.79\times10^{3}$ throughout.
  On 400 random states, $\max|\langle u,J\dot z\rangle|/(\|u\|\|J\dot z\|)
  = 4.28\times10^{-13}$ and $\max\|Pu-u\|/\|u\| = 1.12\times10^{-13}$.
- **The chart fix and `fit_lock_state`** are untouched by this round and their
  tests (`test_sym2_jacobian_has_no_branch_point_at_the_gauge_fold`,
  `test_repaired_chart_reproduces_the_legacy_algebra`,
  `test_fit_lock_state_is_exact_on_the_module_seed`) pass unchanged.
- **T5** at every seed of the sweep: min exact-halving ratios $13.12$–$19.72$
  across the 13 evaluable seeds, all $\ge8$; $15.99$ and $15.73$ at the two
  sweep-window $\alpha'$.
- **Suite** `pytest -m "not network"`: **85 passed** (72 before; the seed-swept
  gate contributes 13 parametrised cases), `ruff check src tests` clean,
  `ruff format --check` clean on both touched files. No tolerance, window,
  band or assertion was weakened anywhere; the regression gate was *widened*
  from 2 configurations to 13 seeds and gained two preconditions it did not
  previously assert.

### 10.8 Status, and the weakest untested assumption

The §9.2 defect is closed **subject to independent adversarial verification**,
which §7 and §9 both showed is not a formality. The mechanism is named, is
measurable on a trajectory generated without the rule under test, is present at
the flagship as well as at the failing seeds, and the repair removes it at 20 of
26 seeds with every remaining failure attributed by a control.

**The weakest untested assumption, stated explicitly (standing rule 7).** It is
*not* the mechanism — that is measured three independent ways. It is this:
**the claim rests on 13 seeds at one shell count, one $\alpha'$ and one window
length, and its worst case clears the gate by 4 %.** Specifically:

1. **The band could be exceeded by a seed not in the sweep.** 26 seeds is twice
   round 2b's 13 and it still found a *new* failure ((0.15,-0.65)) that nobody
   had tried. There is no argument here that 26 is enough; the same sentence
   was true of 13. A sweep of 100 seeds is the obvious next test, and the
   honest prior is that it finds something.
2. **The curvature term repairs first-order blindness, not blindness in
   general.** $|\ddot\theta|$ vanishes at inflection points of $\theta$, where
   $|\dot\theta|$ does not — so the two terms cover each other there — but a
   passage where $\dot\theta$ and $\ddot\theta$ are *both* small while
   $\theta^{(3)}$ is not would defeat both, and no such passage was searched
   for. The sweep contains no seed known to have one.
3. **Nothing here is checked at $N=60$, at the eight other $\alpha'$ of the
   sweep, or at $t_{\max}$ between $1.6$ and $12$**, and §10.5 item 3 shows the
   configuration space where the oracle is *evaluable at all* has narrowed.
4. **$\eta=10^{-3}$ is a convention.** Insensitivity was measured only at
   $10^{-2}/10^{-3}/10^{-4}$ at three seeds, and only over the flagship's
   three-point ladder.

As in §7 and §9: **nothing here counts for or against Conjecture 5.2.** It
removes one more instrument defect, and no more.

---

## 11. §10 adversarially checked — refuted for the third consecutive round, exactly as §10.8 predicted

Independent Opus skeptic, same discipline as §7 and §9. **Verdict: refuted,
on generalization, with high confidence** — and the report's own §10.8 item 1
named the outcome in advance: *"a sweep of 100 seeds is the obvious next
test, and the honest prior is that it finds something."* It did.

### 11.1 What reproduces

Every number checked matched: the flagship spread ($1.0136$ vs $1.014$), all
13 pinned seeds, the duplicate-seed roundoff scatter ($1.9218$ vs $1.9745$
for $(0.2,-0.7)$ vs $(0.7,-0.2)$), and eq. 4.8″'s "before" column exactly
against §8.3. Suite green (85 passed), `ruff check`/`format --check` clean,
gate T1 bit-for-bit, the curvature term provably unreachable for
`lock="none"`. **No tolerance, band, window, threshold, or assertion was
weakened anywhere** — the skeptic checked the $\le2\times$ band, gate T5's
$\ge8\times$, the noise floor, `cond_ceiling`, `rho_ceiling`, `rcond`, the
cfl ladder and $t_{\max}$ individually. The two new `_drift_scan` assertions
are strictly additional constraints. The turning-point mechanism itself
survives: eq. 4.8‴ measurably helps at the seed that refutes it too (spread
$8.32\to5.43$, one fewer sign flip) — this is a real, substantial repair, not
a false one.

### 11.2 What refutes it

Of 51 seeds beyond the reported 26 satisfying the shipped test's *own*
membership rule (all cfl reach $t_{\max}$; $\operatorname{cond}J<10^4$;
drift $\ge200$ ulps), **5 fail the gate**, confirmed by calling the shipped
`_drift_scan`/`_assert_compensated_flatness` directly — the regression test
goes red the moment a seed its own rule admits is added. The worst, roots
$(0.175,-0.675)$, fails gate T5 outright (first halving $3.81\times$ against
the required $8\times$; compensated spread $5.431\times$).

**The §10.5-item-4 defence is falsified, not merely incomplete.** §10
claimed every remaining failure is matched by an equally-bad fixed-$\Delta t$
control (attributing the residual to conditioning, not the rule). At the new
failures the opposite holds — a plain fixed-$\Delta t$ RK4 control is
**clean**:

| seed | fixed-Δt spread | flips | shipped rule spread | shipped halving |
|---|---|---|---|---|
| $(0.175,-0.675)$ | $1.008$ | 0 | $5.431$ | $3.81\times$ |
| $(0.18,-0.68)$ | $1.252$ | 0 | $12.93\times$ worse | — |
| $(0.185,-0.685)$ | $1.255$ | 0 | $12.90\times$ worse | — |
| $(0.195,-0.695)$ | $1.697$ | 0 | $16.02\times$ worse | — |
| $(0.17,-0.67)$ | $1.324$ | 0 | $12.32\times$ worse | — |

This is verbatim the §7.2.A / §9.2 refutation signature — clean control,
dirty adaptive rule, well-conditioned field ($\operatorname{cond}J<10^4$ by
construction of the membership rule) — reproduced against round 2c.

**Why the 26-seed sweep missed it.** The failures sit inside the very
one-parameter family §10 densified to argue its mechanism ($A(0)=0.25$,
i.e. $\lambda+\mu=-0.5$): §10 sampled $b_0=0.0275,0.06,0.0975,0.14,
0.1875,0.2975$; the failures sit at $b_0=0.118,0.1224,0.1267,0.1355$ —
*between* the $0.0975$ sample (passes at $1.36$) and the $0.14$ sample
(passes at $1.92$, §10's disclosed worst case). Not one family either: the
$A(0)=0.3025$ family's two samples ($b_0=-0.0726,\,0.15$) straddle a fifth
failure at $b_0=0.105$. **The failing set has positive measure in seed
space, and 26 samples stepped over it twice.**

A sharper diagnostic on the disclosed worst seed itself: the pinned 3-point
ladder truncates a quantity still moving. At $(0.2,-0.7)$ the compensated
value falls $37\%$ then $17\%$ per halving (local orders $4.68$, $4.30$); a
two-term extrapolation gives an asymptotic spread of $\approx2.09$ — i.e.
the disclosed worst seed *plausibly fails in the limit* and clears the band
at $1.922$ only because the ladder stops at cfl $=0.02$, below which drift
falls under the roundoff floor. Consistent with, and sharper than, §10.5
items 2–3.

### 11.3 Disclosed limits of the skeptic's own check (no silent caps)

Compute was capped: seeds were triaged after cfl $=0.08$ and later cfl
values skipped once a precondition was already violated; `max_steps` was
lowered to $60{,}000$ from the default $300{,}000$. $112$ of $234$ planned
seeds had completed when the check stopped (an earlier full-ladder attempt
was killed at $32/233$ for slowness) — so the 5-failure count is a **lower
bound**. The $N=30$, matched-cost, and $t_{\max}=1.2$ controls were queued
and not run. Two of the five failures sit within $1.4\times$ of the
$200$-ulp precondition (arguable floor proximity); the other three do not
($443$–$697$ ulps, with coarse points at $2$–$10\times10^4$ ulps) and their
fixed-$\Delta t$ controls are clean at comparable ulp counts, which is what
separates "the oracle is noisy here" from "the rule is wrong here." The
skeptic did **not** establish that the five failures share one mechanism
with each other or with the two §9 failures — only that eq. 4.8‴ does not
generalize to them.

### 11.4 Bottom line, and a structural observation for round 2d

**Verdict: real, substantial, honestly-reported improvement (kept); central
generalization claim refuted (third consecutive round).** eq. 4.8‴ helps
broadly and is worth keeping in place of eq. 4.8″. But "the seed-dependence
is diagnosed and repaired" remains a property of the seeds it was measured
at.

**A pattern is now visible across three rounds that is worth naming
explicitly rather than repeating the cycle a fourth time unexamined.** Each
round has (a) found a real, mechanistically-explained, independently
confirmed improvement, (b) swept roughly twice as many seeds as the round
before ($1\to13\to26$), and (c) been refuted by a skeptic who went looking
one seed-density level higher and found the failing set has **positive
measure** — a continuous band between samples, not an isolated outlier. The
skeptic's own prognosis: *"A larger sweep will keep finding failures... until
the criterion is checked on a dense continuum in at least the two $A(0)$
families already known to contain failures — rather than on a hand-picked
seed list — 'compensated flat' should be read as a statement about the
listed seeds only."* If that prognosis is right, **finite seed sampling is
structurally the wrong validation strategy for a closed-form rate rule**,
and round 2d should not simply repeat rounds 2a–2c's shape (bigger sweep,
same rule family). Two directions that would actually change the argument,
neither yet attempted: (i) an analytic bound on the rate function proving it
cannot vanish faster than the error grows, over the whole $(A,b)$ chart, not
sampled points of it; (ii) abandon the closed-form-rate strategy for a
standard embedded-pair adaptive controller with step rejection (candidate
(d) from round 2a's original brief, tried partially in §10.5 item 4(a) and
not carried through) — slower per accepted step, but its correctness would
not depend on which seeds happened to be tested.
