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

**Action required:** check whether the Lean theorem `Reff_tdual` proves the
invariance form (likely, since it was kernel-accepted) and correct the paper's
prose to match. If the Lean statement carries the spurious factor, it cannot
have been proved and the audit needs revisiting.

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
