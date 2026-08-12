# W1 Step 1 — Implementation contract: the Sym²-constrained dyadic shell model

**Workflow:** W1 (Sym²-constrained shell model, the decisive experiment).
**Step:** 1 of 4 — Opus tier: derivation and contract.
**Consumes:** `src/socrates/dualscale/shell.py`, `src/socrates/dualscale/geometry.py`,
`src/socrates/operators/recurrence.py`, `paper/haloalg_spec.md` §2, `docs/FINDINGS.md` §4.
**Produces:** the specification that Step 2 (Sonnet) implements as
`src/socrates/dualscale/shell_sym2.py`.

**Question under test (Conjecture 5.2).** The bare Katz–Pavlović cascade with the
T-dual cutoff has peak enstrophy $\Omega_{\text{peak}} \sim \alpha'^{-0.672}$
(`FINDINGS.md` §4). Does constraining the shell profile to the symmetric-square
spectral structure $\{\lambda^2,\lambda\mu,\mu^2\}$ change that exponent?

---

## 0. How to read this document

Every displayed equation and every numbered claim carries a tier label, per the
programme's standing rule that Tier C claims must never be written in Tier A/B
tone.

| Tier | Meaning in this document |
|---|---|
| **B** | Exact consequence of the certified construction in `operators/recurrence.py`, or elementary algebra/calculus from it. Decidable by `Fraction` arithmetic or by a finite-difference check. A test is specified for each. |
| **C** | A physical-modelling choice, a closure, a numerical convention, or an interpretation of the spec's prose. **Not proven, and possibly not the only reasonable choice.** Where a Tier C choice could plausibly determine the experimental outcome, a matched control is mandated in §7. |

There is no Tier A material here. Nothing in §§3–6 is established mathematics
about Navier–Stokes; it is a definition of a model plus its diagnostics.

---

## 1. What is already certified (Tier B)

`operators/recurrence.py` certifies, in exact `Fraction` arithmetic
(`verify_symmetric_square`, and `FINDINGS.md` §2):

Given the order-2 operator $L_2:\; v_{n+2} = a\,v_{n+1} + b\,v_n$ with
characteristic roots $\lambda,\mu$ (so $\lambda+\mu=a$, $\lambda\mu=-b$), the
order-3 operator

$$u_{n+3} = c_2\,u_{n+2} + c_1\,u_{n+1} + c_0\,u_n,
\qquad (c_0,c_1,c_2) = \bigl(-b^3,\; b(a^2+b),\; a^2+b\bigr)
\tag{1.1 — Tier B}$$

annihilates **every** pointwise product $v^{(1)}_n v^{(2)}_n$ of two $L_2$
solutions, and hence (by linearity) their whole span. Equation (1.1) is exactly
`closed_form_symmetric_square(a, b)` and exactly
`symmetric_square(RecurrenceOperator.of(b, a)).coefficients`.

> **Coefficient-order footgun (Tier B, mechanical).** `RecurrenceOperator`
> stores coefficients in *ascending shift order*: for
> $v_{n+2} = a v_{n+1} + b v_n$ the constructor call is
> `RecurrenceOperator.of(b, a)` — `b` first. The module under this contract
> must call it that way and must assert agreement with
> `closed_form_symmetric_square`.

In elementary-symmetric coordinates $x^3 - e_1x^2 + e_2x - e_3$, so that
$(e_1,e_2,e_3) = (c_2,\,-c_1,\,c_0)$:

$$e_1 = a^2+b,\qquad e_2 = -b(a^2+b),\qquad e_3 = -b^3 .
\tag{1.2 — Tier B}$$

The solution space of (1.1) is spanned by the three geometric sequences

$$\mathrm{Sym}^2(V) \;=\; \operatorname{span}\bigl\{\lambda^{2n},\;(\lambda\mu)^n,\;\mu^{2n}\bigr\},
\qquad V = \operatorname{span}\{\lambda^n,\mu^n\}.
\tag{1.3 — Tier B}$$

### Proposition A — root-free characterisation of the Sym² locus

**Claim (Tier B).** An order-3 constant-coefficient recurrence with
characteristic polynomial $x^3 - e_1x^2 + e_2x - e_3$ is a symmetric square of
an order-2 operator over $\mathbb{C}$ **iff**

$$e_2^{\,3} = e_1^{\,3}\,e_3 .
\tag{1.4 — Tier B}$$

Equivalently: **the three eigenvalues form a geometric progression.**

*Proof.* ($\Rightarrow$) Substitute (1.2): $e_2^3 = -b^3(a^2+b)^3 = e_1^3e_3$.
($\Leftarrow$) Suppose (1.4) and $e_1\neq0$. Put $t = e_2/e_1$; then
$t^3 = e_2^3/e_1^3 = e_3$, and

$$t^3 - e_1t^2 + e_2t - e_3 \;=\; e_3 - e_1t^2 + e_2 t - e_3 \;=\; t\,(e_2 - e_1 t) \;=\; 0,$$

so $t$ is a root. Writing the roots as $\{r_1,t,r_3\}$ we get
$r_1 t r_3 = e_3 = t^3$, hence $r_1 r_3 = t^2$: the roots are in geometric
progression, and setting $\lambda^2 = r_1$, $\mu^2 = r_3$, $\lambda\mu = t$
realises them as a Sym² spectrum. If $e_1 = 0$ then (1.4) forces $e_2 = 0$, the
polynomial is $x^3 = e_3$, and any root $t$ satisfies $r_1r_3 = e_3/t = t^2$. $\square$

**Inverse map (Tier B).** Given $(e_1,e_2,e_3)$ satisfying (1.4), the *real*
order-2 preimage is

$$b = -\,e_3^{1/3}\ \text{(real cube root)},\qquad
a = \pm\sqrt{\,e_1 + e_3^{1/3}\,},
\tag{1.5 — Tier B}$$

which exists in $\mathbb{R}$ iff $e_1 + e_3^{1/3} \ge 0$. The sign of $a$ is a
gauge freedom: $(a,b)$ and $(-a,b)$ give the same $L_3$ (only $a^2$ appears in
(1.2)), i.e. the map $(a,b)\mapsto L_3$ is 2:1. **This gauge redundancy must be
fixed in the implementation** (convention: $a \ge 0$), otherwise the constrained
state has a null direction that pollutes the Jacobian's conditioning.

### Spectral radius, root-free (Tier B)

Let $\rho_3(a,b)$ be the spectral radius of $L_3$, i.e.
$\max(|\lambda|^2,|\lambda\mu|,|\mu|^2) = \max(|\lambda|,|\mu|)^2$. With
$\Delta = a^2 + 4b$:

$$\rho_3(a,b) \;=\;
\begin{cases}
\left(\dfrac{|a| + \sqrt{\Delta}}{2}\right)^{\!2}, & \Delta \ge 0 \quad(\text{real roots}),\\[2ex]
|b|, & \Delta < 0 \quad(\lambda,\mu \text{ conjugate, } |\lambda|^2=|\lambda\mu|=|b|).
\end{cases}
\tag{1.6 — Tier B}$$

---

## 2. What does **not** follow from the certified algebra

This section is the honest core of the derivation. It must survive into the
FINDINGS entry at Step 4.

**Observation 2.1 (Tier B).** $\mathrm{Sym}^2(V)$ is **not closed under
pointwise multiplication.** For $u,w \in \mathrm{Sym}^2(V)$, the product $u_nw_n$
has spectrum contained in
$\{\lambda^4,\lambda^3\mu,\lambda^2\mu^2,\lambda\mu^3,\mu^4\} =
\operatorname{spec}\mathrm{Sym}^4(V)$, which meets
$\{\lambda^2,\lambda\mu,\mu^2\}$ only on a measure-zero set of $(\lambda,\mu)$.

**Consequence 2.2 (Tier B).** The Katz–Pavlović nonlinearity, being quadratic in
$u$, maps the constraint set off itself. Therefore:

> The spec's claim (`haloalg_spec.md` §2, "Spectral Restriction Analysis") that
> "this restriction ensures that enstrophy … cannot blow up independently"
> **does not follow from the Sym² algebra alone.** The restriction is not an
> invariant of the dynamics; it can only be maintained by imposing a closure by
> hand. *The closure is the modelling content, and it is Tier C.*

This is the same failure mode already recorded in `REVIEW_haloalg.md` (the
`sym2_recurrence` name being overloaded onto a bit-width claim it does not
prove): a certified algebraic identity is being used as if it were a dynamical
theorem. It is not. What §§3–4 below construct is a *model* whose behaviour is
measurable — not a proof of anything.

**Remark 2.3 (Tier C, exploratory; not to be implemented in Step 2).** The
class that *is* closed under the nonlinearity is the graded family
"$u_n$ is a homogeneous polynomial of degree $d$ in $(\lambda^n,\mu^n)$", with
the nonlinearity mapping $d \mapsto 2d$. Truncating that grading at $d=2$ is
exactly the Sym² lock; a $d=4$ (Sym⁴) variant would measure sensitivity to the
truncation degree. Deferred; noted so that a null result at $d=2$ is not
over-read as a null result for the whole idea.

---

## 3. The constrained manifold

### 3.1 Half-ladder and the symmetric-square map (Tier B, definitions)

The **half-ladder** is a sequence $v = (v_n)_{n\ge0}$ obeying the order-2
recurrence

$$v_{n+2} = a\,v_{n+1} + b\,v_n,
\tag{3.1 — Tier B (definition)}$$

determined by $(v_0,v_1,a,b)$. Shell amplitudes are built from it by the
symmetric-square map. Two strengths of that map are algebraically distinct and
both are worth measuring:

**(i) Linear Sym² lock** — $u \in \mathrm{Sym}^2(V(a,b))$, i.e. $u$ is a
$\mathbb{R}$-linear combination of products of half-ladder solutions:

$$u_n \;=\; A\,\lambda^{2n} + B\,(\lambda\mu)^n + C\,\mu^{2n}
\;\;\Longleftrightarrow\;\;
u_{n+3} = c_2u_{n+2} + c_1u_{n+1} + c_0u_n \text{ with } (1.1).
\tag{3.2 — Tier B}$$

This is precisely — and only — what `recurrence.py` certifies. Root-free state:

$$z \;=\; (u_0,\,u_1,\,u_2,\,a,\,b) \in \mathbb{R}^5,
\qquad u = \Phi_{\text{sym2}}(z) \in \mathbb{R}^N .
\tag{3.3 — Tier B}$$

**(ii) Veronese (rank-1) lock** — the literal reading "$u$ *is* the square of the
half-ladder":

$$u_n = v_n^2,\qquad z = (v_0,v_1,a,b) \in \mathbb{R}^4 .
\tag{3.4 — Tier B}$$

In the eigenbasis this is $A=\alpha^2$, $B=2\alpha\beta$, $C=\beta^2$, i.e. the
Veronese quadric cone

$$B^2 = 4AC
\tag{3.5 — Tier B}$$

inside (3.2), together with the sign restriction $u_n \ge 0$. Root-free form of
(3.5) in terms of $(u_0,u_1,u_2,a,b)$:
$\bigl(u_2 - a^2u_1 - b^2u_0\bigr)^2 = 4a^2b^2\,u_0u_1$ with $u_0,u_1 \ge 0$.
**Note (Tier B):** (3.4) is *strictly stronger* than (3.2); the squaring map
$v \mapsto v^2$ has 2-dimensional image inside the 3-dimensional
$\mathrm{Sym}^2(V)$. Conflating the two would misstate what is being tested.

### 3.2 The lock ladder (Tier C — experimental design)

Four nested/adjacent models, all integrated by the same machinery, differing
only in the constraint set. The middle two are the experiment; the outer two are
the controls that make it interpretable.

| `lock` | Constraint set $\mathcal{M}$ | dim | Role |
|---|---|---|---|
| `"none"` | $\mathbb{R}^N$ (unconstrained) | $N$ | **Control arm.** Must reproduce `shell.py` exactly. |
| `"order3"` | $u$ obeys *some* order-3 recurrence, $(u_0,u_1,u_2,c_0,c_1,c_2)$ free | 6 | **Dimension-matched control.** Same "low-dimensional geometric profile" closure, *without* the Sym² restriction (1.4). |
| `"sym2"` | (3.2): order-3 **and** $e_2^3=e_1^3e_3$ | 5 | **The hypothesis.** |
| `"veronese"` | (3.4): $u=v^2$ | 4 | Strictest reading of the spec's prose. |

**Why `"order3"` is not optional.** `"sym2"` differs from `"order3"` by exactly
one scalar equation, (1.4). If both arrest the cascade, the arrest is an artifact
of collapsing $N$ degrees of freedom onto a handful of geometric modes — a
Galerkin-truncation effect that any 5-parameter profile family would produce —
and says nothing about symmetric squares. Only a *separation* between `"sym2"`
and `"order3"` is evidence for Conjecture 5.2's stated mechanism. This is the
single most important control in W1.

---

## 4. The constrained right-hand side

### 4.1 The unconstrained field (Tier B — restatement of `shell.py`)

With $k = (k_n)$ the wavenumber ladder (either $k_n = \kappa^n$ or its T-dual
image $k_{\text{eff},n} = \min(k_n, 1/(\alpha' k_n))$ from
`geometry.effective_wavenumber`), and boundary shells closed exactly as in
`shell._rhs`:

$$F_n(u) \;=\; k_{n-1}u_{n-1}^2 \;-\; k_n u_n u_{n+1} \;-\; \nu k_n^2 u_n,
\qquad u_{-1} := 0,\; u_N := 0 .
\tag{4.1 — Tier B}$$

Energy $E = \tfrac12\sum_n u_n^2$ obeys, by exact telescoping of (4.1),

$$\langle u, F(u)\rangle \;=\; -\,\nu\sum_n k_n^2u_n^2 \;=\; -\nu\,\Omega .
\tag{4.2 — Tier B}$$

### 4.2 Reconstruction and its Jacobian (Tier B)

For `lock="sym2"`, $\Phi(z)_n = u_n$ for $n\le2$ and
$\Phi(z)_n = c_2\Phi_{n-1} + c_1\Phi_{n-2} + c_0\Phi_{n-3}$ for $n\ge3$, with
$(c_0,c_1,c_2)$ from (1.1). The Jacobian $J(z) = D\Phi(z) \in \mathbb{R}^{N\times5}$
is generated by the *same* recurrence, differentiated:

- amplitude columns $j\in\{0,1,2\}$: $G^{(j)}_i = \delta_{ij}$ for $i<3$, then
  $G^{(j)}_n = c_2G^{(j)}_{n-1} + c_1G^{(j)}_{n-2} + c_0G^{(j)}_{n-3}$;
- parameter columns $\theta \in \{a,b\}$: $D^\theta_n = 0$ for $n<3$, then

$$D^\theta_n = c_2D^\theta_{n-1} + c_1D^\theta_{n-2} + c_0D^\theta_{n-3}
\;+\; \frac{\partial c_2}{\partial\theta}u_{n-1}
\;+\; \frac{\partial c_1}{\partial\theta}u_{n-2}
\;+\; \frac{\partial c_0}{\partial\theta}u_{n-3},
\tag{4.3 — Tier B}$$

$$\frac{\partial(c_0,c_1,c_2)}{\partial a} = \bigl(0,\;2ab,\;2a\bigr),
\qquad
\frac{\partial(c_0,c_1,c_2)}{\partial b} = \bigl(-3b^2,\;a^2+2b,\;1\bigr).
\tag{4.4 — Tier B}$$

Analogous constructions for `"order3"` ($\partial c_i/\partial c_j = \delta_{ij}$,
so the parameter columns are driven by $u_{n-1},u_{n-2},u_{n-3}$ directly) and
for `"veronese"` ($\Psi_n = v_n^2$, so $\partial_\theta\Psi_n = 2v_n\partial_\theta v_n$
with $\partial_\theta v$ generated by the differentiated order-2 recurrence).

### 4.3 The closure (Tier C — the modelling choice)

By Consequence 2.2, $F(\Phi(z)) \notin T_u\mathcal{M}$ in general. The contract
adopts the **energy-metric Galerkin projection** as the primary closure:

$$\boxed{\;\dot z \;=\; J(z)^{+}\,F\bigl(\Phi(z)\bigr)\;}
\qquad\Longrightarrow\qquad
\dot u \;=\; P\,F(u),\quad P = JJ^{+},
\tag{4.5 — Tier C (closure choice); the identity $\dot u = PF$ is Tier B}$$

where $J^{+}$ is the Moore–Penrose pseudo-inverse (**not** a Tikhonov-regularised
inverse — see Lemma 4.4) and $P$ is the orthogonal projector onto
$\operatorname{range}J = T_u\mathcal{M}$.

Written out for `"sym2"`, (4.5) says: the five numbers
$(u_0,u_1,u_2,a,b)$ evolve so as to track the Katz–Pavlović field as closely as
the constraint permits, in the $L^2$ (= energy) norm on shell space. **The
cascade, in this model, is no longer a front propagating in $n$; it is the drift
of the half-ladder parameters $(a,b)$**, since those set the profile's decay rate.
That reformulation is the derivation's main structural content, and it is
Tier C: it is what the lock *means*, not something the lock is proven to do.

### Lemma 4.4 — the lock preserves the energy budget exactly (Tier B)

**Claim.** Under (4.5), with the pseudo-inverse and any of the four locks,

$$\frac{dE}{dt} \;=\; \langle u, F(u)\rangle \;=\; -\nu\,\Omega ,
\tag{4.6 — Tier B}$$

i.e. **identical to the unconstrained model.** In particular the inviscid locked
model conserves energy exactly, and `shell.py`'s energy-drift oracle remains a
pure measure of *integration* error for the locked runs.

*Proof.* Each reconstruction map is positively homogeneous of degree $d>0$ in a
subset of its coordinates ($d=1$ in $(u_0,u_1,u_2)$ for `"sym2"`/`"order3"`;
$d=2$ in $(v_0,v_1)$ for `"veronese"`; $d=1$ trivially for `"none"`). Euler's
identity gives $\sum_j z_j\,\partial_j\Phi = d\,\Phi$, hence
$u = \Phi(z) \in \operatorname{range}J(z)$ — *for every $z$, including where $J$
is rank-deficient.* Since $P$ is the orthogonal projector onto
$\operatorname{range}J$, $Pu = u$ and $P^{\mathsf T}=P$, so

$$\frac{dE}{dt} = \langle u,\dot u\rangle = \langle u, PF\rangle
= \langle Pu, F\rangle = \langle u, F(u)\rangle . \qquad\square$$

**Corollaries (Tier B).**
1. The lock **cannot** arrest the cascade by leaking energy. Any arrest must come
   from reshaping the profile. This removes the most obvious trivial explanation
   of a positive result.
2. Viscous dissipativity $dE/dt \le 0$ is preserved exactly.
3. The energy metric $W=I$ is the **only** weight in the family
   $\dot z = (J^{\mathsf T}WJ)^{+}J^{\mathsf T}WF$ for which (4.6) holds
   ($P_W$ is $W$-symmetric, not $I$-symmetric, so the step
   $\langle u,P_WF\rangle=\langle P_W^{\mathsf T}u,F\rangle$ fails for $W\neq I$).
   This is why (4.5) is the primary closure and the alternatives of §4.6 are
   robustness probes with a *different* gate.
4. Tikhonov regularisation $(J^{\mathsf T}J+\epsilon I)^{-1}J^{\mathsf T}$ breaks
   $Pu=u$ and therefore **manufactures energy drift that is not integrator
   error**, silently corrupting the project's main oracle. It is forbidden by
   this contract. Rank deficiency is handled by `pinv`/`lstsq` with an explicit
   `rcond`, which preserves (4.6).

### 4.5 The half-ladder right-hand side, written out (Tier B, with caveat)

For the Veronese lock $u_n=v_n^2$, substituting into (4.1) and using
$\dot u_n = 2v_n\dot v_n$ gives the exact pointwise identity

$$\dot v_n \;=\; \frac{1}{2v_n}\Bigl(k_{n-1}v_{n-1}^4 - k_n v_n^2v_{n+1}^2\Bigr) - \frac{\nu}{2}k_n^2 v_n
\;=\; \frac{k_{n-1}v_{n-1}^4}{2v_n} - \frac{k_n v_n v_{n+1}^2}{2} - \frac{\nu}{2}k_n^2v_n .
\tag{4.7 — Tier B}$$

**Caveat (Tier B), and the reason (4.7) is not the model.** (4.7) is $N$
equations for a sequence $v$ that must *also* satisfy the 2-term recurrence (3.1)
— an overdetermined system with no solution for generic data, and singular
wherever $v_n = 0$. It is recorded here because it is the derivation the task
asks for and because it makes the obstruction concrete: **the half-ladder cannot
simply be evolved by the Katz–Pavlović law.** The projected system (4.5) is the
minimal well-posed replacement. Any implementation that integrates (4.7)
directly is wrong.

### 4.6 Alternative closures (Tier C — robustness probes only)

- `closure="collocation"`: solve $J[S,:]\,\dot z = F[S]$ on a 5-index set $S$
  (default: the shells of largest $k_n^2u_n^2$, fallback $\{0,\dots,4\}$). Does
  **not** satisfy (4.6).
- `closure="enstrophy"`: $W = \operatorname{diag}(k_n^2)$. Does **not** satisfy (4.6).

**Gate substitution (mandatory).** For non-Galerkin closures the energy-drift
gate is meaningless as an integrator oracle. It is replaced by: *energy drift
must converge to a nonzero $\alpha'$- and $t$-dependent constant as
$\mathrm{cfl}\to0$* (closure-induced drift is dt-independent; integrator error is
not). A non-Galerkin run whose drift keeps shrinking with cfl has a bug in the
closure; a Galerkin run whose drift does **not** shrink with cfl has a bug in the
projector.

### 4.7 Timestep control (Tier C — numerical convention)

`shell.py` uses $\Delta t = \mathrm{cfl}/\max_n k_n|u_n|$. The locked system has a
second timescale: the drift of the lock parameters. Use

$$\Delta t = \frac{\mathrm{cfl}}{\max\Bigl(\max_n k_n|u_n|,\;
\dfrac{|\dot a|}{|a|+\varepsilon},\;\dfrac{|\dot b|}{|b|+\varepsilon},\;
\nu\max_n k_n^2\Bigr)},\qquad \varepsilon = 10^{-3}.
\tag{4.8 — Tier C}$$

With `lock="none"` this must reduce **exactly** to `shell.py`'s formula (the
parameter terms are absent), which is the basis of gate T1. The convention is
validated only by the dt-refinement control (§7.C1), not by argument.

---

## 5. The enstrophy bridge: from the lock to the measured exponent

This section derives what the α′-sweep of Step 3 should see, *given* the lock.
It supplies the internal consistency check that separates physics from bugs.

### 5.1 Profile and cutoff (Tier B, given a geometric profile)

Let $k_n = \kappa^n$ ($\kappa=2$), $k_{\text{eff},n} = \min(k_n, 1/(\alpha'k_n))$,
and let $n_\star$ be the cutoff shell:

$$\kappa^{n_\star} = \alpha'^{-1/2}
\quad\Longrightarrow\quad
n_\star = -\tfrac12\log_\kappa\alpha' .
\tag{5.1 — Tier B}$$

Above $n_\star$, $k_{\text{eff},n} = \kappa^{2n_\star-n}$ (the dual reflection).
Suppose the locked profile is dominated by one $L_3$ eigenvalue,
$u_n \simeq U\rho_3^{\,n}$. Then with $q := \kappa^2\rho_3^2$,

$$\Omega = \sum_n k_{\text{eff},n}^2u_n^2
= U^2\underbrace{\sum_{n\le n_\star} q^{\,n}}_{\text{sub-cutoff}}
+ U^2\kappa^{4n_\star}\underbrace{\sum_{n>n_\star}(\rho_3^2/\kappa^2)^{n}}_{\text{trans-cutoff}} .
\tag{5.2 — Tier B}$$

Both sums are geometric and both are $\asymp q^{\,n_\star}$ when $q>1$ (the
trans-cutoff series always converges, since $\rho_3^2/\kappa^2<1$ whenever the
profile itself is summable). Hence

$$\Omega_{\text{peak}} \;\asymp\; q^{\,n_\star}
\;=\; \alpha'^{\,-\frac12\log_\kappa(\kappa^2\rho_3^2)} .$$

### 5.2 The master formula (Tier B, given the lock and a single dominant mode)

$$\boxed{\;p \;=\; \frac{d\log\Omega_{\text{peak}}}{d\log\alpha'} \;=\; -1 - \log_\kappa\rho_3\;}
\qquad (\rho_3 > 1/\kappa);\qquad p = 0 \ \ (\rho_3 \le 1/\kappa).
\tag{5.3 — Tier B}$$

Calibration points, $\kappa=2$:

| $\rho_3$ | value | predicted $p$ | reading |
|---|---|---|---|
| $\le 1/\kappa$ | $\le 0.5$ | $0$ | **arrest** — $\Omega$ bounded as $\alpha'\to0$ |
| $\kappa^{-1/3}$ | $0.7937$ | $-2/3$ | **Kolmogorov / lock inert** (matches `FINDINGS.md` §4) |
| $1$ | $1$ | $-1$ | Theorem 4.2 ceiling rate — lock actively *worse* |
| $>1$ | — | $<-1$ | profile grows in $n$: truncation-dominated, **not physical** |

The K41 entry is a genuine consistency check of the whole framework, not a
fitted coincidence: $u_n\sim k_n^{-1/3}$ gives $\rho_3=\kappa^{-1/3}$ and (5.3)
returns exactly $-2/3$, reproducing the measured $-0.672$ of the control arm.

**Marginal case (Tier B).** At $\rho_3 = 1/\kappa$ exactly, $q=1$ and (5.2) gives
$\Omega \asymp n_\star \sim \log(1/\alpha')$: the fitted power-law exponent is $0$
but $\Omega$ is still unbounded. This is a *third* outcome, distinct from both
"arrest" and "inert", and it must be tested for explicitly (§6, D3) rather than
reported as arrest.

### 5.3 Two independent estimators of $\rho_3$ (Tier B / Tier C)

- `spectral_radius` — algebraic, from (1.6) at the time of peak enstrophy. **Tier B.**
- `profile_decay_rate` — empirical, $\exp\bigl(\text{slope of }\log|u_n|\text{ over } 2\le n\le n_\star\bigr)$. **Tier B** as a measurement; its *identification* with $\rho_3$ assumes the dominant mode carries nonzero amplitude (**Tier C**).

If the two disagree, the dominant $L_3$ root is unexcited ($A=0$ in (3.2)). That
is informative structure, not a bug — but it must be reported, and (5.3) must
then be applied with `profile_decay_rate`, not `spectral_radius`.

---

## 6. Falsifiable predictions and the decision table

### 6.1 Operational definition of the measurement

Reproduce the `FINDINGS.md` §4 protocol exactly, changing only the lock:
$N=30$ shells, $\alpha' \in \{10^{-2},\dots,10^{-10}\}$ (9 log-spaced values),
$t_{\max}=12$, $\nu=0$, $\mathrm{cfl}=0.05$, identical seed profile across locks
(§8.4). Fit $p$ by least squares of $\log_{10}\Omega_{\text{peak}}$ on
$\log_{10}\alpha'$ over runs that terminated `t_max`.

### 6.2 What "exponent → 0" means, precisely

**Arrest** is declared **only if all four hold:**

- **A1.** $|p| < 0.05$ over the full sweep;
- **A2.** the local slope over the last three (smallest) $\alpha'$ values also satisfies $|p_{\text{local}}| < 0.05$ (guards against a crossover masquerading as arrest);
- **A3.** $\Omega_{\text{peak}}(10^{-10}) / \Omega_{\text{peak}}(10^{-2}) < 10$ — less than one decade of growth across eight decades of $\alpha'$;
- **A4.** the enstrophy-carrying shell $\arg\max_n k_{\text{eff},n}^2u_n^2$ at $t_{\text{peak}}$ stays **bounded** as $\alpha'\to0$, i.e. it decouples from $n_\star$ of (5.1).

A4 is the mechanistic statement and is the one that cannot be faked by a fit.

### 6.3 Decision table

| Outcome | $p$ | $\rho_3$ at $t_{\text{peak}}$ | peak shell vs $n_\star$ | tangency defect $\dfrac{\|(I-P)F\|}{\|F\|}$ | `"sym2"` vs `"order3"` | Reading |
|---|---|---|---|---|---|---|
| **D1 Arrest** | $\approx 0$, A1–A4 all pass | $\le 1/\kappa$ | bounded, decoupled | $O(1)$ | **separated** | Support for Conjecture 5.2 |
| **D2 Inert (dynamically)** | $\approx -2/3$ | $\approx \kappa^{-1/3}$ | tracks $n_\star$ | $O(1)$ | not separated | Lock removes production but K41 re-establishes itself: K41 is an attractor of a wider class than KP. Conjecture 5.2 needs another mechanism. |
| **D2′ Inert (geometrically)** | $\approx -2/3$ | $\approx \kappa^{-1/3}$ | tracks $n_\star$ | $\approx 0$ | not separated | The KP field is already nearly tangent to $\mathcal{M}$ — a surprising near-invariance worth its own investigation. |
| **D3 Marginal** | $\approx 0$ but $\Omega \sim \log(1/\alpha')$ | $\approx 1/\kappa$ | tracks $n_\star$ | $O(1)$ | either | Boundary case; $\Omega$ unbounded but sub-power-law. **Do not report as arrest.** |
| **D4 Worse** | $\lesssim -1$ | $\ge 1$ | pinned at $N-1$ | any | either | Truncation-dominated; the lock pushes the profile to growth in $n$. Re-run with larger $N$ before interpreting. |
| **D5 Closure artifact** | $\approx 0$ | any | any | $\gtrsim 0.95$ **and** $\|PF\|/\|F\| < 0.05$ | either | The constrained state barely moves. This is "the model is frozen", not arrest. |
| **D6 Bug** | — | — | — | — | — | See §6.4. |

### 6.4 Bug signatures (any one of these invalidates the run)

- **B1.** Fitted $p$ disagrees with $-1-\log_\kappa\rho_3$ from (5.3) by more than the fit standard error plus 0.05, in the regime where the profile is single-mode dominated. *(Two independent routes to the same number; disagreement is arithmetic, not physics.)*
- **B2.** Galerkin, inviscid: energy drift $\ge 10^{-6}$, or drift fails to fall by $\approx 16\times$ when cfl is halved (RK4 is 4th order).
- **B3.** $\|Pu - u\|/\|u\| > 10^{-10}$ at any sampled time (violates Lemma 4.4's Euler identity → the Jacobian or the pseudo-inverse is wrong).
- **B4.** `lock="none"` does not reproduce `simulate_shell_model` to $10^{-12}$.
- **B5.** The reconstructed profile is not annihilated by
  `symmetric_square(RecurrenceOperator.of(b, a))` in exact `Fraction` arithmetic.
- **B6.** `"sym2"` and `"order3"` agree to within $10^{-6}$ in $p$ **and** the
  `"order3"` run's coefficients never leave the Sym² locus (1.4) — that means the
  order-3 control is accidentally locked too (e.g. the $c$-parameters are frozen).
- **B7.** Exponent moves by more than 0.02 when cfl is halved or $N$ is doubled.
- **B8.** $(a,b)$ (or $(c_0,c_1,c_2)$) have zero variance over the run: the lock
  parameters are frozen and the model is a fixed linear profile. See §7.V1.

### 6.5 The separation test (the actual test of Conjecture 5.2)

Report $\Delta p := p(\texttt{sym2}) - p(\texttt{order3})$ with a propagated
standard error. **Conjecture 5.2's stated mechanism predicts $\Delta p$
significantly positive (sym2 closer to 0).** $\Delta p \approx 0$ with both
$\approx 0$ means "low-dimensional closures arrest cascades", which is a known
and uninteresting property of Galerkin truncation and **must not** be reported as
support for the conjecture.

---

## 7. Vacuity guards and mandatory controls

### Vacuity guards (checked and reported every run)

- **V1. Parameter drift.** Report $\operatorname{std}_t(a)$, $\operatorname{std}_t(b)$,
  and $\rho_3(t_{\text{end}})/\rho_3(0)$. If the lock parameters are frozen, the
  model is a fixed linear ODE on a fixed profile and "arrest" is tautological.
  A run with relative parameter drift $<10^{-6}$ is **void**.
- **V2. Live dynamics.** Report $\|PF\|/\|F\|$. Below 0.05 throughout ⇒ D5.
- **V3. Truncation.** Report the fraction of $\Omega$ in the top three shells.
  Above 0.5 ⇒ $N$ too small; re-run with $N \to 2N$ before interpreting.
- **V4. On-manifold initial data.** The seed's fit residual $\|u_0-\Phi(z_0)\|/\|u_0\|$
  must be $<10^{-10}$, else the run starts off the constraint set.
- **V5. Gauge.** Enforce $a\ge0$ (see §1); report the smallest singular value of
  $J$ and flag samples where it falls below `rcond`$\cdot\sigma_{\max}$.
- **V6. Seed coordinate degeneracy (Tier B).** The delta seed $u=e_0$ lies on
  $\mathcal{M}$ for *every* lock, but only via the degenerate corner $b=0$ (resp.
  $c=0$), where the fit does **not** determine $(a,b)$ — every $a$ fits equally
  well, and $\rho_3(0)=a^2$ is then a free parameter that directly sets the
  answer. **The delta seed is therefore not the default**; see §8.4.

### Mandatory controls (standing rule 2: "controls before contrasts")

- **C1. dt-independence.** Every reported exponent measured at cfl $=0.05$ and $0.025$; shift $<0.02$ (loop L1).
- **C2. Shell-count independence.** $N=30$ and $N=60$; shift $<0.02$ (loop L1).
- **C3. Energy oracle.** Galerkin runs: inviscid drift $<10^{-6}$ (workflow gate). Non-Galerkin: the substituted gate of §4.6.
- **C4. Same-seed control arm.** The `"none"` exponent must be re-measured **with the same seed profile** as the locked runs. Comparing a locked run to the published $-0.672$ (which used $u=e_0$) is a confound: the initial condition differs.
- **C5. Dimension-matched control.** `"order3"` run at every $\alpha'$ (§3.2, §6.5).
- **C6. Closure robustness.** The sweep repeated with `closure="collocation"`. A conclusion that flips between closures is a statement about the closure, not the lock, and must be reported as such.
- **C7. Seed sensitivity.** Sweep repeated for at least three seeds $(\lambda_0,\mu_0)$ spanning sub-critical, near-critical and super-critical $\rho_3(0)$ relative to $1/\kappa$. If $p$ depends on the seed, the model has no universal exponent and D1–D4 do not apply; report that instead.

---

## 8. Interface specification — `src/socrates/dualscale/shell_sym2.py`

Sonnet implements exactly this surface at Step 2. It must compose with the
existing validation ladder: `Sym2ShellResult` subclasses `ShellResult` so that
`summary()`, `energy_drift`, `max_enstrophy` and the existing plotting in
`scripts/visualize_results.py` keep working unchanged.

Regime declaration (standing rule 3): **algebraic identities exact
(`Fraction`), dynamics float + oracle, never silently mixed.** `certify_sym2_lock`
and `is_symmetric_square` (exact path) are the exact side; everything with a
`float` in its signature is the dynamical side.

### 8.1 Algebra (exact-friendly, no integration)

```python
def sym2_coefficients(a, b) -> tuple:
    """(c0, c1, c2) = (-b**3, b*(a*a+b), a*a+b) for u_{n+3}=c2 u_{n+2}+c1 u_{n+1}+c0 u_n.

    Accepts float or Fraction and returns the same type.
    MUST equal closed_form_symmetric_square(a, b) and
    symmetric_square(RecurrenceOperator.of(b, a)).coefficients.   [eq. 1.1]
    """

def elementary_symmetric(c0, c1, c2) -> tuple:
    """(e1, e2, e3) = (c2, -c1, c0).                                [eq. 1.2]"""

def is_symmetric_square(c0, c1, c2, *, rtol: float = 1e-12) -> bool:
    """Root-free Sym2 membership test e2**3 == e1**3 * e3.          [eq. 1.4]
    Exact when given Fractions (rtol ignored); relative comparison for floats."""

def order3_to_sym2_params(c0, c1, c2) -> tuple[float, float] | None:
    """Inverse map b = -cbrt(e3), a = +sqrt(e1 + cbrt(e3)); a >= 0 gauge.
    Returns None if e1 + cbrt(e3) < 0 (no real preimage).           [eq. 1.5]"""

def sym2_spectral_radius(a: float, b: float) -> float:
    """rho_3 = max |root| of L_3, closed form via the discriminant. [eq. 1.6]"""

def predicted_enstrophy_exponent(rho3: float, *, base: float = 2.0) -> float:
    """p = 0 if rho3 <= 1/base else -1 - log(rho3)/log(base).       [eq. 5.3]"""

def cutoff_shell(alpha_prime: float, *, base: float = 2.0) -> float:
    """n_star = -log(alpha_prime)/(2 log base).                     [eq. 5.1]"""
```

### 8.2 Reconstruction, Jacobian, projection

```python
LOCKS = ("none", "sym2", "order3", "veronese")
LOCK_DIM = {"none": None, "sym2": 5, "order3": 6, "veronese": 4}

def reconstruct_profile(z: np.ndarray, n_shells: int, *, lock: str) -> np.ndarray:
    """u = Phi(z), shape (n_shells,).                          [eqs. 3.2-3.4]
    lock="none": z is u itself (n_shells,)."""

def reconstruct_jacobian(z: np.ndarray, n_shells: int, *, lock: str) -> np.ndarray:
    """J = dPhi/dz, shape (n_shells, LOCK_DIM[lock]); identity for "none".
                                                              [eqs. 4.3-4.4]"""

def locked_rhs(
    z: np.ndarray, k: np.ndarray, viscosity: float, *,
    lock: str, closure: str = "galerkin", rcond: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float]]:
    """zdot per eq. 4.5, plus per-step diagnostics.

    Returns (zdot, diagnostics) where diagnostics carries:
      "tangency_defect"        ||(I-P)F|| / ||F||
      "live_fraction"          ||P F|| / ||F||                        (V2)
      "removed_production"     2 <K^2 u, (I-P) F>   (>0 = lock suppresses)
      "euler_residual"         ||P u - u|| / ||u||                     (B3)
      "sigma_min", "sigma_max" singular values of J                    (V5)
      "spectral_radius"        rho_3 (sym2/veronese) or max|root| (order3)

    closure="galerkin" uses the Moore-Penrose pseudo-inverse (numpy.linalg.lstsq
    with rcond). Tikhonov regularisation is FORBIDDEN: it breaks Lemma 4.4 and
    turns the energy oracle into a lie.
    """

def fit_lock_state(
    u_target: np.ndarray, *, lock: str, z0: np.ndarray | None = None,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    """Nonlinear least-squares projection onto the constraint set
    (scipy.optimize.least_squares, analytic jac=reconstruct_jacobian).
    Returns (z, relative_residual). Used for seeding and for the soft-lock probe.
    Raises if lock == "none". See V6 on the non-uniqueness at u = e_0."""
```

### 8.3 Result type

```python
@dataclass
class Sym2ShellResult(ShellResult):
    """ShellResult plus the lock diagnostics. All new fields default to empty
    arrays so the parent's field ordering is respected."""
    spectral_radius: np.ndarray = field(default_factory=lambda: np.array([]))
    lock_parameters: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tangency_defect: np.ndarray = field(default_factory=lambda: np.array([]))
    live_fraction: np.ndarray = field(default_factory=lambda: np.array([]))
    removed_production: np.ndarray = field(default_factory=lambda: np.array([]))
    euler_residual: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_min: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_shell_index: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def t_peak(self) -> float: ...
    @property
    def rho3_at_peak(self) -> float: ...
    @property
    def profile_decay_rate(self) -> float:
        """Empirical exp(d log|u_n| / dn) over 2 <= n <= n_star.        (5.3)"""
    @property
    def top_shell_enstrophy_fraction(self) -> float:                  # V3
        ...
    def summary(self) -> dict[str, object]:
        """Parent summary plus: lock, closure, rho3_at_peak,
        profile_decay_rate, predicted_exponent, peak_shell_at_peak,
        max_tangency_defect, min_live_fraction, max_euler_residual,
        param_drift, top_shell_enstrophy_fraction, void (bool, V1/V4 failed)."""
```

New `terminated` values, in addition to `shell.py`'s
(`t_max`, `dt_collapse`, `enstrophy_ceiling`, `max_steps`, `non_finite`,
`degenerate`): `"rho_ceiling"` (spectral radius exceeded `rho_ceiling`, i.e. the
profile grows in $n$ — D4), `"lock_singular"` (`sigma_min/sigma_max < rcond`
persistently), `"off_manifold"` (Euler residual exceeded $10^{-8}$).

### 8.4 Simulation and sweep

```python
def seed_profile(
    n_shells: int, *, lock: str = "sym2",
    roots: tuple[float, float] = (0.5, 0.25),
    amplitudes: tuple[float, float, float] = (1/3, 1/3, 1/3),
    normalize_energy: bool = True, base: float = 2.0,
) -> np.ndarray:
    """Default seed, shared by ALL locks so the comparison has one causal
    variable (mirrors compare_regularization's discipline).

    u_n = A l**(2n) + B (l m)**n + C m**(2n) with (l, m) = roots, renormalised
    to ||u||_2 = 1 so E(0) = 1/2, matching the control arm of FINDINGS 4.
    Defaults give rho_3(0) = 0.25 < 1/kappa = 0.5: sub-critical, so any
    enstrophy growth is produced by the dynamics, not handed to it at t=0.
    For lock="veronese" the equal-amplitude choice is replaced by the nearest
    rank-1 profile (A, B, C) = (1, 2, 1) (satisfies B**2 = 4AC, eq. 3.5).
    """

def simulate_sym2_shell_model(
    wavenumbers: np.ndarray, *,
    lock: str = "sym2",
    closure: str = "galerkin",
    t_max: float = 12.0,
    viscosity: float = 0.0,
    initial_profile: np.ndarray | None = None,   # defaults to seed_profile
    initial_state: np.ndarray | None = None,     # z directly; overrides profile
    cfl: float = 0.05,
    max_steps: int = 300_000,
    enstrophy_ceiling: float = 1e30,
    rho_ceiling: float = 1.0,
    min_dt: float = 1e-12,
    n_samples: int = 2000,
    rcond: float = 1e-10,
    base: float = 2.0,
    label: str = "",
) -> Sym2ShellResult:
    """Adaptive RK4 on z (eq. 4.5), reconstructing u = Phi(z) at each sample.

    Mirrors simulate_shell_model's contract exactly: same adaptive-step
    philosophy (eq. 4.8), same explicit termination reporting, same
    energy-conservation oracle -- which Lemma 4.4 shows is still a pure
    integrator diagnostic under closure="galerkin"."""

def compare_lock(
    n_shells: int = 30, alpha_prime: float = 1e-6, *,
    locks: tuple[str, ...] = ("none", "sym2", "order3", "veronese"),
    t_max: float = 12.0, viscosity: float = 0.0, cfl: float = 0.05,
    closure: str = "galerkin",
) -> dict[str, Sym2ShellResult]:
    """Same ladder, same seed, same integrator, same tolerances; the lock is the
    only causal variable. Direct analogue of compare_regularization()."""

@dataclass
class LockSweep:
    lock: str
    closure: str
    alphas: np.ndarray
    peak_enstrophy: np.ndarray
    exponent: float
    exponent_stderr: float
    rho3_at_peak: np.ndarray
    profile_decay_rate: np.ndarray
    predicted_exponent: np.ndarray        # eq. 5.3 applied per alpha'
    peak_shell: np.ndarray
    n_star: np.ndarray                    # eq. 5.1
    terminated: list[str]
    energy_drift: np.ndarray
    max_tangency_defect: np.ndarray
    min_live_fraction: np.ndarray
    param_drift: np.ndarray
    verdict: str                          # one of D1..D6 / "void"
    def table(self) -> list[dict[str, object]]: ...

def sweep_lock_exponent(
    alphas: np.ndarray | None = None, *,   # default np.logspace(-2, -10, 9)
    lock: str = "sym2", closure: str = "galerkin",
    n_shells: int = 30, t_max: float = 12.0, cfl: float = 0.05,
    base: float = 2.0,
) -> LockSweep:
    """The Hypothesis-U measurement, per-lock. Mirrors
    scripts/experiment_stage1.hypothesis_u_scaling; fits only over runs that
    terminated 't_max' and records which were excluded."""

def classify_sweep(sweep: LockSweep, *, arrest_tol: float = 0.05) -> str:
    """Apply the decision table of section 6.3 -> 'D1_arrest' | 'D2_inert' |
    'D2p_geometrically_inert' | 'D3_marginal' | 'D4_worse' |
    'D5_closure_artifact' | 'void'. Does NOT return 'D6_bug': bug signatures are
    raised by the gates in section 9, not classified here."""

def separation_test(
    sym2: LockSweep, order3: LockSweep,
) -> dict[str, float]:
    """Delta p = p(sym2) - p(order3) with propagated stderr, plus the fraction of
    order3 samples that violated is_symmetric_square (guards B6).  Section 6.5."""
```

### 8.5 Exact certificate (project convention)

```python
def certify_sym2_lock(
    a: Rational = Fraction(3, 4), b: Rational = Fraction(-1, 8), *,
    n_terms: int = 24,
) -> dict[str, object]:
    """Exact-Fraction certificate tying this module to operators/recurrence.py.

    Checks, all in Fraction arithmetic:
      "coefficients_match_closed_form"  sym2_coefficients == closed_form_symmetric_square
      "coefficients_match_construction" == symmetric_square(RecurrenceOperator.of(b, a))
      "profile_annihilated"             L_3.annihilates(reconstruct_profile(...))
      "sym2_locus_holds"                is_symmetric_square(...) exactly True   (1.4)
      "perturbation_detected"           is_symmetric_square fails on c1 -> c1 + 1
      "inverse_map_roundtrips"          order3_to_sym2_params recovers (|a|, b)  (1.5)
      "exact_tangency_family"           see gate T6 below
      "arithmetic": "exact (Fraction)"
    """
```

---

## 9. Gate specification (Step 2 exit criteria)

Tests go in `tests/test_dualscale.py` alongside the existing Sym² tests.

| Gate | Statement | Tier of the checked claim |
|---|---|---|
| **T1** | `lock="none"` reproduces `simulate_shell_model` to $10^{-12}$ in `times`, `energy`, `enstrophy` at matched cfl. | B |
| **T2** | `certify_sym2_lock()["certified"]` is `True` — exact `Fraction`, including that the reconstructed profile is annihilated by `symmetric_square(RecurrenceOperator.of(b, a))`. | B |
| **T3** | `reconstruct_jacobian` matches central finite differences of `reconstruct_profile` to rel. err $<10^{-6}$, for all four locks, over randomised $z$ with $\rho_3<1$. | B |
| **T4** | Euler/projector identity $\|Pu-u\|/\|u\| < 10^{-10}$ for randomised $z$, all locks, **including a rank-deficient case** ($b=0$). (Lemma 4.4.) | B |
| **T5** | Inviscid Galerkin run: `energy_drift` $<10^{-6}$; halving cfl reduces drift by $\ge 8\times$ (RK4 order-4, allowing slack). | B |
| **T6** | *Exact-tangency family.* For $\mu = \sqrt{\kappa}\,\lambda^2$ and $u_n=\lambda^{2n}$, the interior KP field $F_n = k_{n-1}u_{n-1}^2 - k_nu_nu_{n+1}$ is itself a $\mathrm{Sym}^2(V)$ sequence and is annihilated by $L_3(a,b)$ — checked in exact `Fraction` arithmetic on a rational instance. *(Derivation: $u_n = Ar^n$ gives $F_n = A^2(\kappa r^2)^n(\tfrac{1}{\kappa r^2} - r)$, a geometric sequence of ratio $\kappa r^2$; with $r=\lambda^2$ and $\mu^2 = \kappa\lambda^4$ that ratio is $\mu^2 \in \operatorname{spec}L_3$.)* This is a non-trivial analytic witness that the Jacobian and the projector are right. | B |
| **T7** | Viscous run: $dE/dt$ matches $-\nu\Omega$ to $10^{-6}$ relative (Lemma 4.4 corollary 2). | B |
| **T8** | `predicted_enstrophy_exponent(2**(-1/3)) == -2/3` and `predicted_enstrophy_exponent(0.5) == 0.0` (base 2). | B |
| **T9** | `is_symmetric_square` is `True` on `sym2_coefficients(a,b)` and `False` on any single-coefficient perturbation, over a Hypothesis-generated rational grid. | B |
| **T10** | `order3_to_sym2_params(*sym2_coefficients(a,b))` round-trips to $(|a|,b)$ for $a\ge0$; returns `None` when $e_1+e_3^{1/3}<0$. | B |
| **T11** | Non-Galerkin closures: energy drift is dt-*independent* to within 20 % across a cfl halving (§4.6 substituted gate). | C (convention) — the gate itself is the check |
| **T12** | `seed_profile` output satisfies $\|u-\Phi(\texttt{fit\_lock\_state}(u))\|/\|u\| < 10^{-10}$ for every lock (V4). | B |

Workflow-level gates carried from `IMPLEMENTATION_PLAN.md` W1 Step 2–3: every
swept run terminates `t_max`; energy drift $<10^{-6}$ each; suite green.

---

## 10. Tier ledger

| # | Statement | Tier |
|---|---|---|
| 1.1–1.3 | Sym² closed form, elementary symmetric coordinates, solution basis | **B** (certified in `recurrence.py`) |
| 1.4 | Prop. A: Sym² locus $\iff e_2^3=e_1^3e_3$ $\iff$ eigenvalues in geometric progression | **B** (proved above; gate T9) |
| 1.5 | Real inverse map $(e_i)\mapsto(a,b)$ and its existence condition; $a\ge0$ gauge | **B** (gate T10) |
| 1.6 | Closed form for $\rho_3$ | **B** |
| 2.1–2.2 | $\mathrm{Sym}^2$ not multiplicatively closed; the spec's inference does not follow | **B** |
| 2.3 | Graded $\mathrm{Sym}^{2d}$ family as the closed object | **C** (exploratory) |
| 3.1–3.5 | Half-ladder; linear vs Veronese lock; $B^2=4AC$ | **B** (definitions + algebra) |
| §3.2 | The four-lock ladder as the experimental design | **C** (design choice) |
| 4.1–4.2 | KP field and its exact energy identity | **B** (restated from `shell.py`) |
| 4.3–4.4 | Reconstruction Jacobian and its recurrences | **B** (gate T3) |
| 4.5 | Galerkin projection **as the model** | **C** (closure choice) |
| 4.5 | The identity $\dot u = PF$ given (4.5) | **B** |
| 4.6 / Lemma 4.4 | Energy budget preserved exactly; $W=I$ is the unique such weight; Tikhonov forbidden | **B** (gates T4, T5, T7) |
| 4.7 | Half-ladder RHS (4.7) as an exact substitution; and its overdetermination | **B** |
| 4.8 | Timestep rule | **C** (numerical convention; validated by C1 only) |
| §4.6 alt. closures | Collocation / enstrophy-metric variants and their substituted gate | **C** |
| 5.1–5.2 | Cutoff shell; master exponent formula $p=-1-\log_\kappa\rho_3$; marginal log case | **B**, *conditional on* a single dominant geometric mode (**C** assumption, tested by §5.3) |
| §5.3 | Identification of the empirical decay rate with $\rho_3$ | **C** |
| §6 | Operational definitions A1–A4, decision table D1–D6, bug signatures B1–B8 | **C** (protocol) — but each individual gate is decidable |
| §6.5 | $\Delta p$ separation test as *the* test of Conjecture 5.2 | **C** (this is the contract's central interpretive claim) |
| §7 | Vacuity guards V1–V6, controls C1–C7 | **C** (methodology), except V6 which is **B** |

### Standing caution for Step 4 (adjudication)

A positive result (D1) from this contract licenses exactly one sentence:
*"under the energy-metric Galerkin closure, the five-parameter Sym²-constrained
dyadic model has bounded peak enstrophy as $\alpha'\to0$, whereas the
six-parameter unrestricted order-3 closure does not."* It does **not** license
"the Sym² lock prevents blow-up in Navier–Stokes", nor any claim about
$w_{1+\infty}$, celestial holography, or the spec's asserted mechanism — those
remain Tier C exactly as `REVIEW_haloalg.md` records. A negative result (D2) is
equally publishable and is the outcome the control arm's clean K41 exponent
should lead us to expect by default.
