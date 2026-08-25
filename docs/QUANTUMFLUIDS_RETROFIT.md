# Incoming retrofit from SocrateAI-Scientific-QuantumFluids

**Status: INCOMING CROSS-STREAM MATERIAL — nothing here is a claim of THIS repo until it
passes this repo's own adversarial verification.** Source stream:
`~/xdev/SocrateAI-Scientific-QuantumFluids` (github.com/xaviercallens/SocrateAI-Scientific-QuantumFluids).
Tier labels are the SOURCE stream's; this repo re-tiers on its own audit.

Date: 2026-08. Companion Lean file: `lean/QuantumFluidsShell.lean` — eleven theorems,
**built against THIS repo's pin (v4.32.2)**, all `[propext, Classical.choice, Quot.sound]`,
registered in `lean/AxiomCheck.lean`.

> **Read §R0 first.** The first draft of this retrofit led with a prediction that your
> §4 headline exponent was a horizon artifact. That prediction was **tested in your own
> code and refuted**. It is retracted below, and what the test found instead is more
> useful to you than the prediction would have been.

---

## R0. RETRACTED: "your −0.672 is a fixed-horizon transient"

**The claim (withdrawn).** That at ν = 0 your truncated model must relax toward absolute
equilibrium, so re-running §4 at longer horizons would drive β from −0.672 toward −1.

**The test.** Your §4 protocol re-run verbatim (N=30, ν=0, cfl=0.05, 9 α′ over eight
decades) at t_max ∈ {6, 12, 24, 48}, plus a 4-seed control at fixed t_max = 12.
Script: source stream `exploration/socrates_horizon_test.py`, output `.out`.

| t_max | 6 | 12 | 24 | 48 |
|---|---|---|---|---|
| β | −0.6721 | −0.6721 | −0.6721 | −0.6721 |

Horizon drift over an 8× range: **−0.0001**. Seed spread: **0.0006**
(−0.6721/−0.6720/−0.6725/−0.6719). Your published −0.672 reproduces exactly.

**Verdict: the prediction is refuted. β is horizon-stable and seed-stable.** Your §4
number is robust on both axes that were challenged. The retraction is recorded in the
source stream's LEDGER.

**Why the transfer was invalid — and this is the part worth keeping.** The
absolute-equilibrium argument requires a **volume-preserving (Liouville)** flow. The
source stream established that the real Katz–Pavlović flow has phase-space divergence
`−Σₙkₙaₙ₊₁ ≠ 0` — it is volume-**contracting**, so it may carry attracting structure and
is under no obligation to reach equipartition. Only the *complexified* model (§R3) is
Liouville, and that is where β → −1 was measured. Likewise, the source stream's
CV 23–49% scatter came from randomising initial **phases**; your state vector is real by
construction and **has no phase degree of freedom to randomise**, which is exactly why
your seed spread is 0.0006 and not 30%.

**The transferable lesson is therefore conditional, and this is the corrected form of
what §R2 below used to assert:** thermalization and phase-chaos results apply to models
with phase freedom and Liouville structure. Your real-amplitude cascade has neither. The
real/complex distinction is not cosmetic — it is the difference between β stable at
−2/3 and β drifting to −1.

## R1. A real defect in your sup-enstrophy instrument (found by the refuted test)

`ShellResult.max_enstrophy` is `np.max(self.enstrophy)` — a maximum over **recorded
samples** — while `sample_times = np.linspace(0.0, t_max, n_samples)` with `n_samples`
fixed at 2000. **The sampling interval is `t_max/2000`, so it coarsens in proportion to
the horizon.** Measured consequence at α′ = 10⁻⁶, everything else fixed:

| t_max | 12 | 48 | 200 | 800 |
|---|---|---|---|---|
| reported `max_enstrophy` | 11425.5 | 11425.5 | 10950.9 | 9885.8 |

A supremum over a **nested, growing** window cannot decrease. The −13.5% is the
instrument, not the dynamics: the sampled grid steps over a peak it used to resolve.
(Energy drift is 1.34×10⁻⁷ in all four — the integration is fine; only the readout is.)

**Fix is one line's worth of state:** the per-step `current_enstrophy` needed for a true
running maximum is *already computed* at `shell.py:183` for the `enstrophy_ceiling` test
and simply discarded. Track its max alongside.

**Does this bias the published §4 number? No — tested.** Varying only your public
`n_samples` parameter at your exact protocol (2 000 vs 200 000, all nine α′): worst peak
miss **0.051%**, and **β = −0.6721 either way (Δβ = 0.0000)**. At t_max = 12 the enstrophy
peak is broad enough that 2000 samples resolve it. **§4 needs no re-measurement.**
Script/output: source stream `exploration/socrates_sampled_max_defect.py(.out)`.

So this is a *latent* fragility, not a live error: the sampling interval is tied to the
horizon rather than to the feature being measured, so it bites whenever t_max grows or the
peak sharpens — e.g. any future Sym²-locked run whose peak is narrower, or any run that
extends the horizon to chase convergence. Worth fixing before it is load-bearing rather
than after.

**Related, same class:** MechanicaFluidorum's `data/dyadic_omega_sup.csv` mixes
sum-enstrophy (INFEASIBLE rows) and max-enstrophy (OK rows) in one column — full report
at QuantumFluids `docs/DEFECT_REPORT_MF_ENSTROPHY.md`. If this repo ingests that CSV it
inherits the defect.

## R1b. A point in your design's favour, found while testing R0

The same test was then run against MechanicaFluidorum's dyadic model (raw `kₙ = 2ⁿ`, no
dual cap), and there the degeneracy **is** real: `sup_t Ω` climbs to 51.7% → 95.0% →
99.6% → **99.90%** of the trivial ceiling `k_N²E` at T = 2, 8, 32, 64, energy drift
≤ 2.7×10⁻¹³. Their ν = 0 control really does end up measuring the cutoff.

**Your model does not do this, and the reason is your regularization.** `k_eff =
min(k, 1/(α′k))` makes trans-cutoff shells *soft* — at α′ = 10⁻⁶, N = 30 the top shell
carries k_eff = 0.0019 against a peak of 976.6 at shell 10 — so the highest shells cannot
dominate the enstrophy, and there is no ceiling for `sup_t Ω` to degenerate onto. The
T-dual cap **protects against a degeneracy that plain dyadic truncation suffers.**

That is a positive structural property of your Stage-1 design, and it was found by trying
to refute your headline number rather than by looking for something nice to say. It is
also the concrete reason §4's exponent is a dynamical measurement where the same protocol
on an uncapped model would return the truncation.

## R2. Ensemble discipline — scoped correctly this time

The source stream measured, at fixed parameters and identical energy, varying only
initial phases: CV **23–49%** (thermalization time), **25–84%** (time-averaged
enstrophy — time-averaging did *not* self-average it). Four measurement rounds were
retracted on this basis. Their six-criterion validation battery missed it because every
criterion tested *deterministic* reproducibility (dt-refinement, sampling, neighbouring
parameter) and none tested *statistical* reproducibility across realisations.

**Scope, per R0:** this applies to models with phase freedom. It does **not** impugn your
§4 measurement, which was checked here and is seed-stable — nor MechanicaFluidorum's runs,
which were tested for the same reason and came back at CV 0.15%. It *does* apply to any future
work in this repo that adopts complex amplitudes, and the battery-design lesson — that
a dt-refinement passing at 0.00% is silent about ensemble scatter, because it is the
wrong limit — is method-level and transfers regardless.

Your §9–10 seed-dependence of the timestep estimator (2/13, then 6/26 seeds) is a
*different* phenomenon from phase chaos, and the R0 test supports that reading: β itself
is seed-stable even where the estimator is not.

## R3. The seam theorem — what an energy-conserving T-dual "bounce" can be (Tier A)

Kernel-checked both directions (`seam_conserves_iff`, in the ported file): in the
complexified model with k_N ≠ 0, a boundary value at shell N+1 conserves the energy
pairing **iff** `Re(conj(v_N)²·v_{N+1}) = 0`. Corollaries also kernel-checked: truncation
conserves; the family `v_{N+1} = iμv_N²` conserves. Numerically, every seam that reads a
*neighbouring* shell — the geometric mirror `±v_{N−1}`, its conjugate, its i-rotation —
leaks at |dE/dt| ~ 10²–10³.

**Consequence for the programme's bounce principle: an energy-conserving T-dual bounce
cannot be a spatial reflection about the self-dual scale; it must be local phase rotation
at the cutoff (GPE-like self-phase-modulation).** `Reff_bounce` in
`CallensDualScale.lean` — the scalar `max(R, α/R)` geometry — is untouched; this
constrains its *dynamical realization* in a cascade.

Note your `effective_wavenumber` already encodes the spatial reading:
`k_eff = min(k, 1/(α′k))` makes trans-cutoff modes *soft* (at α′ = 10⁻⁶, N = 30 the top
shell has k_eff = 0.0019 versus peak 976.6 at shell 10). That is a legitimate design and
is not what R3 forbids — R3 is about a *seam that hands amplitude back*, not about a
non-monotonic k_eff profile.

## R4. The Liouville dichotomy (Tier A blocks + numerics)

Real Katz–Pavlović: divergence `−Σₙkₙaₙ₊₁ ≠ 0`, volume-contracting. The conjugated
complexification `Bₙ = kₙ₋₁vₙ₋₁² − kₙ·conj(vₙ)·vₙ₊₁`: divergence-free, conserves
`½Σ|vₙ|²` exactly, reduces **exactly** to the real model on real data (the reals are an
invariant subspace, so Katz–Pavlović blow-up solutions remain solutions), and reproduces
MechanicaFluidorum's `dyadic_cascade.py` bit-for-bit (0.00e+00, 9 configs, 2.4×10⁶ RK4
steps).

Per R0 this dichotomy is now *load-bearing*, not decorative: it is the reason the
thermalization transfer failed. If Sym²-lock work wants a setting where equilibrium
statistical mechanics legitimately applies, the complexified model is that setting — and
if it wants attracting structure, the real model is.

**Open, flagged not claimed:** whether the real model's volume contraction is
mechanistically tied to self-similar blow-up attraction (Katz–Pavlović). The R0 result —
a genuinely non-thermalizing conservative-energy cascade — makes this sharper, not
weaker.

## R5. Empirical constraint on any coupling law

From Godfrin et al. PRB 103:104516's author-published all-pressure table (7 pressures,
per-point errors; deterministic weighted fits, P = 0 cross-checking an independent fit to
0.05%): the roton parameters are **linear in P at low pressure** (Δ: −0.67%/bar;
Q_m: +0.475%/bar; constant across three independent points) and **not power laws in P**
(windowed slopes disagree, r² ≈ 0.75). Any dual-scale coupling-law exponent tested
against these parameters must be stated in the physical variable (density — requires an
EOS, not in the file) and cannot be a pure power law in pressure.

## R6. Housekeeping

- Your §4 has β = −2/3 as **measured**; MechanicaFluidorum's `OP2_LITE_CANDIDATES.md`
  cites it as a **pre-registered threshold**. One line in the authoritative doc would stop
  the drift — the source stream mis-cited it once already, and R0's test confirms yours is
  a measurement.
- Your §4 remark that Theorem 4.2's bound `Ω ≤ 2E/α′` "is not tight": R0 confirms it is
  genuinely not tight *here* — the earlier draft's claim that it becomes tight on a
  thermalized state does not apply, because your model does not thermalize.

---

**Reproducibility:** source stream `scripts/verify.sh` (143 tests + Lean gate with axiom
audit), `paper/quantumfluids_tdual.pdf`, `LEDGER.md`. The source stream's own measurement
programme failed and says so; R0 above is the same discipline applied to its own
cross-stream prediction.
