"""Round-3 independent skeptic verification of the N8d graph-level consensus rule.

Everything here is built from scratch by this script:

  * my own kick-drift-kick leapfrog and my own RK4 (no `socrates.solvers` import),
  * my own AGM-based complete elliptic integral K(m) (no `scipy.special.ellipk`),
  * my own Weyl / bit-reversal index sequences,
  * my own numpy re-derivation of dimension / R^2 / shell CV / slope bound,
  * my own re-implementation of the three acceptance policies (pre-N8, the
    round-2 node-local gate, and the shipped consensus rule) and of the
    "stable convergence" minimum-n search.

No module under scripts/hypergraph_benchmark/round3/ written by the repair
agent is imported. The production library is used only where the point of the
check is what production code does (`local_dimension`, `mean_dimension`,
`near_constant_consensus`, `poly_algebraic_minimum_points`), and every such
number is cross-checked against the re-derivation above.

Usage: python n8e_round3_skeptic.py [anchor|prob06|prob05|prob02|prob10|prob03|all]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.comparison import poly_algebraic_minimum_points  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    local_dimension,
    mean_dimension,
    near_constant_consensus,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

# --------------------------------------------------------------------------
# My own integrators.
# --------------------------------------------------------------------------


def my_leapfrog(force, q0, v0, *, dt, n_steps):
    """Kick-drift-kick leapfrog, written here rather than imported."""
    q = np.asarray(q0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    qs = np.empty((n_steps + 1, q.size))
    vs = np.empty_like(qs)
    qs[0], vs[0] = q, v
    a = force(q)
    for i in range(1, n_steps + 1):
        v_half = v + 0.5 * dt * a
        q = q + dt * v_half
        a = force(q)
        v = v_half + 0.5 * dt * a
        qs[i], vs[i] = q, v
    return np.arange(n_steps + 1) * dt, qs, vs


def my_rk4_step(deriv, state, t, dt):
    s = np.asarray(state, dtype=float)
    k1 = np.asarray(deriv(s, t))
    k2 = np.asarray(deriv(s + 0.5 * dt * k1, t + 0.5 * dt))
    k3 = np.asarray(deriv(s + 0.5 * dt * k2, t + 0.5 * dt))
    k4 = np.asarray(deriv(s + dt * k3, t + dt))
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def my_ellipk(m: float) -> float:
    """Complete elliptic integral of the first kind, via the AGM.

    K(m) = pi / (2 * AGM(1, sqrt(1 - m))). Independent of scipy.special.
    """
    a, b = 1.0, math.sqrt(1.0 - m)
    for _ in range(60):
        a, b = 0.5 * (a + b), math.sqrt(a * b)
        if abs(a - b) < 1e-16 * a:
            break
    return math.pi / (2.0 * a)


def my_weyl_indices(n: int, n_steps_total: int) -> np.ndarray:
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    fracs = (np.arange(n) * phi) % 1.0
    return np.clip(np.floor(fracs * n_steps_total).astype(int), 0, n_steps_total)


def my_bitrev(m_bits: int) -> list[int]:
    size = 1 << m_bits
    out = []
    for i in range(size):
        r = 0
        for b in range(m_bits):
            if i >> b & 1:
                r |= 1 << (m_bits - 1 - b)
        out.append(r)
    return out


# --------------------------------------------------------------------------
# My own re-derivation of every quantity the gate uses.
# --------------------------------------------------------------------------


def my_fit(shells):
    """(dimension, r_squared, shell_cv, slope_bound) from a shell sequence."""
    s = np.asarray(shells, dtype=float)
    r = np.arange(1.0, len(s) + 1.0)
    lx, ly = np.log(r), np.log(s)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 if ss_tot <= 1e-24 * max(1.0, float(np.sum(ly**2))) else 1.0 - ss_res / ss_tot
    cv = float(np.std(s) / np.mean(s))
    band = float(ly.max() - ly.min())
    var_x = float(np.sum((lx - lx.mean()) ** 2))
    bound = band * float(np.sum(np.abs(lx - lx.mean()))) / (2.0 * var_x)
    return float(slope) + 1.0, r2, cv, bound


MIN_FIT_LENGTH = 5
CV_MAX = 0.15
SLOPE_BOUND_MAX = 0.25
CONSENSUS_TOL = 0.25
CONSENSUS_FRAC = 0.5


def my_near_degenerate(shells):
    s = np.asarray(shells, dtype=float)
    if len(s) < 2:
        return False
    _dim, _r2, cv, bound = my_fit(s)
    ly = np.log(s)
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    degenerate = ss_tot <= 1e-24 * max(1.0, float(np.sum(ly**2)))
    return (
        (not degenerate)
        and len(s) >= MIN_FIT_LENGTH
        and cv <= CV_MAX
        and bound <= SLOPE_BOUND_MAX
    )


def my_consensus(ests, threshold=0.9):
    """Independent re-implementation of the shipped graph-level rule."""
    if not ests:
        return False
    conf = [e for e in ests if e.r_squared >= threshold]
    if conf:
        return abs(sum(e.dimension for e in conf) / len(conf) - 1.0) <= CONSENSUS_TOL
    nd = sum(1 for e in ests if e.near_degenerate)
    return nd / len(ests) >= CONSENSUS_FRAC


def policy_means(points, k, max_radius, samples=40):
    """(pre_n8, node_local_round2, shipped_consensus) sampled means + diagnostics.

    Recomputed from `local_dimension` outputs by hand -- no call to
    `mean_dimension` -- so the three policies are compared on identical nodes.
    """
    hg = knn_hypergraph(points, k=k, dedupe=True)
    nodes = sorted(hg.nodes)
    nsamp = min(len(points), samples)
    if nsamp < len(nodes):
        step = max(1, len(nodes) // nsamp)
        nodes = nodes[::step][:nsamp]
    ests = [local_dimension(hg, n, max_radius=max_radius) for n in nodes]

    pre = [e.dimension for e in ests if e.r_squared >= 0.9]
    nodelocal = [e.dimension for e in ests if e.r_squared >= 0.9 or e.near_degenerate]
    cons = my_consensus(ests)
    shipped = nodelocal if cons else pre

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "estimates": ests,
        "n_sampled": len(ests),
        "pre_n8": mean(pre),
        "node_local": mean(nodelocal),
        "shipped": mean(shipped),
        "consensus": cons,
        "n_confident": len(pre),
        "near_deg_frac": sum(1 for e in ests if e.near_degenerate) / len(ests),
        "n_kept_pre": len(pre),
        "n_kept_nodelocal": len(nodelocal),
    }


def my_min_n(points, true_dim, tol, *, k, n_grid, max_radius, policy):
    """My own 'first n that is within tolerance and STAYS within it' search."""
    rows = []
    for n in n_grid:
        if n > len(points) or k >= n:
            break
        m = policy_means(points[:n], k, max_radius)
        rows.append((n, m[policy], m))
    for i, (_n, dim, _m) in enumerate(rows):
        if not math.isfinite(dim) or abs(dim - true_dim) > tol:
            continue
        if all(math.isfinite(d) and abs(d - true_dim) <= tol for _, d, _ in rows[i:]):
            return rows[i][0], rows
    return None, rows


# --------------------------------------------------------------------------
# Point clouds -- each built here from its own physics.
# --------------------------------------------------------------------------


def cloud_prob01(n_max=3200):
    """Harmonic oscillator, one period, golden-ratio Weyl subsample."""
    dt = 2e-4
    n_steps = int(round(2 * math.pi / dt))
    t, q, v = my_leapfrog(lambda x: -x, [1.0], [0.0], dt=dt, n_steps=n_steps)
    err_x = float(np.max(np.abs(q[:, 0] - np.cos(t))))
    err_v = float(np.max(np.abs(v[:, 0] + np.sin(t))))
    idx = my_weyl_indices(n_max, len(t) - 1)
    assert len(np.unique(idx)) == n_max
    pts = list(zip(q[idx, 0].tolist(), v[idx, 0].tolist(), strict=True))
    return pts, {"max_err_x": err_x, "max_err_v": err_v}


def cloud_prob02(m=4096):
    """Nonlinear pendulum theta0=2.0, exactly one period, bit-reversal order."""
    theta0 = 2.0
    period = 4.0 * my_ellipk(math.sin(theta0 / 2.0) ** 2)
    dt = period / m
    t, q, v = my_leapfrog(lambda x: -np.sin(x), [theta0], [0.0], dt=dt, n_steps=m)
    closure = math.hypot(q[m, 0] - theta0, v[m, 0])
    energy = 0.5 * v[:, 0] ** 2 + (1.0 - np.cos(q[:, 0]))
    drift = float(np.max(np.abs(energy - energy[0])))
    time_ordered = [(float(q[i, 0]), float(v[i, 0])) for i in range(m)]
    perm = my_bitrev(m.bit_length() - 1)
    pts = [time_ordered[perm[i]] for i in range(m)]
    return pts, {"period": period, "closure": closure, "energy_drift": drift}


def cloud_prob05(n_max=12800):
    """Quasiperiodic 2-torus, closed form, constant time density."""
    density = 500 / 1800.0
    t_max = n_max / density
    w2 = math.sqrt(2.0)
    return [
        (math.cos(t_max * i / n_max), math.cos(w2 * t_max * i / n_max)) for i in range(n_max)
    ], {"t_max": t_max}


def cloud_prob06():
    """Planar Brownian motion, seed 42, 80k unit-variance increments, stride 3."""
    rng = np.random.default_rng(42)
    inc = rng.normal(loc=0.0, scale=1.0, size=(80_000, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(inc, axis=0)], axis=0)
    # MSD scaling check (my own).
    lags = np.arange(10, 501, 10)
    msd = np.array([np.mean(np.sum((path[lag:] - path[:-lag]) ** 2, axis=1)) for lag in lags])
    slope = float(np.polyfit(np.log(lags.astype(float)), np.log(msd), 1)[0])
    stride = max(1, len(path) // 25_600)
    pts = [tuple(p) for p in path[::stride]]
    return pts, {"msd_slope": slope, "stride": stride, "n_points": len(pts)}


def cloud_prob10(m=4096):
    """Driven damped pendulum, mode-locked period-1, one drive period, bitrev."""
    gamma, f_drive, om = 0.5, 0.5, 0.6
    t_drive = 2.0 * math.pi / om

    def deriv(s, t):
        return np.array([s[1], -gamma * s[1] - math.sin(s[0]) + f_drive * math.cos(om * t)])

    steps = 500
    dt_v = t_drive / steps
    s = np.array([0.2, 0.0])
    t = 0.0
    for _ in range(200 * steps):
        s = my_rk4_step(deriv, s, t, dt_v)
        t += dt_v
    strobe = []
    for _ in range(50):
        for _ in range(steps):
            s = my_rk4_step(deriv, s, t, dt_v)
            t += dt_v
        strobe.append((((s[0] + math.pi) % (2 * math.pi)) - math.pi, s[1]))
    tail = np.array(strobe[len(strobe) // 2 :])
    spread = (float(np.ptp(tail[:, 0])), float(np.ptp(tail[:, 1])))

    dt = t_drive / m
    s0 = s.copy()
    time_ordered = []
    for _ in range(m):
        s = my_rk4_step(deriv, s, t, dt)
        t += dt
        time_ordered.append((((s[0] + math.pi) % (2 * math.pi)) - math.pi, float(s[1])))
    closure = math.hypot(s[0] - s0[0], s[1] - s0[1])
    perm = my_bitrev(m.bit_length() - 1)
    pts = [time_ordered[perm[i]] for i in range(m)]
    return pts, {"strobe_spread": spread, "closure": closure}


def cloud_kepler(n_max=6400):
    """Kepler ellipse e=0.6, my own leapfrog-equivalent RK4 in (x,y,vx,vy),
    sampled uniformly in TIME over an integer number of periods."""
    ecc = 0.6
    a = 1.0
    mu = 1.0
    r0 = a * (1 - ecc)
    v0 = math.sqrt(mu * (1 + ecc) / (a * (1 - ecc)))
    period = 2 * math.pi * math.sqrt(a**3 / mu)

    def deriv(s, _t):
        x, y, vx, vy = s
        r3 = (x * x + y * y) ** 1.5
        return np.array([vx, vy, -mu * x / r3, -mu * y / r3])

    n_steps = 200_000
    dt = period / (n_steps / 1.0)
    s = np.array([r0, 0.0, 0.0, v0])
    t = 0.0
    pos = np.empty((n_steps + 1, 2))
    pos[0] = s[:2]
    for i in range(1, n_steps + 1):
        s = my_rk4_step(deriv, s, t, dt)
        t += dt
        pos[i] = s[:2]
    closure = float(np.hypot(*(pos[-1] - pos[0])))
    idx = np.linspace(0, n_steps, num=n_max, endpoint=False).astype(int)
    return [(float(pos[i, 0]), float(pos[i, 1])) for i in idx], {
        "period": period,
        "closure": closure,
    }


# --------------------------------------------------------------------------
# Checks.
# --------------------------------------------------------------------------


def cross_check_estimates(ests, label):
    """Every production estimate re-derived from its own volume sequence."""
    worst = 0.0
    for e in ests:
        vols = list(e.volumes)
        shells = [vols[0] - 1] + [vols[i] - vols[i - 1] for i in range(1, len(vols))]
        fit = []
        for s in shells:
            if s <= 0:
                break
            fit.append(float(s))
        if len(fit) < 2:
            continue
        dim, r2, cv, bound = my_fit(fit)
        worst = max(
            worst,
            abs(dim - e.dimension),
            abs(r2 - e.r_squared),
            abs(cv - e.shell_cv),
            abs(bound - e.slope_bound),
        )
        assert my_near_degenerate(fit) == e.near_degenerate, (label, fit, e)
    print(f"  [cross-check] {label}: max |mine - production| over dim/R^2/CV/bound = {worst:.3e}")
    assert worst < 1e-9, worst


def check_anchor():
    print("=== [1] ANCHOR: problem 01 cloud, n=200, k=6, max_radius=6 ===")
    pts, info = cloud_prob01()
    print(f"  my leapfrog vs exact: max|x-cos t|={info['max_err_x']:.3e} "
          f"max|v+sin t|={info['max_err_v']:.3e}")
    sub = pts[:200]
    hg = knn_hypergraph(sub, k=6, dedupe=True)
    nodes = sorted(hg.nodes)
    ests = [local_dimension(hg, n, max_radius=6) for n in nodes]
    cross_check_estimates(ests, "anchor")

    target = None
    for e in ests:
        vols = list(e.volumes)
        shells = tuple([vols[0] - 1] + [vols[i] - vols[i - 1] for i in range(1, len(vols))])
        if shells == (6, 5, 5, 6, 6, 6):
            target = (e, shells)
            break
    assert target is not None, "documented anchor shell sequence NOT FOUND"
    e, shells = target
    print(f"  anchor node volumes={e.volumes} shells={shells}")
    print(f"    dim={e.dimension:.10f} R^2={e.r_squared:.10f} cv={e.shell_cv:.6f} "
          f"bound={e.slope_bound:.6f} near_degenerate={e.near_degenerate} deg={e.degenerate}")

    prod_cons = near_constant_consensus(ests, threshold=0.9)
    mine_cons = my_consensus(ests)
    nconf = sum(1 for x in ests if x.r_squared >= 0.9)
    frac = sum(1 for x in ests if x.near_degenerate) / len(ests)
    print(f"  graph: n_confident={nconf}  near_degenerate_fraction={frac:.4f}")
    print(f"  consensus: production={prod_cons}  my re-implementation={mine_cons}")
    assert prod_cons == mine_cons is True

    kept = [x.dimension for x in ests if x.is_well_fit(0.9, near_constant_consensus=prod_cons)]
    kept_r2only = [x.dimension for x in ests if x.r_squared >= 0.9]
    md = mean_dimension(hg, samples=200, max_radius=6)
    print(f"  kept by gate {len(kept)}/{len(ests)}   "
          f"kept by R^2 alone {len(kept_r2only)}/{len(ests)}")
    print(f"  mean_dimension (production) = {md:.6f}   my recompute = {sum(kept)/len(kept):.6f}")
    worst_kept = max(abs(d - 1.0) for d in kept)
    print(f"  worst |dim - 1| among kept  = {worst_kept:.6f} (tolerance +/-0.15)")
    assert e.near_degenerate and not e.degenerate
    assert e.r_squared < 0.9  # kept BY the gate, not by R^2
    assert len(kept) == 196 and abs(md - 0.994266) < 1e-5
    print("  -> ANCHOR PRESERVED (196/200, mean 0.994266), kept by the branch, not by R^2.\n")
    return True


def check_prob06():
    print("=== [2] PROBLEM 06 CULPRIT: Brownian, my own cloud, n=400, k=10, R=6 ===")
    pts, info = cloud_prob06()
    print(f"  my cloud: msd log-log slope={info['msd_slope']:.4f} (Brownian => ~1), "
          f"stride={info['stride']}, n_points={info['n_points']}")
    sub = pts[:400]
    hg = knn_hypergraph(sub, k=10, dedupe=True)
    m = policy_means(sub, 10, 6)
    ests = m["estimates"]
    cross_check_estimates(ests, "prob06 n=400")

    culprit = None
    for e in ests:
        vols = list(e.volumes)
        shells = tuple([vols[0] - 1] + [vols[i] - vols[i - 1] for i in range(1, len(vols))])
        if shells == (13, 17, 13, 13, 17, 15):
            culprit = (e, shells)
    assert culprit is not None, "round-2 culprit shell sequence NOT FOUND on my cloud"
    e, shells = culprit
    print(f"  culprit volumes={e.volumes} shells={shells}")
    print(f"    dim={e.dimension:.4f} R^2={e.r_squared:.4f} cv={e.shell_cv:.4f} "
          f"bound={e.slope_bound:.4f} near_degenerate={e.near_degenerate}  "
          f"|dim-2|={abs(e.dimension - 2):.4f}")
    assert e.near_degenerate, "culprit is no longer even a node-local candidate?"

    prod_cons = near_constant_consensus(ests, threshold=0.9)
    print(f"  graph: n_confident={m['n_confident']} near_deg_frac={m['near_deg_frac']:.4f}")
    conf = [x.dimension for x in ests if x.r_squared >= 0.9]
    conf_mean = sum(conf) / len(conf)
    print(f"  confident mean = {conf_mean:.4f}  |mean-1| = {abs(conf_mean - 1):.4f} "
          f"(tolerance {CONSENSUS_TOL})")
    print(f"  consensus: production={prod_cons}  mine={m['consensus']}")
    assert prod_cons is False and m["consensus"] is False

    md = mean_dimension(hg, samples=40, max_radius=6)
    print(f"  means at n=400:  pre-N8={m['pre_n8']:.4f}  round2-node-local={m['node_local']:.4f}  "
          f"shipped={m['shipped']:.4f}  production mean_dimension={md:.4f}")
    assert abs(md - m["shipped"]) < 1e-12
    assert abs(md - m["pre_n8"]) < 1e-12
    print(f"  |shipped - 2.0| = {abs(md-2.0):.4f} vs tolerance 0.3 -> "
          f"{'IN' if abs(md-2.0)<=0.3 else 'OUT'}")

    n_grid = (100, 200, 400, 800, 1600, 3200, 6400, 12_800, 25_600)
    prod_min = poly_algebraic_minimum_points(pts, 2.0, 0.3, k=10, n_grid=n_grid, max_radius=6)
    mine_pre, rows_pre = my_min_n(pts, 2.0, 0.3, k=10, n_grid=n_grid[:6], max_radius=6,
                                  policy="pre_n8")
    mine_nl, _ = my_min_n(pts, 2.0, 0.3, k=10, n_grid=n_grid[:6], max_radius=6,
                          policy="node_local")
    mine_sh, _ = my_min_n(pts, 2.0, 0.3, k=10, n_grid=n_grid[:6], max_radius=6,
                          policy="shipped")
    print("  per-n (n<=3200): " + " ".join(f"{n}:{d:.4f}" for n, d, _ in rows_pre))
    print(f"  poly_algebraic_min_n: production(shipped tree)={prod_min}")
    print(f"    my re-implementation over n<=3200: pre-N8={mine_pre} "
          f"node-local={mine_nl} shipped={mine_sh}")
    print("  docs/MENSURA_BENCHMARK.md records 400 for problem 06.")
    assert prod_min == 400 and mine_pre == 400 and mine_sh == 400 and mine_nl == 800
    print("  -> CULPRIT REJECTED by the graph verdict; recorded 400 restored.\n")
    return True


def check_prob05():
    print("=== [3] PROBLEM 05: quasiperiodic torus, my own closed-form cloud ===")
    pts, info = cloud_prob05()
    n_grid = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
    prod_min = poly_algebraic_minimum_points(pts, 2.0, 0.3, k=10, n_grid=n_grid, max_radius=6)
    mine_pre, rows = my_min_n(pts, 2.0, 0.3, k=10, n_grid=n_grid[:6], max_radius=6, policy="pre_n8")
    mine_nl, _ = my_min_n(pts, 2.0, 0.3, k=10, n_grid=n_grid[:6], max_radius=6, policy="node_local")
    mine_sh, _ = my_min_n(pts, 2.0, 0.3, k=10, n_grid=n_grid[:6], max_radius=6, policy="shipped")
    print(f"  t_max={info['t_max']:.1f}")
    print("   n   pre_N8   node_local  shipped  near_deg_frac  n_conf  consensus")
    for n, _d, m in rows:
        print(f"  {n:5d} {m['pre_n8']:8.4f} {m['node_local']:11.4f} {m['shipped']:8.4f} "
              f"{m['near_deg_frac']:14.4f} {m['n_confident']:7d}  {m['consensus']}")
    print(f"  poly_algebraic_min_n: production={prod_min}  mine(pre)={mine_pre} "
          f"mine(node_local)={mine_nl} mine(shipped)={mine_sh}   (docs record 400)")
    assert prod_min == 400 and mine_pre == 400 and mine_sh == 400
    print("  -> PROBLEM 05 UNCHANGED at 400.\n")
    return True


def _spotcheck(label, pts, true_dim, tol, k, max_radius, n_grid, expect_pre, expect_nl,
               reported_fracs=None):
    print(f"=== SPOT-CHECK {label} (k={k}, max_radius={max_radius}, tol={tol}) ===")
    prod_min = poly_algebraic_minimum_points(pts, true_dim, tol, k=k, n_grid=n_grid,
                                             max_radius=max_radius)
    mine_pre, rows = my_min_n(pts, true_dim, tol, k=k, n_grid=n_grid, max_radius=max_radius,
                              policy="pre_n8")
    mine_nl, _ = my_min_n(pts, true_dim, tol, k=k, n_grid=n_grid, max_radius=max_radius,
                          policy="node_local")
    mine_sh, _ = my_min_n(pts, true_dim, tol, k=k, n_grid=n_grid, max_radius=max_radius,
                          policy="shipped")
    print("      n   pre_N8  node_local   shipped  near_deg_frac  n_conf  conf_mean  consensus")
    for n, _d, m in rows:
        conf = [x.dimension for x in m["estimates"] if x.r_squared >= 0.9]
        cm = sum(conf) / len(conf) if conf else float("nan")
        print(f"  {n:6d} {m['pre_n8']:8.4f} {m['node_local']:11.4f} {m['shipped']:9.4f} "
              f"{m['near_deg_frac']:14.4f} {m['n_confident']:7d} {cm:10.4f}  {m['consensus']}")
    print(f"  poly_algebraic_min_n: production(shipped)={prod_min}  "
          f"mine: pre-N8={mine_pre} node-local={mine_nl} shipped={mine_sh}")
    if reported_fracs is not None:
        print("  repair agent's reported near_degenerate_fraction vs mine:")
        for n, rep in reported_fracs.items():
            got = next((m["near_deg_frac"] for nn, _d, m in rows if nn == n), None)
            status = "MATCH" if got is not None and abs(got - rep) < 1e-9 else "MISMATCH"
            print(f"    n={n:6d}  reported={rep:.4f}  measured={got}  {status}")
            assert status == "MATCH", (n, rep, got)
    assert prod_min == mine_sh == expect_pre, (prod_min, mine_sh, expect_pre)
    assert mine_pre == expect_pre, (mine_pre, expect_pre)
    assert mine_nl == expect_nl, (mine_nl, expect_nl)
    print(f"  -> shipped == pre-N8 == {expect_pre}; "
          f"round-2 node-local tree would give {expect_nl}.\n")
    return True


def check_prob02():
    pts, info = cloud_prob02()
    print(f"[prob02 cloud] my AGM period={info['period']:.10f} closure={info['closure']:.3e} "
          f"energy drift={info['energy_drift']:.3e}")
    return _spotcheck(
        "PROBLEM 02 nonlinear pendulum", pts, 1.0, 0.15, 6, 6,
        (32, 64, 128, 256, 512, 1024, 2048, 4096), 64, 32,
        reported_fracs={32: 0.25, 64: 0.0, 128: 0.0, 256: 0.0, 512: 0.0,
                        1024: 0.0, 2048: 0.0, 4096: 0.0},
    )


def check_prob10():
    pts, info = cloud_prob10()
    print(f"[prob10 cloud] stroboscopic tail spread={info['strobe_spread']} "
          f"one-period closure={info['closure']:.3e}")
    return _spotcheck(
        "PROBLEM 10 driven pendulum", pts, 1.0, 0.2, 6, 6,
        (32, 64, 128, 256, 512, 1024, 2048, 4096), 64, 32,
        reported_fracs={32: 0.125, 64: 0.0, 128: 0.0, 256: 0.0, 512: 0.0,
                        1024: 0.0, 2048: 0.0, 4096: 0.0},
    )


def check_prob03():
    pts, info = cloud_kepler()
    print(f"[kepler cloud] period={info['period']:.6f} one-period closure={info['closure']:.3e}")
    return _spotcheck(
        "KEPLER-TYPE 1D ORBIT (my own e=0.6 ellipse)", pts, 1.0, 0.15, 6, 6,
        (100, 200, 400, 800, 1600, 3200, 6400), 100, 100,
    )


CHECKS = {
    "anchor": check_anchor,
    "prob06": check_prob06,
    "prob05": check_prob05,
    "prob02": check_prob02,
    "prob10": check_prob10,
    "prob03": check_prob03,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(CHECKS) if which == "all" else [which]
    for name in names:
        CHECKS[name]()
    print("ALL REQUESTED N8e SKEPTIC CHECKS PASSED")
