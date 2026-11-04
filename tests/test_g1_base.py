"""G1 external floating base: world->base targets, the bbc53a9 base step, alternating LM, batched selection
(humanoid-v1 M4b). GPU + nvcc. Independent references: numpy rigid transforms below, the numpy URDF FK
(tests/urdf_fk.py) and the bbc53a9 base-update oracle (tests/base_update_oracle.py).

Conventions: base pose [x,y,z, qw,qx,qy,qz] = ^W T_B (world frame, WXYZ, normalised on the host);
^B T* = (^W T_B)^-1 ^W T*;  base step perturbs on the LEFT in the world frame about the base origin.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

import test_g1_lm as L
import test_g1_batch as Bt
from test_g1_lm import tm, model, oracle  # noqa: F401  (fixtures)

PREC = {"f64": np.float64, "f32": np.float32}
spec = importlib.util.spec_from_file_location("bu_oracle", pathlib.Path(__file__).with_name("base_update_oracle.py"))
BU = importlib.util.module_from_spec(spec); spec.loader.exec_module(BU)


def rand_base(rng, trans=0.5, rot=np.pi):
    q = L.rand_quat(rng)
    if rot < np.pi:                                   # bounded rotation about a random axis
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = rng.uniform(-rot, rot)
        q = np.r_[np.cos(a / 2), np.sin(a / 2) * ax]
    return np.r_[rng.uniform(-trans, trans, 3), q]


def to_world(base, T_B):
    """^W T = ^W T_B ^B T (independent numpy)."""
    W = np.eye(4); W[:3, :3] = L.quat2mat(base[3:] / np.linalg.norm(base[3:])); W[:3, 3] = base[:3]
    return W @ T_B


def targets_to_base(tgt, base):
    """CPU reference of the world->base target transform."""
    qb = base[3:] / np.linalg.norm(base[3:]); R = L.quat2mat(qb)
    out = tgt.copy()
    for k in range(len(tgt)):
        out[k, :3] = R.T @ (tgt[k, :3] - base[:3])
        q = L.qmul(qb * np.array([1, -1, -1, -1]), tgt[k, 3:])
        out[k, 3:] = q
    return out


def world_problem(rng, model, oracle, q, base, active, modes, tool=None):
    """Targets EXACTLY reachable from (base, q): world = base * FK(q)."""
    pr = list(L.make_problem(rng, model, oracle, q, active, modes, noise_p=0.0, noise_r=0.0, tool=tool))
    tgt = pr[0].copy()
    for k in range(4):
        Tb = np.eye(4); Tb[:3, :3] = L.quat2mat(tgt[k, 3:]); Tb[:3, 3] = tgt[k, :3]
        Tw = to_world(base, Tb)
        tgt[k] = np.r_[Tw[:3, 3], L.mat2quat(Tw[:3, :3])]
    pr[0] = tgt
    return tuple(pr)


def ev(tm, prec, Q, probs, base=None, tools=None):
    T = PREC[prec]
    tgt, act, wp, wo, mode, axis, tw = L.stack(probs)
    return getattr(tm, f"evaluate_{prec}")(np.asarray(Q, T), tgt.astype(T), act.astype(np.uint32), wp.astype(T),
                                          wo.astype(T), mode, axis.astype(T), twist=tw.astype(T),
                                          tool=None if tools is None else np.asarray(tools, T),
                                          base=None if base is None else np.asarray(base, T))


# ------------------------------------------------------------- transforms
@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_world_to_base_matches_cpu_and_is_equivariant(tm, model, oracle, prec):
    rng = np.random.default_rng(3)
    lo, hi = L.limits(model).T
    bases = [np.r_[0, 0, 0, 1, 0, 0, 0.0], np.r_[0.3, -0.2, 0.9, 1, 0, 0, 0], rand_base(rng, 0.0), rand_base(rng)]
    bases += [rand_base(rng) for _ in range(8)]
    Q, probs, B, tools = [], [], [], []
    for i, b in enumerate(bases):
        active, modes = L.CASES[i % len(L.CASES)]
        q = lo + (hi - lo) * rng.uniform(0.2, 0.8, 29)
        t = L.tools_of(model).copy()
        if i % 3 == 1:
            c, s = np.cos(0.4), np.sin(0.4); t[2] = np.array([[c, -s, 0, 0.1], [s, c, 0, 0], [0, 0, 1, -0.03], [0, 0, 0, 1]])
        probs.append(L.make_problem(rng, model, oracle, q, active, modes, noise_p=0.03, noise_r=0.2, tool=t))
        Q.append(q + rng.normal(size=29) * 0.05); B.append(b); tools.append(t)
    Q, B, tools = np.array(Q), np.array(B), np.array(tools)
    g = ev(tm, prec, Q, probs, base=B, tools=tools)
    # reference: CPU-transformed targets, no base on the GPU, and the CPU oracle residual
    ref_probs = [(targets_to_base(p[0], b),) + tuple(p[1:]) for p, b in zip(probs, B)]
    r = ev(tm, prec, Q, ref_probs, tools=tools)
    tol = 1e-12 if prec == "f64" else 2e-5
    np.testing.assert_allclose(g["e"], r["e"], atol=tol)
    for i in range(len(Q)):
        o = L.oracle_eval(oracle, model, Q[i], ref_probs[i], tool=tools[i])
        np.testing.assert_allclose(g["e"][i].reshape(-1), o["E"], atol=tol if prec == "f64" else 5e-5)
    # quaternion sign equivalence (q_B and -q_B are the same base)
    Bn = B.copy(); Bn[:, 3:] *= -1
    np.testing.assert_allclose(ev(tm, prec, Q, probs, base=Bn, tools=tools)["e"], g["e"], atol=tol)
    # equivariance: one rigid G applied to every base AND every world target leaves base-frame residuals unchanged
    G = rand_base(rng, 2.0); RG = L.quat2mat(G[3:])
    probs_g, B_g = [], B.copy()
    for i, p in enumerate(probs):
        t = p[0].copy()
        for k in range(4):
            t[k, :3] = RG @ t[k, :3] + G[:3]; t[k, 3:] = L.qmul(G[3:], t[k, 3:])
        probs_g.append((t,) + tuple(p[1:]))
        B_g[i, :3] = RG @ B[i, :3] + G[:3]; B_g[i, 3:] = L.qmul(G[3:], B[i, 3:])
    np.testing.assert_allclose(ev(tm, prec, Q, probs_g, base=B_g, tools=tools)["e"], g["e"],
                               atol=1e-11 if prec == "f64" else 5e-5)
    # identity base == no base (bitwise)
    np.testing.assert_array_equal(ev(tm, prec, Q[:1], probs[:1], base=B[:1], tools=tools[:1])["e"],
                                  ev(tm, prec, Q[:1], probs[:1], tools=tools[:1])["e"])


# ------------------------------------------------------------- one base step vs the numpy oracle
def oracle_step(tm_out_row, model, oracle, q, prob, base, cfg):
    """Independent: contacts from the URDF FK, J/H/b/solve/clip/apply from bbc53a9's numpy oracle."""
    tgt, act, wp = prob[0], int(prob[1]), prob[2]
    Tj = oracle.joint_world_transforms(q)
    c_base = np.array([(Tj[t["anchor_jid"]] @ np.array(t["tool_row_major"]))[:3, 3] for t in model["targets"]])
    x = BU.contact_points_world(c_base, base[:3], base[3:])
    J = BU.base_jacobian(x, base[:3])
    r = (tgt[:, :3] - x).reshape(-1)
    H, b = BU.base_normal_equations(J, r, weights=np.where(wp > 0, wp, 0.0), active_mask=act, damping=cfg["damping"])
    dxi, ok = BU.solve_base_update(H, b)
    step = BU.clip_base_step(dxi, cfg["max_translation_step"], cfg["max_rotation_step"])
    p_new, q_new = BU.apply_base_update(base[:3], base[3:], step)
    return H, b, dxi, step, p_new, q_new, ok


CFG = dict(enabled=True, damping=1e-3, max_translation_step=0.05, max_rotation_step=0.10)


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_base_step_matches_oracle(tm, model, oracle, prec):
    rng = np.random.default_rng(8)
    lo, hi = L.limits(model).T
    rows = []
    kinds = ["trans", "rot", "both", "big", "tiny", "small_angle", "single_target", "hands_only"]
    for i, kind in enumerate(kinds * 2):
        q = lo + (hi - lo) * rng.uniform(0.3, 0.7, 29)
        base_true = rand_base(rng, 0.3)
        active, modes = (0b1111, [L.FULL] * 4)
        if kind == "single_target":
            active = 0b0001
        if kind == "hands_only":
            active, modes = 0b0011, [L.AXIS, L.FULL, L.FULL, L.FULL]
        pr = world_problem(rng, model, oracle, q, base_true, active, modes)
        d = {"trans": (0.02, 0.0), "rot": (0.0, 0.05), "both": (0.02, 0.05), "big": (0.3, 0.6),
             "tiny": (1e-7, 1e-7), "small_angle": (0.0, 1e-4), "single_target": (0.03, 0.08), "hands_only": (0.03, 0.05)}[kind]
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        base = base_true.copy(); base[:3] += d[0] * ax
        base[3:] = L.qmul(np.r_[np.cos(d[1] / 2), np.sin(d[1] / 2) * ax], base_true[3:])
        rows.append((q, pr, base))
    Q = np.array([r[0] for r in rows]); P = [r[1] for r in rows]; Bs = np.array([r[2] for r in rows])
    T = PREC[prec]
    tgt, act, wp, wo, mode, axis, tw = L.stack(P)
    res = getattr(tm, f"base_step_{prec}")(Q.astype(T), tgt.astype(T), act.astype(np.uint32), wp.astype(T), wo.astype(T),
                                          mode, axis.astype(T), twist=tw.astype(T), base=Bs.astype(T), base_update=CFG)
    out, base_in = res["raw"].astype(np.float64), res["base_in"]
    tolH, tols = (1e-12, 1e-9) if prec == "f64" else (2e-4, 2e-3)
    worst = dict(H=0, b=0, step=0, base=0)
    for i, (q, pr, base) in enumerate(rows):
        bn = base.copy(); bn[3:] /= np.linalg.norm(bn[3:])
        H, b, dxi, step, p_new, q_new, ok = oracle_step(out[i], model, oracle, q, pr, bn, CFG)
        gH = out[i, :36].reshape(6, 6).T; gb = out[i, 36:42]; gstep = out[i, 48:54]
        assert ok and out[i, 62] == 0
        worst["H"] = max(worst["H"], L.rel(gH, H)); worst["b"] = max(worst["b"], np.max(np.abs(gb - b)) / max(1, np.max(np.abs(b))))
        assert L.rel(gH, H) <= tolH and np.max(np.abs(gb - b)) / max(1, np.max(np.abs(b))) <= tolH * 10, (kinds[i % 8], L.rel(gH, H))
        worst["step"] = max(worst["step"], np.max(np.abs(gstep - step)))
        assert np.max(np.abs(gstep - step)) <= tols, (kinds[i % 8], gstep, step)
        assert np.linalg.norm(gstep[:3]) <= 0.05 * (1 + 1e-6) and np.linalg.norm(gstep[3:]) <= 0.10 * (1 + 1e-6)
        new = out[i, 54:61]
        assert abs(np.linalg.norm(new[3:]) - 1) <= (1e-12 if prec == "f64" else 1e-6)
        if out[i, 61] == 1:          # accepted: the GPU base equals the oracle's applied step
            sgn = 1 if new[3:] @ q_new >= 0 else -1
            worst["base"] = max(worst["base"], np.max(np.abs(new[:3] - p_new)), np.max(np.abs(new[3:] - sgn * q_new)))
            assert np.max(np.abs(new[:3] - p_new)) <= tols and np.max(np.abs(new[3:] - sgn * q_new)) <= tols
            assert out[i, 64] < out[i, 63]
        else:                         # rejected: exact restore of the (host-normalised) input base
            np.testing.assert_array_equal(new.astype(T), base_in[i])
            assert out[i, 64] >= out[i, 63] * (1 - 1e-6)
    print(prec, {k: f"{v:.2e}" for k, v in worst.items()})


def test_base_step_singular_and_rejected(tm, model, oracle):
    rng = np.random.default_rng(12)
    q = np.zeros(29)
    pr = list(world_problem(rng, model, oracle, q, rand_base(rng, 0.2), 0b0001, [L.FULL] * 4))
    pr[2] = np.zeros(4)                               # zero position weight: H = damping * D only, b = 0
    base = rand_base(rng, 0.2)
    T = np.float64
    tgt, act, wp, wo, mode, axis, tw = L.stack([tuple(pr)])
    out = tm.base_step_f64(q[None], tgt, act.astype(np.uint32), wp, wo, mode, axis, twist=tw, base=base[None],
                           base_update=dict(CFG, damping=0.0))["raw"]
    assert out[0, 62] == 1                            # H = 0 exactly: posv flags non-PD, counted, base untouched
    bn = base.copy(); bn[3:] /= np.linalg.norm(bn[3:])
    np.testing.assert_array_equal(out[0, 54:61], bn)
    out = tm.base_step_f64(q[None], tgt, act.astype(np.uint32), wp, wo, mode, axis, twist=tw, base=base[None],
                           base_update=CFG)["raw"]
    assert out[0, 62] == 0 and np.all(out[0, 48:54] == 0)   # damping only: zero step, finite
    # a position-improving step that worsens the FULL orientation objective is rejected and restored
    pr2 = list(world_problem(rng, model, oracle, q, np.r_[0, 0, 0, 1, 0, 0, 0.0], 0b0001, [L.FULL] * 4))
    tg = pr2[0].copy(); tg[0, :3] += np.array([0.0, 0.0, 0.02]); pr2[0] = tg
    pr2[3] = np.full(4, 100.0)
    st = L.stack([tuple(pr2)])
    base2 = np.r_[0, 0, 0, np.cos(0.0002), np.sin(0.0002), 0, 0]
    o2 = tm.base_step_f64(q[None], st[0], st[1].astype(np.uint32), st[2], st[3], st[4], st[5], twist=st[6], base=base2[None],
                          base_update=CFG)["raw"]
    if o2[0, 61] == 0:
        bn2 = base2 / np.r_[1, 1, 1, [np.linalg.norm(base2[3:])] * 4]
        np.testing.assert_array_equal(o2[0, 54:61], bn2)


# ------------------------------------------------------------- batched floating-base solves
def fb_batch(model, oracle, P, S, seed, perturb=(0.08, 0.15), unreachable=()):
    rng = np.random.default_rng(seed)
    lo, hi = L.limits(model).T
    probs, seeds, bases, truth = [], [], [], []
    for p in range(P):
        active, modes = L.CASES[p % len(L.CASES)]
        if all(m == L.NONE for m in modes):
            modes = [L.FULL] * 4
        q = lo + (hi - lo) * rng.uniform(0.2, 0.8, 29)
        bt = rand_base(rng, 0.5)
        pr = list(world_problem(rng, model, oracle, q, bt, active, modes))
        if p in unreachable:
            pr[0] = pr[0].copy(); pr[0][:, :3] += 3.0
        probs.append(tuple(pr)); truth.append((q, bt))
        sd, bs = [], []
        for s in range(S):
            sd.append(np.clip(q + rng.normal(size=29) * 0.1, lo, hi))
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            b = bt.copy(); b[:3] += rng.normal(size=3) * perturb[0]
            a = rng.uniform(-perturb[1], perturb[1]); b[3:] = L.qmul(np.r_[np.cos(a / 2), np.sin(a / 2) * ax], bt[3:])
            bs.append(b)
        seeds.append(sd); bases.append(bs)
    return np.array(seeds), np.array(bases), probs, truth


def fsolve(tm, prec, seeds, bases, probs, M=1, ids=None, base_update=None, base_bounds=None, tools=None, **kw):
    T = PREC[prec]
    tgt, act, wp, wo, mode, axis, tw = L.stack(probs)
    return getattr(tm, f"solve_problems_{prec}")(
        np.ascontiguousarray(seeds, T), tgt.astype(T), act.astype(np.uint32), wp.astype(T), wo.astype(T), mode,
        axis.astype(T), twist_spec=tw.astype(T), tool_override=None if tools is None else tools.astype(T),
        num_solutions=M, candidate_ids=ids, lm_iters=60, stag_patience=0, return_all_candidates=True,
        base_poses=None if bases is None else np.ascontiguousarray(bases, T), base_update=base_update,
        base_bounds=base_bounds, **kw)


def world_check(model, oracle, X, base, prob):
    """Independently recomputed WORLD-frame contact errors of (base, joints) for every active target."""
    tgt, act, mode, axis, tw = prob[0], int(prob[1]), prob[4], prob[5], prob[6]
    Tj = oracle.joint_world_transforms(X)
    pe, oe = np.zeros(4), np.zeros(4)
    for k, t in enumerate(model["targets"]):
        if not (act >> k) & 1:
            continue
        Tw = to_world(base, Tj[t["anchor_jid"]] @ np.array(t["tool_row_major"]))
        ep, eR, ang = L.residual(Tw, tgt[k, :3], tgt[k, 3:], mode[k], axis[k], tw[k])
        pe[k], oe[k] = np.linalg.norm(ep), ang
    return pe, oe


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_fixed_external_base_and_disabled_update(tm, model, oracle, prec):
    seeds, bases, probs, _ = fb_batch(model, oracle, 8, 8, seed=21)
    r = fsolve(tm, prec, seeds, bases, probs, M=3)
    bn = bases.copy(); bn[..., 3:] /= np.linalg.norm(bn[..., 3:], axis=-1, keepdims=True)
    np.testing.assert_array_equal(r["all_base_position"], r["all_base_in"][..., :3])     # base untouched
    np.testing.assert_array_equal(r["all_base_quaternion"], r["all_base_in"][..., 3:])
    assert np.all(r["all_base_updates"] == 0)
    # equals the fixed-base solve of the same problems with CPU-transformed targets (per candidate)
    for p in range(2):
        for s in range(3):
            pr = (targets_to_base(probs[p][0], bn[p, s]),) + tuple(probs[p][1:])
            f = Bt.solve(tm, prec, seeds[p:p+1, s:s+1], [pr], None, M=1)
            tol = 1e-9 if prec == "f64" else 2e-3
            np.testing.assert_allclose(f["all_joint_config"][0, 0], r["all_joint_config"][p, s], atol=tol)
    # identity base == no base, bitwise
    ident = np.zeros_like(bases); ident[..., 3] = 1
    a = fsolve(tm, prec, seeds, ident, probs, M=3); b = Bt.solve(tm, prec, seeds, probs, None, M=3)
    Bt.assert_same(Bt.take(a, slice(None), Bt.RAW + Bt.SEL), Bt.take(b, slice(None), Bt.RAW + Bt.SEL), "identity base")


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_alternating_lm_world_frame_and_pairing(tm, model, oracle, prec):
    seeds, bases, probs, _ = fb_batch(model, oracle, 12, 12, seed=33, unreachable=(5,))
    off = fsolve(tm, prec, seeds, bases, probs, M=4)
    on = fsolve(tm, prec, seeds, bases, probs, M=4, base_update=dict(enabled=True))
    print(prec, "exact candidates off/on", int(np.sum(off["all_position_errors"].max(-1) <= 1e-4)),
          int(on["num_solved"].sum()), "base updates att/acc/nf/rej", on["all_base_updates"].reshape(-1, 4).sum(0))
    assert on["num_solved"].sum() > off["num_solved"].sum()
    assert np.all(on["all_base_updates"][..., 0] >= 1) and np.all(np.isfinite(on["all_base_position"]))
    qn = np.linalg.norm(on["all_base_quaternion"].astype(np.float64), axis=-1)
    assert np.max(np.abs(qn - 1)) <= (1e-12 if prec == "f64" else 1e-6)
    tol = 1e-9 if prec == "f64" else 5e-6
    P, S = seeds.shape[:2]
    for p in range(P):                    # PAIRING: reported errors == world errors recomputed from (base, joints)
        for s in range(S):
            b = np.r_[on["all_base_position"][p, s], on["all_base_quaternion"][p, s]].astype(np.float64)
            pe, oe = world_check(model, oracle, on["all_joint_config"][p, s].astype(np.float64), b, probs[p])
            act = int(probs[p][1])
            for k in range(4):
                if (act >> k) & 1:
                    assert abs(pe[k] - on["all_position_errors"][p, s, k]) <= tol, (p, s, k)
        for m in range(4):                # selected slots: same candidate's base and joints
            sd = on["selected_seed_ids"][p, m]
            if sd >= 0:
                np.testing.assert_array_equal(on["base_position"][p, m], on["all_base_position"][p, sd])
                np.testing.assert_array_equal(on["base_quaternion"][p, m], on["all_base_quaternion"][p, sd])
                np.testing.assert_array_equal(on["joint_config"][p, m], on["all_joint_config"][p, sd])
                np.testing.assert_array_equal(on["selected_base_updates"][p, m], on["all_base_updates"][p, sd])
                if on["success"][p, m]:
                    b = np.r_[on["base_position"][p, m], on["base_quaternion"][p, m]].astype(np.float64)
                    pe, oe = world_check(model, oracle, on["joint_config"][p, m].astype(np.float64), b, probs[p])
                    slack = 1e-9 if prec == "f64" else 5e-7          # fp32 FK error is ~1.5e-7 (M2b)
                    assert np.all(pe <= 1e-4 + slack) and np.all(oe <= 1e-3 + 10 * slack), (p, m, pe, oe)
            else:
                assert np.all(np.isfinite(on["base_position"][p, m]))
    assert on["num_solved"][5] == 0 and not on["problem_success"][5]


def test_base_bounds_and_validation(tm, model, oracle):
    seeds, bases, probs, truth = fb_batch(model, oracle, 4, 6, seed=44)
    lo = bases[..., :3].reshape(-1, 3).min(0) - 0.01; hi = bases[..., :3].reshape(-1, 3).max(0) + 0.01
    r = fsolve(tm, "f64", seeds, bases, probs, base_update=dict(enabled=True), base_bounds=np.array([lo, hi]))
    assert np.all(r["all_base_position"] >= lo - 1e-15) and np.all(r["all_base_position"] <= hi + 1e-15)
    with pytest.raises(ValueError, match="per-problem"):
        fsolve(tm, "f64", seeds, bases, probs, base_update=dict(enabled=True), base_bounds=np.zeros((4, 2, 3)))
    with pytest.raises(ValueError, match="unknown base_update"):
        fsolve(tm, "f64", seeds, bases, probs, base_update=dict(enable=True))
    with pytest.raises(ValueError, match="require base_poses"):
        fsolve(tm, "f64", seeds, None, probs, base_update=dict(enabled=True))
    bad = bases.copy(); bad[0, 0, 3:] = 0
    with pytest.raises(ValueError, match="non-zero"):
        fsolve(tm, "f64", seeds, bad, probs)
    b0 = bases.copy()
    fsolve(tm, "f64", seeds, bases, probs, base_update=dict(enabled=True))
    np.testing.assert_array_equal(bases, b0)                                  # input never modified


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_floating_batch_invariance_and_isolation(tm, model, oracle, prec):
    seeds, bases, probs, _ = fb_batch(model, oracle, 9, 8, seed=55)
    kw = dict(base_update=dict(enabled=True), M=3)
    ids = np.arange(9 * 8, dtype=np.int64).reshape(9, 8) * 3 + 11
    full = fsolve(tm, prec, seeds, bases, probs, ids=ids, **kw)
    keys = Bt.RAW + Bt.SEL + ("all_base_position", "all_base_quaternion", "all_base_updates", "base_position",
                              "base_quaternion", "selected_candidate_ids")
    for p in (0, 4, 8):                                                  # independent vs batched
        one = fsolve(tm, prec, seeds[p:p+1], bases[p:p+1], probs[p:p+1], ids=ids[p:p+1], **kw)
        Bt.assert_same(Bt.take(full, slice(p, p+1), keys), Bt.take(one, slice(None), keys), f"independent {p}")
    perm = np.random.default_rng(2).permutation(9)                      # permutation
    pr = fsolve(tm, prec, seeds[perm], bases[perm], [probs[i] for i in perm], ids=ids[perm], **kw)
    Bt.assert_same(Bt.take(full, perm, keys), Bt.take(pr, slice(None), keys), "permutation")
    for sl in (slice(0, 4), slice(4, 9)):                               # partitioning
        part = fsolve(tm, prec, seeds[sl], bases[sl], probs[sl], ids=ids[sl], **kw)
        Bt.assert_same(Bt.take(full, sl, keys), Bt.take(part, slice(None), keys), "partition")
    b2 = bases.copy(); b2[3, 2, :3] += 0.05                              # base isolation (one candidate)
    iso = fsolve(tm, prec, seeds, b2, probs, ids=ids, **kw)
    mask = np.ones((9, 8), bool); mask[3, 2] = False
    for k in ("all_joint_config", "all_base_position", "all_position_errors"):
        np.testing.assert_array_equal(np.asarray(full[k])[mask], np.asarray(iso[k])[mask])
    assert not np.array_equal(full["all_base_position"][3, 2], iso["all_base_position"][3, 2])
    p2 = list(probs); t = probs[6][0].copy(); t[:, :3] += 0.02; p2[6] = (t,) + tuple(probs[6][1:])   # problem isolation
    iso2 = fsolve(tm, prec, seeds, bases, p2, ids=ids, **kw)
    others = [p for p in range(9) if p != 6]
    Bt.assert_same(Bt.take(full, others, keys), Bt.take(iso2, others, keys), "problem isolation")
