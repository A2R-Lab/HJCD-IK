"""G1 multi-target coarse search (humanoid-v1 M5; port of bbc53a9 coarse_search_mt_kernel). GPU + nvcc.

CPU references: coordinate proposals from the independent numpy oracle (tests/test_g1_lm.oracle_eval: J, residual,
row scales); subtree FK checked in-kernel against a full FK of the same state; contacts re-checked in the world frame.
"""
import numpy as np
import pytest

import test_g1_lm as L
import test_g1_base as TB
import test_g1_batch as Bt
from test_g1_lm import tm, model, oracle  # noqa: F401  (fixtures)

PREC = {"f64": np.float64, "f32": np.float32}
LAM, HMIN, MAXSTEP = 1e-6, 1e-9, 0.35


def cpu_proposals(model, oracle, q, prob):
    """bbc53a9 coord_proposal, written independently: g, h over affected ACTIVE targets with W = w s^2."""
    o = L.oracle_eval(oracle, model, q, prob)
    tgt, act, wp, wo, mode = prob[0], int(prob[1]), prob[2], prob[3], prob[4]
    lo, hi = L.limits(model).T
    out = np.zeros((29, 3)); out[:, 0] = q; out[:, 2] = -1
    for j in range(29):
        g = h = 0.0; aff = False
        for k, t in enumerate(model["targets"]):
            if not ((act >> k) & 1 and (t["ancestor_mask"] >> j) & 1):
                continue
            aff = True
            wok = 0.0 if mode[k] == L.NONE else wo[k]
            for r in range(6):
                W = (wp[k] if r < 3 else wok) * o["scale"][6 * k + r] ** 2
                g += W * o["J"][6 * k + r, j] * o["E"][6 * k + r]
                h += W * o["J"][6 * k + r, j] ** 2
        if not aff or not h > HMIN:
            continue
        d = np.clip(g / (h + LAM), -MAXSTEP, MAXSTEP)
        v = np.clip(q[j] + d, lo[j], hi[j]); d = v - q[j]
        if d == 0:
            continue
        pred = 2 * d * g - d * d * h
        if pred > 0:
            out[j] = (v, d, pred)
    return out


def props(tm, prec, Q, probs, base=None):
    T = np.float64 if prec == "f64" else np.float32
    tgt, act, wp, wo, mode, axis, tw = L.stack(probs)
    return getattr(tm, f"proposals_{prec}")(np.asarray(Q, T), tgt.astype(T), act.astype(np.uint32), wp.astype(T),
                                            wo.astype(T), mode, axis.astype(T), twist=tw.astype(T),
                                            base=None if base is None else np.asarray(base, T)).astype(np.float64)


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_proposals_match_cpu_and_subtree_fk_matches_full(tm, model, oracle, prec):
    rng = np.random.default_rng(1)
    lo, hi = L.limits(model).T
    Q, probs = [], []
    for i in range(27):
        active, modes = L.CASES[i % len(L.CASES)]
        q = lo + (hi - lo) * rng.uniform(0.1, 0.9, 29)
        if i % 9 == 4:
            q[[3, 9, 18, 25]] = hi[[3, 9, 18, 25]]                # at joint limits: outward proposals clip to 0
        wp = rng.choice([0.0, 0.5, 2.0], 4) if i % 4 == 3 else None
        noise = 0.0 if i % 9 == 8 else 0.04                        # exact solution: zero gradient -> no proposals
        probs.append(L.make_problem(rng, model, oracle, q, active, modes, noise_p=noise, noise_r=noise * 5, wp=wp))
        Q.append(q if noise == 0 else np.clip(q + rng.normal(size=29) * 0.05, lo, hi))
    Q = np.array(Q)
    g = props(tm, prec, Q, probs)
    worst = dict(v=0.0, d=0.0, pred_rel=0.0, subtree_pose=0.0, subtree_cost=0.0)
    tol = 1e-10 if prec == "f64" else 5e-4
    for i in range(len(Q)):
        c = cpu_proposals(model, oracle, Q[i], probs[i])
        valid_c, valid_g = c[:, 2] > 0, g[i, :, 2] > 0
        exact = i % 9 == 8
        if exact:   # zero residual: g is rounding noise, so validity is ill-posed; gains must be negligible
            assert np.max(g[i, :, 2]) <= (1e-18 if prec == "f64" else 1e-6) * max(1.0, g[i, 0, 3]), (i, np.max(g[i, :, 2]))
            continue
        if prec == "f64":
            np.testing.assert_array_equal(valid_g, valid_c, err_msg=f"case {i}")
        both = valid_c & valid_g
        worst["v"] = max(worst["v"], np.max(np.abs(g[i, both, 0] - c[both, 0]), initial=0))
        worst["d"] = max(worst["d"], np.max(np.abs(g[i, both, 1] - c[both, 1]), initial=0))
        if both.any():   # relative to the case's largest gain (tiny gains are rounding-dominated)
            worst["pred_rel"] = max(worst["pred_rel"], np.max(np.abs(g[i, both, 2] - c[both, 2])) / np.max(c[both, 2]))
        assert worst["d"] <= tol, (i, worst)
        act = int(probs[i][1])
        for j in range(29):                                         # unrelated joints never propose
            if not any((act >> k) & 1 and (t["ancestor_mask"] >> j) & 1 for k, t in enumerate(model["targets"])):
                assert g[i, j, 2] == -1 and g[i, j, 1] == 0
        worst["subtree_pose"] = max(worst["subtree_pose"], g[i, :, 4].max())
        worst["subtree_cost"] = max(worst["subtree_cost"], np.max(g[i, :, 5] / np.maximum(1, g[i, :, 3])))
    assert worst["pred_rel"] <= (1e-8 if prec == "f64" else 2e-2)
    assert worst["subtree_pose"] == 0.0                              # subtree FK == full FK, bitwise
    assert worst["subtree_cost"] <= (1e-12 if prec == "f64" else 1e-5)
    print(prec, {k: f"{v:.2e}" for k, v in worst.items()})


def test_predicted_gain_matches_actual_change_for_small_steps(tm, model, oracle):
    """2 delta g - delta^2 h is the Gauss-Newton model of the (frozen-scale) cost decrease. With POSITION-only targets
    J is the exact derivative of e, so actual/predicted -> 1 as delta -> 0 (FULL uses the GN approximation J = jw,
    exact only at zero residual, so it is not a valid probe of the model)."""
    rng = np.random.default_rng(2)
    lo, hi = L.limits(model).T
    q = lo + (hi - lo) * rng.uniform(0.3, 0.7, 29)
    pr = L.make_problem(rng, model, oracle, q, 0b1111, [L.NONE] * 4, noise_p=0.002, noise_r=0.0)
    o0 = L.oracle_eval(oracle, model, q, pr)
    W = np.zeros(24)
    for k in range(4):
        for r in range(6):
            W[6 * k + r] = (pr[2][k] if r < 3 else pr[3][k]) * o0["scale"][6 * k + r] ** 2
    g = props(tm, "f64", q[None], [pr])[0]
    ratios = []
    for j in np.where(g[:, 2] > 0)[0]:
        q1 = q.copy(); q1[j] = g[j, 0]
        o1 = L.oracle_eval(oracle, model, q1, pr)
        actual = np.sum(W * o0["E"] ** 2) - np.sum(W * o1["E"] ** 2)
        if abs(g[j, 1]) < 0.02 and g[j, 2] > 1e-6 * g[:, 2].max():
            ratios.append(actual / g[j, 2])
    assert ratios and np.all(np.abs(np.array(ratios) - 1) < 0.05), ratios
    print("gain ratio actual/pred:", np.round(ratios, 4))


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_proposals_with_fixed_base_equal_transformed_targets(tm, model, oracle, prec):
    rng = np.random.default_rng(3)
    seeds, bases, probs, _ = TB.fb_batch(model, oracle, 6, 1, seed=91)
    Q, B = seeds[:, 0], bases[:, 0]
    a = props(tm, prec, Q, probs, base=B)
    ref = [(TB.targets_to_base(p[0], b),) + tuple(p[1:]) for p, b in zip(probs, B)]
    r = props(tm, prec, Q, ref)
    np.testing.assert_allclose(a[..., :3], r[..., :3], atol=1e-11 if prec == "f64" else 2e-3)


def coarse(tm, prec, seeds, bases, probs, M=2, mode="multi_target", lm_iters=60, iters=120, ps=None, ids=None, **kw):
    T = np.float64 if prec == "f64" else np.float32
    tgt, act, wp, wo, md, axis, tw = L.stack(probs)
    return getattr(tm, f"solve_problems_{prec}")(
        np.ascontiguousarray(seeds, T), tgt.astype(T), act.astype(np.uint32), wp.astype(T), wo.astype(T), md,
        axis.astype(T), twist_spec=tw.astype(T), num_solutions=M, candidate_ids=ids, lm_iters=lm_iters, stag_patience=0,
        return_all_candidates=True, base_poses=None if bases is None else np.ascontiguousarray(bases, T),
        coarse_mode=mode, coarse_iters=iters, problem_seeds=ps, **kw)


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_coarse_fixed_base_handoff_identity_and_invariance(tm, model, oracle, prec):
    seeds, bases, probs, _ = TB.fb_batch(model, oracle, 8, 8, seed=101, perturb=(0.05, 0.1), unreachable=(3,))
    ps = np.arange(8, dtype=np.uint32) * 7919 + 3
    ids = np.arange(64, dtype=np.int64).reshape(8, 8) * 13 + 5
    # LM budget 0: the final joints ARE the coarse joints, bases untouched, identities intact
    r0 = coarse(tm, prec, seeds, bases, probs, lm_iters=0, ps=ps, ids=ids)
    np.testing.assert_array_equal(r0["all_joint_config"], r0["all_coarse_joint_config"])
    np.testing.assert_array_equal(r0["all_base_position"], r0["all_base_in"][..., :3])
    np.testing.assert_array_equal(r0["all_base_quaternion"], r0["all_base_in"][..., 3:])
    lo, hi = L.limits(model).T
    assert np.all(r0["all_coarse_joint_config"] >= lo.astype(r0["all_coarse_joint_config"].dtype)) and \
        np.all(r0["all_coarse_joint_config"] <= hi.astype(r0["all_coarse_joint_config"].dtype))
    # coarse never worsens the physical objective (best state on E_phys), checked in the WORLD frame
    for p in range(8):
        for s in range(8):
            b = r0["all_base_in"][p, s].astype(np.float64)
            pe0, oe0 = TB.world_check(model, oracle, seeds[p, s], b, probs[p])
            pe1, oe1 = TB.world_check(model, oracle, r0["all_coarse_joint_config"][p, s].astype(np.float64), b, probs[p])
            E0 = np.sum((pe0 / 1e-4) ** 2 + (oe0 / 1e-3) ** 2); E1 = np.sum((pe1 / 1e-4) ** 2 + (oe1 / 1e-3) ** 2)
            assert E1 <= E0 * (1 + 1e-5) + 1e-6, (p, s, E0, E1)
            np.testing.assert_allclose(pe1, r0["all_coarse_position_errors"][p, s], atol=1e-9 if prec == "f64" else 5e-6)
    # full pipeline; selected slot pairs joints + base + id of the same candidate; world-frame exactness
    r = coarse(tm, prec, seeds, bases, probs, ps=ps, ids=ids, base_update=dict(enabled=True))
    for p in range(8):
        for m in range(2):
            s = r["selected_seed_ids"][p, m]
            if s < 0:
                continue
            assert r["selected_candidate_ids"][p, m] == ids[p, s]
            np.testing.assert_array_equal(r["base_position"][p, m], r["all_base_position"][p, s])
            np.testing.assert_array_equal(r["joint_config"][p, m], r["all_joint_config"][p, s])
            if r["success"][p, m]:
                pe, oe = TB.world_check(model, oracle, r["joint_config"][p, m].astype(np.float64),
                                        np.r_[r["base_position"][p, m], r["base_quaternion"][p, m]].astype(np.float64), probs[p])
                slack = 1e-9 if prec == "f64" else 5e-7
                assert np.all(pe <= 1e-4 + slack) and np.all(oe <= 1e-3 + 10 * slack)
    assert r["num_solved"][3] == 0
    # deterministic; independent == batched; permutation with problem_seeds + ids follows bitwise
    keys = Bt.RAW + Bt.SEL + ("all_coarse_joint_config", "all_coarse_kicks", "all_base_position", "selected_candidate_ids")
    Bt.assert_same(r, coarse(tm, prec, seeds, bases, probs, ps=ps, ids=ids, base_update=dict(enabled=True)), "repeat", keys)
    one = coarse(tm, prec, seeds[5:6], bases[5:6], probs[5:6], ps=ps[5:6], ids=ids[5:6], base_update=dict(enabled=True))
    Bt.assert_same(Bt.take(r, slice(5, 6), keys), Bt.take(one, slice(None), keys), "independent")
    perm = np.random.default_rng(4).permutation(8)
    rp = coarse(tm, prec, seeds[perm], bases[perm], [probs[i] for i in perm], ps=ps[perm], ids=ids[perm],
                base_update=dict(enabled=True))
    Bt.assert_same(Bt.take(r, perm, keys), Bt.take(rp, slice(None), keys), "permutation")


def test_coarse_mode_dispatch_and_validation(tm, model, oracle):
    seeds, bases, probs, _ = TB.fb_batch(model, oracle, 9, 4, seed=111)
    r = coarse(tm, "f64", seeds, bases, probs, mode="auto")
    popc = np.array([bin(int(p[1])).count("1") for p in probs])
    np.testing.assert_array_equal(r["all_use_coarse"], np.repeat((popc >= 2)[:, None], 4, axis=1))
    single = np.where(popc == 1)[0]
    assert single.size and np.all(r["all_coarse_iterations"][single] == 0)
    none = coarse(tm, "f64", seeds, bases, probs, mode="none")                     # == M4b path
    m4b = TB.fsolve(tm, "f64", seeds, bases, probs, M=2)
    Bt.assert_same(Bt.take(none, slice(None), Bt.RAW + Bt.SEL), Bt.take(m4b, slice(None), Bt.RAW + Bt.SEL), "coarse none")
    z = coarse(tm, "f64", seeds, bases, probs, iters=0)
    assert not z["coarse_ran"]
    with pytest.raises(ValueError, match="coarse_mode"):
        coarse(tm, "f64", seeds, bases, probs, mode="legacy")
    with pytest.raises(ValueError, match="problem_seeds"):
        coarse(tm, "f64", seeds, bases, probs, mode="none", ps=np.zeros(9, np.uint32))


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_coarse_improves_lm_on_hard_seeds(tm, model, oracle, prec):
    seeds, bases, probs, _ = TB.fb_batch(model, oracle, 16, 16, seed=121, perturb=(0.02, 0.04))
    rng = np.random.default_rng(5)
    lo, hi = L.limits(model).T
    seeds = lo + (hi - lo) * rng.uniform(size=seeds.shape)                         # uninformed joint seeds
    # (near-correct seeds leave nothing for coarse to do: 243-244 vs 244 of 256 measured; see the M5 report)
    lm = coarse(tm, prec, seeds, bases, probs, mode="none")
    cl = coarse(tm, prec, seeds, bases, probs, mode="multi_target")
    print(prec, "exact candidates LM-only", int(lm["all_lm_converged"].sum()), "coarse+LM", int(cl["all_lm_converged"].sum()),
          "coarse-only", int(cl["all_coarse_success"].sum()), "kicks", int(cl["all_coarse_kicks"].sum()))
    assert cl["all_lm_converged"].sum() > lm["all_lm_converged"].sum()
    assert np.all(np.isfinite(cl["all_joint_config"]))
