"""humanoid-v1 M6b: native GPU self-collision on the M6a 621-sphere G1 model.

Independent references: sphere centres from the M2a numpy URDF FK on each sphere's own link; the body-pair policy
re-derived here from the URDF joint graph (not via hjcdik._tree_collision); clearances by direct numpy sphere/sphere
distances over every eligible pair (no broad phase); selection by a numpy re-implementation of the documented key.
"""
import numpy as np
import pytest

import test_g1_lm as L
import test_g1_base as TB
import test_g1_env_collision as E
from test_g1_lm import tm, model, oracle  # noqa: F401  (fixtures)
from test_g1_env_collision import table, link_spheres, PREC, FREE, COLL, INVALID, NOT_EVAL  # noqa: F401
from test_g1_gpu_fk import URDF

N_BODIES, N_EXCLUDED, N_CHECKED = 28, 26, 352


# ------------------------------------------------------------------ independent policy + oracle
def body_of_link(oracle):
    """link -> movable joint whose child (via fixed joints) carries it, or None for the root assembly."""
    parent_joint = {j["child"]: j for j in oracle.joints.values()}
    out = {}
    for link in set(parent_joint) | {oracle.root_link}:
        cur = link
        while cur in parent_joint and parent_joint[cur]["type"] == "fixed":
            cur = parent_joint[cur]["parent"]
        out[link] = parent_joint[cur]["name"] if cur in parent_joint else None
    return out


@pytest.fixture(scope="module")
def policy(oracle, link_spheres):
    """(body per sphere [NS] as joint name/None, checked body-pair set) from the documented rule."""
    links, _ = link_spheres
    bol = body_of_link(oracle)
    body = [bol[l] for l in links]
    parent = {n: bol[oracle.joints[n]["parent"]] for n in oracle.joint_order()}
    bodies = sorted(set(body), key=lambda b: (b is not None, b or ""))
    checked = set()
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            if not (parent.get(a, "x") == b or parent.get(b, "x") == a):
                checked.add(frozenset((a, b)))
    return body, checked, bodies


@pytest.fixture(scope="module")
def pairs(model, table):
    from hjcdik._tree_collision import self_collision_pairs
    return self_collision_pairs(URDF, [j["name"] for j in model["joints"]], table[0])


def eligible_index(policy):
    body, checked, _ = policy
    n = len(body)
    I, J = np.triu_indices(n, 1)
    ok = np.array([body[i] != body[j] and frozenset((body[i], body[j])) in checked for i, j in zip(I, J)])
    return I[ok], J[ok]


@pytest.fixture(scope="module")
def elig(policy):
    return eligible_index(policy)


def cpu_self(oracle, link_spheres, elig, q, margin=0.0):
    links, local = link_spheres
    c = E.cpu_centres(oracle, links, local, q)
    I, J = elig
    d = np.linalg.norm(c[I] - c[J], axis=1) - local[I, 3] - local[J, 3]
    if len(d) == 0:
        return FREE, np.inf, (-1, -1), d
    k = np.lexsort((J, I, d))[0]                                    # min d, ties -> lowest (i, j)
    return (COLL if d.min() < margin else FREE), d[k], (int(I[k]), int(J[k])), d


def self_call(tm, prec, Q, tab, pr, margin=0.0, base=None):
    T = PREC[prec]
    return getattr(tm, f"self_collision_{prec}")(np.ascontiguousarray(Q, T), tab.astype(T), pr, margin,
                                                 None if base is None else np.ascontiguousarray(base, T))


def corpora(model, rng, n=24):
    """Named joint distributions: random, neutral, near limits, single arm, crossed arms, waist, legs."""
    lo, hi = L.limits(model).T
    mid = np.clip(np.zeros(29), lo, hi)
    names = [j["name"] for j in model["joints"]]
    idx = lambda *keys: [i for i, nm in enumerate(names) if any(k in nm for k in keys)]
    out = {"random": lo + (hi - lo) * rng.uniform(size=(n, 29)),
           "neutral": np.clip(mid + rng.normal(size=(n, 29)) * 0.05, lo, hi),
           "near_limits": np.where(rng.uniform(size=(n, 29)) < 0.5, lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo))}
    for name, keys in (("left_arm", ("left_shoulder", "left_elbow", "left_wrist")), ("both_arms", ("shoulder", "elbow", "wrist")),
                       ("waist", ("waist",)), ("legs", ("hip", "knee", "ankle"))):
        Q = np.tile(mid, (n, 1)); ix = idx(*keys)
        Q[:, ix] = lo[ix] + (hi[ix] - lo[ix]) * rng.uniform(size=(n, len(ix)))
        out[name] = Q
    return out


# ------------------------------------------------------------------ policy
def test_policy_table(model, oracle, table, pairs, policy):
    tab, links = table
    pr, audit = pairs
    names = [j["name"] for j in model["joints"]]
    body, checked, bodies = policy
    assert len(bodies) == N_BODIES and len(audit) == N_BODIES * (N_BODIES - 1) // 2
    assert len(pr) == N_CHECKED and sum(not r["checked"] for r in audit) == N_EXCLUDED
    nm = lambda a: None if a < 0 else names[a]
    assert {frozenset((nm(a), nm(b))) for a, b in pr} == checked              # == independent re-derivation
    for r in audit:                                                            # every exclusion is parent/child
        if not r["checked"]:
            pa = {n: body_of_link(oracle)[oracle.joints[n]["parent"]] for n in oracle.joint_order()}
            a, b = nm(r["a"]), nm(r["b"])
            assert pa.get(a, "x") == b or pa.get(b, "x") == a
    roots = [i for i, a in enumerate(tab[:, 0]) if a < 0]
    assert len(roots) == 23 and (tab[roots, 0] == -1).all()                    # pelvis spheres take part
    # torso/logo/head (fixed-welded) form ONE body; torso vs pelvis (two massless waist bodies apart) is CHECKED
    assert len({body[i] for i, l in enumerate(links) if l in ("torso_link", "logo_link", "head_link")}) == 1
    wp = names.index("waist_pitch_joint")
    assert any((a, b) in ((-1, wp), (wp, -1)) for a, b in pr)


# ------------------------------------------------------------------ numerical correctness
@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_gpu_matches_cpu_oracle(tm, model, oracle, table, link_spheres, pairs, elig, prec):
    tab, _ = table; pr, _ = pairs
    rng = np.random.default_rng(31)
    tol = 1e-9 if prec == "f64" else 2e-5
    seen = {FREE: 0, COLL: 0}
    for name, Q in corpora(model, rng, 12 if prec == "f64" else 8).items():
        o = self_call(tm, prec, Q, tab, pr)
        assert o["eligible_pairs"] == len(elig[0]) == 160585
        for i, q in enumerate(Q):
            st, dmin, (pi, pj), d = cpu_self(oracle, link_spheres, elig, q)
            np.testing.assert_allclose(o["min_clearance"][i], dmin, atol=tol, err_msg=name)
            if abs(dmin) > 10 * tol:
                assert o["status"][i] == st, name
            gap = np.partition(d, 1)[1] - d.min() if len(d) > 1 else np.inf
            if prec == "f64" and gap > 1e-9:                                   # no near-tie: same closest pair
                assert (o["pair_i"][i], o["pair_j"][i]) == (pi, pj), name
                assert tuple(o["body_pair"][i]) == (int(tab[pi, 0]), int(tab[pj, 0]))
            seen[int(o["status"][i])] += 1
            assert 0 < o["evaluated_pairs"][i] <= 160585
    assert seen[COLL] > 0                                                      # (M6b: no FREE pose exists under this model; see REPORT)


def test_margin_boundary_zero_pairs_and_invalid(tm, model, oracle, table, link_spheres, pairs, elig):
    tab, _ = table; pr, _ = pairs
    rng = np.random.default_rng(32)
    Q = corpora(model, rng, 40)["neutral"]
    o = self_call(tm, "f64", Q, tab, pr)
    i = int(np.argmax(o["min_clearance"]))
    d0 = cpu_self(oracle, link_spheres, elig, Q[i])[1]
    if d0 > 2e-7:                                                              # margin either side of the true minimum
        for m, want in ((d0 - 1e-7, FREE), (d0 + 1e-7, COLL)):
            assert self_call(tm, "f64", Q[i:i + 1], tab, pr, margin=m)["status"][0] == want
    for m in (0.0, 0.002, 0.005, 0.01):                                        # margin == count of d < m
        om = self_call(tm, "f64", Q, tab, pr, margin=m)
        want = np.array([cpu_self(oracle, link_spheres, elig, q, m)[0] for q in Q])
        far = np.abs(om["min_clearance"] - m) > 1e-9
        np.testing.assert_array_equal(om["status"][far], want[far])
    z = self_call(tm, "f64", Q[:3], tab, np.zeros((0, 2), np.int32))           # zero eligible pairs
    assert (z["status"] == FREE).all() and np.isinf(z["min_clearance"]).all() and (z["pair_i"] == -1).all()
    assert z["eligible_pairs"] == 0 and (z["evaluated_pairs"] == 0).all()
    Qb = Q[:3].copy(); Qb[1, 7] = np.inf
    b = self_call(tm, "f64", Qb, tab, pr)
    assert b["status"][1] == INVALID and np.isnan(b["min_clearance"][1]) and not b["collision_free"][1]
    for bad in (np.array([[0, 0]]), np.array([[-2, 3]]), np.array([[3, 29]])):
        with pytest.raises(ValueError):
            self_call(tm, "f64", Q[:1], tab, bad)
    with pytest.raises(ValueError):
        self_call(tm, "f64", Q[:1], tab, pr, margin=-1e-3)


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_base_invariance_batching_and_repeat(tm, model, table, pairs, prec):
    tab, _ = table; pr, _ = pairs
    rng = np.random.default_rng(33)
    Q = corpora(model, rng, 40)["random"]
    B = np.array([TB.rand_base(rng, 2.0) for _ in Q])
    a = self_call(tm, prec, Q, tab, pr)
    keys = ("status", "min_clearance", "pair_i", "pair_j", "evaluated_pairs")
    for k in keys:
        np.testing.assert_array_equal(self_call(tm, prec, Q, tab, pr, base=B)[k], a[k])     # base: exactly irrelevant
        np.testing.assert_array_equal(self_call(tm, prec, Q, tab, pr)[k], a[k])             # repeat
    perm = rng.permutation(len(Q))
    p = self_call(tm, prec, Q[perm], tab, pr)
    parts = [self_call(tm, prec, Q[s], tab, pr) for s in (slice(0, 1), slice(1, 33), slice(33, None))]
    for k in keys:
        np.testing.assert_array_equal(p[k], a[k][perm])
        np.testing.assert_array_equal(np.concatenate([x[k] for x in parts]), a[k])
    Q2 = Q.copy(); Q2[5, 20] += 0.3                                             # change one candidate only
    c = self_call(tm, prec, Q2, tab, pr)
    other = np.arange(len(Q)) != 5
    np.testing.assert_array_equal(c["min_clearance"][other], a["min_clearance"][other])


def test_fp32_matches_fp64(tm, model, table, pairs):
    tab, _ = table; pr, _ = pairs
    rng = np.random.default_rng(34)
    Q = np.concatenate(list(corpora(model, rng, 40).values()))
    for m in (0.0, 0.005):
        d = self_call(tm, "f64", Q, tab, pr, margin=m); f = self_call(tm, "f32", Q, tab, pr, margin=m)
        np.testing.assert_allclose(f["min_clearance"], d["min_clearance"], atol=5e-5)
        far = np.abs(d["min_clearance"] - m) > 1e-4
        np.testing.assert_array_equal(f["status"][far], d["status"][far])


# ------------------------------------------------------------------ selection (two channels)
def cpu_select2(Q, pe, oe, active, est, sst, M, esel, ssel, eps_p=1e-4, eps_o=1e-3):
    """M6b key: class (M5; a GATE channel with status != FREE -> class 2), then 3*env_rank + self_rank (rank channels
    only), then E_phys, then seed. Counts on the M5 class."""
    P, S = Q.shape[:2]
    seeds, counts = [], []
    for p in range(P):
        keys, c = [], dict(solved=0, sfree=0, scoll=0, sinv=0, all_free=0)
        for s in range(S):
            fin = np.isfinite(Q[p, s]).all(); e = 0.0; within = True
            for k in range(4):
                if not (active[p] >> k) & 1:
                    continue
                if not (np.isfinite(pe[p, s, k]) and np.isfinite(oe[p, s, k])):
                    fin = False; continue
                e += (pe[p, s, k] / eps_p) ** 2 + (oe[p, s, k] / eps_o) ** 2
                within &= pe[p, s, k] <= eps_p and oe[p, s, k] <= eps_o
            cls = 2 if not fin else (0 if within else 1)
            if not fin: e = np.inf
            ok = (est is None or est[p, s] == FREE) and sst[p, s] == FREE
            c["solved"] += cls == 0; c["all_free"] += cls == 0 and ok
            c["sfree"] += sst[p, s] == FREE; c["scoll"] += sst[p, s] == COLL; c["sinv"] += sst[p, s] == INVALID
            coll = 0
            if est is not None:
                if esel == "gate" and est[p, s] != FREE: cls, e = 2, np.inf
                elif esel == "rank": coll += 3 * int(est[p, s])
            if ssel == "gate" and sst[p, s] != FREE: cls, e = 2, np.inf
            elif ssel == "rank": coll += int(sst[p, s])
            keys.append((cls, coll, e, s))
        order = sorted(keys)
        seeds.append([k[3] if k[0] != 2 else -1 for k in order[:M]])
        counts.append(c)
    return np.array(seeds), counts


@pytest.mark.parametrize("prec", ["f64", "f32"])
@pytest.mark.parametrize("esel,ssel", [(None, "gate"), (None, "rank"), (None, "annotate"), ("gate", "gate"),
                                       ("rank", "rank"), ("gate", "rank"), ("rank", "gate"), ("annotate", "rank")])
def test_two_channel_selection_matches_cpu(tm, prec, esel, ssel):
    T = PREC[prec]
    rng = np.random.default_rng(35)
    P, S, M = 7, 40, 6
    Q, pe, oe, est = E.make_pool(rng, P, S, all_coll=(4,), zero_exact=(5,))
    sst = rng.choice([FREE, COLL, INVALID], size=(P, S), p=[0.5, 0.4, 0.1]).astype(np.uint8)
    sst[~np.isfinite(Q).all(-1)] = INVALID
    sst[6] = np.where(sst[6] == INVALID, INVALID, COLL)                        # a problem where self kills everything
    Q, pe, oe = Q.astype(T), pe.astype(T), oe.astype(T)
    active = np.array([15, 3, 12, 1, 15, 9, 6], np.uint32)
    kw = dict(self_status=sst, self_selection=ssel)
    if esel is not None:
        kw.update(env_status=est, env_selection=esel)
    o = getattr(tm, f"select_{prec}")(Q, pe, oe, np.zeros((P, S), T), np.ones((P, S), np.uint8), np.zeros_like(Q), active, M, **kw)
    want, counts = cpu_select2(Q.astype(float), pe.astype(float), oe.astype(float), active, None if esel is None else est, sst, M, esel, ssel)
    np.testing.assert_array_equal(o["selected_seed_ids"], want)
    for p in range(P):
        assert o["num_solved"][p] == counts[p]["solved"]
        assert o["num_self_free"][p] == counts[p]["sfree"] and o["num_self_colliding"][p] == counts[p]["scoll"]
        assert o["num_self_invalid"][p] == counts[p]["sinv"] and o["num_solved_collision_free"][p] == counts[p]["all_free"]
        for m in range(M):
            s = want[p, m]
            assert o["selected_self_status"][p, m] == (NOT_EVAL if s < 0 else sst[p, s])
    if ssel == "gate":
        assert (o["selected_seed_ids"][6] == -1).all()
    if ssel == "annotate" and esel is None:                                   # == the M5 order
        m5 = getattr(tm, f"select_{prec}")(Q, pe, oe, np.zeros((P, S), T), np.ones((P, S), np.uint8), np.zeros_like(Q), active, M)
        np.testing.assert_array_equal(o["selected_seed_ids"], m5["selected_seed_ids"])


# ------------------------------------------------------------------ pipeline
@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_pipeline_channels_independent(tm, model, oracle, table, pairs, prec):
    tab, _ = table; pr, _ = pairs
    T = PREC[prec]
    seeds, bases, probs, _ = TB.fb_batch(model, oracle, 6, 12, seed=411, perturb=(0.05, 0.1), unreachable=(2,))
    ids = np.arange(72, dtype=np.int64).reshape(6, 12) * 5 + 1
    kw = dict(M=4, ids=ids, base_update={"enabled": True}, coarse_mode="multi_target", problem_seeds=np.arange(6, dtype=np.uint32) + 9)
    env = {k: np.asarray(v, T) for k, v in E.wall_floor(bases).items()}
    off = TB.fsolve(tm, prec, seeds, bases, probs, **kw)
    envonly = TB.fsolve(tm, prec, seeds, bases, probs, environment_collision="final", environment=env, robot_spheres=tab.astype(T),
                        environment_margin=0.01, environment_selection="rank", **kw)
    raw = ("all_joint_config", "all_base_position", "all_base_quaternion", "all_position_errors", "all_orientation_errors",
           "all_lm_converged", "all_iterations", "all_coarse_joint_config", "num_solved", "num_valid", "num_converged")
    sk = dict(self_collision="final", self_pairs=pr, self_margin=0.0)
    for ssel in ("annotate", "rank", "gate"):
        sonly = TB.fsolve(tm, prec, seeds, bases, probs, robot_spheres=tab.astype(T), self_selection=ssel, **sk, **kw)
        both = TB.fsolve(tm, prec, seeds, bases, probs, environment_collision="final", environment=env, robot_spheres=tab.astype(T),
                         environment_margin=0.01, environment_selection="rank", self_selection=ssel, **sk, **kw)
        for r in (sonly, both):
            for k in raw:
                np.testing.assert_array_equal(r[k], off[k], err_msg=k)       # the IK state is never touched
        # self channel == standalone on the final joints; env channel unchanged by self
        X = sonly["all_joint_config"].reshape(-1, 29)
        sa = self_call(tm, prec, X, tab, pr)
        for r in (sonly, both):
            np.testing.assert_array_equal(r["all_self_status"].reshape(-1), sa["status"])
            np.testing.assert_array_equal(r["all_self_min_clearance"].reshape(-1), sa["min_clearance"])
        for k in ("all_environment_status", "all_environment_min_clearance"):
            np.testing.assert_array_equal(both[k], envonly[k])
        assert sonly["collision_channels"] == ["self"] and both["collision_channels"] == ["environment", "self"]
        assert "all_environment_status" not in sonly and sonly["environment_evaluated"] is False
        np.testing.assert_array_equal(both["all_collision_free"], (both["all_environment_status"] == FREE) & (both["all_self_status"] == FREE))
        np.testing.assert_array_equal(sonly["all_collision_free"], sonly["all_self_status"] == FREE)
        for r, esel in ((sonly, None), (both, "rank")):
            want, _ = cpu_select2(r["all_joint_config"].astype(float), r["all_position_errors"].astype(float),
                                  r["all_orientation_errors"].astype(float), E.probs_active(probs),
                                  None if esel is None else r["all_environment_status"], r["all_self_status"], 4, esel, ssel)
            np.testing.assert_array_equal(r["selected_seed_ids"], want)
            for p in range(6):
                for m in range(4):
                    s = want[p, m]
                    if s < 0:
                        assert r["selected_candidate_ids"][p, m] == -1 and r["selected_self_status"][p, m] == NOT_EVAL
                        continue
                    assert r["selected_candidate_ids"][p, m] == ids[p, s]
                    np.testing.assert_array_equal(r["joint_config"][p, m], r["all_joint_config"][p, s])
                    np.testing.assert_array_equal(r["base_quaternion"][p, m], r["all_base_quaternion"][p, s])
                    assert r["selected_self_status"][p, m] == r["all_self_status"][p, s]
        if ssel == "annotate":
            np.testing.assert_array_equal(sonly["selected_seed_ids"], off["selected_seed_ids"])
    # changing only the environment, or only the floating-base seeds' frame, leaves self status alone
    env2 = {k: v + 0.0 for k, v in env.items()}; env2["planes"] = np.asarray([[0, 0, 1, -50.0]], T)
    b2 = TB.fsolve(tm, prec, seeds, bases, probs, environment_collision="final", environment=env2, robot_spheres=tab.astype(T),
                   self_selection="annotate", **sk, **kw)
    np.testing.assert_array_equal(b2["all_self_status"], sonly["all_self_status"])


def test_mode_validation(tm, model, oracle, table, pairs):
    tab, _ = table; pr, _ = pairs
    seeds, bases, probs, _ = TB.fb_batch(model, oracle, 2, 4, seed=5)
    for bad in ("hard", "gate", "both"):
        with pytest.raises(ValueError):
            TB.fsolve(tm, "f64", seeds, bases, probs, self_collision=bad, robot_spheres=tab, self_pairs=pr)
    with pytest.raises(ValueError):
        TB.fsolve(tm, "f64", seeds, bases, probs, self_pairs=pr)                # pairs without the mode
    with pytest.raises(ValueError):
        TB.fsolve(tm, "f64", seeds, bases, probs, robot_spheres=tab)            # robot table with no channel on
    with pytest.raises(ValueError):
        TB.fsolve(tm, "f64", seeds, bases, probs, self_collision="final", robot_spheres=tab)   # no pair policy
    with pytest.raises(ValueError):
        TB.fsolve(tm, "f64", seeds, bases, probs, self_collision="final", self_pairs=pr)      # no robot table
    o = TB.fsolve(tm, "f64", seeds, bases, probs, self_collision="final", robot_spheres=tab, self_pairs=pr)
    assert o["self_selection"] == "annotate" and o["self_evaluated"] and o["all_self_status"].shape == (2, 4)
