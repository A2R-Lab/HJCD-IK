"""Batched fixed-base multi-target solve: P problems x S seeds, segmented top-M (humanoid-v1 M4a). GPU + nvcc.

Selection contract (port of bbc53a9 segmented_topM_kernel, no collision channel):
  class 0: finite q and every ACTIVE target's reported errors within (position_tol, orientation_tol)
  class 1: finite, not within;  class 2: non-finite q or error
  key = (class, E_phys (double), seed index); M rounds of masked argmin -> distinct seeds;
  slots whose winner is class 2 are pads: seed -1, errors +inf, valid False, q = first seed (zeros if non-finite).
"""
import numpy as np
import pytest

import test_g1_lm as L
import test_g1_model as g1
from test_g1_lm import tm, model, oracle  # noqa: F401  (fixtures)

PREC = {"f64": np.float64, "f32": np.float32}


def build_batch(model, oracle, P, S, seed=0, unreachable=(), tools=False, noise=0.2):
    """P problems (mixed masks/modes/weights; some near limits; optional per-problem tool overrides)."""
    rng = np.random.default_rng(seed)
    lo, hi = L.limits(model).T
    probs, seeds = [], []
    base_tools = L.tools_of(model)
    tool_arr = np.repeat(base_tools[None], P, axis=0)
    for p in range(P):
        active, modes = L.CASES[p % len(L.CASES)]
        if all(m == L.NONE for m in modes):
            modes = [L.FULL] * 4
        q = lo + (hi - lo) * rng.uniform(0.15, 0.85, 29)
        if p % 7 == 3:
            q = lo + (hi - lo) * np.where(rng.uniform(size=29) < 0.5, 0.02, 0.98)       # near joint limits
        wp = rng.choice([0.5, 1.0, 2.0], size=4) if p % 4 == 1 else None
        if tools and p % 3 == 2:
            c, s = np.cos(0.3 * p), np.sin(0.3 * p)
            tool_arr[p, 2] = np.array([[c, -s, 0, 0.1], [s, c, 0, 0.0], [0, 0, 1, -0.03], [0, 0, 0, 1]])
        pr = list(L.make_problem(rng, model, oracle, q, active, modes, noise_p=0.0, noise_r=0.0, wp=wp,
                                 tool=tool_arr[p]))
        if p in unreachable:
            pr[0] = pr[0].copy(); pr[0][:, :3] += 3.0                                 # 3 m away: no solution
        if p % 9 == 5:                                                               # rank-deficient AXIS hand
            pr[1] = np.uint32(0b0001); pr[4] = np.array([L.AXIS, L.NONE, L.NONE, L.NONE], np.int32)
            pr[5] = pr[5].copy(); pr[5][0] = model["joints"][21]["axis"]
        probs.append(tuple(pr))
        seeds.append(np.clip(q + rng.normal(size=(S, 29)) * noise, lo, hi))
    return np.array(seeds), probs, (tool_arr if tools else None)


def solve(tm, prec, seeds, probs, tools=None, M=1, ids=None, **kw):
    T = PREC[prec]
    tgt, act, wp, wo, mode, axis, tw = L.stack(probs)
    return getattr(tm, f"solve_problems_{prec}")(
        np.ascontiguousarray(seeds, T), tgt.astype(T), act.astype(np.uint32), wp.astype(T), wo.astype(T), mode,
        axis.astype(T), twist_spec=tw.astype(T), tool_override=None if tools is None else tools.astype(T),
        num_solutions=M, candidate_ids=ids, lm_iters=60, stag_patience=0, return_all_candidates=True, **kw)


RAW = ("all_joint_config", "all_position_errors", "all_orientation_errors", "all_cost_lm", "all_lm_converged",
       "all_iterations", "all_solve_failures", "all_kicks")
SEL = ("joint_config", "position_errors", "orientation_errors", "selected_seed_ids", "success", "valid", "cost_physical")


def take(r, idx, keys):
    return {k: np.asarray(r[k])[idx] for k in keys}


def assert_same(a, b, what, keys=None):
    for k in (keys or a.keys()):
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=f"{what}: {k}")


@pytest.fixture(scope="module")
def batch(model, oracle):
    return build_batch(model, oracle, P=12, S=16, seed=4, unreachable=(7,), tools=True)


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_batched_equals_independent_and_m3(tm, model, oracle, batch, prec):
    """Test A: each problem solved alone (P=1, all its seeds) == the same problem inside the batch, bitwise;
    and the per-seed results equal M3's standalone lm_* on the same rows."""
    seeds, probs, tools = batch
    full = solve(tm, prec, seeds, probs, tools, M=4)
    for p in range(len(probs)):
        one = solve(tm, prec, seeds[p:p+1], probs[p:p+1], None if tools is None else tools[p:p+1], M=4)
        assert_same(take(full, slice(p, p+1), RAW + SEL), take(one, slice(0, 1), RAW + SEL), f"{prec} problem {p}")
    T = PREC[prec]
    rows = [(p, s) for p in range(len(probs)) for s in range(seeds.shape[1])]
    no_tool = [p for p in range(len(probs)) if tools is None or np.array_equal(tools[p], L.tools_of(model))]
    sub = [(p, s) for p, s in rows if p in no_tool]
    st = L.stack([probs[p] for p, _ in sub])
    m3 = getattr(tm, f"lm_{prec}")(np.array([seeds[p, s] for p, s in sub], T), st[0].astype(T), st[1].astype(np.uint32),
                                    st[2].astype(T), st[3].astype(T), st[4], st[5].astype(T), twist=st[6].astype(T),
                                    k_max=60, stag_patience=0)
    for i, (p, s) in enumerate(sub):
        np.testing.assert_array_equal(m3["x"][i], full["all_joint_config"][p, s])
        np.testing.assert_array_equal(m3["success"][i], full["all_lm_converged"][p, s])


def test_problem_permutation(tm, batch):
    """Test B: permuting problems permutes results exactly; identities follow."""
    seeds, probs, tools = batch
    ids = np.arange(seeds.shape[0] * seeds.shape[1], dtype=np.int64).reshape(seeds.shape[:2]) * 7 + 1000
    a = solve(tm, "f64", seeds, probs, tools, M=3, ids=ids)
    perm = np.random.default_rng(1).permutation(len(probs))
    b = solve(tm, "f64", seeds[perm], [probs[i] for i in perm], tools[perm], M=3, ids=ids[perm])
    assert_same(take(a, perm, RAW + SEL + ("selected_candidate_ids", "num_solved", "num_valid", "num_converged")),
                take(b, slice(None), RAW + SEL + ("selected_candidate_ids", "num_solved", "num_valid", "num_converged")),
                "permutation")


def test_batch_partitioning(tm, batch):
    """Test C: one batch == concatenation of smaller batches over the same problems."""
    seeds, probs, tools = batch
    a = solve(tm, "f32", seeds, probs, tools, M=2)
    parts = [slice(0, 5), slice(5, 6), slice(6, 12)]
    for sl in parts:
        b = solve(tm, "f32", seeds[sl], probs[sl], tools[sl], M=2)
        assert_same(take(a, sl, RAW + SEL), take(b, slice(None), RAW + SEL), f"partition {sl}")


def test_input_isolation(tm, batch):
    """Test D: perturbing one problem's targets, weights and tool leaves every other problem's raw results unchanged."""
    seeds, probs, tools = batch
    a = solve(tm, "f64", seeds, probs, tools, M=2)
    probs2 = list(probs); tools2 = tools.copy()
    tgt, act, wp, wo, mode, axis, tw = probs[4]
    tgt = tgt.copy(); tgt[:, :3] += 0.05
    probs2[4] = (tgt, act, wp * 0.5, wo, mode, axis, tw)
    tools2[4, 0, :3, 3] += 0.02
    b = solve(tm, "f64", seeds, probs2, tools2, M=2)
    others = [p for p in range(len(probs)) if p != 4]
    assert_same(take(a, others, RAW + SEL), take(b, others, RAW + SEL), "isolation")
    assert not np.array_equal(a["all_joint_config"][4], b["all_joint_config"][4])


def cpu_select(q, pe, oe, active, M, eps_p, eps_o, seeds):
    """Independent CPU reference of the documented selection policy (no GPU sort reused)."""
    P, S = q.shape[:2]
    out = dict(seed=np.full((P, M), -1), exact=np.zeros((P, M), bool), valid=np.zeros((P, M), bool),
               ns=np.zeros(P, int), nv=np.zeros(P, int), q=np.zeros((P, M, q.shape[2])))
    for p in range(P):
        keys = []
        for s in range(S):
            fin = bool(np.all(np.isfinite(q[p, s].astype(np.float64))))
            eph, within = 0.0, True
            for k in range(4):
                if not (int(active[p]) >> k) & 1:
                    continue
                ep, eo = float(pe[p, s, k]), float(oe[p, s, k])
                if not (np.isfinite(ep) and np.isfinite(eo)):
                    fin = False
                    continue
                eph += (ep / eps_p) ** 2 + (eo / eps_o) ** 2
                within &= ep <= eps_p and eo <= eps_o
            cls = 2 if not fin else (0 if within else 1)
            keys.append((cls, np.inf if cls == 2 else eph, s))
        out["ns"][p] = sum(k[0] == 0 for k in keys); out["nv"][p] = sum(k[0] <= 1 for k in keys)
        ranked = sorted(keys)
        for m in range(M):
            cls, eph, s = ranked[m]
            if cls == 2:
                q0 = seeds[p, 0]
                out["q"][p, m] = q0 if np.all(np.isfinite(q0)) else 0.0
                continue
            out["seed"][p, m] = s; out["exact"][p, m] = cls == 0; out["valid"][p, m] = True; out["q"][p, m] = q[p, s]
    return out


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_topM_matches_cpu_reference_with_ties_and_invalid(tm, prec):
    """Test G + F: random pools with exact ties, NaN/inf candidates, all-invalid and zero-success problems."""
    T = PREC[prec]
    rng = np.random.default_rng(17)
    P, S = 9, 37
    q = rng.normal(size=(P, S, 29)).astype(T)
    pe = (10 ** rng.uniform(-6, -2, (P, S, 4))).astype(T)
    oe = (10 ** rng.uniform(-5, -1, (P, S, 4))).astype(T)
    pe[0, :, :] = T(5e-5); oe[0, :, :] = T(5e-4)                    # problem 0: all exact ties -> seed order
    pe[1, 3], pe[1, 9] = pe[1, 20], pe[1, 20]; oe[1, 3], oe[1, 9] = oe[1, 20], oe[1, 20]  # 3-way tie
    q[2, ::2, 5] = np.nan                                           # problem 2: half non-finite q
    pe[3, :, 1] = np.inf                                            # problem 3: every candidate invalid
    q[4] = np.nan; seeds4 = None                                    # problem 4: all NaN q (pad = first seed)
    pe[5] *= T(100); oe[5] *= T(100)                                # problem 5: no exact candidate
    active = np.array([15, 1, 3, 2, 15, 12, 5, 10, 15], np.uint32)
    seeds = rng.normal(size=(P, S, 29)).astype(T)
    seeds[6, 0, 0] = np.nan                                         # (selection-only call may carry a NaN seed)
    cost = rng.uniform(size=(P, S)).astype(T)
    succ = (rng.uniform(size=(P, S)) < 0.5).astype(np.uint8)
    for M in (1, 4, S):
        g = getattr(tm, f"select_{prec}")(q, pe, oe, cost, succ, seeds, active, M)
        c = cpu_select(q, pe, oe, active, M, 1e-4, 1e-3, seeds)
        np.testing.assert_array_equal(g["selected_seed_ids"], c["seed"], err_msg=f"M={M}")
        np.testing.assert_array_equal(g["success"], c["exact"])
        np.testing.assert_array_equal(g["valid"], c["valid"])
        np.testing.assert_array_equal(g["num_solved"], c["ns"]); np.testing.assert_array_equal(g["num_valid"], c["nv"])
        np.testing.assert_array_equal(g["joint_config"], c["q"].astype(T))
        assert np.all(np.isfinite(g["joint_config"]))                     # pads never carry NaN
        sel = g["selected_seed_ids"]
        for p in range(P):                                               # distinct real seeds per problem
            real = sel[p][sel[p] >= 0]
            assert len(set(real.tolist())) == len(real)
            assert np.all(np.isinf(g["position_errors"][p][sel[p] < 0]))
    assert list(g["selected_seed_ids"][0][:5]) == [0, 1, 2, 3, 4]        # exact ties -> seed order
    assert g["num_solved"][3] == 0 and g["num_valid"][3] == 0 and not g["problem_success"][3]


def test_identity_and_zero_success_in_solve(tm, model, oracle):
    """Test E + F on the real solver: candidate ids map to the originating seed; an unreachable problem
    reports zero solved/converged, no fabricated success, finite outputs."""
    seeds, probs, tools = build_batch(model, oracle, P=6, S=24, seed=9, unreachable=(2,))
    ids = (np.arange(6 * 24, dtype=np.int64).reshape(6, 24) * 31 + 5)[:, ::-1].copy()
    r = solve(tm, "f64", seeds, probs, None, M=5, ids=ids)
    for p in range(6):
        for m in range(5):
            s = r["selected_seed_ids"][p, m]
            if s < 0:
                assert r["selected_candidate_ids"][p, m] == -1 and not r["valid"][p, m]
                continue
            assert r["selected_candidate_ids"][p, m] == ids[p, s]
            np.testing.assert_array_equal(r["joint_config"][p, m], r["all_joint_config"][p, s])
            np.testing.assert_array_equal(r["position_errors"][p, m], r["all_position_errors"][p, s])
            assert r["selected_iterations"][p, m] == r["all_iterations"][p, s]
        # the documented order holds on the real pool
        cls = [0 if r["success"][p, m] else 1 for m in range(5) if r["valid"][p, m]]
        assert cls == sorted(cls)
        eph = r["cost_physical"][p][r["valid"][p]]
        same = [r["success"][p, m] for m in range(5) if r["valid"][p, m]]
        for i in range(len(eph) - 1):
            if same[i] == same[i + 1]:
                assert eph[i] <= eph[i + 1]
    assert r["num_solved"][2] == 0 and r["num_converged"][2] == 0 and not r["problem_success"][2]
    assert not np.any(r["success"][2]) and np.all(r["valid"][2])          # finite but unsolved: class 1, real seeds
    assert np.all(np.isfinite(r["joint_config"]))
    lo, hi = L.limits(model).T
    assert np.all(r["all_joint_config"] >= lo - 1e-9) and np.all(r["all_joint_config"] <= hi + 1e-9)
    # reported exactness agrees with the independent oracle on every returned slot
    for p in range(6):
        for m in range(5):
            if r["valid"][p, m]:
                o = L.oracle_eval(oracle, model, r["joint_config"][p, m], probs[p])
                act = int(probs[p][1])
                exact = all(o["pn"][k] <= 1e-4 * (1 + 1e-9) + 1e-12 and o["on"][k] <= 1e-3 * (1 + 1e-9) + 1e-12
                            for k in range(4) if (act >> k) & 1)
                assert bool(r["success"][p, m]) == exact, (p, m)


def test_repeatability_and_validation(tm, batch):
    seeds, probs, tools = batch
    a = solve(tm, "f32", seeds, probs, tools, M=2)
    b = solve(tm, "f32", seeds, probs, tools, M=2)
    assert_same(a, b, "repeat", RAW + SEL)
    s0 = seeds.copy()
    solve(tm, "f32", seeds, probs, tools, M=2)
    np.testing.assert_array_equal(seeds, s0)                              # inputs never modified
    bad = seeds.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve(tm, "f32", bad, probs, tools)
    with pytest.raises(ValueError, match="num_solutions"):
        solve(tm, "f32", seeds, probs, tools, M=seeds.shape[1] + 1)
    with pytest.raises(ValueError, match="candidate_ids"):
        solve(tm, "f32", seeds, probs, tools, ids=np.zeros((2, 2), np.int64))
