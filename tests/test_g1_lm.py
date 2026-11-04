"""G1 multi-target residuals, Jacobians, normal equations and fixed-base LM (humanoid-v1 M3). GPU + nvcc.

The CPU oracle below is independent numpy: FK from the M2a URDF oracle (tests/urdf_fk.py), residuals
and Jacobians written from the humanoid-branch (bbc53a9) conventions:
  e_p = p* - p;  FULL e_R = Log(q* (x) q^-1) (world/spatial, short way);  AXIS e_R = R* a - R a (chord);
  NONE e_R = 0;  BOUNDED_TWIST = FULL against the nearest admissible twist of R* about d = R* a.
  J = d(prediction)/dq: Jv = a_j x (p - o_j), FULL Jw = a_j, AXIS Jw = a_j x n,
  BOUNDED Jw = a_j - (1 - slope)(k . a_j) d;  zero outside the target's ancestor mask.
  A = J^T W J, b = J^T W e, W = w * s^2 (s = 1/||J_row||), null-column pin, (A + lam diag A) dq = b.
"""
import os
import shutil

import numpy as np
import pytest

import test_g1_model as g1
from test_g1_gpu_fk import URDF, SPEC, NAME

NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
NONE, AXIS, FULL, BOUNDED = 0, 1, 2, 3


@pytest.fixture(scope="module")
def tm(tmp_path_factory):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    from hjcdik import _tree_build as tb
    store = os.environ.get("HJCD_TREE_CACHE") or str(tmp_path_factory.mktemp("tree-store"))
    return tb.import_tree(tb.build_tree(URDF, SPEC, name=NAME, cache_dir=store))


@pytest.fixture(scope="module")
def model():
    return g1.model.describe_tree_robot(g1.parse(), URDF, g1.TARGETS)


@pytest.fixture(scope="module")
def oracle():
    return g1.UrdfFK(URDF)


# ---------------------------------------------------------------- numpy oracle
def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def mat2quat(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    if w > 1e-3:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4*w), (R[0, 2] - R[2, 0]) / (4*w), (R[1, 0] - R[0, 1]) / (4*w)])
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4); q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def quat2mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def qlog_err(q_cur, q_goal):
    """log(q_goal (x) q_cur^-1) as a world rotation vector, short way."""
    if q_cur @ q_goal < 0:
        q_cur = -q_cur
    e = qmul(q_goal, q_cur * np.array([1, -1, -1, -1]))
    if e[0] < 0:
        e = -e
    v = np.linalg.norm(e[1:])
    return np.zeros(3) if v < 1e-300 else 2 * np.arctan2(v, e[0]) * e[1:] / v


def deadzone(phi, lo, hi):
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    off = (phi - centre + np.pi) % (2 * np.pi) - np.pi
    if off <= -np.pi:
        off = np.pi
    out = abs(off) - half
    return (0.0, 0.0) if not out > 0 else ((out if off > 0 else -out), 1.0)


def twist_aux(R, qt, a, tw):
    d, t, h = quat2mat(qt) @ a, quat2mat(qt) @ tw[:3], R @ tw[:3]
    b = np.cross(d, t)
    u, w = b @ h, t @ h
    v, slope = deadzone(np.arctan2(u, w), tw[3], tw[4])
    sc = u*u + w*w
    kdir = np.zeros(3) if not sc > 1e-12 else (u * np.cross(t, h) - w * np.cross(b, h)) / sc
    return d, kdir, slope, np.arctan2(u, w), v


def residual(T, p, qt, mode, a, tw):
    ep = p - T[:3, 3]
    R = T[:3, :3]
    if mode == NONE:
        return ep, np.zeros(3), 0.0
    if mode == AXIS:
        eR = quat2mat(qt) @ a - R @ a
        return ep, eR, 2 * np.arcsin(min(np.linalg.norm(eR) / 2, 1.0))
    if mode == BOUNDED:
        d, _, _, phi, v = twist_aux(R, qt, a, tw)
        hp = 0.5 * (phi - v)
        qeff = qmul(np.r_[np.cos(hp), np.sin(hp) * d], qt)
        eR = qlog_err(mat2quat(R), qeff)
        return ep, eR, np.linalg.norm(eR)
    eR = qlog_err(mat2quat(R), qt)
    return ep, eR, np.linalg.norm(eR)


def tools_of(model, tool=None):
    return np.array([t["tool_row_major"] for t in model["targets"]]) if tool is None else np.asarray(tool)


def oracle_eval(oracle, model, q, prob, tool=None, lam=5e-3):
    tgt, act, wp, wo, mode, axis, tw = prob
    Tj = oracle.joint_world_transforms(q)
    tools = tools_of(model, tool)
    n = len(q)
    out = dict(pose=[], e=[], pn=[], on=[], J=np.zeros((24, n)))
    for k, t in enumerate(model["targets"]):
        T = Tj[t["anchor_jid"]] @ tools[k]
        out["pose"].append(T)
        if not (act >> k) & 1:
            out["e"].append(np.zeros(6)); out["pn"].append(0.0); out["on"].append(0.0)
        else:
            ep, eR, ang = residual(T, tgt[k, :3], tgt[k, 3:], mode[k], axis[k], tw[k])
            out["e"].append(np.r_[ep, eR]); out["pn"].append(np.linalg.norm(ep)); out["on"].append(ang)
        R, p = T[:3, :3], T[:3, 3]
        nk = R @ axis[k]
        aux = twist_aux(R, tgt[k, 3:], axis[k], tw[k]) if mode[k] == BOUNDED else None
        for j in range(n):
            if not (t["ancestor_mask"] >> j) & 1:
                continue
            ax = Tj[j][:3, :3] @ np.asarray(model["joints"][j]["axis"])
            jv = np.cross(ax, p - Tj[j][:3, 3])
            if mode[k] == AXIS:
                jo = np.cross(ax, nk)
            elif mode[k] == BOUNDED:
                d, kd, sl = aux[0], aux[1], aux[2]
                jo = ax - (1 - sl) * (kd @ ax) * d
            else:
                jo = ax
            out["J"][6*k:6*k+3, j] = jv
            out["J"][6*k+3:6*k+6, j] = jo
    E = np.concatenate(out["e"])
    J = out["J"]
    w = np.zeros(24); wraw = np.zeros(24)
    scale = np.ones(24)
    for k in range(4):
        if not (act >> k) & 1:
            continue
        wo_k = 0.0 if mode[k] == NONE else wo[k]
        for r in range(6):
            nrm2 = J[6*k + r] @ J[6*k + r]
            scale[6*k + r] = 1 / np.sqrt(nrm2) if nrm2 > 1e-18 else 1.0
            wraw[6*k + r] = wp[k] if r < 3 else wo_k
    w = wraw * scale**2
    A_raw, b_raw = J.T @ (wraw[:, None] * J), J.T @ (wraw * E)
    A, b = J.T @ (w[:, None] * J), J.T @ (w * E)
    col = np.array([np.sum(wraw * J[:, j]**2) for j in range(n)])
    act_j = 0
    for k in range(4):
        if (act >> k) & 1:
            act_j |= model["targets"][k]["ancestor_mask"]
    anc = np.array([(act_j >> j) & 1 for j in range(n)], bool)
    cmax = col[anc].max() if anc.any() else 0.0
    pin = ~anc | (anc & (cmax > 0) & (col <= 1e-12 * cmax))
    Ap, bp = A.copy(), b.copy()
    Ap[pin, :] = 0; Ap[:, pin] = 0; Ap[pin, pin] = 1; bp[pin] = 0
    dq = np.linalg.solve(Ap + lam * np.diag(np.diag(Ap)), bp)
    out.update(E=E, A_raw=A_raw, b_raw=b_raw, A=A, b=b, scale=scale, colnorm2=col, pin=pin, Ap=Ap, bp=bp, dq=dq,
               pose=np.array(out["pose"]), pn=np.array(out["pn"]), on=np.array(out["on"]))
    return out


# ---------------------------------------------------------------- problem builders
def limits(model):
    return np.array([j["limits"] for j in model["joints"]])


def rand_quat(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    return q * (1 if q[0] >= 0 else -1)


def make_problem(rng, model, oracle, q_true, active, modes, noise_p=0.05, noise_r=0.3, wp=None, wo=None, tool=None):
    """Targets near FK(q_true) (noise 0 => exactly reachable)."""
    Tj = oracle.joint_world_transforms(q_true)
    tools = tools_of(model, tool)
    tgt = np.zeros((4, 7)); axis = np.zeros((4, 3)); tw = np.zeros((4, 5))
    for k, t in enumerate(model["targets"]):
        T = Tj[t["anchor_jid"]] @ tools[k]
        rv = rng.normal(size=3) * noise_r
        ang = np.linalg.norm(rv)
        dqn = np.r_[np.cos(ang / 2), np.sin(ang / 2) * rv / ang] if ang > 0 else np.r_[1.0, 0, 0, 0]
        qt = qmul(dqn, mat2quat(T[:3, :3]))
        tgt[k] = np.r_[T[:3, 3] + rng.normal(size=3) * noise_p, qt / np.linalg.norm(qt)]
        axis[k] = [0, 0, 1]
        tw[k] = [1, 0, 0, -0.4, 0.4]
    return (tgt, np.uint32(active), np.ones(4) if wp is None else np.asarray(wp, float),
            np.ones(4) if wo is None else np.asarray(wo, float), np.asarray(modes, np.int32), axis, tw)


def stack(probs):
    return [np.ascontiguousarray(np.stack([p[i] for p in probs])) for i in range(7)]


CASES = [  # (active mask, modes)
    (0b1111, [FULL, FULL, FULL, FULL]), (0b0001, [FULL, NONE, NONE, NONE]), (0b0011, [AXIS, AXIS, FULL, FULL]),
    (0b1100, [NONE, NONE, FULL, AXIS]), (0b0101, [BOUNDED, FULL, AXIS, NONE]), (0b1010, [FULL, BOUNDED, NONE, FULL]),
    (0b1111, [AXIS, BOUNDED, NONE, AXIS]), (0b1111, [NONE, NONE, NONE, NONE]), (0b0110, [FULL, AXIS, BOUNDED, FULL]),
]


@pytest.fixture(scope="module")
def corpus(model, oracle):
    rng = np.random.default_rng(31)
    lo, hi = limits(model).T
    qs, probs = [], []
    for i in range(72):
        active, modes = CASES[i % len(CASES)]
        q = rng.uniform(lo, hi) * 0.6
        wp = rng.choice([0.0, 0.5, 1.0, 2.0], size=4) if i % 5 == 4 else None
        probs.append(make_problem(rng, model, oracle, q, active, modes, wp=wp))
        qs.append(q + rng.normal(size=29) * 0.05)
    return np.clip(np.array(qs), lo, hi), probs


def run_eval(tm, prec, Q, probs, tool=None):
    T = np.float64 if prec == "f64" else np.float32
    tgt, act, wp, wo, mode, axis, tw = stack(probs)
    f = getattr(tm, f"evaluate_{prec}")
    return f(Q.astype(T), tgt.astype(T), act.astype(np.uint32), wp.astype(T), wo.astype(T), mode, axis.astype(T),
             twist=tw.astype(T), tool=None if tool is None else np.asarray(tool, T))


TOL = {"f64": dict(e=1e-12, J=1e-12, A=1e-10, dq=1e-8), "f32": dict(e=5e-5, J=5e-5, A=5e-4, dq=None)}


def rel(a, b):
    return np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b)))


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_residuals_jacobians_normal_equations(tm, model, oracle, corpus, prec):
    Q, probs = corpus
    g = run_eval(tm, prec, Q, probs)
    t = TOL[prec]
    worst = dict(e=0, J=0, A=0, b=0, dq=0, A_raw=0)
    for i, (q, pr) in enumerate(zip(Q, probs)):
        o = oracle_eval(oracle, model, q.astype(np.float64), pr)
        worst["e"] = max(worst["e"], np.max(np.abs(g["e"][i].reshape(-1) - o["E"])))
        assert np.allclose(g["e"][i].reshape(-1), o["E"], atol=t["e"], rtol=0), i
        np.testing.assert_allclose(g["pn"][i], o["pn"], atol=t["e"])
        np.testing.assert_allclose(g["on"][i], o["on"], atol=max(t["e"], 2e-4 if prec == "f32" else 0))
        worst["J"] = max(worst["J"], rel(g["J"][i], o["J"]))
        assert rel(g["J"][i], o["J"]) <= t["J"], (i, rel(g["J"][i], o["J"]))
        for key in ("A_raw", "A"):
            worst[key] = max(worst[key], rel(g[key][i], o[key]))
            assert rel(g[key][i], o[key]) <= t["A"], (i, key, rel(g[key][i], o[key]))
        worst["b"] = max(worst["b"], rel(g["b"][i], o["b"]))
        assert rel(g["b"][i], o["b"]) <= t["A"]
        pinned = np.array([(int(g["pinned"][i]) >> j) & 1 for j in range(29)], bool)
        if prec == "f64":
            np.testing.assert_array_equal(pinned, o["pin"])
            assert g["fail"][i] == 0
            worst["dq"] = max(worst["dq"], rel(g["dq"][i], o["dq"]))
            assert rel(g["dq"][i], o["dq"]) <= t["dq"], (i, rel(g["dq"][i], o["dq"]))
        assert np.all(np.isfinite(g["dq"][i]))
    print(prec, {k: f"{v:.2e}" for k, v in worst.items()})


def test_jacobian_matches_finite_differences(tm, model, oracle):
    """d(prediction)/dq by central differences of the CPU FK: Jv vs dp, FULL vs angular velocity,
    AXIS vs dn; BOUNDED at zero residual inside the arc vs -de_R/dq."""
    rng = np.random.default_rng(5)
    lo, hi = limits(model).T
    worst = 0.0
    for trial in range(6):
        q = rng.uniform(lo, hi) * 0.6
        modes = [FULL, AXIS, BOUNDED, FULL] if trial % 2 else [AXIS, FULL, FULL, BOUNDED]
        pr = make_problem(rng, model, oracle, q, 0b1111, modes, noise_p=0.0, noise_r=0.0)
        g = run_eval(tm, "f64", q[None], [pr])
        J = g["J"][0]
        h = 1e-6
        tools = tools_of(model)
        for j in range(29):
            dqv = np.zeros(29); dqv[j] = h
            Tp, Tm = oracle.joint_world_transforms(q + dqv), oracle.joint_world_transforms(q - dqv)
            for k, t in enumerate(model["targets"]):
                A_, B_ = Tp[t["anchor_jid"]] @ tools[k], Tm[t["anchor_jid"]] @ tools[k]
                fd_p = (A_[:3, 3] - B_[:3, 3]) / (2 * h)
                W = (A_[:3, :3] - B_[:3, :3]) / (2 * h) @ (0.5 * (A_[:3, :3] + B_[:3, :3])).T
                fd_w = np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]) / 2
                if modes[k] == AXIS:
                    fd_o = (A_[:3, :3] @ pr[5][k] - B_[:3, :3] @ pr[5][k]) / (2 * h)
                elif modes[k] == BOUNDED:
                    ep, eR_p, _ = residual(A_, pr[0][k, :3], pr[0][k, 3:], BOUNDED, pr[5][k], pr[6][k])
                    ep, eR_m, _ = residual(B_, pr[0][k, :3], pr[0][k, 3:], BOUNDED, pr[5][k], pr[6][k])
                    fd_o = -(eR_p - eR_m) / (2 * h)
                else:
                    fd_o = fd_w
                err = max(np.max(np.abs(J[6*k:6*k+3, j] - fd_p)), np.max(np.abs(J[6*k+3:6*k+6, j] - fd_o)))
                worst = max(worst, err)
                assert err <= 2e-8, (trial, j, k, modes[k], err)
    print("fd worst abs", worst)


def test_ancestor_masking_in_jacobian(tm, model, corpus):
    Q, probs = corpus
    g = run_eval(tm, "f64", Q[:9], probs[:9])
    for i in range(9):
        for k, t in enumerate(model["targets"]):
            for j in range(29):
                if not (t["ancestor_mask"] >> j) & 1:
                    assert np.all(g["J"][i][6*k:6*k+6, j] == 0.0)


def test_inactive_and_zero_weight_targets_contribute_nothing(tm, model, oracle, corpus):
    Q, probs = corpus
    tgt, act, wp, wo, mode, axis, tw = [p.copy() for p in probs[0]]
    base = run_eval(tm, "f64", Q[:1], [(tgt, np.uint32(0b0011), wp, wo, mode, axis, tw)])
    far = tgt.copy(); far[2:, :3] += 10.0                       # inactive feet moved far away
    moved = run_eval(tm, "f64", Q[:1], [(far, np.uint32(0b0011), wp, wo, mode, axis, tw)])
    for key in ("A", "b", "A_raw", "b_raw", "dq"):
        np.testing.assert_array_equal(base[key], moved[key])
    zw, zo = wp.copy(), wo.copy(); zw[1] = zo[1] = 0.0          # right hand weightless == inactive
    a = run_eval(tm, "f64", Q[:1], [(tgt, np.uint32(0b0011), zw, zo, mode, axis, tw)])
    b = run_eval(tm, "f64", Q[:1], [(tgt, np.uint32(0b0001), wp, wo, mode, axis, tw)])
    for key in ("A_raw", "b_raw"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-14)
    nm = mode.copy(); nm[0] = NONE                              # NONE: exactly no orientation term
    c = run_eval(tm, "f64", Q[:1], [(tgt, np.uint32(0b0001), wp, wo, nm, axis, tw)])
    assert np.all(c["e"][0, 0, 3:] == 0) and c["on"][0, 0] == 0


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_tool_override_replaces_model_tool(tm, model, oracle, prec):
    rng = np.random.default_rng(9)
    q = rng.uniform(*limits(model).T) * 0.5
    tools = tools_of(model).copy()
    c, s = np.cos(0.7), np.sin(0.7)
    tools[2] = np.array([[c, -s, 0, 0.11], [s, c, 0, 0.0], [0, 0, 1, -0.03], [0, 0, 0, 1]])   # rotated + translated
    tools[0] = np.array([[1, 0, 0, 0.165], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])        # translated
    pr = make_problem(rng, model, oracle, q, 0b1111, [FULL, AXIS, FULL, BOUNDED])
    T = np.float64 if prec == "f64" else np.float32
    g_over = run_eval(tm, prec, q[None], [pr], tool=tools)
    g_def = run_eval(tm, prec, q[None], [pr])
    again = run_eval(tm, prec, q[None], [pr])                  # a later call without override is unaffected
    np.testing.assert_array_equal(g_def["pose"], again["pose"])
    o = oracle_eval(oracle, model, q, pr, tool=tools)
    tol = 1e-12 if prec == "f64" else 5e-5
    np.testing.assert_allclose(g_over["pose"][0], o["pose"], atol=tol)
    np.testing.assert_allclose(g_over["e"][0].reshape(-1), o["E"], atol=tol)
    assert rel(g_over["J"][0], o["J"]) <= (1e-12 if prec == "f64" else 5e-5)
    Tj = oracle.joint_world_transforms(q)                      # replaces (not composes with) the model tool
    np.testing.assert_allclose(g_over["pose"][0, 2], Tj[5] @ tools[2], atol=tol)
    with pytest.raises(ValueError, match="orthonormal"):
        bad = tools.copy(); bad[1][:3, :3] *= 1.1
        run_eval(tm, prec, q[None], [pr], tool=bad)


def test_null_columns_axis_hand(tm, model, oracle):
    """bbc53a9 regression: an AXIS hand whose axis is the wrist-yaw axis gives that joint a zero column
    (the target sits on the joint origin, jw x n = 0). It must be pinned, not solved, and nothing is NaN."""
    rng = np.random.default_rng(2)
    q = rng.uniform(*limits(model).T) * 0.4
    pr = list(make_problem(rng, model, oracle, q, 0b0001, [AXIS, NONE, NONE, NONE]))
    pr[5] = pr[5].copy(); pr[5][0] = model["joints"][21]["axis"]
    g = run_eval(tm, "f64", q[None], [tuple(pr)])
    g32 = run_eval(tm, "f32", q[None], [tuple(pr)])
    o = oracle_eval(oracle, model, q, tuple(pr))
    assert np.all(g["J"][0][:, 21] == 0) or np.max(np.abs(g["J"][0][:, 21])) < 1e-15
    for gg in (g, g32):
        pinned = int(gg["pinned"][0])
        assert (pinned >> 21) & 1 and all((pinned >> j) & 1 for j in range(29) if not (0x3FF000 >> j) & 1)
        assert gg["fail"][0] == 0 and np.all(np.isfinite(gg["dq"][0]))
        assert gg["dq"][0][21] == 0 and np.all(gg["dq"][0][[j for j in range(29) if not (0x3FF000 >> j) & 1]] == 0)
    np.testing.assert_array_equal(np.array([(int(g["pinned"][0]) >> j) & 1 for j in range(29)], bool), o["pin"])


# ---------------------------------------------------------------- LM
def lm_corpus(model, oracle, n=96, seed=11, noise=0.15):
    rng = np.random.default_rng(seed)
    lo, hi = limits(model).T
    Q0, probs, Qtrue = [], [], []
    for i in range(n):
        active, modes = CASES[i % len(CASES)]
        if active and all(m == NONE for m in modes):
            modes = [FULL] * 4
        q = lo + (hi - lo) * rng.uniform(0.2, 0.8, 29)
        probs.append(make_problem(rng, model, oracle, q, active, modes, noise_p=0.0, noise_r=0.0))
        Qtrue.append(q)
        Q0.append(np.clip(q + rng.normal(size=29) * noise, lo, hi))
    return np.array(Q0), probs, np.array(Qtrue)


def run_lm(tm, prec, Q0, probs, **kw):
    T = np.float64 if prec == "f64" else np.float32
    tgt, act, wp, wo, mode, axis, tw = stack(probs)
    return getattr(tm, f"lm_{prec}")(Q0.astype(T), tgt.astype(T), act.astype(np.uint32), wp.astype(T), wo.astype(T),
                                     mode, axis.astype(T), twist=tw.astype(T), **kw)


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_lm_known_feasible(tm, model, oracle, prec):
    Q0, probs, _ = lm_corpus(model, oracle)
    r = run_lm(tm, prec, Q0, probs, k_max=60, stag_patience=0)
    lo, hi = limits(model).T
    assert np.all(np.isfinite(r["x"]))
    if prec == "f64":
        assert np.all(r["solve_fail"] == 0)
    else:   # fp32: posv may flag a non-PD system once lambda reaches 1e-12 near convergence; the step is
        # rejected (dq = 0) and lambda grows, so it must never cost convergence
        assert np.all(r["success"][r["solve_fail"] > 0])
    assert np.all(r["x"] >= lo - 1e-6) and np.all(r["x"] <= hi + 1e-6)
    # reported success == independently re-checked contact exactness (per active, weighted component)
    for i, pr in enumerate(probs):
        o = oracle_eval(oracle, model, r["x"][i].astype(np.float64), pr)
        act = pr[1]
        exact = all(o["pn"][k] <= 1e-4 + 1e-6 and (pr[4][k] == NONE or o["on"][k] <= 1e-3 + 1e-5)
                    for k in range(4) if (act >> k) & 1)
        assert bool(r["success"][i]) == exact or not r["success"][i], i
    rate = float(np.mean(r["success"]))
    print(prec, "LM success", rate, "iters median", np.median(r["iters"]))
    assert rate >= 0.9, rate


def test_lm_matches_old_humanoid_outcomes(tm, model, oracle):
    """Paired fixed-base comparison against bbc53a9 (results recorded by temp/humanoid_v1_m3/compare/run_old.py in the
    baseline clone's venv). Skips unless HJCD_OLD_LM_RESULTS points at old.npz + corpus.npz."""
    path = os.environ.get("HJCD_OLD_LM_RESULTS")
    if not path:
        pytest.skip("HJCD_OLD_LM_RESULTS not set")
    old, c = np.load(os.path.join(path, "old.npz")), np.load(os.path.join(path, "corpus.npz"))
    for name in ("easy", "hard"):
        st = [c[f"{name}_{k}"] for k in ("tgt", "act", "wp", "wo", "mode", "axis", "tw")]
        for prec, oprec in (("f64", "float64"), ("f32", "float32")):
            T = np.float64 if prec == "f64" else np.float32
            r = getattr(tm, f"lm_{prec}")(c[f"{name}_q0"].astype(T), st[0].astype(T), st[1].astype(np.uint32),
                                           st[2].astype(T), st[3].astype(T), st[4], st[5].astype(T),
                                           twist=st[6].astype(T), k_max=60, stag_patience=0)
            np.testing.assert_array_equal(r["success"], old[f"{name}_{oprec}_success"].astype(bool), err_msg=f"{name} {prec}")


def test_lm_keeps_unrelated_joints_fixed(tm, model, oracle):
    Q0, probs, _ = lm_corpus(model, oracle, n=9)
    feet = [i for i, p in enumerate(probs) if p[1] == 0b1100]
    r = run_lm(tm, "f64", Q0[feet], [probs[i] for i in feet], k_max=60, stag_patience=0)
    for row, i in enumerate(feet):
        if r["kicks"][row] == 0:   # the bbc53a9 stall kick deliberately perturbs every joint
            np.testing.assert_array_equal(r["x"][row][12:], Q0[i][12:])     # waist + arms untouched by feet-only
