"""humanoid-v1 M6a: environment collision on the G1 tree path (GRiD 0.6.0 grid_collision primitives).

Independent references (no CUDA collision function is its own oracle):
  * sphere world centres: the M2a numpy URDF FK (tests/urdf_fk.py) on the sphere's own LINK, composed with the
    candidate base in numpy (never through the anchor table the kernel uses);
  * clearances: closed-form numpy sphere/sphere, sphere/segment, sphere/oriented-box, sphere/half-space;
  * selection: a numpy re-implementation of the documented ranking.
"""
from pathlib import Path

import numpy as np
import pytest

import test_g1_lm as L
import test_g1_base as TB
from test_g1_lm import tm, model, oracle  # noqa: F401  (fixtures)
from test_g1_gpu_fk import URDF

ROOT = Path(__file__).resolve().parents[1]
SPHERES = ROOT / "csrc/collision/g1_29dof_contacts_spheres.urdf"
PREC = {"f64": np.float64, "f32": np.float32}
FREE, COLL, INVALID, NOT_EVAL = 0, 1, 2, 3


@pytest.fixture(scope="module")
def table(model):
    from hjcdik._tree_collision import robot_sphere_table
    names = [j["name"] for j in model["joints"]]
    return robot_sphere_table(SPHERES, URDF, names)


@pytest.fixture(scope="module")
def link_spheres():
    from hjcdik._tree_collision import parse_sphere_model
    m = parse_sphere_model(SPHERES)
    links, local = [], []
    for link, sph in m.items():
        for s in sph:
            links.append(link); local.append(s)
    return links, np.array(local)


# ------------------------------------------------------------------ independent numpy geometry
def cpu_centres(oracle, links, local, q, base=None):
    T = oracle.fk(q)
    c = np.array([(T[l] @ np.r_[x[:3], 1.0])[:3] for l, x in zip(links, local)])
    if base is not None:
        c = c @ L.quat2mat(base[3:] / np.linalg.norm(base[3:])).T + base[:3]
    return c


def quat_R(q):
    return L.quat2mat(np.asarray(q, float) / np.linalg.norm(q))


def cpu_clearance(c, r, env):
    """(NS,) signed clearance to the nearest obstacle (inf if none). Box inside = -(face depth) - r."""
    d = np.full(len(c), np.inf)
    for s in env.get("spheres", np.zeros((0, 4))):
        d = np.minimum(d, np.linalg.norm(c - s[:3], axis=1) - r - s[3])
    for k in env.get("capsules", np.zeros((0, 7))):
        a, b = k[:3], k[3:6]; ab = b - a
        t = np.clip(((c - a) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(c - (a + t[:, None] * ab), axis=1) - r - k[6])
    for bx in env.get("cuboids", np.zeros((0, 10))):
        R = quat_R(bx[3:7]); h = bx[7:10] / 2
        p = (c - bx[:3]) @ R                    # box-frame coordinates
        ex = np.abs(p) - h
        out = np.linalg.norm(np.maximum(ex, 0), axis=1)
        inside = np.all(ex <= 0, axis=1)
        d = np.minimum(d, np.where(inside, -np.min(-ex, axis=1), out) - r)
    for pl in env.get("planes", np.zeros((0, 4))):
        n = pl[:3] / np.linalg.norm(pl[:3])
        d = np.minimum(d, c @ n - pl[3] / np.linalg.norm(pl[:3]) - r)
    return d


def env_call(tm, prec, Q, table, env=None, base=None, margin=0.0, per_sphere=True):
    T = PREC[prec]
    return getattr(tm, f"env_collision_{prec}")(np.ascontiguousarray(Q, T), table.astype(T),
                                                None if env is None else {k: np.asarray(v, T) for k, v in env.items()},
                                                None if base is None else np.ascontiguousarray(base, T), margin, per_sphere)


def rand_q(model, rng, n):
    lo, hi = L.limits(model).T
    return lo + (hi - lo) * rng.uniform(size=(n, 29))


def rand_env(rng, centre, scale=0.6, n=3):
    """Mixed primitives scattered around `centre` (world)."""
    sp = np.c_[centre + rng.uniform(-scale, scale, (n, 3)), rng.uniform(0.03, 0.15, n)]
    a = centre + rng.uniform(-scale, scale, (n, 3)); cap = np.c_[a, a + rng.normal(size=(n, 3)) * 0.2, rng.uniform(0.02, 0.08, n)]
    qs = rng.normal(size=(n, 4)); qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    cub = np.c_[centre + rng.uniform(-scale, scale, (n, 3)), qs, rng.uniform(0.05, 0.4, (n, 3))]
    nn = rng.normal(size=(1, 3)); nn /= np.linalg.norm(nn)
    pl = np.c_[nn, nn @ centre - rng.uniform(0.6, 1.0, 1)]
    return dict(spheres=sp, capsules=cap, cuboids=cub, planes=pl)


# ------------------------------------------------------------------ geometry and frames
def test_sphere_table_frames(model, oracle, table, link_spheres):
    """Anchor/offset table == link-frame spheres under the independent URDF FK; root (pelvis) spheres kept;
    non-root anchors equal GRiD's sphere_anchor_frames joint mapping."""
    tab, tlinks = table
    links, local = link_spheres
    assert tlinks == links and len(tab) == 621
    np.testing.assert_array_equal(tab[:, 4], local[:, 3])                      # radii untouched
    rng = np.random.default_rng(1)
    for q in rand_q(model, rng, 4):
        Tj = oracle.joint_world_transforms(q)
        via_table = np.array([(Tj[int(a)] @ np.r_[o, 1.0])[:3] if a >= 0 else o for a, o in zip(tab[:, 0], tab[:, 1:4])])
        np.testing.assert_allclose(via_table, cpu_centres(oracle, links, local, q), atol=1e-12)
    roots = {l for l, a in zip(tlinks, tab[:, 0]) if a < 0}
    assert roots == {"pelvis_contour_link"}                                      # floating base: not a pedestal
    from grid_codegen.algorithms._collision import sphere_anchor_frames
    from hjcdik import _tree_build as tb
    robot = tb._parse(URDF)
    gf = sphere_anchor_frames(robot, sorted(set(links)), str(URDF))
    names = [j["name"] for j in model["joints"]]
    for l, a in zip(tlinks, tab[:, 0]):
        ga, _ = gf[l]
        if a >= 0:
            assert names[int(a)] == robot.get_joint_by_id(ga).get_name(), l
        else:
            assert ga == -1


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_world_centres_and_clearances_match_cpu(tm, model, oracle, table, link_spheres, prec):
    """Per-sphere world centres and clearances for random joints, arbitrary base rotations and mixed primitives."""
    tab, _ = table; links, local = link_spheres
    rng = np.random.default_rng(2)
    Q = rand_q(model, rng, 6)
    B = np.array([TB.rand_base(rng, 1.0) for _ in range(6)])                   # full SO(3) rotations
    for i in range(6):
        env = rand_env(rng, B[i, :3])
        o = env_call(tm, prec, Q[i:i + 1], tab, env, B[i:i + 1])
        c = cpu_centres(oracle, links, local, Q[i], B[i])
        d = cpu_clearance(c, local[:, 3], env)
        tol = 1e-9 if prec == "f64" else 2e-5
        np.testing.assert_allclose(o["sphere_world"][0], c, atol=tol)
        np.testing.assert_allclose(o["sphere_clearance"][0], d, atol=tol)
        np.testing.assert_allclose(o["min_clearance"][0], d.min(), atol=tol)
        if abs(d.min()) > 10 * tol:
            assert o["status"][0] == (COLL if d.min() < 0 else FREE)
            assert o["argmin_sphere"][0] == int(np.argmin(d))


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_empty_separated_and_penetrating(tm, model, oracle, table, link_spheres, prec):
    tab, _ = table; links, local = link_spheres
    rng = np.random.default_rng(3)
    Q = rand_q(model, rng, 8); B = np.array([TB.rand_base(rng, 0.5) for _ in range(8)])
    e = env_call(tm, prec, Q, tab, None, B)                                     # A: empty
    assert (e["status"] == FREE).all() and np.isinf(e["min_clearance"]).all() and (e["argmin_sphere"] == -1).all()
    far = dict(spheres=[[50, 50, 50, 1.0]], cuboids=[[-40, 0, 0, 1, 0, 0, 0, 1, 1, 1]], planes=[[0, 0, 1, -30]])
    f = env_call(tm, prec, Q, tab, far, B)                                      # B: far away
    assert (f["status"] == FREE).all() and (f["min_clearance"] > 20).all()
    for i in range(8):                                                          # C: obstacle at a robot sphere centre
        c = cpu_centres(oracle, links, local, Q[i], B[i])
        k = rng.integers(len(c))
        for kind, row in (("spheres", [*c[k], 0.01]), ("cuboids", [*c[k], 1, 0, 0, 0, 0.02, 0.02, 0.02]),
                          ("capsules", [*c[k], *(c[k] + 0.01), 0.005])):
            o = env_call(tm, prec, Q[i:i + 1], tab, {kind: [row]}, B[i:i + 1])
            assert o["status"][0] == COLL and o["min_clearance"][0] < 0


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_tangency_and_margin(tm, model, oracle, table, link_spheres, prec):
    """Collision iff min clearance < margin (strict). The obstacle is sized so the GLOBAL min clearance is delta."""
    tab, _ = table; links, local = link_spheres
    rng = np.random.default_rng(4)
    eps = 1e-7 if prec == "f64" else 2e-4
    rows = []
    for i in range(6):
        q = rand_q(model, rng, 1)[0]; b = TB.rand_base(rng, 0.5)
        c = cpu_centres(oracle, links, local, q, b)
        p = c[rng.integers(len(c))] + rng.normal(size=3) * 0.4                 # obstacle centre near the robot
        gap = np.linalg.norm(c - p, axis=1) - local[:, 3]                      # clearance to a point
        for margin in (0.0, 0.02):
            for delta, want in ((margin + eps, FREE), (margin - eps, COLL)):
                rad = gap.min() - delta
                if rad <= 1e-3:
                    continue
                o = env_call(tm, prec, q[None], tab, dict(spheres=[[*p, rad]]), b[None], margin=margin)
                assert o["status"][0] == want, (i, margin, delta)
                np.testing.assert_allclose(o["min_clearance"][0], delta, atol=eps / 4 if prec == "f64" else 5e-5)
                rows.append((margin, delta, int(o["status"][0])))
        # plane tangency: n.p >= d with d chosen so the lowest sphere surface sits exactly at margin
        n = np.array([0.0, 0.0, 1.0])
        for margin in (0.0, 0.02):
            dd = (c @ n - local[:, 3]).min() - margin
            for shift, want in ((-eps, FREE), (eps, COLL)):
                o = env_call(tm, prec, q[None], tab, dict(planes=[[*n, dd + shift]]), b[None], margin=margin)
                assert o["status"][0] == want
    assert len(rows) >= 12


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_common_rigid_transform_invariance(tm, model, table, prec):
    """F/G: translating/rotating robot base and environment together leaves the verdict and clearances unchanged."""
    tab, _ = table
    rng = np.random.default_rng(5)
    Q = rand_q(model, rng, 12); B = np.array([TB.rand_base(rng, 0.3) for _ in range(12)])
    env = rand_env(rng, np.zeros(3), 0.5, n=4)
    base = env_call(tm, prec, Q, tab, env, B, per_sphere=False)
    for _ in range(3):
        G = TB.rand_base(rng, 2.0); R = quat_R(G[3:]); t = G[:3]
        B2 = np.array([np.r_[R @ b[:3] + t, L.qmul(G[3:], b[3:])] for b in B])
        e2 = dict(spheres=np.c_[env["spheres"][:, :3] @ R.T + t, env["spheres"][:, 3]],
                  capsules=np.c_[env["capsules"][:, :3] @ R.T + t, env["capsules"][:, 3:6] @ R.T + t, env["capsules"][:, 6]],
                  cuboids=np.c_[env["cuboids"][:, :3] @ R.T + t, [L.qmul(G[3:], x) for x in env["cuboids"][:, 3:7]], env["cuboids"][:, 7:]],
                  planes=np.c_[env["planes"][:, :3] @ R.T, env["planes"][:, 3] + (env["planes"][:, :3] @ R.T) @ t])
        o = env_call(tm, prec, Q, tab, e2, B2, per_sphere=False)
        tol = 1e-9 if prec == "f64" else 5e-5
        np.testing.assert_allclose(o["min_clearance"], base["min_clearance"], atol=tol)
        ok = np.abs(base["min_clearance"]) > 10 * tol
        np.testing.assert_array_equal(o["status"][ok], base["status"][ok])


def test_bases_and_joints_are_per_candidate(tm, model, oracle, table, link_spheres):
    """H/I: fixed joints with distinct bases, and single-joint moves: each candidate's spheres follow its own state;
    a joint move displaces exactly the spheres of its descendant links."""
    tab, _ = table; links, local = link_spheres
    rng = np.random.default_rng(6)
    q = rand_q(model, rng, 1)[0]
    B = np.array([TB.rand_base(rng, 1.0) for _ in range(16)])
    o = env_call(tm, "f64", np.repeat(q[None], 16, 0), tab, None, B)
    for i in range(16):
        np.testing.assert_allclose(o["sphere_world"][i], cpu_centres(oracle, links, local, q, B[i]), atol=1e-9)
    anc = oracle.ancestor_matrix()                                               # A[k, j]: j moves k
    Q = np.repeat(q[None], 29, 0); lo, hi = L.limits(model).T
    for j in range(29):
        Q[j, j] = lo[j] + 0.5 * (hi[j] - lo[j]) if abs(q[j] - (lo[j] + hi[j]) / 2) > 0.2 else lo[j] + 0.05
    base = np.tile(B[0], (29, 1))
    o2 = env_call(tm, "f64", Q, tab, None, base)
    ref = env_call(tm, "f64", q[None], tab, None, B[:1])["sphere_world"][0]
    names = [jj["name"] for jj in model["joints"]]
    for j in range(29):
        moved = np.linalg.norm(o2["sphere_world"][j] - ref, axis=1) > 1e-9
        anchors = tab[:, 0].astype(int)
        desc = np.array([a >= 0 and bool(anc[a, j]) for a in anchors]) if anc is not None else None
        np.testing.assert_allclose(o2["sphere_world"][j], cpu_centres(oracle, links, local, Q[j], base[j]), atol=1e-9)
        if desc is not None:
            assert not moved[~desc].any(), names[j]                            # unrelated links bitwise still


def test_invalid_and_validation(tm, model, table):
    tab, _ = table
    rng = np.random.default_rng(7)
    Q = rand_q(model, rng, 3); Q[1, 4] = np.nan
    env = dict(spheres=[[0, 0, 0, 5.0]])                                         # everything collides
    o = env_call(tm, "f64", Q, tab, env)
    assert list(o["status"]) == [COLL, INVALID, COLL] and np.isnan(o["min_clearance"][1]) and not o["collision_free"][1]
    with pytest.raises(ValueError):
        env_call(tm, "f64", Q[:1], tab, dict(spheres=[[0, 0, 0, -1.0]]))         # GRiD row validation
    with pytest.raises(ValueError):
        env_call(tm, "f64", Q[:1], tab, dict(cuboids=[[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]))   # zero quaternion
    with pytest.raises((ValueError, RuntimeError)):
        tm.env_collision_f64(Q[:1], tab, {"boxes": np.zeros((1, 10))})
    bad = tab.copy(); bad[0, 0] = 29
    with pytest.raises(ValueError):
        env_call(tm, "f64", Q[:1], bad, env)
    with pytest.raises(ValueError):
        env_call(tm, "f64", Q[:1], tab, env, margin=-0.01)


@pytest.mark.parametrize("prec", ["f64", "f32"])
def test_batch_isolation_and_environment_changes(tm, model, table, prec):
    tab, _ = table
    rng = np.random.default_rng(8)
    Q = rand_q(model, rng, 37); B = np.array([TB.rand_base(rng, 0.3) for _ in range(37)])
    eA = rand_env(rng, np.zeros(3), 0.5, 4); eB = rand_env(rng, np.zeros(3), 0.5, 4)
    a = env_call(tm, prec, Q, tab, eA, B)
    perm = rng.permutation(37)
    p = env_call(tm, prec, Q[perm], tab, eA, B[perm])
    for k in ("status", "min_clearance", "argmin_sphere", "sphere_world", "sphere_clearance"):
        np.testing.assert_array_equal(p[k], a[k][perm])
    parts = [env_call(tm, prec, Q[s], tab, eA, B[s]) for s in (slice(0, 1), slice(1, 20), slice(20, 37))]
    np.testing.assert_array_equal(np.concatenate([x["min_clearance"] for x in parts]), a["min_clearance"])
    np.testing.assert_array_equal(np.concatenate([x["status"] for x in parts]), a["status"])
    b = env_call(tm, prec, Q, tab, eB, B)
    a2 = env_call(tm, prec, Q, tab, eA, B)                                      # A, B, A: no cross-call leakage
    np.testing.assert_array_equal(a2["sphere_clearance"], a["sphere_clearance"])
    assert not np.array_equal(b["sphere_clearance"], a["sphere_clearance"])


def test_fp32_classification_matches_fp64(tm, model, table):
    tab, _ = table
    rng = np.random.default_rng(9)
    Q = rand_q(model, rng, 400); B = np.array([TB.rand_base(rng, 0.3) for _ in range(400)])
    env = rand_env(rng, np.zeros(3), 0.6, 6)
    for margin in (0.0, 0.01):
        d = env_call(tm, "f64", Q, tab, env, B, margin=margin, per_sphere=False)
        f = env_call(tm, "f32", Q, tab, env, B, margin=margin, per_sphere=False)
        np.testing.assert_allclose(f["min_clearance"], d["min_clearance"], atol=5e-5)
        clear = np.abs(d["min_clearance"] - margin) > 1e-4
        np.testing.assert_array_equal(f["status"][clear], d["status"][clear])
        assert 0 < (d["status"] == COLL).sum() < 400


# ------------------------------------------------------------------ selection
def cpu_select(Q, pe, oe, active, st, M, sel, eps_p=1e-4, eps_o=1e-3):
    """Documented ranking: class (0 exact / 1 finite / 2 non-finite; GATE: status != FREE -> 2), then (RANK only)
    status, then E_phys, then seed. Returns per problem the M seeds (-1 pads) and counts."""
    P, S = Q.shape[:2]
    seeds, counts = [], []
    for p in range(P):
        keys = []
        c = dict(free=0, coll=0, inv=0, solved_free=0, solved=0)
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
            c["solved"] += cls == 0
            c["free"] += st[p, s] == FREE; c["coll"] += st[p, s] == COLL; c["inv"] += st[p, s] == INVALID
            c["solved_free"] += cls == 0 and st[p, s] == FREE
            coll = 0
            if sel == "gate" and st[p, s] != FREE:
                cls, e = 2, np.inf
            elif sel == "rank":
                coll = int(st[p, s])
            keys.append((cls, coll, e, s))
        order = sorted(keys)
        seeds.append([k[3] if k[0] != 2 else -1 for k in order[:M]])
        counts.append(c)
    return np.array(seeds), counts


def make_pool(rng, P, S, all_coll=(), zero_exact=()):
    Q = rng.normal(size=(P, S, 29))
    pe = np.where(rng.uniform(size=(P, S, 4)) < 0.5, rng.uniform(0, 1e-4, (P, S, 4)), rng.uniform(1e-4, 1e-2, (P, S, 4)))
    oe = rng.uniform(0, 1.2e-3, (P, S, 4))
    st = rng.choice([FREE, COLL, INVALID], size=(P, S), p=[0.5, 0.4, 0.1]).astype(np.uint8)
    nf = rng.uniform(size=(P, S)) < 0.08
    Q[nf, 3] = np.nan; st[nf] = INVALID                                         # non-finite => never evaluated
    pe[1, 2] = pe[1, 3]; oe[1, 2] = oe[1, 3]                                     # exact E_phys tie -> seed order
    for p in all_coll: st[p] = np.where(st[p] == INVALID, INVALID, COLL)
    for p in zero_exact: pe[p] += 1e-3
    return Q, pe, oe, st


@pytest.mark.parametrize("prec", ["f64", "f32"])
@pytest.mark.parametrize("sel", ["gate", "rank", "annotate"])
def test_selection_matches_cpu_reference(tm, prec, sel):
    T = PREC[prec]
    rng = np.random.default_rng(10)
    P, S, M = 7, 40, 6
    Q, pe, oe, st = make_pool(rng, P, S, all_coll=(4,), zero_exact=(5,))
    Q, pe, oe = Q.astype(T), pe.astype(T), oe.astype(T)
    active = np.array([15, 3, 12, 1, 15, 9, 6], np.uint32)
    seeds0 = np.zeros_like(Q)
    o = getattr(tm, f"select_{prec}")(Q, pe, oe, np.zeros((P, S), T), np.ones((P, S), np.uint8), seeds0, active, M,
                                      env_status=st, env_selection=sel)
    want, counts = cpu_select(Q.astype(np.float64), pe.astype(np.float64), oe.astype(np.float64), active, st, M, sel)
    np.testing.assert_array_equal(o["selected_seed_ids"], want)
    for p in range(P):
        assert o["num_solved"][p] == counts[p]["solved"]                        # contact-exactness, M5 meaning
        assert o["num_environment_free"][p] == counts[p]["free"]
        assert o["num_environment_colliding"][p] == counts[p]["coll"]
        assert o["num_environment_invalid"][p] == counts[p]["inv"]
        assert o["num_solved_environment_free"][p] == counts[p]["solved_free"]
        for m in range(M):
            s = want[p, m]
            assert o["selected_environment_status"][p, m] == (NOT_EVAL if s < 0 else st[p, s])
            assert bool(o["valid"][p, m]) == (s >= 0)                            # pads are never valid/free
    if sel == "gate":
        assert (o["selected_seed_ids"][4] == -1).all()                           # all-colliding problem -> pads
        assert (o["selected_environment_status"][o["valid"]] == FREE).all()
    if sel == "annotate":                                                        # == the M5 ranking
        m5 = getattr(tm, f"select_{prec}")(Q, pe, oe, np.zeros((P, S), T), np.ones((P, S), np.uint8), seeds0, active, M)
        np.testing.assert_array_equal(o["selected_seed_ids"], m5["selected_seed_ids"])


# ------------------------------------------------------------------ integrated pipeline
def wall_floor(bases):
    """A wall slab just in front of the mean base and a floor plane under it (world frame)."""
    c = bases.reshape(-1, 7)[:, :3].mean(0)
    return dict(cuboids=[[c[0] + 0.25, c[1], c[2], 1, 0, 0, 0, 0.1, 2.0, 2.0]], planes=[[0, 0, 1, c[2] - 0.75]],
                spheres=[[c[0] - 0.2, c[1] + 0.3, c[2] + 0.2, 0.1]])


@pytest.mark.parametrize("prec", ["f64", "f32"])
@pytest.mark.parametrize("coarse", ["none", "multi_target"])
def test_post_lm_check_pairs_final_state(tm, model, oracle, table, link_spheres, prec, coarse):
    tab, _ = table; links, local = link_spheres
    seeds, bases, probs, _ = TB.fb_batch(model, oracle, 6, 12, seed=211, perturb=(0.05, 0.1), unreachable=(2,))
    ids = np.arange(72, dtype=np.int64).reshape(6, 12) * 11 + 3
    ps = np.arange(6, dtype=np.uint32) * 7 + 1 if coarse != "none" else None
    T = PREC[prec]
    env = {k: np.asarray(v, T) for k, v in wall_floor(bases).items()}
    kw = dict(M=4, ids=ids, base_update={"enabled": True}, coarse_mode=coarse, problem_seeds=ps)
    off = TB.fsolve(tm, prec, seeds, bases, probs, **kw)
    res = {}
    for sel in ("gate", "rank", "annotate"):
        on = TB.fsolve(tm, prec, seeds, bases, probs, environment_collision="final", environment=env,
                       robot_spheres=tab.astype(T), environment_margin=0.01, environment_selection=sel, **kw)
        res[sel] = on
        # the check never alters the IK: every raw output bitwise equal to the disabled call
        for k in ("all_joint_config", "all_base_position", "all_base_quaternion", "all_position_errors",
                  "all_orientation_errors", "all_lm_converged", "all_iterations", "num_solved", "num_valid", "num_converged"):
            np.testing.assert_array_equal(on[k], off[k], err_msg=k)
        assert on["environment_evaluated"] and not off["environment_evaluated"] and "all_environment_status" not in off
        # status is the standalone evaluator on the FINAL (joints, base) pair of the same candidate ...
        X = on["all_joint_config"].reshape(-1, 29)
        Bf = np.concatenate([on["all_base_position"], on["all_base_quaternion"]], -1).reshape(-1, 7)
        sa = env_call(tm, prec, X, tab, env, Bf, margin=0.01, per_sphere=False)
        np.testing.assert_array_equal(on["all_environment_status"].reshape(-1), sa["status"])
        np.testing.assert_array_equal(on["all_environment_min_clearance"].reshape(-1), sa["min_clearance"])
        # ... and agrees with the independent CPU geometry away from the margin
        tol = 1e-9 if prec == "f64" else 5e-5
        for i in range(0, len(X), 5):
            d = cpu_clearance(cpu_centres(oracle, links, local, X[i].astype(float), Bf[i].astype(float)), local[:, 3],
                              {k: v.astype(float) for k, v in env.items()}).min()
            np.testing.assert_allclose(sa["min_clearance"][i], d, atol=tol)
            if abs(d - 0.01) > 10 * tol:
                assert sa["status"][i] == (COLL if d < 0.01 else FREE)
        # selection == CPU reference over the raw outputs; identity/pairing of each slot
        want, _ = cpu_select(on["all_joint_config"].astype(float), on["all_position_errors"].astype(float),
                             on["all_orientation_errors"].astype(float), probs_active(probs),
                             on["all_environment_status"], 4, sel)
        np.testing.assert_array_equal(on["selected_seed_ids"], want)
        for p in range(6):
            for m in range(4):
                s = want[p, m]
                if s < 0:
                    assert on["selected_environment_status"][p, m] == NOT_EVAL and on["selected_candidate_ids"][p, m] == -1
                    continue
                assert on["selected_candidate_ids"][p, m] == ids[p, s]
                np.testing.assert_array_equal(on["joint_config"][p, m], on["all_joint_config"][p, s])
                np.testing.assert_array_equal(on["base_position"][p, m], on["all_base_position"][p, s])
                assert on["selected_environment_status"][p, m] == on["all_environment_status"][p, s]
                assert bool(on["success"][p, m]) == bool(exact(on, probs)[p, s])   # contact-exactness, unchanged meaning
        np.testing.assert_array_equal(on["num_solved_environment_free"],
                                      ((on["all_environment_status"] == FREE) & exact(on, probs)).sum(1))
    np.testing.assert_array_equal(res["annotate"]["selected_seed_ids"], off["selected_seed_ids"])
    st = res["gate"]["all_environment_status"]
    assert (st == FREE).any() and (st == COLL).any(), "corpus must exercise both verdicts"
    assert (res["gate"]["selected_environment_status"][res["gate"]["valid"]] == FREE).all()


def probs_active(probs):
    return np.array([int(p[1]) for p in probs], np.uint32)


def exact(o, probs):
    act = probs_active(probs)
    out = np.ones(o["all_position_errors"].shape[:2], bool)
    for p in range(len(act)):
        for k in range(4):
            if (act[p] >> k) & 1:
                out[p] &= (o["all_position_errors"][p, :, k] <= 1e-4) & (o["all_orientation_errors"][p, :, k] <= 1e-3)
    return out & np.isfinite(o["all_joint_config"]).all(-1)


def test_mode_validation(tm, model, oracle, table):
    tab, _ = table
    seeds, bases, probs, _ = TB.fb_batch(model, oracle, 2, 4, seed=5)
    env = wall_floor(bases)
    for bad in ("hard", "both", "coarse", "soft"):
        with pytest.raises(ValueError):
            TB.fsolve(tm, "f64", seeds, bases, probs, environment_collision=bad, environment=env, robot_spheres=tab)
    with pytest.raises(ValueError):                                             # env args without the mode
        TB.fsolve(tm, "f64", seeds, bases, probs, environment=env, robot_spheres=tab)
    with pytest.raises(ValueError):                                             # requested but no robot model
        TB.fsolve(tm, "f64", seeds, bases, probs, environment_collision="final", environment=env)
    with pytest.raises(ValueError):                                             # bad environment rows: no silent skip
        TB.fsolve(tm, "f64", seeds, bases, probs, environment_collision="final", robot_spheres=tab,
                  environment=dict(spheres=[[0, 0, 0, 0.0]]))
    with pytest.raises(ValueError):
        TB.fsolve(tm, "f64", seeds, bases, probs, environment_collision="final", robot_spheres=tab, environment=env,
                  environment_selection="penalty")
    o = TB.fsolve(tm, "f64", seeds, None, probs, environment_collision="final", robot_spheres=tab, environment=env)
    assert o["environment_evaluated"] and o["all_environment_status"].shape == (2, 4)   # fixed base: identity


def test_tool_override_moves_targets_not_collision_geometry(tm, model, oracle, table):
    """A tool override changes the IK contact targets (so the solve), but the collision verdict of every final
    candidate is still the standalone evaluator's on that candidate's (joints, base): the tool never enters it."""
    tab, _ = table
    seeds, bases, probs, _ = TB.fb_batch(model, oracle, 4, 8, seed=77, perturb=(0.05, 0.1))
    t = L.tools_of(model).copy()
    t[0][:3, 3] += [0.05, 0.0, 0.02]; t[2][:3, 3] += [0.0, 0.03, -0.04]
    env = wall_floor(bases)
    kw = dict(M=2, base_update={"enabled": True}, environment_collision="final", environment=env, robot_spheres=tab)
    a = TB.fsolve(tm, "f64", seeds, bases, probs, **kw)
    b = TB.fsolve(tm, "f64", seeds, bases, probs, tools=t, **kw)
    assert not np.array_equal(a["all_joint_config"], b["all_joint_config"])      # the override did act on the IK
    for r in (a, b):
        X = r["all_joint_config"].reshape(-1, 29)
        Bf = np.concatenate([r["all_base_position"], r["all_base_quaternion"]], -1).reshape(-1, 7)
        sa = env_call(tm, "f64", X, tab, env, Bf, per_sphere=False)
        np.testing.assert_array_equal(r["all_environment_status"].reshape(-1), sa["status"])
        np.testing.assert_array_equal(r["all_environment_min_clearance"].reshape(-1), sa["min_clearance"])
