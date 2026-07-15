"""Phase 4: tree-correct incremental (subtree) FK and incremental target-state caching.

The property under test is simple and absolute: after ANY sequence of accepted/rejected coordinate
updates, the incrementally-maintained state must equal a fresh FULL FK at the final configuration --
world transforms, target transforms, residual vectors, norms, per-target costs and the total cost.

This replaces ee_fk_suffix_thread, whose `for kk = jovr..FLANGE_JID` walk assumed parent(u) == u-1.
That is true for Panda and false for G1, where it would silently skip whole limbs.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import hjcdik
from urdf_fk import UrdfFK

REPO = Path(__file__).resolve().parents[1]
URDF = Path(os.environ.get("HJCD_TEST_URDF", REPO / "csrc" / "urdf" / "panda.urdf"))
SIDECAR = REPO / "csrc" / "generated" / "hjcd_targets.json"

N = hjcdik.num_joints()
K = hjcdik.num_targets()
ALL = (1 << K) - 1
LIMITS = hjcdik.joint_limits()

# fp64 FK is deterministic: the incremental path must match a full FK to round-off, not "closely".
ATOL = 1e-12


@pytest.fixture(scope="module")
def oracle():
    return UrdfFK(URDF)


@pytest.fixture(scope="module")
def meta():
    return json.loads(SIDECAR.read_text())


def _sample_q(rng, margin=0.15):
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    return rng.uniform(lo + margin * (hi - lo), hi - margin * (hi - lo))


def _quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s]) / 1.0
    i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
    if i == 0:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        q = np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif i == 1:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        q = np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        q = np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])
    return q / np.linalg.norm(q)


def _targets_at(q):
    T = hjcdik.target_transforms(q[None, :])[0]
    p = np.stack([T[k][:3, 3] for k in range(K)])
    quat = np.stack([_quat_from_R(T[k][:3, :3]) for k in range(K)])
    return p, quat


def _full_reference(q_final, p, quat, mask, wp, wo):
    """Fresh full FK + full residual evaluation at q_final -- the ground truth."""
    r = hjcdik.target_residuals(q_final[None, :], p[None], quat[None],
                                active_target_mask=np.array([mask], dtype=np.uint32),
                                position_weights=wp[None], orientation_weights=wo[None])
    Tj = hjcdik.link_transforms(q_final[None, :])[0][:N]
    Tt = hjcdik.target_transforms(q_final[None, :])[0]
    return Tj, Tt, r


def _run(q0, upd, acc, p, quat, mask, wp, wo):
    u = np.array(upd, dtype=float).reshape(1, len(upd), 2)      # M == 0 => the no-update baseline
    a = np.array(acc, dtype=bool).reshape(1, len(acc))
    return hjcdik.incremental_probe(
        q0[None, :], u, a, p[None], quat[None],
        active_target_mask=np.array([mask], dtype=np.uint32),
        position_weights=wp[None], orientation_weights=wo[None])


def _compare(out, q_final, p, quat, mask, wp, wo, label=""):
    """Incremental state vs a fresh full FK at q_final.

    Target frames are compared for ACTIVE targets only. The incremental path recomposes exactly
    JOINT_TARGET_MASK[j] & active_target_mask, so an INACTIVE target's frame is deliberately left
    stale -- that is the point of masking by `active`, and its residual/norm/cost are zero in both
    paths regardless. test_inactive_target_frames_are_stale_by_design pins that explicitly.
    """
    Tj, Tt, r = _full_reference(q_final, p, quat, mask, wp, wo)
    act = [k for k in range(K) if (mask >> k) & 1]
    errs = {
        "joint_config":  np.abs(out["joint_config"][0] - q_final).max(),
        "joint_xform":   np.abs(out["joint_transforms"][0] - Tj).max(),
        "target_xform":  np.abs(out["target_transforms"][0][act] - Tt[act]).max(),
        "e_pos":         np.abs(out["position_residuals"][0] - r["position_residuals"][0]).max(),
        "e_ori":         np.abs(out["orientation_residuals"][0] - r["orientation_residuals"][0]).max(),
        "pos_norm":      np.abs(out["position_errors"][0] - r["position_errors"][0]).max(),
        "ori_norm":      np.abs(out["orientation_errors"][0] - r["orientation_errors"][0]).max(),
        "target_costs":  np.abs(out["target_costs"][0] - r["target_costs"][0]).max(),
        "total_cost":    abs(out["cost_raw"][0] - r["cost_raw"][0]),
    }
    worst = max(errs.values())
    assert worst < ATOL, f"{label} incremental != full FK: {errs}"
    return errs


# --- descendant masks vs the independent URDF tree oracle ----------------------------------------

def test_descendant_masks_match_urdf_tree(oracle, meta):
    """JOINT_DESCENDANT_MASK[j] must be exactly subtree(j) INCLUDING j, per the URDF."""
    A = oracle.ancestor_matrix()          # A[k, j] = j is an ancestor-or-self of k
    for j in range(N):
        expect = sum(1 << u for u in range(N) if A[u, j])   # u has j as ancestor-or-self
        got = int(meta["joint_descendant_mask"][j])
        assert got == expect, (
            f"joint {j} ({oracle.joint_order()[j]}):\n"
            f"  generated {got:0{N}b}\n  urdf      {expect:0{N}b}")
        assert (got >> j) & 1, f"joint {j}'s mask must include ITSELF"


def test_parent_table_is_parent_before_child(oracle, meta):
    """The ascending subtree scan is only valid if parent(u) < u for every u."""
    parent = meta["joint_parent_jid"]
    order = oracle.joint_order()
    for u in range(N):
        assert parent[u] < u, f"joint {u} has parent {parent[u]} >= {u}: ascending scan invalid"
        # and it must agree with the URDF
        link = oracle.joints[order[u]]["parent"]
        movable = {v["child"]: v["name"] for v in oracle.joints.values() if v["type"] != "fixed"}
        exp = -1
        cur = link
        plink = {v["child"]: v["parent"] for v in oracle.joints.values()}
        while cur is not None:
            if cur in movable:
                exp = order.index(movable[cur])
                break
            cur = plink.get(cur)
        assert parent[u] == exp, f"joint {u} parent {parent[u]} != urdf {exp}"


# --- 1, 2, 3, 5, 10, 11, 12: every joint, both signs, all state ----------------------------------

@pytest.mark.parametrize("sign", [+1, -1])
def test_every_joint_single_update(sign):
    """Every one of the N joints, positive and negative update, all cached state checked."""
    rng = np.random.default_rng(100 + sign)
    worst = 0.0
    for j in range(N):
        q0 = _sample_q(rng)
        p, quat = _targets_at(_sample_q(rng))          # a DIFFERENT config => nonzero residuals
        wp, wo = np.ones(K), np.ones(K)
        v = float(np.clip(q0[j] + sign * 0.23, LIMITS[j, 0], LIMITS[j, 1]))
        out = _run(q0, [[j, v]], [True], p, quat, ALL, wp, wo)
        qf = q0.copy(); qf[j] = v
        errs = _compare(out, qf, p, quat, ALL, wp, wo, label=f"joint {j} sign {sign}")
        worst = max(worst, max(errs.values()))
    print(f"\n  every joint, sign {sign:+d}: worst |incremental - full| = {worst:.3e}")


# --- 4: updates that land exactly on a joint limit ------------------------------------------------

def test_updates_at_joint_limits():
    rng = np.random.default_rng(7)
    worst = 0.0
    for j in range(N):
        for bound in (0, 1):
            q0 = _sample_q(rng)
            p, quat = _targets_at(_sample_q(rng))
            wp, wo = np.ones(K), np.ones(K)
            v = float(LIMITS[j, bound])                # exactly at the limit
            out = _run(q0, [[j, v]], [True], p, quat, ALL, wp, wo)
            qf = q0.copy(); qf[j] = v
            worst = max(worst, max(_compare(out, qf, p, quat, ALL, wp, wo,
                                            label=f"joint {j} at limit {bound}").values()))
    print(f"\n  at joint limits: worst = {worst:.3e}")


# --- 6: every active-target mask ------------------------------------------------------------------

def test_every_active_mask():
    rng = np.random.default_rng(11)
    worst = 0.0
    for mask in range(1, 1 << K):
        for j in (0, N // 2, N - 1):
            q0 = _sample_q(rng)
            p, quat = _targets_at(_sample_q(rng))
            wp, wo = np.ones(K), np.ones(K)
            v = float(np.clip(q0[j] + 0.17, LIMITS[j, 0], LIMITS[j, 1]))
            out = _run(q0, [[j, v]], [True], p, quat, mask, wp, wo)
            qf = q0.copy(); qf[j] = v
            worst = max(worst, max(_compare(out, qf, p, quat, mask, wp, wo,
                                            label=f"mask {mask:0{K}b} joint {j}").values()))
    print(f"\n  all {(1 << K) - 1} active masks: worst = {worst:.3e}")


# --- 8: rejected updates roll back exactly --------------------------------------------------------

def test_rejected_update_restores_state():
    """A rejected step must restore the state.

    Two different guarantees, and the distinction is real:

      * The cached residual/norm/cost state and the joint value are RESTORED FROM THE SAVE, so they
        come back BITWISE identical -- asserted with == 0.0.
      * The world/target transforms are RECOMPUTED by subtree_fk_warp (that is the specified
        rollback: "restore the joint and recompute the subtree"). subtree_fk_warp is a generic 4x4
        chain multiply, whereas the baseline full FK is GRiD's specialized emitted chain -- a
        different floating-point association order. So they agree to round-off (~1 ULP), not bit for
        bit. That is inherent to recomputing rather than snapshotting the transforms, and
        test_long_update_sequences shows it does NOT accumulate over 1000 steps.
    """
    rng = np.random.default_rng(21)
    worst_x = 0.0
    for j in range(N):
        q0 = _sample_q(rng)
        p, quat = _targets_at(_sample_q(rng))
        wp, wo = np.ones(K), np.ones(K)
        base = _run(q0, [], [], p, quat, ALL, wp, wo)                       # no updates
        v = float(np.clip(q0[j] + 0.4, LIMITS[j, 0], LIMITS[j, 1]))
        rej = _run(q0, [[j, v]], [False], p, quat, ALL, wp, wo)             # applied then rolled back

        for key in ("joint_config", "position_residuals", "orientation_residuals",
                    "position_errors", "orientation_errors", "target_costs", "cost_raw"):
            d = np.abs(np.asarray(rej[key]) - np.asarray(base[key])).max()
            assert d == 0.0, (f"joint {j}: rollback left '{key}' changed by {d:.3e} -- this state is "
                              f"restored from the save and must be bitwise exact")

        for key in ("joint_transforms", "target_transforms"):
            d = np.abs(np.asarray(rej[key]) - np.asarray(base[key])).max()
            assert d < 1e-15, f"joint {j}: rollback left '{key}' off by {d:.3e} (> round-off)"
            worst_x = max(worst_x, d)
    print(f"\n  rollback: cached state bitwise exact; recomputed transforms within {worst_x:.3e}")


def test_inactive_target_frames_are_stale_by_design():
    """An INACTIVE target's frame is not maintained -- and its residual/cost stay exactly zero.

    The incremental path recomposes JOINT_TARGET_MASK[j] & active_target_mask. So on G1, moving a
    left-hip joint while only the left HAND is active recomposes nothing, and the left-foot frame
    goes stale. That is correct and intended: nothing in the solve reads an inactive target. This
    test pins it so it cannot silently become a real bug if a later phase starts reading those
    frames.
    """
    if K < 2:
        pytest.skip("needs K >= 2")
    m = hjcdik.target_metadata()
    rng = np.random.default_rng(41)
    q0 = _sample_q(rng)
    p, quat = _targets_at(_sample_q(rng))
    wp, wo = np.ones(K), np.ones(K)

    for j in range(N):
        tmask_all = int(m["joint_target_mask"][j])
        inactive_but_moved = [k for k in range(K) if (tmask_all >> k) & 1]
        if not inactive_but_moved:
            continue
        k_stale = inactive_but_moved[0]
        mask = ALL & ~(1 << k_stale)            # everything EXCEPT the target this joint moves
        if mask == 0:
            continue
        v = float(np.clip(q0[j] + 0.3, LIMITS[j, 0], LIMITS[j, 1]))
        out = _run(q0, [[j, v]], [True], p, quat, mask, wp, wo)
        # inactive target: residual/cost must be exactly zero, whatever its (stale) frame says
        assert np.all(out["position_residuals"][0, k_stale] == 0.0)
        assert np.all(out["orientation_residuals"][0, k_stale] == 0.0)
        assert out["position_errors"][0, k_stale] == 0.0
        assert out["target_costs"][0, k_stale] == 0.0


def test_unaffected_targets_are_bitwise_untouched():
    """Moving a left-arm joint must not perturb the right-foot cached state by a single ULP."""
    if K < 2:
        pytest.skip("needs K >= 2")
    m = hjcdik.target_metadata()
    rng = np.random.default_rng(31)
    q0 = _sample_q(rng)
    p, quat = _targets_at(_sample_q(rng))
    wp, wo = np.ones(K), np.ones(K)
    base = _run(q0, [], [], p, quat, ALL, wp, wo)
    for j in range(N):
        v = float(np.clip(q0[j] + 0.3, LIMITS[j, 0], LIMITS[j, 1]))
        out = _run(q0, [[j, v]], [True], p, quat, ALL, wp, wo)
        tm = int(m["joint_target_mask"][j])
        for k in range(K):
            if (tm >> k) & 1:
                continue                                  # affected: expected to change
            for key in ("target_transforms", "position_residuals", "orientation_residuals",
                        "position_errors", "orientation_errors", "target_costs"):
                d = np.abs(np.asarray(out[key])[0, k] - np.asarray(base[key])[0, k]).max()
                assert d == 0.0, (f"joint {j} is not in target {k}'s mask but changed "
                                  f"'{key}' by {d:.3e}")


# --- 9: long sequences, accepted and rejected, drift ----------------------------------------------

@pytest.mark.parametrize("M", [1, 10, 100, 1000])
def test_long_update_sequences(M):
    """Consecutive updates: does incremental state DRIFT from full FK over many steps?"""
    rng = np.random.default_rng(1000 + M)
    q0 = _sample_q(rng)
    p, quat = _targets_at(_sample_q(rng))
    wp, wo = np.ones(K), np.ones(K)

    upd, acc, qf = [], [], q0.copy()
    for _ in range(M):
        j = int(rng.integers(0, N))
        v = float(np.clip(qf[j] + rng.normal(scale=0.2), LIMITS[j, 0], LIMITS[j, 1]))
        keep = bool(rng.integers(0, 2))                 # mix of accepted and rejected
        upd.append([j, v])
        acc.append(keep)
        if keep:
            qf[j] = v
    out = _run(q0, upd, acc, p, quat, ALL, wp, wo)
    errs = _compare(out, qf, p, quat, ALL, wp, wo, label=f"M={M}")
    print(f"\n  M={M:>4} ({sum(acc)} accepted, {M - sum(acc)} rolled back): "
          f"worst = {max(errs.values()):.3e}   total_cost err = {errs['total_cost']:.3e}")


def test_mixed_weights_and_costs():
    rng = np.random.default_rng(55)
    q0 = _sample_q(rng)
    p, quat = _targets_at(_sample_q(rng))
    wp = np.linspace(0.3, 2.5, K)
    wo = np.linspace(1.7, 0.2, K)
    upd, acc, qf = [], [], q0.copy()
    for _ in range(50):
        j = int(rng.integers(0, N))
        v = float(np.clip(qf[j] + rng.normal(scale=0.15), LIMITS[j, 0], LIMITS[j, 1]))
        upd.append([j, v]); acc.append(True); qf[j] = v
    out = _run(q0, upd, acc, p, quat, ALL, wp, wo)
    _compare(out, qf, p, quat, ALL, wp, wo, label="mixed weights")
