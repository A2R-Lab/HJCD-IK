"""Phase 1B: the generated multi-target metadata and the target-frame FK.

Everything here is checked against the DEVICE's view of the metadata (hjcdik.target_metadata() dumps
the __device__ constexpr arrays from a kernel, not a host copy) and against the independent numpy
URDF oracle -- never against the codegen's own JSON, which would be circular.

The hand targets get the strongest possible check: their expected pose is the world pose of the
URDF's OWN `*_rubber_hand` link, which the whole-tree oracle produces by applying the fixed joint
itself. That is fully independent of the tool transform we baked.

Robot-agnostic via HJCD_TEST_URDF (defaults to Panda). scripts/dev/g1_check.sh drives the G1 case.
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

# What each robot's target set is REQUIRED to be. This is the spec, written out independently of the
# codegen, so a silent change to ordering, anchors or offsets fails here.
#   name -> (anchor joint name, expected tool xyz, expected tool rpy)
G1_SOLE = (0.035, 0.0, -0.035)     # = (x-centroid, y-centroid, min_i(z_i - r_i)) of the 4 foot spheres
EXPECTED = {
    "g1_29dof_rev_1_0": [
        ("left_hand",  "left_wrist_yaw_joint",    (0.0415,  0.003, 0.0)),
        ("right_hand", "right_wrist_yaw_joint",   (0.0415, -0.003, 0.0)),
        ("left_foot",  "left_ankle_roll_joint",   G1_SOLE),
        ("right_foot", "right_ankle_roll_joint",  G1_SOLE),
    ],
    "panda": [
        ("panda_grasptarget_hand", "panda_joint7", None),   # tool checked via the URDF link instead
    ],
}

FK_ATOL = 1e-6   # GRiD rationalizes origins with nsimplify(1e-6); see tests/test_joint_axis.py


@pytest.fixture(scope="module")
def oracle():
    fk = UrdfFK(URDF)
    fk.assert_matches_grid(URDF)
    assert len(fk.joint_order()) == hjcdik.num_joints(), (
        f"built robot has {hjcdik.num_joints()} joints but {URDF.name} has "
        f"{len(fk.joint_order())}; set HJCD_TEST_URDF to this build's URDF")
    return fk


@pytest.fixture(scope="module")
def meta():
    return hjcdik.target_metadata()          # dumped from device code


@pytest.fixture(scope="module")
def sidecar():
    return json.loads(SIDECAR.read_text())   # host-side names only


@pytest.fixture(scope="module")
def spec(sidecar):
    robot = sidecar["robot"]
    assert robot in EXPECTED, f"no expectation table for robot '{robot}'"
    return EXPECTED[robot]


@pytest.fixture(scope="module")
def configs(oracle):
    rng = np.random.default_rng(11)
    return rng.uniform(-0.9, 0.9, size=(12, hjcdik.num_joints()))


def _xform(xyz):
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


# --- 4. target-frame ordering -------------------------------------------------------------------

def test_target_ordering_and_count(meta, sidecar, spec):
    names = [t["name"] for t in sidecar["targets"]]
    assert names == [s[0] for s in spec], f"target ORDER changed: {names}"
    assert meta["num_targets"] == len(spec)
    assert hjcdik.num_targets() == len(spec)


def test_anchor_jids(meta, sidecar, spec, oracle):
    """Anchor joint ids must be the GRiD index of the named anchor joint."""
    order = oracle.joint_order()
    for k, (name, anchor_name, _) in enumerate(spec):
        expect = order.index(anchor_name)
        assert meta["anchor_jid"][k] == expect, (
            f"target {k} ({name}): device anchor_jid={meta['anchor_jid'][k]}, "
            f"expected {expect} ({anchor_name})")
        assert sidecar["targets"][k]["anchor_name"] == anchor_name


# --- 2 & 3. tool transforms: non-identity hand offsets, sole-center foot offsets -----------------

def test_tool_transforms(meta, spec):
    """The device's baked tool 4x4s must equal the specified offsets exactly."""
    for k, (name, _, xyz) in enumerate(spec):
        if xyz is None:
            continue                       # Panda's grasptarget: validated by FK against the URDF link
        got = np.asarray(meta["tool_xform"][k])
        np.testing.assert_allclose(
            got[:3, 3], xyz, atol=1e-12,
            err_msg=f"target {k} ({name}): tool translation {got[:3,3]} != specified {xyz}")
        np.testing.assert_allclose(
            got[:3, :3], np.eye(3), atol=1e-12,
            err_msg=f"target {k} ({name}): tool rotation is not identity")
        np.testing.assert_allclose(got[3], [0, 0, 0, 1], atol=1e-12)


# --- 1. all target transforms vs the independent numpy URDF FK -----------------------------------

def test_target_fk_matches_reference(oracle, meta, sidecar, configs):
    """The composed target frames vs the whole-tree URDF oracle, per target.

    For a target whose tool came from a URDF fixed joint (both hands, and Panda's grasptarget) the
    expected pose is the world pose of that fixed joint's CHILD LINK, which the oracle computes from
    the URDF directly -- so the baked tool transform is never used to build the expectation.
    """
    T = hjcdik.target_transforms(configs)                       # (B, K, 4, 4)
    per_target = {}
    for k, t in enumerate(sidecar["targets"]):
        worst = 0.0
        for b, q in enumerate(configs):
            links = oracle.fk(q)
            if t["fixed_joint"]:
                ref = links[oracle.joints[t["fixed_joint"]]["child"]]
            else:
                anchor_link = oracle.joints[t["anchor_name"]]["child"]
                ref = links[anchor_link] @ np.asarray(meta["tool_xform"][k])
            worst = max(worst, np.abs(T[b, k] - ref).max())
        per_target[t["name"]] = worst
    for name, err in per_target.items():
        assert err < FK_ATOL, f"target '{name}': max |T_kernel - T_ref| = {err:.3e}"
    print("\n  per-target FK error:  " +
          "  ".join(f"{n}={e:.2e}" for n, e in per_target.items()))


# --- 5 & 6. exact ancestor masks and per-joint dependency masks ----------------------------------

def test_target_ancestor_masks(oracle, meta, sidecar):
    """TARGET_ANCESTOR_MASK[k] must be exactly the ancestor-or-self set of target k's anchor."""
    A = oracle.ancestor_matrix()
    N = hjcdik.num_joints()
    for k, t in enumerate(sidecar["targets"]):
        anchor = meta["anchor_jid"][k]
        expect = sum(1 << j for j in range(N) if A[anchor, j])
        got = int(meta["target_ancestor_mask"][k])
        assert got == expect, (
            f"target {k} ({t['name']}):\n  device   {got:0{N}b}\n  expected {expect:0{N}b}")


def test_joint_target_masks(oracle, meta, sidecar):
    """JOINT_TARGET_MASK[j] must be the exact transpose of TARGET_ANCESTOR_MASK."""
    A = oracle.ancestor_matrix()
    N, K = hjcdik.num_joints(), meta["num_targets"]
    for j in range(N):
        expect = sum(1 << k for k in range(K) if A[meta["anchor_jid"][k], j])
        got = int(meta["joint_target_mask"][j])
        assert got == expect, (
            f"joint {j} ({oracle.joint_order()[j]}):\n"
            f"  device   {got:0{K}b}\n  expected {expect:0{K}b}")
        # transpose consistency with the other mask, independently of the oracle
        for k in range(K):
            a = bool(int(meta["target_ancestor_mask"][k]) >> j & 1)
            b = bool(got >> k & 1)
            assert a == b, f"masks disagree at joint {j}, target {k}: {a} vs {b}"


# --- 7. unrelated branches do not affect a target ------------------------------------------------

def test_unrelated_branches_do_not_move_a_target(oracle, meta, sidecar, configs):
    """Ground truth, by finite difference: a joint OUTSIDE target k's ancestor mask must not move it.

    This is the property the masks encode, verified against the actual kinematics rather than
    against the codegen. On G1 it is what stops a left-leg joint from getting a bogus Jacobian
    column for a right-hand target.
    """
    N, K = hjcdik.num_joints(), meta["num_targets"]
    eps = 1e-6
    q = configs[0]

    Q = np.repeat(q[None, :], 2 * N, axis=0)
    for j in range(N):
        Q[2 * j, j] += eps
        Q[2 * j + 1, j] -= eps
    T = hjcdik.target_transforms(Q)                             # (2N, K, 4, 4)

    worst_out, min_in = 0.0, np.inf
    n_out = n_in = 0
    for k in range(K):
        for j in range(N):
            d = np.abs(T[2 * j, k] - T[2 * j + 1, k]).max() / (2 * eps)
            inside = bool(int(meta["joint_target_mask"][j]) >> k & 1)
            if inside:
                min_in = min(min_in, d); n_in += 1
            else:
                worst_out = max(worst_out, d); n_out += 1

    assert worst_out < 1e-7, (
        f"a joint OUTSIDE the mask moves its target by {worst_out:.3e} -- the mask is too tight")
    if n_out:
        print(f"\n  masked-out joint-target pairs: {n_out}, max |dT/dq| = {worst_out:.2e}"
              f"   (in-mask pairs: {n_in})")
