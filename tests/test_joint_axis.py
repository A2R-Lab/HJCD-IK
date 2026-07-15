"""Phase 1A: whole-body FK + per-joint world-axis correctness for the CURRENTLY BUILT robot.

The kernel used to read each joint's rotation axis as the z-column of its world transform. GRiD does
not rotate joint axes onto local z (it keeps the URDF <axis> in the motion subspace S), so that was
only correct because Panda happens to be all-z. These tests pin the replacement: the generated
grid::JOINT_AXIS_COL / JOINT_AXIS_SIGN metadata, and the FK it is read from.

Robot-agnostic: set HJCD_TEST_URDF to the URDF the current build's grid.cuh was generated from.
Defaults to Panda (the committed build). scripts/dev/g1_check.sh drives the G1 case.
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

import hjcdik
from urdf_fk import UrdfFK

REPO = Path(__file__).resolve().parents[1]
URDF = Path(os.environ.get("HJCD_TEST_URDF", REPO / "csrc" / "urdf" / "panda.urdf"))
GRID_CUH = REPO / "csrc" / "generated" / "grid.cuh"

N = hjcdik.num_joints()


def _generated_int_array(name):
    """Pull a `constexpr int NAME[n] = {...};` out of the built grid.cuh."""
    text = GRID_CUH.read_text()
    m = re.search(rf"constexpr int {name}\[\d+\]\s*=\s*\{{([^}}]*)\}}", text)
    assert m, f"{name} not found in {GRID_CUH} — regenerate with scripts/codegen/generate_grid.py"
    return [int(v) for v in m.group(1).split(",")]


@pytest.fixture(scope="module")
def oracle():
    fk = UrdfFK(URDF)
    fk.assert_matches_grid(URDF)          # ordering drift fails loudly, not silently
    assert len(fk.joint_order()) == N, (
        f"built robot has {N} joints but {URDF.name} has {len(fk.joint_order())}; "
        f"set HJCD_TEST_URDF to the URDF this build was generated from"
    )
    return fk


@pytest.fixture(scope="module")
def configs(oracle):
    rng = np.random.default_rng(0)
    return rng.uniform(-0.9, 0.9, size=(16, N))


def test_axis_metadata_matches_urdf(oracle):
    """JOINT_AXIS_COL/SIGN must be the cardinal decomposition of the URDF <axis>."""
    col = _generated_int_array("JOINT_AXIS_COL")
    sign = _generated_int_array("JOINT_AXIS_SIGN")
    prism = _generated_int_array("JOINT_IS_PRISMATIC")
    assert len(col) == len(sign) == len(prism) == N

    for j, name in enumerate(oracle.joint_order()):
        jv = oracle.joints[name]
        a = jv["axis"] / np.linalg.norm(jv["axis"])
        expect = np.zeros(3)
        expect[col[j]] = sign[j]
        np.testing.assert_allclose(
            a, expect, atol=1e-12,
            err_msg=f"joint {j} ({name}): URDF axis {a} != metadata col={col[j]} sign={sign[j]}")
        assert prism[j] == (1 if jv["type"] == "prismatic" else 0)


# GRiD rationalizes every origin transform with sympy nsimplify(tolerance=1e-6) to strip URDF noise
# (URDFParser/Joint.py:492), so its baked kinematics is a <=1e-6 approximation of the raw URDF. On G1
# this bites at the shoulders, whose origin rpy carries tiny terms (5.5e-05, -1.9e-04): the dropped
# second-order cosine is 1.99e-08 and the observed kernel-vs-oracle gap is 1.9e-08, matching exactly.
# Panda has no such joint, which is why it agrees to 1e-11. This is an upstream modelling choice, not
# solver error; a genuine axis/FK bug is O(0.1-1), so this bound still catches everything that counts.
FK_ATOL = 1e-6


def test_full_fk_matches_reference(oracle, configs):
    """Every movable joint's world transform, kernel vs numpy oracle."""
    T = hjcdik.link_transforms(configs)                 # (B, F, 4, 4)
    worst = 0.0
    for b, q in enumerate(configs):
        ref = oracle.joint_world_transforms(q)          # (N, 4, 4)
        for j in range(N):
            worst = max(worst, np.abs(T[b, j] - ref[j]).max())
    assert worst < FK_ATOL, f"max |T_kernel - T_ref| = {worst:.3e} over {len(configs)}x{N} transforms"


def test_world_axis_matches_reference(oracle, configs):
    """axis_world = SIGN * column(COL) of the kernel's world transform == oracle R @ axis_local."""
    col = _generated_int_array("JOINT_AXIS_COL")
    sign = _generated_int_array("JOINT_AXIS_SIGN")
    T = hjcdik.link_transforms(configs)
    worst = 0.0
    for b, q in enumerate(configs):
        ref = oracle.joint_world_axes(q)                # (N, 3)
        for j in range(N):
            got = sign[j] * T[b, j][:3, col[j]]         # the kernel's joint_world_axis()
            worst = max(worst, np.abs(got - ref[j]).max())
    assert worst < FK_ATOL, f"max |axis_kernel - axis_ref| = {worst:.3e}"


def test_axis_sign_via_finite_difference(oracle):
    """The deep check: does +q_j actually rotate about +axis_world[j]?

    A frame-vs-pose convention error would give a correct-looking axis with the wrong SIGN, which the
    static checks above cannot catch. Finite-difference the world position of each leaf frame w.r.t.
    every joint and compare against the analytic geometric-Jacobian column built from the metadata.

    Only ANCESTOR joints get a column: on a branched robot, moving a joint in another limb cannot
    move the tip, so the analytic formula axis x (p_tip - p_j) applies solely where the ancestor mask
    is set. Non-ancestors are asserted to have a genuinely zero derivative — that is the ground truth
    the Phase-3 ancestor masks must reproduce.
    """
    col = _generated_int_array("JOINT_AXIS_COL")
    sign = _generated_int_array("JOINT_AXIS_SIGN")
    prism = _generated_int_array("JOINT_IS_PRISMATIC")
    anc = oracle.ancestor_matrix()

    rng = np.random.default_rng(7)
    q = rng.uniform(-0.8, 0.8, size=N)
    eps = 1e-6

    T0 = hjcdik.link_transforms(q[None, :])[0]
    Q = np.repeat(q[None, :], 2 * N, axis=0)
    for j in range(N):
        Q[2 * j, j] += eps
        Q[2 * j + 1, j] -= eps
    Tp = hjcdik.link_transforms(Q)

    # every leaf joint is a tip worth checking (for G1: both wrists and both ankles)
    tips = [k for k in range(N) if all(not anc[m, k] for m in range(N) if m != k)]

    worst_anc, worst_zero = 0.0, 0.0
    for tip_j in tips:
        p_tip = T0[tip_j][:3, 3]
        for j in range(N):
            num = (Tp[2 * j][tip_j][:3, 3] - Tp[2 * j + 1][tip_j][:3, 3]) / (2 * eps)
            if not anc[tip_j, j]:
                worst_zero = max(worst_zero, np.abs(num).max())
                continue
            axis = sign[j] * T0[j][:3, col[j]]
            ana = axis if prism[j] else np.cross(axis, p_tip - T0[j][:3, 3])
            worst_anc = max(worst_anc, np.abs(num - ana).max())

    assert worst_zero < 1e-7, (
        f"a NON-ancestor joint moves a tip by {worst_zero:.3e} — the ancestor structure is wrong")
    assert worst_anc < 1e-5, (
        f"analytic Jacobian column disagrees with finite differences by {worst_anc:.3e} "
        f"— joint axis convention (likely the SIGN) is wrong")
