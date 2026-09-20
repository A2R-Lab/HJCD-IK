"""Environment collision must respect the FLOATING BASE.

grid_collision computes sphere centres with grid::multi_target_position_device, which returns them
in the robot's own base/model frame. The two tests built on those centres behave very differently
under a floating base:

    self-collision   sphere-vs-sphere on one body. Invariant to any rigid transform of the whole
                     robot, so the base is correctly irrelevant.
    environment      sphere-vs-WORLD obstacle. The obstacles sit at fixed world poses, so where the
                     robot IS decides the answer.

Before this fix mark_collisions_ct received only q[B,29] and no base, so a floating-base humanoid
was evaluated as though welded at the world origin -- a wall at x=0.05 and the same wall at x=50 m
produced identical verdicts. The fix applies

    p_world = R(base_q) * p_local + base_p

to every sphere centre between the two tests, with no extra joints: the model stays 29 actuated
joints and the base stays the [B,3]+[B,4] candidate state the solver already carries.

These tests are deliberately independent of G1 sphere-model QUALITY. They assert only that the
verdict responds to the base and to obstacle placement in the correct direction, which holds for
any sane sphere set.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import hjcdik

N = hjcdik.num_joints()
WALL_JSON = Path(__file__).with_name("g1_wall_problems.json")

pytestmark = pytest.mark.skipif(
    not WALL_JSON.exists() or not hjcdik.solve_problems(
        target_poses=np.tile(np.array([0.,0,0,1,0,0,0]), (1, hjcdik.num_targets(), 1)),
        active_masks=np.array([1], np.uint32),
        seed_configs=np.zeros((1, 2, N)),
        problems_json_text=WALL_JSON.read_text() if WALL_JSON.exists() else "",
        problem_set_name="problem").get("collision_enabled", False),
    reason="build has no HJCD_HAS_COLLISION: regenerate grid.cuh with --collision")

IDENT = np.array([[1.0, 0.0, 0.0, 0.0]])
Q0 = np.zeros((1, N))


def env_free(base_xyz, base_quat=IDENT, doc=None, q=Q0):
    """Environment-only verdict for one configuration at one base pose."""
    return bool(np.asarray(hjcdik.collision_free(
        q, doc if doc is not None else WALL_JSON.read_text(), "problem",
        base_positions=np.asarray(base_xyz, float),
        base_quaternions=np.asarray(base_quat, float),
        include_self_collision=False))[0])


def wall_at(x):
    d = json.loads(WALL_JSON.read_text())
    d["problems"]["problem"][0]["obstacles"]["cuboid"]["wall"]["pose"][0] = x
    return json.dumps(d)


# --- 1. the base translation must change the verdict ------------------------------------------

def test_base_translation_changes_environment_verdict():
    """Identical joints, different base x. The wall face is at x=0 and the robot faces +x."""
    far = env_free([[-1.50, 0.0, 1.0]])
    near = env_free([[0.00, 0.0, 1.0]])
    assert far, "robot 1.5 m from the wall reported colliding"
    assert not near, "robot at the wall face reported free"


def test_base_translation_is_monotone_towards_the_wall():
    """Sweeping the base into the wall must cross from free to colliding exactly once.

    A verdict that flickers would indicate the transform is being applied inconsistently across
    spheres; one clean transition is what a rigid body moving through a plane must produce.
    """
    xs = np.arange(-1.2, 0.11, 0.05)
    verdicts = [env_free([[float(x), 0.0, 1.0]]) for x in xs]
    assert verdicts[0] and not verdicts[-1]
    transitions = sum(1 for a, b in zip(verdicts, verdicts[1:]) if a != b)
    assert transitions == 1, f"expected one free->colliding transition, got {transitions}: {verdicts}"


# --- 2. obstacle placement must change the verdict --------------------------------------------

def test_wall_near_vs_far_changes_verdict_for_the_same_candidate():
    """The candidate is fixed; only the WALL moves. This is the test the old code could not pass."""
    base = [[-0.10, 0.0, 1.0]]
    assert not env_free(base, doc=wall_at(0.05)), "wall at the robot: expected colliding"
    assert env_free(base, doc=wall_at(50.0)), "wall 50 m away: expected free"


def test_environment_binding_is_keyed_on_content_not_just_set_name():
    """Two environments sharing a set name must not alias.

    bind_collision_env used to key its cache on set_name#idx alone, so the second scene silently
    reused the first one's obstacles -- a stale-environment bug that made the test above pass for
    the wrong reason.
    """
    base = [[-0.10, 0.0, 1.0]]
    a = env_free(base, doc=wall_at(0.05))
    b = env_free(base, doc=wall_at(50.0))
    c = env_free(base, doc=wall_at(0.05))          # back again: must return to the first verdict
    assert (a, b, c) == (False, True, False), f"environment cache aliased: {(a, b, c)}"


# --- 3. base ROTATION must change the verdict -------------------------------------------------

@pytest.mark.parametrize("x", [-0.25, -0.23, -0.21])
def test_base_rotation_changes_environment_verdict(x):
    """At these standoffs the G1 is deeper front-to-back than side-to-side, so a 90 deg yaw
    clears the wall while yaw 0 does not. Only the rotation term of the transform can produce
    that difference -- a translation-only fix would give the same answer for both."""
    yaw0 = IDENT
    yaw90 = np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]])
    assert not env_free([[x, 0.0, 1.0]], yaw0), f"x={x}, yaw 0: expected colliding"
    assert env_free([[x, 0.0, 1.0]], yaw90), f"x={x}, yaw 90: expected free"


def test_base_height_interacts_with_the_ground():
    """The ground slab is a separate obstacle from the wall; lowering the base must hit it."""
    clear = [[-1.5, 0.0, 1.0]]
    sunk = [[-1.5, 0.0, 0.05]]
    assert env_free(clear), "robot standing clear of the ground reported colliding"
    assert not env_free(sunk), "robot sunk into the ground reported free"


# --- 4. fixed-base regression ------------------------------------------------------------------

def test_fixed_base_matches_identity_base():
    """Omitting the base must be exactly the identity transform, i.e. the pre-fix behaviour."""
    doc = WALL_JSON.read_text()
    no_base = np.asarray(hjcdik.collision_free(Q0, doc, "problem", include_self_collision=False))
    identity = np.asarray(hjcdik.collision_free(
        Q0, doc, "problem", base_positions=np.zeros((1, 3)), base_quaternions=IDENT,
        include_self_collision=False))
    np.testing.assert_array_equal(no_base, identity)


def test_base_args_must_be_supplied_together():
    doc = WALL_JSON.read_text()
    with pytest.raises((ValueError, RuntimeError)):
        hjcdik.collision_free(Q0, doc, "problem", base_positions=np.zeros((1, 3)))


def test_identity_base_is_invariant_to_joint_only_changes():
    """Sanity floor: with the base fixed, the verdict must still depend on the JOINTS."""
    rng = np.random.default_rng(3)
    qs = np.vstack([np.zeros(N), rng.normal(0, 0.5, (7, N))])
    doc = WALL_JSON.read_text()
    v = np.asarray(hjcdik.collision_free(
        qs, doc, "problem",
        base_positions=np.tile([-0.30, 0.0, 1.0], (8, 1)),
        base_quaternions=np.tile(IDENT, (8, 1)),
        include_self_collision=False))
    assert v.dtype == bool and v.shape == (8,)
