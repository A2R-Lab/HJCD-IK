"""FK-equivalence tests for HJCD-IK.

Validates the GRiD-generated forward kinematics used by the solver against an *independent*
reference computed in pure numpy directly from the URDF (no GRiD codegen involved). This guards
the FK migration (bespoke ``X_warp`` -> stock ``grid::ee_pose_inner_warp`` + grasptarget tool
offset) against silent convention / tool-offset bugs.

``generate_solutions`` returns both the solved joint configuration and the EE pose the kernel
computes for it; we recompute the grasptarget world pose from the URDF and compare position +
orientation (quaternion, with sign disambiguation).

Skips cleanly when the built extension or the URDF is unavailable.
"""
import os
import xml.etree.ElementTree as ET

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(REPO, "csrc", "urdf", "panda.urdf")

POS_ATOL_M = 1e-4      # 0.1 mm
QUAT_ATOL = 1e-3


def _rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_R(axis, q):
    a = axis / np.linalg.norm(axis)
    c, s = np.cos(q), np.sin(q)
    x, y, z = a
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def _parse_joints():
    root = ET.parse(URDF).getroot()
    joints = {}
    for j in root.findall("joint"):
        o, a = j.find("origin"), j.find("axis")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()])
        ax = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None
        joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                     child=j.find("child").get("link"), xyz=xyz, rpy=rpy, axis=ax)
    return joints


def _fk_grasptarget(q):
    """World 4x4 transform of panda_grasptarget_hand for actuated config q (independent numpy FK)."""
    joints = _parse_joints()
    by_child = {v["child"]: (k, v) for k, v in joints.items()}
    target_joint = next(k for k in joints if "grasptarget" in k)
    chain = []
    cur = joints[target_joint]["child"]
    while cur in by_child:
        name, v = by_child[cur]
        chain.append(v)
        cur = v["parent"]
    chain.reverse()  # base -> grasptarget
    T = np.eye(4)
    qi = 0
    for v in chain:
        Tj = np.eye(4)
        Tj[:3, :3] = _rpy_to_R(*v["rpy"])
        Tj[:3, 3] = v["xyz"]
        if v["type"] in ("revolute", "continuous"):
            Rj = np.eye(4); Rj[:3, :3] = _axis_R(v["axis"], q[qi]); qi += 1
            Tj = Tj @ Rj
        elif v["type"] == "prismatic":
            d = np.eye(4); d[:3, 3] = v["axis"] * q[qi]; qi += 1
            Tj = Tj @ d
        T = T @ Tj
    return T, qi


def _quat_from_R(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _kernel_qpose_pairs(num=12, seed=7, batch_size=4000):
    """(q, kernel_pose) pairs: solve sampled targets and read back the solution + its EE pose."""
    if not os.path.exists(URDF):
        pytest.skip(f"URDF not found at {URDF}")
    pairs = []
    for t in hjcdik.sample_targets(num_targets=num, seed=seed):
        out = hjcdik.generate_solutions(t, batch_size=batch_size, num_solutions=1)
        if out["count"] < 1:
            continue
        q = np.asarray(out["joint_config"])[0]
        pose = np.asarray(out["pose"])[0]  # [x,y,z, qw,qx,qy,qz]
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(pose)):
            continue
        pairs.append((q, pose))
    if not pairs:
        pytest.skip("solver returned no finite solutions to check")
    return pairs


def test_ee_position_matches_reference():
    worst = 0.0
    for q, pose in _kernel_qpose_pairs():
        T, used = _fk_grasptarget(q)
        assert used == hjcdik.num_joints(), f"actuated-joint count mismatch: {used}"
        worst = max(worst, float(np.linalg.norm(T[:3, 3] - pose[:3])))
    assert worst < POS_ATOL_M, f"max EE position error {worst*1e3:.4f} mm exceeds {POS_ATOL_M*1e3:.4f} mm"


def test_ee_orientation_matches_reference():
    worst = 0.0
    for q, pose in _kernel_qpose_pairs():
        T, _ = _fk_grasptarget(q)
        quat = _quat_from_R(T[:3, :3])
        kq = pose[3:7]
        if np.dot(quat, kq) < 0:
            quat = -quat
        worst = max(worst, float(np.linalg.norm(quat - kq)))
    assert worst < QUAT_ATOL, f"max EE quaternion error {worst:.2e} exceeds {QUAT_ATOL:.2e}"


# ===================================================================================================
# Cranley-Patterson scramble equivalence: benchmark/gen_targets.py's cranley-patterson path must be an
# exact transcription of HJCD's internal sample_targets (csrc/kernel/hjcd_kernel.cu:sample_q_halton_kernel
# + csrc/kernel/device_utils.cuh:wanghash). These checks are pure-numpy (no GPU); they run wherever this
# module runs. See Milestone 1 spec.
# ===================================================================================================
import sys as _sys  # noqa: E402

_sys.path.insert(0, os.path.join(REPO, "benchmark"))
import gen_targets as _gt  # noqa: E402


def _ref_wang_hash(a):
    """Independent uint32 transcription of device_utils.cuh:wanghash (not the module under test)."""
    M = 0xFFFFFFFF
    a &= M
    a = ((a ^ 61) ^ (a >> 16)) & M
    a = (a * 9) & M
    a = (a ^ (a >> 4)) & M
    a = (a * 0x27D4EB2D) & M
    a = (a ^ (a >> 15)) & M
    return a


def _ref_internal_q(seed, i, lo, hi):
    """Independent transcription of sample_q_halton_kernel for one 0-based config index i."""
    M = 0xFFFFFFFF
    n = 1 + i
    hseed = (seed ^ 0x9E3779B97F4A7C15) & M
    q = np.empty(len(lo))
    for j in range(len(lo)):
        base = _gt._PRIMES[j]
        inv = 1.0 / base
        f = inv
        u = 0.0
        nn = n
        while nn > 0:
            u += (nn % base) * f
            nn //= base
            f *= inv
        sh = _ref_wang_hash((hseed + j * 0x9E3779B9) & M)
        shift = (sh & 0xFFFFFF) / float(0x1000000)
        u = u + shift
        u = u - np.floor(u)
        q[j] = lo[j] + u * (hi[j] - lo[j])
    return q


def test_cp_wang_hash_known_values():
    assert _gt._wang_hash_u32(0) == 3232319850
    assert _gt._wang_hash_u32(1) == 663891101
    assert _gt._wang_hash_u32(0x9E3779B9) == 1125925706
    # 64-bit input must wrap to its low 32 bits before hashing.
    assert _gt._wang_hash_u32((1 << 32) | 0x9E3779B9) == 1125925706


def test_cp_shifts_known_values_and_range():
    sh = _gt._cranley_patterson_shifts(0, 7)
    assert sh.shape == (7,)
    assert abs(sh[0] - 0.9202864766120911) < 1e-15
    assert abs(sh[1] - 0.7119667530059814) < 1e-15
    assert np.all((sh >= 0.0) & (sh < 1.0))


def test_cp_shifts_dimension_specific_and_seed_dependent():
    sh = _gt._cranley_patterson_shifts(0, 7)
    assert len(set(sh.tolist())) == 7                          # a distinct scalar per dimension
    assert not np.allclose(sh, _gt._cranley_patterson_shifts(1, 7))   # a different seed => different shifts


def test_cp_deterministic_for_fixed_seed():
    a = _gt._halton(100, 7, 0, scramble=True, seed=0, method="cranley-patterson")
    b = _gt._halton(100, 7, 0, scramble=True, seed=0, method="cranley-patterson")
    assert np.array_equal(a, b)


def test_cp_seed_changes_sequence():
    a = _gt._halton(50, 7, 0, scramble=True, seed=0, method="cranley-patterson")
    b = _gt._halton(50, 7, 0, scramble=True, seed=1, method="cranley-patterson")
    assert not np.allclose(a, b)


def test_cp_unit_interval_and_index_convention():
    u = _gt._halton(100, 7, 0, scramble=True, seed=0, method="cranley-patterson")
    assert u.shape == (100, 7)
    assert np.all((u >= 0.0) & (u < 1.0))
    # skip=0 => first emitted index is n=1 (origin n=0 excluded), matching the internal offset=1.
    sh = _gt._cranley_patterson_shifts(0, 7)
    for j in range(7):
        expect = (_gt._radical_inverse(1, _gt._PRIMES[j]) + sh[j])
        expect = expect - np.floor(expect)
        assert abs(u[0, j] - expect) < 1e-15


def test_no_scramble_is_raw_halton_and_ignores_method():
    # raw Halton (identity digits, division-chain) must be unchanged and independent of `method`.
    u = _gt._halton(20, 7, 0, scramble=False, seed=0, method="cranley-patterson")
    ref = np.empty((20, 7))
    for k in range(20):
        for d in range(7):
            base, f, r, ii = _gt._PRIMES[d], 1.0, 0.0, k + 1
            while ii > 0:
                f /= base
                r += f * (ii % base)
                ii //= base
            ref[k, d] = r
    assert np.array_equal(u, ref)


def test_legacy_digit_permutation_unchanged():
    # Regression guard: the default path is byte-identical to the pre-change implementation.
    u = _gt._halton(30, 7, 0, scramble=True, seed=0, method="digit-permutation")
    rng = np.random.default_rng(0)
    perms = [rng.permutation(_gt._PRIMES[d]) for d in range(7)]
    ref = np.empty((30, 7))
    for k in range(30):
        for d in range(7):
            base, perm, f, r, ii = _gt._PRIMES[d], perms[d], 1.0, 0.0, k + 1
            while ii > 0:
                f /= base
                r += f * perm[ii % base]
                ii //= base
            ref[k, d] = r
    assert np.array_equal(u, ref)


def test_cp_q_matches_direct_cuda_transcription():
    # The generator's CP config q must equal an independent transcription of the CUDA kernel (bitwise).
    joints = _gt._parse_joints(URDF)
    chain = _gt._chain_to_target(joints, "panda_hand_joint")
    lo, hi = _gt._actuated_limits(chain)
    u = _gt._halton(100, len(lo), 0, scramble=True, seed=0, method="cranley-patterson")
    q = lo + u * (hi - lo)
    worst = max(float(np.max(np.abs(q[i] - _ref_internal_q(0, i, lo, hi)))) for i in range(100))
    assert worst == 0.0, f"CP q differs from direct transcription by {worst} (expected bitwise-equal)"


def test_cp_generated_poses_finite_and_normalized():
    joints = _gt._parse_joints(URDF)
    chain = _gt._chain_to_target(joints, "panda_hand_joint")
    lo, hi = _gt._actuated_limits(chain)
    u = _gt._halton(16, len(lo), 0, scramble=True, seed=0, method="cranley-patterson")
    for q in (lo + u * (hi - lo)):
        T = _gt._fk(chain, q)
        quat = _gt._quat_from_R(T[:3, :3])
        assert np.all(np.isfinite(T)) and np.all(np.isfinite(quat))
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-9
