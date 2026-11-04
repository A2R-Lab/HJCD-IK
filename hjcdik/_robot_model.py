"""Shared, CUDA-free validation of the serial models HJCD can solve.

The source generator loads this module by path so generating a first header does not
depend on an installed CUDA extension. Keep imports independent of the hjcdik package.
"""
import math
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile


def validate_solver_robot(robot, ee_link=None):
    """Reject models outside the solver's current kinematic and sampling contract."""
    if robot.floating_base or not robot.is_serial_chain():
        raise ValueError("HJCD-IK requires a fixed-base serial chain")
    if not 1 <= robot.get_num_joints() <= 32:
        raise ValueError("HJCD-IK requires 1 to 32 actuated joints (one joint per warp lane)")
    for joint in robot.get_joints_ordered_by_id():
        if joint.is_mimic or joint.jtype not in ("revolute", "continuous", "prismatic"):
            raise ValueError("HJCD-IK requires independent revolute, continuous or prismatic joints")
        subspace = [float(x) for x in robot.get_S_by_id(joint.jid).reshape(-1)]
        axis, rest = (subspace[3:], subspace[:3]) if joint.jtype == "prismatic" else (subspace[:3], subspace[3:])
        if (not all(math.isfinite(x) for x in subspace) or any(x != 0 for x in rest)
                or not math.isclose(sum(x*x for x in axis), 1.0, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f"joint {joint.name!r} requires a finite unit axis")
        if joint.jtype != "continuous":
            limits = joint.get_joint_limits()
            if (len(limits) != 2 or not all(math.isfinite(float(x)) for x in limits)
                    or not float(limits[0]) < float(limits[1])
                    or any(abs(float(x)) > 1e30 for x in limits)):
                raise ValueError(f"joint {joint.name!r} requires finite, strictly ordered limits")
    # GRiD 0.6.0's homogeneous-loader trig scan reaches a NumPy fixed transform
    # on an all-prismatic chain and calls .atoms on it. Keep this upstream limit
    # explicit until a released generator supports the case (GRiD ask G13).
    if all(j.jtype == "prismatic" for j in robot.get_joints_ordered_by_id()):
        raise ValueError("GRiD 0.6.0 cannot generate all-prismatic chains; at least one revolute or continuous joint is required")
    if ee_link is not None:
        target = robot.get_fixed_joint_by_name(ee_link)
        if target is None:
            raise ValueError(f"no fixed joint named {ee_link!r} (the end-effector target frame)")
        last = robot.get_joint_by_id(robot.get_num_joints() - 1)
        if target.parent_name != last.name:
            raise ValueError("HJCD-IK requires a fixed end-effector target attached to the final actuated joint")


def describe_robot(robot, urdf_path, ee_link):
    """Canonical model metadata in the same joint order as generated GRiD kernels."""
    validate_solver_robot(robot, ee_link)
    joints = []
    for joint in robot.get_joints_ordered_by_id():
        s = [float(x) for x in robot.get_S_by_id(joint.jid).reshape(-1)]
        slider = joint.jtype == "prismatic"
        limits = [-math.pi, math.pi] if joint.jtype == "continuous" else list(map(float, joint.get_joint_limits()))
        joints.append(dict(name=joint.name, type=joint.jtype, unit="m" if slider else "rad",
                           axis=s[3:] if slider else s[:3], limits=limits))
    return dict(schema=1, urdf_sha256=hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
                ee_joint=ee_link, joints=joints)


# ---------------------------------------------------------------------------------------------
# Fixed-base kinematic TREE with several target frames (humanoid contract, opt-in).
#
# This path is separate from validate_solver_robot/describe_robot, which keep the serial
# single-target contract byte-for-byte. Nothing in the solver or build calls it yet: it fixes the
# model contract the multi-target kernels will be generated against. The floating base stays
# OUTSIDE the GRiD topology (no virtual base joints); a target is a URDF fixed joint whose child
# link is the contact frame, rigidly attached to an actuated ancestor.
# ---------------------------------------------------------------------------------------------

TREE_MAX_JOINTS = 32    # one joint per warp lane; uint32 ancestor masks
TREE_MAX_TARGETS = 32   # uint32 joint->target masks


def _tree_joint_axis(robot, joint):
    subspace = [float(x) for x in robot.get_S_by_id(joint.jid).reshape(-1)]
    axis, rest = (subspace[3:], subspace[:3]) if joint.jtype == "prismatic" else (subspace[:3], subspace[3:])
    if (not all(math.isfinite(x) for x in subspace) or any(x != 0 for x in rest)
            or not math.isclose(sum(x*x for x in axis), 1.0, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError(f"joint {joint.name!r} requires a finite unit axis")
    return axis


def _tree_limits(joint):
    if joint.jtype == "continuous":
        return [-math.pi, math.pi]
    limits = joint.get_joint_limits()
    if (len(limits) != 2 or not all(math.isfinite(float(x)) for x in limits)
            or not float(limits[0]) < float(limits[1]) or any(abs(float(x)) > 1e30 for x in limits)):
        raise ValueError(f"joint {joint.name!r} requires finite, strictly ordered limits")
    return [float(limits[0]), float(limits[1])]


def _rigid_tool(fixed, name):
    tool = [[float(x) for x in row] for row in fixed.Xmat_hom.tolist()]
    flat = [x for row in tool for x in row]
    if len(tool) != 4 or any(len(row) != 4 for row in tool) or not all(math.isfinite(x) for x in flat):
        raise ValueError(f"target {name!r}: tool transform must be a finite 4x4")
    if tool[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError(f"target {name!r}: tool transform must be homogeneous (last row 0 0 0 1)")
    R = [row[:3] for row in tool[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(R[k][i] * R[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-9:
                raise ValueError(f"target {name!r}: tool rotation is not orthonormal")
    det = (R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1]) - R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])
           + R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]))
    if abs(det - 1.0) > 1e-9:
        raise ValueError(f"target {name!r}: tool rotation is not proper (det {det:.6g})")
    return tool


def validate_tree_robot(robot, targets):
    """Validate a fixed-base tree with named target frames; return the canonical description pieces.

    `targets` is an ordered sequence of (target_name, fixed_joint_name). The order is the device
    target index. Raises ValueError for anything outside the contract instead of accepting it.
    """
    if robot.floating_base:
        raise ValueError("HJCD-IK tree models require a fixed-base URDF; the floating base is external")
    n = robot.get_num_joints()
    if not 1 <= n <= TREE_MAX_JOINTS:
        raise ValueError(f"HJCD-IK tree models require 1 to {TREE_MAX_JOINTS} actuated joints")
    joints = list(robot.get_joints_ordered_by_id())
    if [j.jid for j in joints] != list(range(n)):
        raise ValueError("actuated joint ids must be contiguous 0..N-1 in GRiD order")
    names = [j.name for j in joints] + list(robot.get_fixed_joint_names())
    if len(set(names)) != len(names):
        raise ValueError("joint names must be unique across actuated and fixed joints")
    parents = []
    for joint in joints:
        if joint.is_mimic or joint.jtype not in ("revolute", "continuous", "prismatic"):
            raise ValueError(f"joint {joint.name!r}: HJCD-IK requires independent revolute, continuous or prismatic joints")
        parent = int(robot.get_parent_id(joint.jid))
        if not -1 <= parent < joint.jid:
            raise ValueError(f"joint {joint.name!r}: parent id {parent} breaks the topological (DFS) order")
        parents.append(parent)
    if all(j.jtype == "prismatic" for j in joints):
        raise ValueError("GRiD 0.6.0 cannot generate all-prismatic chains; at least one revolute or continuous joint is required")
    targets = [(str(t), str(f)) for t, f in targets]
    if not 1 <= len(targets) <= TREE_MAX_TARGETS:
        raise ValueError(f"HJCD-IK tree models require 1 to {TREE_MAX_TARGETS} targets")
    if len({t for t, _ in targets}) != len(targets) or len({f for _, f in targets}) != len(targets):
        raise ValueError("target names and target frames must be unique")
    by_name = {j.name: j.jid for j in joints}
    out_targets = []
    for name, frame in targets:
        fixed = robot.get_fixed_joint_by_name(frame)
        if fixed is None:
            raise ValueError(f"target {name!r}: no fixed joint named {frame!r}")
        anchor = by_name.get(fixed.parent_name)
        if anchor is None:
            raise ValueError(f"target {name!r}: frame {frame!r} has no actuated ancestor (attached to the root)")
        mask, j = 0, anchor
        while j != -1:
            mask |= 1 << j
            j = parents[j]
        out_targets.append(dict(name=name, fixed_joint=frame, anchor_jid=anchor, anchor_joint=fixed.parent_name,
                                tool_row_major=_rigid_tool(fixed, name), ancestor_mask=mask))
    described = []
    for joint, parent in zip(joints, parents):
        slider = joint.jtype == "prismatic"
        described.append(dict(name=joint.name, type=joint.jtype, unit="m" if slider else "rad",
                              axis=_tree_joint_axis(robot, joint), limits=_tree_limits(joint), parent=parent))
    joint_target = [sum(1 << k for k, t in enumerate(out_targets) if t["ancestor_mask"] >> j & 1) for j in range(n)]
    return described, out_targets, joint_target


def describe_tree_robot(robot, urdf_path, targets):
    """Canonical tree-model metadata (schema 1, kind fixed_base_tree) in GRiD joint order."""
    joints, out_targets, joint_target = validate_tree_robot(robot, targets)
    return dict(schema=1, kind="fixed_base_tree", base="external",
                urdf_sha256=hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
                joints=joints, targets=out_targets, joint_target_mask=joint_target)


def fixed_frame_index(header, target):
    """Read GRiD's resolved frame using its identifier escaping, not raw URDF names."""
    from grid_codegen.helpers._code_generation_helpers import ident_suffix
    symbol = "end_effector_pose_inner" + ident_suffix(target) + "("
    for start in re.finditer(re.escape(symbol), header):
        match = re.search(r"s_Xhom\[16\s*\*\s*(\d+)\s*\+\s*ind\]", header[start.start():start.start()+4000])
        if match:
            return int(match.group(1))
    raise ValueError(f"could not resolve the fixed-frame index of {target!r} in the generated header")


def joint_header(model, grid_sha="runtime-store"):
    """Emit host metadata and device switch accessors without dynamic host-array indexing."""
    joints = model["joints"]
    encoded = json.dumps(model, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    all_z = all(j["type"] != "prismatic" and j["axis"] == [0, 0, 1] for j in joints)
    lines = ["// Generated by HJCD-IK; do not edit.", f"// grid-sha256: {grid_sha}", "#pragma once",
             "namespace hjcd_model {", f"constexpr int NUM_JOINTS = {len(joints)};",
             f"constexpr bool ALL_REVOLUTE_Z = {'true' if all_z else 'false'};",
             f'constexpr const char* MODEL_SHA256 = "{digest}";',
             f"constexpr const char* MODEL_JSON = {json.dumps(encoded)};", "#if defined(__CUDACC__)",
             "__device__ __forceinline__ bool is_prismatic(int joint) {", "  switch (joint) {"]
    lines += [f"    case {i}: return true;" for i, j in enumerate(joints) if j["type"] == "prismatic"]
    lines += ["    default: return false;", "  }", "}",
              "template<class T> __device__ __forceinline__ void local_axis(int joint, T* a) {",
              "  switch (joint) {"]
    for i, j in enumerate(joints):
        axis = [format(x, ".17g") for x in j["axis"]]
        lines.append(f"    case {i}: a[0]=T({axis[0]}); a[1]=T({axis[1]}); a[2]=T({axis[2]}); return;")
    lines += ["    default: a[0]=a[1]=a[2]=T(0); return;", "  }", "}", "#endif", "}", ""]
    return "\n".join(lines)


def stage_joint_header(model, cache_dir):
    """Publish an immutable content-addressed input; GRiD still owns compiled artifacts."""
    data = joint_header(model).encode()
    digest = hashlib.sha256(data).hexdigest()
    directory = Path(cache_dir).expanduser().resolve() / "hjcdik-inputs" / digest
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "hjcd_joint_model.h"
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"generated joint header is corrupt: {path}")
        return path
    fd, staged = tempfile.mkstemp(prefix=".joint-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)
    return path


# ---------------------------------------------------------------------------------------------
# Tree-model code generation inputs (humanoid-v1 M2b). GRiD 0.6.0 bakes a single named fixed
# target per header (grid_rbd honors only ee_joint_names[0]); the per-joint homogeneous transform
# table s_XmatsHom (actuated joints first, then every fixed frame) is the same whichever target is
# named, so HJCD resolves each target's fixed-frame slot once and composes all targets itself.
# ---------------------------------------------------------------------------------------------

def tree_fixed_frame_index(header, frame):
    """Slot of `frame` in s_XmatsHom, read from GRiD's branched-robot EE epilogue (`eeInd = ... * k`)."""
    from grid_codegen.helpers._code_generation_helpers import ident_suffix
    symbol = "end_effector_pose_inner" + ident_suffix(frame) + "("
    for start in re.finditer(re.escape(symbol), header):
        match = re.search(r"int eeInd = \(ind < 16\) \* (\d+);", header[start.start():start.start()+4000])
        if match:
            return int(match.group(1))
    raise ValueError(f"could not resolve the tree fixed-frame index of {frame!r} in the generated header")


def tree_depths(model):
    depth = []
    for joint in model["joints"]:
        depth.append(0 if joint["parent"] < 0 else depth[joint["parent"]] + 1)
    return depth


def model_sha256(model):
    return hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def targets_header(model, fixed_indices):
    """hjcd_targets.cuh for a fixed-base tree: target order, anchors, GRiD fixed-frame slots, full
    rigid tools (column-major, both precisions), ancestry/dependency masks and the parent table."""
    targets, joints = model["targets"], model["joints"]
    if len(fixed_indices) != len(targets):
        raise ValueError("one fixed-frame index per target is required")
    depth = tree_depths(model)
    ints = lambda xs: "{" + ", ".join(str(int(x)) for x in xs) + "}"
    hexes = lambda xs: "{" + ", ".join(f"0x{int(x):08x}u" for x in xs) + "}"
    col = []
    for t in targets:
        R = t["tool_row_major"]
        col.append([R[r][c] for c in range(4) for r in range(4)])        # cell = 4*col + row
    dbl = ",\n    ".join(", ".join(format(x, ".17g") for x in c) for c in col)
    flt = ",\n    ".join(", ".join(format(x, ".9g") + "f" if "." in format(x, ".9g") or "e" in format(x, ".9g")
                                   else format(x, ".9g") + ".0f" for x in c) for c in col)
    lines = [
        "// Generated by HJCD-IK (hjcdik._robot_model.targets_header); do not edit.",
        "// Fixed-base tree target set. The floating base is external: these are base-frame poses.",
        "// Tools are FULL rigid 4x4 transforms anchor-link -> target frame, COLUMN-MAJOR (cell = 4*col+row),",
        "// the layout of GRiD's s_XmatsHom; a later per-call tool override REPLACES (never composes with) them.",
        "#pragma once",
        "namespace hjcd_targets {",
        f'constexpr const char* MODEL_SHA256 = "{model_sha256(model)}";',
        f"constexpr int NUM_JOINTS = {len(joints)};",
        f"constexpr int NUM_TARGETS = {len(targets)};",
        f"constexpr int MAX_DEPTH = {max(depth)};",
    ]
    lines += [f"//   [{k}] {t['name']}: anchor {t['anchor_joint']} (jid {t['anchor_jid']}), frame "
              f"{t['fixed_joint']} (s_XmatsHom slot {fixed_indices[k]})" for k, t in enumerate(targets)]
    lines += [
        "constexpr const char* TARGET_NAMES[NUM_TARGETS] = {" + ", ".join(f'"{t["name"]}"' for t in targets) + "};",
        "constexpr const char* TARGET_FRAMES[NUM_TARGETS] = {" + ", ".join(f'"{t["fixed_joint"]}"' for t in targets) + "};",
        "constexpr int H_TARGET_ANCHOR_JID[NUM_TARGETS] = " + ints(t["anchor_jid"] for t in targets) + ";",
        "constexpr int H_TARGET_FIXED_FRAME_IDX[NUM_TARGETS] = " + ints(fixed_indices) + ";",
        "constexpr unsigned int H_TARGET_ANCESTOR_MASK[NUM_TARGETS] = " + hexes(t["ancestor_mask"] for t in targets) + ";",
        "constexpr unsigned int H_JOINT_TARGET_MASK[NUM_JOINTS] = " + hexes(model["joint_target_mask"]) + ";",
        "constexpr int H_JOINT_PARENT[NUM_JOINTS] = " + ints(j["parent"] for j in joints) + ";",
        "constexpr int H_JOINT_DEPTH[NUM_JOINTS] = " + ints(depth) + ";",
        "constexpr double H_TARGET_TOOL[NUM_TARGETS * 16] = {\n    " + dbl + "};",
        "constexpr double H_JOINT_LOWER[NUM_JOINTS] = {" + ", ".join(format(j["limits"][0], ".17g") for j in joints) + "};",
        "constexpr double H_JOINT_UPPER[NUM_JOINTS] = {" + ", ".join(format(j["limits"][1], ".17g") for j in joints) + "};",
        "#if defined(__CUDACC__)",
        "__device__ constexpr int TARGET_ANCHOR_JID[NUM_TARGETS] = " + ints(t["anchor_jid"] for t in targets) + ";",
        "__device__ constexpr int TARGET_FIXED_FRAME_IDX[NUM_TARGETS] = " + ints(fixed_indices) + ";",
        "__device__ constexpr int JOINT_PARENT[NUM_JOINTS] = " + ints(j["parent"] for j in joints) + ";",
        "__device__ constexpr int JOINT_DEPTH[NUM_JOINTS] = " + ints(depth) + ";",
        "__device__ constexpr unsigned int TARGET_ANCESTOR_MASK[NUM_TARGETS] = " + hexes(t["ancestor_mask"] for t in targets) + ";",
        "__device__ constexpr double JOINT_LOWER[NUM_JOINTS] = {" + ", ".join(format(j["limits"][0], ".17g") for j in joints) + "};",
        "__device__ constexpr double JOINT_UPPER[NUM_JOINTS] = {" + ", ".join(format(j["limits"][1], ".17g") for j in joints) + "};",
        "__device__ constexpr double TARGET_TOOL[NUM_TARGETS * 16] = {\n    " + dbl + "};",
        "__device__ constexpr float TARGET_TOOL_F[NUM_TARGETS * 16] = {\n    " + flt + "};",
        "template<class T> __device__ __forceinline__ T tool(int i);",
        "template<> __device__ __forceinline__ double tool<double>(int i) { return TARGET_TOOL[i]; }",
        "template<> __device__ __forceinline__ float tool<float>(int i) { return TARGET_TOOL_F[i]; }",
        "#endif",
        "}",
        "",
    ]
    return "\n".join(lines)


def stage_tree_headers(model, fixed_indices, cache_dir):
    """Publish hjcd_joint_model.h + hjcd_targets.cuh for a tree model under one content-addressed directory."""
    files = {"hjcd_joint_model.h": joint_header(model).encode(),
             "hjcd_targets.cuh": targets_header(model, fixed_indices).encode()}
    digest = hashlib.sha256(b"".join(name.encode() + b"\0" + data for name, data in sorted(files.items()))).hexdigest()
    directory = Path(cache_dir).expanduser().resolve() / "hjcdik-inputs" / ("tree-" + digest)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = directory / name
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"generated tree header is corrupt: {path}")
            continue
        fd, staged = tempfile.mkstemp(prefix=".tree-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.replace(staged, path)
        finally:
            if os.path.exists(staged):
                os.unlink(staged)
    return directory
