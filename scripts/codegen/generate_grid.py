#!/usr/bin/env python3
"""Generate the GRiD CUDA header (grid.cuh) for HJCD-IK from a URDF.

Uses the modern GRiD codegen API (URDFParser + GRiDCodeGenerator.gen_all_code);
the legacy top-level generateGRiD.py was removed upstream. The GRiD submodule
(external/GRiD) and its nested GRiDCodeGenerator / URDFParser must be initialized
(run scripts/setup/setup_dev.sh, or git submodule update --init --recursive).

By default writes to csrc/generated/grid.cuh (the committed, build-default header).
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID_DIR = REPO_ROOT / "external" / "GRiD"
DEFAULT_OUT = REPO_ROOT / "csrc" / "generated" / "grid.cuh"
DEFAULT_TARGETS_OUT = REPO_ROOT / "csrc" / "generated" / "hjcd_targets.cuh"

# Compile-time ceiling on the target count. Bumping it costs nothing until targets are actually
# declared (loops are bounded by NUM_TARGETS), but the per-problem active mask is one bit per target.
MAX_TARGETS = 4


def main():
    ap = argparse.ArgumentParser(description="Generate grid.cuh via GRiD codegen.")
    ap.add_argument("urdf_path", help="Path to the URDF (relative to repo root, or absolute).")
    ap.add_argument("-t", "--fixed-target-name", default="panda_grasptarget_hand",
                    help="Fixed end-effector target frame (default: panda_grasptarget_hand).")
    ap.add_argument("-n", "--namespace", default="grid", help="File namespace (default: grid).")
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT), help="Output path for grid.cuh.")
    ap.add_argument("-f", "--floating-base", action="store_true", help="Add a floating base.")
    ap.add_argument("--collision", action="store_true",
                    help="Bake the grid_collision namespace (URDF->spheres) into grid.cuh. Off by "
                         "default => byte-identical to a no-collision header.")
    ap.add_argument("--collision-res", default="0.05",
                    help="Sphere spacing in meters (smaller=finer). A single value => one tier; a "
                         "comma list (e.g. 0.10,0.05) => a broad->fine cascade. Used with --collision "
                         "when spherizing the URDF's own collision geometry (default: 0.05). Ignored "
                         "when --spherized-urdf is given.")
    ap.add_argument("--spherized-urdf", default=None,
                    help="Path(s) to a pre-spherized URDF (foam interchange format: <collision> as "
                         "<sphere>). Use this when the kinematic URDF's collision meshes can't be "
                         "resolved (e.g. Panda). Spheres are read directly and bound to the robot's "
                         "GRiD frames -- no re-spherization. A comma list => coarsest->finest tiers.")
    ap.add_argument("--target", action="append", default=None, metavar="SPEC",
                    help="An end-effector target frame; repeat (in order) for multi-target. Each SPEC "
                         "is semicolon-separated key=value:\n"
                         "  name=<label>                       required, host-side only\n"
                         "  fixed=<fixed_joint_name>           tool transform + anchor from the URDF's "
                         "fixed joint\n"
                         "  anchor=<movable_joint_name>        anchor explicitly (required without "
                         "'fixed=')\n"
                         "  xyz=x,y,z   rpy=r,p,y              explicit tool offset (default identity)\n"
                         'e.g. --target "name=left_hand;fixed=left_hand_palm_joint"\n'
                         '     --target "name=left_foot;anchor=left_ankle_roll_joint;'
                         'xyz=0.035,0,-0.035"\n'
                         "Omit entirely to get a single target from -t (the pre-existing behavior).")
    ap.add_argument("--targets-output", default=str(DEFAULT_TARGETS_OUT),
                    help="Output path for the generated target-metadata header.")
    args = ap.parse_args()

    urdf = Path(args.urdf_path)
    if not urdf.is_absolute():
        urdf = (REPO_ROOT / urdf).resolve()
    if not urdf.exists():
        print(f"[error] URDF not found: {urdf}", file=sys.stderr)
        sys.exit(1)
    if not (GRID_DIR / "GRiDCodeGenerator").exists():
        print(f"[error] GRiD codegen not initialized at {GRID_DIR}. Run scripts/setup/setup_dev.sh", file=sys.stderr)
        sys.exit(1)

    # Import from the GRiD submodule.
    sys.path.insert(0, str(GRID_DIR))
    from URDFParser import URDFParser            # noqa: E402
    from GRiDCodeGenerator import GRiDCodeGenerator  # noqa: E402

    robot = URDFParser().parse(str(urdf), floating_base=args.floating_base)
    print(f"[generate_grid] robot={robot.name} dof={robot.get_num_joints()} target={args.fixed_target_name}")

    # --- optional collision spec (URDF -> covering spheres -> grid_collision namespace) ---
    collision_spec = None
    if args.collision:
        from GRiDCodeGenerator.algorithms._collision import (  # noqa: E402
            collision_spec_from_urdf, multi_tier_collision_spec_from_urdf, build_sphere_tiers)
        if args.spherized_urdf:
            # Read spheres directly from pre-spherized (foam) URDF(s); bind to the robot's GRiD frames.
            # Used when the kinematic URDF's collision meshes don't resolve (Panda) -- gives the exact
            # baked sphere model (e.g. Panda's 59-sphere paper model) rather than re-spherizing.
            sph_paths = [p.strip() for p in str(args.spherized_urdf).split(",") if p.strip()]
            for p in sph_paths:
                sp = Path(p)
                if not sp.is_absolute():
                    sp = (REPO_ROOT / p).resolve()
                if not sp.exists():
                    print(f"[error] spherized URDF not found: {sp}", file=sys.stderr)
                    sys.exit(1)
            resolved = [str((REPO_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p))
                        for p in sph_paths]
            if len(resolved) == 1:
                tier = build_sphere_tiers(robot, {"all": resolved[0]})["all"]
                collision_spec = {"anchor": tier["anchor"], "offset": tier["offset"],
                                  "radius": tier["radius"], "self_cc_ranges": tier["self_cc_ranges"]}
            else:
                # coarsest->finest, named broad/fine (2) or tier0..tierK (>2)
                names = (["broad", "fine"] if len(resolved) == 2
                         else [f"tier{i}" for i in range(len(resolved))])
                tiers = build_sphere_tiers(robot, {n: p for n, p in zip(names, resolved)})
                collision_spec = {"tiers": [{"name": n, **tiers[n]} for n in names]}
            print(f"[generate_grid] collision ON from spherized URDF(s)={resolved}")
        else:
            resolutions = [float(r) for r in str(args.collision_res).split(",") if r.strip()]
            if len(resolutions) == 1:
                collision_spec = collision_spec_from_urdf(robot, str(urdf), resolution=resolutions[0])
            else:
                collision_spec = multi_tier_collision_spec_from_urdf(robot, str(urdf), resolutions)
            print(f"[generate_grid] collision ON, spherizing {urdf.name} at resolution(s)={resolutions}m")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True,
                                FILE_NAMESPACE=args.namespace)
    codegen.gen_all_code(
        include_homogenous_transforms=True,
        fixed_target_name=args.fixed_target_name,
        output_path=str(out),
        collision_spec=collision_spec,
    )
    print(f"[generate_grid] wrote {out}")

    # --- inject a collision-presence sentinel (compile guard for the kernel) ---
    # grid.cuh only carries the grid_collision namespace when --collision was passed; the kernel and
    # grid_env.cuh reference grid_collision:: symbols only under #if defined(HJCD_HAS_COLLISION), so a
    # no-collision header (e.g. the DoF-scaling regens) still compiles. This is codegen output, not a
    # hand-edit — every regen re-emits it deterministically.
    if args.collision:
        text = out.read_text()
        if "HJCD_HAS_COLLISION" not in text:
            out.write_text("#define HJCD_HAS_COLLISION 1\n" + text)
        print("[generate_grid] collision sentinel: #define HJCD_HAS_COLLISION 1")

    # --- resolve + inject the EE fixed-frame index (per-robot, not hardcoded) ---
    # GRiD emits end_effector_pose_inner_<target>() whose epilogue composes the EE offset from a
    # single s_Xhom frame: `s_temp[ind] = s_Xhom[16*IDX + ind];`. That IDX is where the named EE
    # frame lands and it SHIFTS with DoF, so the kernel must read it from the header rather than
    # hardcode it. We expose it as grid::EE_FIXED_FRAME_IDX (consumed by hjcd_settings.h).
    if args.fixed_target_name:
        text = out.read_text()
        # Resolve the frame index from the parsed robot, not by scraping the emitted text. GRiD gives
        # fixed joints ids NUM_JOINTS..NUM_JOINTS+NFJ-1, which index s_Xhom directly. (The former
        # regex scrape of `s_Xhom[16*IDX + ind]` searched a fixed 4000-char window after the target
        # function's opening brace; that window is far too small for a 29-joint robot like G1, where
        # the epilogue it looks for sits thousands of lines further down, so it silently failed.)
        fj = robot.get_fixed_joint_by_name(args.fixed_target_name)
        if fj is None:
            names = robot.get_fixed_joint_names()
            print(f"[generate_grid][ERROR] '{args.fixed_target_name}' is not a fixed joint in this "
                  f"URDF. Available: {names}", file=sys.stderr)
            sys.exit(2)
        idx = fj.get_id()
        # Cross-check against the emitted text when the epilogue is reachable, so a GRiD indexing
        # change can't drift away from us unnoticed.
        m0 = text.find(f"end_effector_pose_inner_{args.fixed_target_name}(")
        if m0 >= 0:
            m = re.search(r"s_Xhom\[16\s*\*\s*(\d+)\s*\+\s*ind\]", text[m0:])
            if m and int(m.group(1)) != idx:
                print(f"[generate_grid][ERROR] EE frame index mismatch: URDFParser says {idx}, "
                      f"emitted code uses {m.group(1)}", file=sys.stderr)
                sys.exit(2)
        if "EE_FIXED_FRAME_IDX" not in text:
            inj = (f"    // codegen-resolved EE fixed-frame index for target "
                   f"'{args.fixed_target_name}' (shifts with DoF; consumed by hjcd_settings.h)\n"
                   f"    constexpr int EE_FIXED_FRAME_IDX = {idx};\n")
            new_text, n = re.subn(r"(const int NUM_JOINTS = \d+;\n)", r"\1" + inj, text, count=1)
            if n != 1:
                print("[generate_grid][ERROR] could not find NUM_JOINTS anchor to inject "
                      "EE_FIXED_FRAME_IDX", file=sys.stderr)
                sys.exit(2)
            out.write_text(new_text)
        print(f"[generate_grid] EE_FIXED_FRAME_IDX = {idx}  (target '{args.fixed_target_name}')")
    else:
        print("[generate_grid][warn] no fixed target — EE_FIXED_FRAME_IDX not injected "
              "(kernel grasptarget path requires a fixed EE frame).")

    inject_joint_axis_metadata(robot, out)
    emit_target_metadata(robot, args, Path(args.targets_output))


# ---------------------------------------------------------------------------
# Target-frame metadata (the multi-target set). Names and URDF lookups live HERE, on the host, at
# codegen time; the device only ever sees indices, bitmasks, and baked 4x4s.
# ---------------------------------------------------------------------------

def _rpy_to_R(rpy):
    import math
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]]


def _parse_target_spec(spec):
    d = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"[generate_grid] bad --target field '{part}' (want key=value)")
        k, v = part.split("=", 1)
        d[k.strip()] = v.strip()
    if "name" not in d:
        raise SystemExit(f"[generate_grid] --target '{spec}' is missing name=")
    if "fixed" not in d and "anchor" not in d:
        raise SystemExit(f"[generate_grid] --target '{d['name']}' needs fixed= or anchor=")
    return d


def _resolve_targets(robot, args):
    """-> ordered list of {name, anchor_jid, anchor_name, tool (row-major 4x4), source}."""
    specs = args.target
    if not specs:
        # Backward compatible: no --target => the single -t fixed frame (Panda and friends).
        if not args.fixed_target_name:
            return []
        specs = [f"name={args.fixed_target_name};fixed={args.fixed_target_name}"]

    if len(specs) > MAX_TARGETS:
        raise SystemExit(f"[generate_grid] {len(specs)} targets exceeds MAX_TARGETS={MAX_TARGETS}")

    out = []
    for spec in specs:
        d = _parse_target_spec(spec)
        tool = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        source = "identity"

        if "fixed" in d:
            fj = robot.get_fixed_joint_by_name(d["fixed"])
            if fj is None:
                raise SystemExit(f"[generate_grid] '{d['fixed']}' is not a fixed joint. "
                                 f"Available: {robot.get_fixed_joint_names()}")
            # GRiD folds fixed joints onto their nearest MOVABLE parent and pre-composes the
            # transform, so this 4x4 is already anchor->tool (row-major pose, not a Featherstone
            # frame transform). That makes it directly usable as our tool offset.
            tool = [[float(v) for v in row] for row in fj.get_transformation_matrix_hom()]
            anchor_name = d.get("anchor", fj.get_parent())
            source = f"fixed joint '{d['fixed']}'"
            if anchor_name == -1 or anchor_name is None:
                raise SystemExit(f"[generate_grid] fixed joint '{d['fixed']}' hangs off the world "
                                 f"root; it has no movable anchor.")
        else:
            anchor_name = d["anchor"]

        if "xyz" in d or "rpy" in d:
            if "fixed" in d:
                raise SystemExit(f"[generate_grid] target '{d['name']}': xyz/rpy cannot be combined "
                                 f"with fixed= (the fixed joint already defines the transform)")
            xyz = [float(v) for v in d.get("xyz", "0,0,0").split(",")]
            rpy = [float(v) for v in d.get("rpy", "0,0,0").split(",")]
            R = _rpy_to_R(rpy)
            tool = [[R[i][j] for j in range(3)] + [xyz[i]] for i in range(3)] + [[0, 0, 0, 1]]
            source = f"explicit xyz={xyz} rpy={rpy}"

        aj = robot.get_joint_by_name(anchor_name)
        if aj is None:
            raise SystemExit(f"[generate_grid] anchor '{anchor_name}' is not a movable joint")
        out.append(dict(name=d["name"], anchor_jid=aj.get_id(), anchor_name=anchor_name,
                        tool=tool, source=source, fixed_joint=d.get("fixed")))
    return out


def _ancestors_of(robot, jid):
    """Ancestor-or-self joint ids of `jid` (GRiD's parent chain; -1 terminates at the world)."""
    out, cur = set(), jid
    while cur is not None and cur >= 0:
        out.add(cur)
        cur = robot.get_parent_id(cur)
    return out


def emit_target_metadata(robot, args, out_path):
    import json
    targets = _resolve_targets(robot, args)
    n = robot.get_num_joints()
    if not targets:
        print("[generate_grid][warn] no targets resolved; skipping target metadata")
        return
    if n > 32:
        raise SystemExit(f"[generate_grid] {n} joints > 32: the ancestor masks are uint32_t "
                         f"(one bit per joint). A wider mask type would be needed.")

    # M[k] = set of joints that can move target k = ancestors-or-self of its anchor.
    anc = [_ancestors_of(robot, t["anchor_jid"]) for t in targets]
    target_ancestor_mask = [sum(1 << j for j in a) for a in anc]
    joint_target_mask = [sum(1 << k for k in range(len(targets)) if j in anc[k]) for j in range(n)]

    # Parent table + descendant masks for the incremental (subtree) FK.
    parent = [robot.get_parent_id(j) for j in range(n)]

    # PARENT-BEFORE-CHILD is the whole correctness argument for the ascending mask scan
    #     for (u = 0; u < N; ++u) if (desc & 1<<u) X[u] = X[parent[u]] * Xloc[u];
    # GRiD renumbers joints by a DFS pre-order from the root (URDFParser.renumber_linksJoints ->
    # dfs_order_update), so parent(u) < u always. Assert it rather than trust it: if a future GRiD
    # ever renumbered differently, this scan would read a stale parent transform and be silently wrong.
    for u in range(n):
        if not (parent[u] < u):
            raise SystemExit(f"[generate_grid][ERROR] joint {u} has parent {parent[u]} >= {u}: GRiD's "
                             f"ids are not parent-before-child, so the ascending subtree scan is "
                             f"invalid. A topological order would have to be emitted explicitly.")

    # Subtree of j INCLUDING j itself: the scan recomputes X_world[j] from its (unchanged) parent and
    # its (updated) local transform, so j must be in its own mask.
    descendant_mask = [0] * n
    for j in range(n):
        m = 1 << j
        for u in range(j + 1, n):          # parent < child, so a descendant always has a larger id
            if parent[u] >= 0 and (m >> parent[u]) & 1:
                m |= 1 << u
        descendant_mask[j] = m

    def hexu(v):
        return f"0x{v:08x}u"

    lines = [
        "// GENERATED by scripts/codegen/generate_grid.py -- DO NOT HAND-EDIT.",
        "//",
        "// The multi-target end-effector set. Target NAMES and all URDF lookups are resolved here at",
        "// codegen time; the CUDA hot path sees only indices, bitmasks and baked 4x4s -- never a string.",
        "//",
        "// Include AFTER grid.cuh (which has no include guard of its own, so this header must not pull",
        "// it in a second time). csrc/kernel/hjcd_settings.h is the intended -- and only -- includer.",
        "#pragma once",
        "",
        "namespace hjcd_gen {",
        "",
        f"constexpr int MAX_TARGETS = {MAX_TARGETS};",
        f"constexpr int NUM_TARGETS = {len(targets)};",
        "static_assert(NUM_TARGETS <= MAX_TARGETS, \"too many targets\");",
        "static_assert(grid::NUM_JOINTS <= 32, \"ancestor masks are uint32_t: one bit per joint\");",
        "",
        "// Ordered target set (index == device-side target id):",
    ]
    for k, t in enumerate(targets):
        lines.append(f"//   [{k}] {t['name']:<12s} anchor={t['anchor_name']} (jid {t['anchor_jid']}), "
                     f"tool from {t['source']}")
    lines += [
        "",
        "__device__ constexpr int TARGET_ANCHOR_JID[NUM_TARGETS] = {"
        + ", ".join(str(t["anchor_jid"]) for t in targets) + "};",
        "",
        "// Tool offset anchor->target, COLUMN-MAJOR 4x4 (cell = 16*k + 4*col + row), matching the",
        "// layout of grid's s_XmatsHom / s_jointXforms so it composes with them directly.",
        "//",
        "// Emitted in BOTH precisions. An fp32 kernel reading the double table would issue a 64-bit",
        "// load plus an F2F.F32.F64 conversion for every one of the 16 cells, per target, on every",
        "// target composition -- FP64-pipe work in the hot loop on a GPU that runs FP64 at 1/64 rate.",
        "// tool_xform<T>() below picks the matching table, so an fp32 solve never touches the double one.",
        "__device__ constexpr double TARGET_TOOL_XFORM[NUM_TARGETS * 16] = {",
    ]
    for t in targets:
        cm = [t["tool"][r][c] for c in range(4) for r in range(4)]   # row-major -> column-major
        lines.append("    " + ", ".join(f"{v:.17g}" for v in cm) + f",   // {t['name']}")
    lines += [
        "};",
        "",
        "__device__ constexpr float TARGET_TOOL_XFORM_F[NUM_TARGETS * 16] = {",
    ]
    def _fl(v):
        # "%.9g" of 0.0 is "0", and "0f" is not a valid float literal -- force a decimal point.
        s = f"{v:.9g}"
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s + "f"

    for t in targets:
        cm = [t["tool"][r][c] for c in range(4) for r in range(4)]
        lines.append("    " + ", ".join(_fl(v) for v in cm) + f",   // {t['name']}")
    lines += [
        "};",
        "",
        "// The tool transform in the kernel's own compute type. Specialised, never converted.",
        "template<typename T> __device__ __forceinline__ T tool_xform(int i);",
        "template<> __device__ __forceinline__ double tool_xform<double>(int i) "
        "{ return TARGET_TOOL_XFORM[i]; }",
        "template<> __device__ __forceinline__ float  tool_xform<float>(int i) "
        "{ return TARGET_TOOL_XFORM_F[i]; }",
        "",
        "// TARGET_ANCESTOR_MASK[k]: bit j set iff joint j can move target k (j is an ancestor-or-self",
        "// of target k's anchor). A joint outside this mask contributes a ZERO Jacobian column for k --",
        "// on a branched robot that is a correctness requirement, not an optimization.",
        "__device__ constexpr unsigned int TARGET_ANCESTOR_MASK[NUM_TARGETS] = {"
        + ", ".join(hexu(m) for m in target_ancestor_mask) + "};",
        "",
        "// JOINT_TARGET_MASK[j]: bit k set iff joint j affects target k. The transpose of the above;",
        "// lets the hot path do  affected = JOINT_TARGET_MASK[j] & active_target_mask  in one AND.",
        "__device__ constexpr unsigned int JOINT_TARGET_MASK[grid::NUM_JOINTS] = {"
        + ", ".join(hexu(m) for m in joint_target_mask) + "};",
        "",
        "constexpr unsigned int ALL_TARGETS_MASK = (NUM_TARGETS >= 32) ? 0xffffffffu",
        "                                                              : ((1u << NUM_TARGETS) - 1u);",
        "",
        "// --- incremental (subtree) FK ---------------------------------------------------------",
        "// JOINT_PARENT_JID[u]: u's parent joint, -1 for a joint attached to the world root.",
        "// GRiD numbers joints in DFS pre-order, so parent(u) < u ALWAYS (asserted at codegen). That",
        "// is what makes the ascending scan below parent-before-child without any explicit topo order:",
        "//     for (u = 0; u < NUM_JOINTS; ++u)",
        "//         if (JOINT_DESCENDANT_MASK[j] & (1u << u))",
        "//             X_world[u] = X_world[JOINT_PARENT_JID[u]] * X_local[u];",
        "__device__ constexpr int JOINT_PARENT_JID[grid::NUM_JOINTS] = {"
        + ", ".join(str(p) for p in parent) + "};",
        "",
        "// JOINT_DESCENDANT_MASK[j]: bit u set iff u is in j's subtree. INCLUDES j ITSELF -- the scan",
        "// must recompute X_world[j] too, since j's LOCAL transform changed even though its parent's",
        "// world transform did not.",
        "__device__ constexpr unsigned int JOINT_DESCENDANT_MASK[grid::NUM_JOINTS] = {"
        + ", ".join(hexu(m) for m in descendant_mask) + "};",
        "",
        "}  // namespace hjcd_gen",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

    # Host-side sidecar: target names + the same numbers, for tests and the Python arg validation
    # layer. Never read by device code.
    meta = dict(
        robot=robot.name, num_joints=n, max_targets=MAX_TARGETS,
        targets=[dict(name=t["name"], anchor_jid=t["anchor_jid"], anchor_name=t["anchor_name"],
                      tool_row_major=t["tool"], source=t["source"], fixed_joint=t["fixed_joint"])
                 for t in targets],
        target_ancestor_mask=target_ancestor_mask,
        joint_target_mask=joint_target_mask,
        joint_parent_jid=parent,
        joint_descendant_mask=descendant_mask,
    )
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"[generate_grid] targets ({len(targets)}): "
          + ", ".join(f"{k}:{t['name']}@jid{t['anchor_jid']}" for k, t in enumerate(targets)))
    for k, t in enumerate(targets):
        print(f"[generate_grid]   [{k}] {t['name']:<12s} ancestors={bin(target_ancestor_mask[k])}")
    print(f"[generate_grid] wrote {out_path} (+ .json sidecar)")


def inject_joint_axis_metadata(robot, out):
    """Inject per-joint motion-axis metadata (grid::JOINT_AXIS_*) into grid.cuh.

    GRiD does NOT rotate joint axes onto local z — it keeps the URDF <axis> and records it in the
    motion subspace S (URDFParser/Joint.py set_type). So a joint's WORLD axis is not universally the
    z-column of its world transform; it is:

        axis_world = JOINT_AXIS_SIGN[j] * column(JOINT_AXIS_COL[j]) of s_jointXforms[16*j]

    which holds because R_world[j] = R_parent * R_origin * Rot(a_local, q_j) and Rot(a,q)*e_a == e_a,
    so the a-th column of R_world[j] is exactly R_parent*R_origin*e_a (q-independent, as an axis must
    be). Panda is all-z (S index 2, sign +1) so this reduces to the old hardcoded Ci[8..10].

    S index 0..2 => revolute about local axis e_index; 3..5 => prismatic along e_(index-3).
    Skew (non-cardinal) axes have no single signed index; get_S_index_by_id raises on them, which is
    the correct outcome — the kernel's column-selection identity does not hold for a skew axis.
    """
    n = robot.get_num_joints()
    cols, signs, prism = [], [], []
    for jid in range(n):
        try:
            s_idx = robot.get_S_index_by_id(jid)
            s_sgn = robot.get_S_sign_by_id(jid)
        except ValueError as e:
            print(f"[generate_grid][ERROR] joint {jid} has no single signed motion axis: {e}\n"
                  f"  HJCD's Jacobian/coordinate-descent need one. Skew-axis robots are unsupported.",
                  file=sys.stderr)
            sys.exit(2)
        is_prism = 1 if s_idx >= 3 else 0
        cols.append(s_idx - 3 if is_prism else s_idx)
        signs.append(s_sgn)
        prism.append(is_prism)

    def arr(name, vals):
        return (f"    __device__ constexpr int {name}[{n}] = "
                f"{{{', '.join(str(v) for v in vals)}}};\n")

    text = out.read_text()
    if "JOINT_AXIS_COL" in text:
        return
    # __device__ constexpr (not __constant__): PTX shows .const serializes when lanes read DIFFERENT
    # elements (the hot case — lane i reads joint i), whereas .global + ld.global.nc coalesces AND
    # still constant-folds to zero loads under a compile-time index. See the Phase-1A report.
    inj = ("    // Per-joint motion axis (codegen-resolved from the URDF <axis> via GRiD's motion\n"
           "    // subspace S). axis_world[j] = JOINT_AXIS_SIGN[j] * column(JOINT_AXIS_COL[j]) of the\n"
           "    // world transform of joint j. NOT always z: Panda is all-z, G1 is 13y/9x/7z.\n"
           "    // JOINT_IS_PRISMATIC[j] selects the prismatic Jacobian column (Jv = axis, Jw = 0);\n"
           "    // HAS_PRISMATIC lets an all-revolute robot compile that branch out entirely (the\n"
           "    // per-lane index is a runtime value, so without it the compiler must keep both arms\n"
           "    // of the ternary live — measured at +26 registers on lm_tuner<double>).\n"
           + arr("JOINT_AXIS_COL", cols)
           + arr("JOINT_AXIS_SIGN", signs)
           + arr("JOINT_IS_PRISMATIC", prism)
           + f"    constexpr bool HAS_PRISMATIC = {'true' if any(prism) else 'false'};\n"
           # All-z robots (Panda and its DoF variants, the Fetch arm) keep the old compile-time
           # constant column offsets 8/9/10 — the indexed read costs +26 registers on lm_tuner<double>
           # and buys nothing there, so the pre-existing numerical path stays bit-identical.
           + f"    constexpr bool ALL_AXIS_Z = "
             f"{'true' if all(c == 2 and s == 1 for c, s in zip(cols, signs)) else 'false'};\n")
    new_text, k = re.subn(r"(const int NUM_JOINTS = \d+;\n)", r"\1" + inj, text, count=1)
    if k != 1:
        print("[generate_grid][ERROR] could not find NUM_JOINTS anchor to inject joint-axis metadata",
              file=sys.stderr)
        sys.exit(2)
    out.write_text(new_text)
    hist = {c: cols.count(c) for c in sorted(set(cols))}
    print(f"[generate_grid] joint axes: cols={hist} (0=x,1=y,2=z)  "
          f"prismatic={sum(prism)}/{n}  signs={{'+': {signs.count(1)}, '-': {signs.count(-1)}}}")


if __name__ == "__main__":
    main()
