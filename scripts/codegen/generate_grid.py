#!/usr/bin/env python3
"""Generate the GRiD CUDA header (grid.cuh) for HJCD-IK from a URDF.

Uses the modern GRiD codegen API (URDFParser + grid_codegen.GRiDCodeGenerator.gen_all_code);
post packaging-fold layout: the grid_codegen package sits at the GRiD root and
URDFParser is a nested submodule under external/GRiD/external/. Both must be
initialized (run scripts/setup/setup_dev.sh, or git submodule update --init --recursive).

By default writes to csrc/generated/grid.cuh (the committed, build-default header).
"""
import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID_DIR = REPO_ROOT / "external" / "GRiD"
DEFAULT_OUT = REPO_ROOT / "csrc" / "generated" / "grid.cuh"


def validate_solver_robot(robot):
    """Reject GRiD models that HJCD's warp solver cannot represent correctly."""
    if robot.floating_base or not robot.is_serial_chain():
        raise ValueError("HJCD-IK requires a fixed-base serial chain")
    if not 1 <= robot.get_num_joints() <= 32:
        raise ValueError("HJCD-IK requires 1 to 32 actuated joints (one joint per warp lane)")
    for joint in robot.get_joints_ordered_by_id():
        # The geometric Jacobian uses each joint frame's +Z axis as a revolute axis.
        # GRiD supports more joint types, but accepting them here silently produces wrong IK.
        if joint.is_mimic or joint.jtype not in ("revolute", "continuous"):
            raise ValueError("HJCD-IK requires independent revolute or continuous joints")
        if list(robot.get_S_by_id(joint.jid).reshape(-1)) != [0, 0, 1, 0, 0, 0]:
            raise ValueError("HJCD-IK requires each actuated joint to rotate around local +Z")


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
    ap.add_argument("--profile", choices=("kinematics", "all"), default="kinematics",
                    help="GRiD algorithms to emit (default: kinematics, which is all HJCD-IK uses).")
    ap.add_argument("--glass-revision", default=None,
                    help="GLASS commit label for Git-free source archives; verified against Git when available.")
    args = ap.parse_args()
    if args.floating_base:
        ap.error("HJCD-IK requires a fixed-base serial chain")
    if args.namespace != "grid":
        ap.error("HJCD-IK requires the grid namespace")
    if not args.fixed_target_name:
        ap.error("HJCD-IK requires a fixed end-effector target frame")

    urdf = Path(args.urdf_path)
    if not urdf.is_absolute():
        urdf = (REPO_ROOT / urdf).resolve()
    if not urdf.exists():
        print(f"[error] URDF not found: {urdf}", file=sys.stderr)
        sys.exit(1)
    if not (GRID_DIR / "grid_codegen").exists():
        print(f"[error] GRiD codegen not initialized at {GRID_DIR}. Run scripts/setup/setup_dev.sh", file=sys.stderr)
        sys.exit(1)

    # Import from the GRiD submodule (post packaging-fold layout: grid_codegen package at the
    # GRiD root, URDFParser a nested submodule under external/).
    sys.path.insert(0, str(GRID_DIR))
    sys.path.insert(0, str(GRID_DIR / "external"))
    from URDFParser import URDFParser            # noqa: E402
    from grid_codegen import GRiDCodeGenerator   # noqa: E402

    robot = URDFParser().parse(str(urdf), floating_base=args.floating_base)
    try:
        validate_solver_robot(robot)
    except ValueError as error:
        ap.error(str(error))
    fixed_target = robot.get_fixed_joint_by_name(args.fixed_target_name)
    if fixed_target is None or fixed_target.parent_name != robot.get_joint_by_id(robot.get_num_joints() - 1).name:
        ap.error("HJCD-IK requires a fixed end-effector target attached to the final actuated joint")
    print(f"[generate_grid] robot={robot.name} dof={robot.get_num_joints()} target={args.fixed_target_name}")

    # --- optional collision spec (URDF -> covering spheres -> grid_collision namespace) ---
    collision_spec = None
    if args.collision:
        from grid_codegen.algorithms._collision import (  # noqa: E402
            collision_spec_from_urdf, multi_tier_collision_spec_from_urdf, build_sphere_tiers)
        if args.spherized_urdf:
            # Read spheres directly from pre-spherized (foam) URDF(s); bind to the robot's GRiD frames.
            # Avoid re-spherizing missing meshes (Panda). Fixed-link poses come from the
            # kinematic URDF, so a different gripper opening changes the placed sphere model.
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
    # Keep the previous buildable header until every generation/postprocessing check succeeds.
    with tempfile.TemporaryDirectory(prefix=".hjcd-codegen-", dir=out.parent) as temporary:
        staged = Path(temporary) / out.name
        emit_header(codegen, args, staged, collision_spec)
        staged.replace(out)
    print(f"[generate_grid] wrote {out}")
    write_collision_tables(out.parent / "hjcd_collision_tables.cuh", collision_spec)


def emit_header(codegen, args, out, collision_spec):
    # vendor_glass=False: the generated kinematics consume the top-level external/GLASS
    # (`#include "glass.cuh"` + `namespace glass = ::glass;`) instead of inlining a second,
    # pinned GLASS snapshot under grid::glass. HJCD-IK already links that GLASS, so this keeps
    # ONE GLASS per translation unit and drops ~9k vendored lines from grid.cuh. The nested
    # GRiD GLASS pin must match external/GLASS (bootstrap.sh checks the gitlinks).
    codegen.gen_all_code(
        codegen_profile=args.profile,
        glass_revision=args.glass_revision,
        vendor_glass=False,
        include_homogenous_transforms=True,
        fixed_target_name=args.fixed_target_name,
        output_path=str(out),
        collision_spec=collision_spec,
    )

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
    import re
    if args.fixed_target_name:
        text = out.read_text()
        idx = None
        for m0 in re.finditer(re.escape(f"end_effector_pose_inner_{args.fixed_target_name}("), text):
            m = re.search(r"s_Xhom\[16\s*\*\s*(\d+)\s*\+\s*ind\]", text[m0.start():m0.start() + 4000])
            if m:
                idx = int(m.group(1))
                break
        if idx is None:
            print(f"[generate_grid][ERROR] could not resolve EE_FIXED_FRAME_IDX for target "
                  f"'{args.fixed_target_name}' — kernel EE will be wrong.", file=sys.stderr)
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


    # --- sidecar: the sphere batch in a form the kernel can place from its OWN joint transforms ---
    # grid_collision bakes the anchor/offset tables inside its FK extractor (function-local), so the
    # warp-scoped collision check in hjcd_kernel.cu (collision-aware early stop, repair round) gets the
    # same batch here: anchor = GRiD movable-joint slot of s_jointX, offset = local xyz in that frame,
    # radius; base spheres already dropped, order identical to grid_collision's. Always rewritten (an
    # open-world regen leaves an EMPTY table so the kernel's static_assert stays honest).
    # Normalize generator formatting once, keeping reproducible headers and clean diffs.
    out.write_text("".join(line.rstrip() + "\n" for line in out.read_text().splitlines()))


def write_collision_tables(path, collision_spec):
    tier = None
    if collision_spec:
        tier = collision_spec["tiers"][-1] if "tiers" in collision_spec else collision_spec   # finest tier
    anchor = list(tier["anchor"]) if tier else []
    offset = [float(v) for v in tier["offset"]] if tier else []
    radius = [float(v) for v in tier["radius"]] if tier else []
    n = len(anchor)
    assert len(offset) == 3 * n and len(radius) == n, "collision spec arrays disagree"
    fmt = lambda vals, conv: ", ".join(conv(v) for v in vals) if vals else "0"
    text = (
        "// GENERATED by scripts/codegen/generate_grid.py alongside grid.cuh — do not hand-edit.\n"
        "// Collision sphere batch for the kernel's warp-scoped checks: anchor = GRiD movable-joint slot\n"
        "// (index into s_jointX[16*slot]), offset = local xyz in that frame (welded frames folded in),\n"
        "// radius in metres. Same order and count as grid_collision's baked batch (base spheres dropped).\n"
        "#pragma once\n"
        "namespace hjcd_cc {\n"
        f"    constexpr int NUM_SPHERES = {n};\n"
        f"    __device__ __constant__ int   sphere_anchor[{max(n, 1)}] = {{ {fmt(anchor, lambda v: str(int(v)))} }};\n"
        f"    __device__ __constant__ float sphere_offset[{max(3 * n, 1)}] = {{ {fmt(offset, lambda v: repr(float(v)) + 'f')} }};\n"
        f"    __device__ __constant__ float sphere_radius[{max(n, 1)}] = {{ {fmt(radius, lambda v: repr(float(v)) + 'f')} }};\n"
        "}\n"
    )
    Path(path).write_text(text)
    print(f"[generate_grid] wrote {path} ({n} spheres)")


if __name__ == "__main__":
    main()
