#!/usr/bin/env python3
"""Generate the GRiD CUDA header (grid.cuh) for HJCD-IK from a URDF.

Uses the modern GRiD codegen API (URDFParser + GRiDCodeGenerator.gen_all_code);
the legacy top-level generateGRiD.py was removed upstream. The GRiD submodule
(external/GRiD) and its nested GRiDCodeGenerator / URDFParser must be initialized
(run scripts/setup_dev.sh, or git submodule update --init --recursive).

By default writes to include/test_cuh/grid.cuh (the committed, build-default header).
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GRID_DIR = REPO_ROOT / "external" / "GRiD"
DEFAULT_OUT = REPO_ROOT / "include" / "test_cuh" / "grid.cuh"


def main():
    ap = argparse.ArgumentParser(description="Generate grid.cuh via GRiD codegen.")
    ap.add_argument("urdf_path", help="Path to the URDF (relative to repo root, or absolute).")
    ap.add_argument("-t", "--fixed-target-name", default="panda_grasptarget_hand",
                    help="Fixed end-effector target frame (default: panda_grasptarget_hand).")
    ap.add_argument("-n", "--namespace", default="grid", help="File namespace (default: grid).")
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT), help="Output path for grid.cuh.")
    ap.add_argument("-f", "--floating-base", action="store_true", help="Add a floating base.")
    args = ap.parse_args()

    urdf = Path(args.urdf_path)
    if not urdf.is_absolute():
        urdf = (REPO_ROOT / urdf).resolve()
    if not urdf.exists():
        print(f"[error] URDF not found: {urdf}", file=sys.stderr)
        sys.exit(1)
    if not (GRID_DIR / "GRiDCodeGenerator").exists():
        print(f"[error] GRiD codegen not initialized at {GRID_DIR}. Run scripts/setup_dev.sh", file=sys.stderr)
        sys.exit(1)

    # Import from the GRiD submodule.
    sys.path.insert(0, str(GRID_DIR))
    from URDFParser import URDFParser            # noqa: E402
    from GRiDCodeGenerator import GRiDCodeGenerator  # noqa: E402

    robot = URDFParser().parse(str(urdf), floating_base=args.floating_base)
    print(f"[generate_grid] robot={robot.name} dof={robot.get_num_joints()} target={args.fixed_target_name}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True,
                                FILE_NAMESPACE=args.namespace)
    codegen.gen_all_code(
        include_homogenous_transforms=True,
        fixed_target_name=args.fixed_target_name,
        output_path=str(out),
    )
    print(f"[generate_grid] wrote {out}")

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


if __name__ == "__main__":
    main()
