#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GRID_DIR  = REPO_ROOT / "external" / "GRiD"
SCRIPT    = GRID_DIR / "generateGRiD.py"


def main():
    parser = argparse.ArgumentParser(
        description="Run generateGRiD.py and place grid.cuh in external/GRiD/."
    )
    parser.add_argument(
        "urdf_path",
        help="Path to the URDF file (relative to repo root, or absolute)",
    )
    parser.add_argument(
        "-t", "--fixed-target-names", default="",
        help="Fixed joint kinematic target name (e.g. panda_grasptarget_hand)",
    )
    parser.add_argument(
        "-n", "--namespace", default="grid",
        help="File namespace name (default: grid)",
    )
    parser.add_argument("-d", "--debug",         action="store_true", help="Enable debug mode")
    parser.add_argument("-f", "--floating-base", action="store_true", help="Add a floating base")
    args = parser.parse_args()

    urdf = Path(args.urdf_path)
    if not urdf.is_absolute():
        urdf = (REPO_ROOT / urdf).resolve()
    if not urdf.exists():
        print(f"[error] URDF not found: {urdf}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(SCRIPT), str(urdf)]
    if args.fixed_target_names:
        cmd += ["-t", args.fixed_target_names]
    if args.namespace != "grid":
        cmd += ["-n", args.namespace]
    if args.debug:
        cmd.append("-d")
    if args.floating_base:
        cmd.append("-f")

    print(f"[generate_grid] cwd : {GRID_DIR}")
    print(f"[generate_grid] cmd : {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd, cwd=GRID_DIR).returncode)


if __name__ == "__main__":
    main()
