#!/usr/bin/env python3
"""Fail if the committed grid.cuh is stale w.r.t. current GRiD codegen.

Regenerates the Panda header (via scripts/generate_grid.py, which uses the GRiD
submodule's URDFParser + GRiDCodeGenerator) into a temp file and diffs it against
the committed include/test_cuh/grid.cuh. Run in CI and locally after touching the
URDF or bumping the GRiD submodule.

Exit codes: 0 = fresh, 1 = stale (prints a unified diff summary), 2 = codegen could not run.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
URDF = REPO / "include" / "test_urdf" / "panda.urdf"
TARGET = "panda_grasptarget_hand"
COMMITTED = REPO / "include" / "test_cuh" / "grid.cuh"
GEN = REPO / "scripts" / "generate_grid.py"
GRID_CODEGEN = REPO / "external" / "GRiD" / "GRiDCodeGenerator"


def main() -> int:
    if not GRID_CODEGEN.exists():
        print(f"[stale-check] GRiD codegen not initialized at {GRID_CODEGEN} — run scripts/bootstrap.sh",
              file=sys.stderr)
        return 2
    if not COMMITTED.exists():
        print(f"[stale-check] committed header missing: {COMMITTED}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        regenerated = Path(tmp) / "grid.cuh"
        rc = subprocess.run(
            [sys.executable, str(GEN), str(URDF), "-t", TARGET, "-o", str(regenerated)],
        ).returncode
        if rc != 0 or not regenerated.exists():
            print("[stale-check] codegen failed", file=sys.stderr)
            return 2

        diff = subprocess.run(
            ["diff", "-u", str(COMMITTED), str(regenerated)],
            capture_output=True, text=True,
        )
        if diff.returncode == 0:
            print("[stale-check] grid.cuh is fresh ✓")
            return 0
        print("[stale-check] grid.cuh is STALE — regenerate and commit:")
        print("\n".join(diff.stdout.splitlines()[:40]))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
