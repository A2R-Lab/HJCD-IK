#!/usr/bin/env python3
"""Collect the full suite and update its reviewed GPU-proof test manifest."""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-o", "addopts="],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if run.returncode or re.search(r"\b\d+ skipped\b", run.stdout):
        sys.exit("Full collection failed or skipped modules; install .[dev,codegen].\n"
                 + run.stdout + run.stderr)
    nodes = sorted(line for line in run.stdout.splitlines()
                   if line.startswith("tests/") and "::" in line)
    if not nodes:
        sys.exit("Collection returned no test IDs")
    output = ROOT / "tests/gpu-proof-tests.txt"
    output.write_text("\n".join(nodes) + "\n")
    print(f"Wrote {len(nodes)} test IDs to {output}; review and commit before recording proof.")


if __name__ == "__main__":
    main()
