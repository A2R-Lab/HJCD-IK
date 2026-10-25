"""Bounded audit reproductions, not a benchmark or regression gate.

Run from the checkout using its venv: python /path/to/this.py --out /tmp/NEW
Creates a dedicated cache and generated fixtures only beneath a new output directory.
Retains evidence. Does not modify the default cache, source, dependencies or timing stages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET

import hjcdik
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
root = Path.cwd()
assert (root / "scripts/codegen/generate_grid.py").is_file(), "run from HJCD-IK"
report = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}

# Same initialization clamp, different selection: inaccurate candidates can outrank accurate ones.
solver = hjcdik.Solver()
targets = solver.sample_targets(4, seed=31)
report["warm_ranking"] = {}
for label, q in (("large", np.full((1, 7), 1e7)),
                 ("clipped", np.array(solver.info.joint_limits)[:, 1][None, :])):
    result = solver.solve(targets, collision=None, batch_size=64, num_solutions=4, q_init=q, stats=True)
    report["warm_ranking"][label] = {
        "accurate_refined": result.stats.n_ik_accurate.tolist(),
        "accurate_returned": result.stats.n_returned_ik_accurate.tolist(),
        "pos_mm": result.pos_err_mm.tolist(), "ori_rad": result.ori_err_rad.tolist(),
    }

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    arrays = hjcdik.world_arrays(cylinders=[[0, 0, 0, 1e-200, 0, 0, 0, .1, 1]])
report["tiny_cylinder_quaternion_finite"] = bool(np.isfinite(arrays["capsules"]).all())
report["accepted_negative_device"] = hjcdik.Solver(device=-1).info.device
report["fractional_sample_shape"] = list(solver.sample_targets(1.9).shape)

# Independent store, original and changed requested URDF share a display name.
urdf = out / "arm.urdf"
urdf.write_bytes((root / "examples/mixed_chain.urdf").read_bytes())
with (out / "build.log").open("w") as log:
    command = [sys.executable, "-m", "hjcdik", "build", str(urdf), "--ee-link", "tool",
               "--spheres", str(urdf), "--cache-dir", str(out / "cache")]
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
original_sha = hashlib.sha256(urdf.read_bytes()).hexdigest()
tree = ET.parse(urdf)
tree.getroot().find("joint[@name='j1']/limit").set("upper", "1.5")
tree.write(urdf)
loaded = hjcdik.Solver(urdf, ee_link="tool", spheres=urdf, allow_build=False, cache_dir=out / "cache")
report["changed_urdf_load"] = {
    "original_sha": original_sha, "requested_sha": hashlib.sha256(urdf.read_bytes()).hexdigest(),
    "requested_slider_upper": 1.5, "loaded_slider_upper": loaded.info.joint_limits[1][1],
    "artifact": loaded.info.artifact,
}

# The generator accepts a sphere model without its separate enabling switch.
header = out / "generated/grid.cuh"
preset = hjcdik.presets()["ur5e"]
command = [sys.executable, "scripts/codegen/generate_grid.py", str(preset["urdf"]),
           "-t", preset["ee_link"], "--spherized-urdf", str(hjcdik.sphere_presets()["ur5e"]),
           "-o", str(header)]
with (out / "codegen.log").open("w") as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
text = header.read_text()
report["sphere_option_without_collision_switch"] = {
    "exit": 0, "collision_sentinel": "#define HJCD_HAS_COLLISION 1" in text,
    "collision_namespace": "namespace grid_collision" in text,
}

# Execute the actual CMake target-metadata parser on the actual generated header.
cmake = (root / "CMakeLists.txt").read_text().splitlines()
parsing = next(line.strip() for line in cmake if 'string(REGEX MATCH "for target ' in line)
program = out / "metadata.cmake"
program.write_text('file(STRINGS "' + str(header) + '" HJCDIK_TARGET_LINE REGEX "codegen-resolved EE fixed-frame index for target")\n'
                   + parsing + '\nmessage("PARSED_TARGET=${CMAKE_MATCH_1}")\n')
result = subprocess.run(["cmake", "-P", str(program)], capture_output=True, text=True, check=True)
report["cmake_target_metadata"] = {"requested": preset["ee_link"], "observed": result.stderr.strip()}

# Trace unchanged implementation; numerical output is diagnostic, never timing evidence.
report["sources"] = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in (
    "hjcdik/_solver.py", "hjcdik/_build.py", "csrc/kernel/hjcd_kernel.cu",
    "scripts/codegen/generate_grid.py", "CMakeLists.txt")}
(out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(report, indent=2, allow_nan=False))
