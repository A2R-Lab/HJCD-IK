"""02 — Collision-free IK against a MotionBenchMaker scene.

Install the obstacles of one dataset problem as the solver's world, then solve with the default
collision policy (``"hard"``: self- and environment-collision filtering). The scene and goal come
from ``tests/mb_problems.json`` (the sets the benchmark uses). Dataset goals describe ``panda_hand``;
the fixed tool offset is applied explicitly below to preserve the requested hand pose.

Run: ``python examples/02_collision_free_solve.py``
"""
import json
import sys
from pathlib import Path

import hjcdik

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from query_results import panda_hand_target, best_candidate

PROBLEMS = Path(__file__).resolve().parents[1] / "tests" / "mb_problems.json"
PROBLEM_SET = "box_panda"
PROBLEM_IDX = 0

problem = hjcdik.worlds.select_problem(json.loads(PROBLEMS.read_text()), PROBLEM_SET, PROBLEM_IDX)

solver = hjcdik.Solver()
solver.set_world(**hjcdik.worlds.from_mbm_problem(problem))    # boxes, cylinders, spheres -> GPU
print("world:", solver.world_counts)

# Goal pose for this problem, expressed for the compiled tool frame.
target = panda_hand_target(hjcdik.worlds.goal_pose(problem), solver.info.ee_frame)

out = solver.solve(target, batch_size=2000, num_solutions=4, collision="hard")
r = out[0]

print(f"set={PROBLEM_SET} idx={PROBLEM_IDX}: {r.count} collision-free solutions")
if r.count:
    best = best_candidate(r)
    print("selected candidate position error (mm):  ", float(r.pos_err_mm[best]))
    print("selected candidate orientation error (rad):", float(r.ori_err_rad[best]))
    print("within 5 mm and 0.05 rad:", bool(r.ok(pos_mm=5, ori_rad=0.05)[best]))
else:
    print("No collision-free candidate was found in this batch.")
