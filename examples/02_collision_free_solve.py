"""02 — Collision-free IK against a MotionBenchMaker scene.

Solve IK while the GPU filters candidates against the obstacles in a problem set. The scene + goal come
from ``tests/mb_problems.json`` (the same sets the benchmark uses). Requires HJCD-IK built with the
``panda_grasptarget_hand`` frame (the committed default), which is the frame these problems are posed in.

Run: ``python examples/02_collision_free_solve.py``
"""
import json
from pathlib import Path

from hjcdik import generate_solutions

PROBLEMS = Path(__file__).resolve().parents[1] / "tests" / "mb_problems.json"
PROBLEM_SET = "box_panda"
PROBLEM_IDX = 0

problems_text = PROBLEMS.read_text()
problem = json.loads(problems_text)["problems"][PROBLEM_SET][PROBLEM_IDX]

# Goal pose for this problem: [x, y, z, qw, qx, qy, qz].
gp = problem["goal_pose"]
target = [*gp["position_xyz"], *gp["quaternion_wxyz"]]

out = generate_solutions(
    target,
    batch_size=2000,
    num_solutions=4,
    collision_free=True,
    problems_json_text=problems_text,   # the GPU reads obstacles from this set...
    problem_set_name=PROBLEM_SET,
    problem_idx=PROBLEM_IDX,             # ...for this specific scene
)

print(f"set={PROBLEM_SET} idx={PROBLEM_IDX}: {out['count']} collision-free solutions")
print("best position error (m):   ", float(out["pos_errors"].min()))
print("best orientation error (rad):", float(out["ori_errors"].min()))
