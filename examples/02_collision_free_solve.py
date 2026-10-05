"""02 — Collision-free IK against a MotionBenchMaker scene.

Solve IK while the GPU filters candidates against the obstacles in a problem set. The scene + goal come
from ``tests/mb_problems.json`` (the same sets the benchmark uses). Requires HJCD-IK built with the
``panda_grasptarget_hand`` frame (the committed default). Dataset goals describe panda_hand;
the fixed tool offset is applied explicitly below to preserve the requested hand pose.

Run: ``python examples/02_collision_free_solve.py``
"""
import json
import sys
from pathlib import Path

from hjcdik import generate_solutions, build_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from query_results import panda_hand_target, best_candidate

PROBLEMS = Path(__file__).resolve().parents[1] / "tests" / "mb_problems.json"
PROBLEM_SET = "box_panda"
PROBLEM_IDX = 0

problems_text = PROBLEMS.read_text()
problem = json.loads(problems_text)["problems"][PROBLEM_SET][PROBLEM_IDX]

# Goal pose for this problem: [x, y, z, qw, qx, qy, qz].
gp = problem["goal_pose"]
target = [*gp["position_xyz"], *gp["quaternion_wxyz"]]
target = panda_hand_target(target, build_info()["ee_target"])

out = generate_solutions(
    target,
    batch_size=2000,
    num_solutions=4,
    collision_free=True,
    collision_mode="hard",
    problems_json_text=problems_text,   # the GPU reads obstacles from this set...
    problem_set_name=PROBLEM_SET,
    problem_idx=PROBLEM_IDX,             # ...for this specific scene
)

print(f"set={PROBLEM_SET} idx={PROBLEM_IDX}: {out['count']} collision-free solutions")
if out["count"]:
    best = best_candidate(out)
    print("selected candidate position error (mm):  ", float(out["pos_errors"][best]))
    print("selected candidate orientation error (rad):", float(out["ori_errors"][best]))
    print("within 5 mm and 0.05 rad:", bool(out["pos_errors"][best] < 5 and out["ori_errors"][best] < .05))
else:
    print("No collision-free candidate was found in this batch.")
