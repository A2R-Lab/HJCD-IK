"""Build a small analytic mixed-joint robot, install a world, and warm-start a solve.

Run: python examples/mixed_joint_warm_start.py
Needs hjcdik[build] and nvcc on first use. The generated artifact is reused later.
This example models only its explicitly supplied spheres, not a physical robot.
"""
from pathlib import Path
import hjcdik


def run():
    urdf = Path(__file__).with_name("mixed_chain.urdf")
    artifact = hjcdik.build(urdf,ee_link="tool",spheres=urdf,name="example-mixed-chain")
    solver = hjcdik.Solver(artifact.name)
    print(list(zip(solver.info.joint_names,solver.info.joint_types,
                   solver.info.joint_units,solver.info.joint_limits)))
    solver.set_world(spheres=[[10,10,10,.1]])
    targets = solver.sample_targets(4,seed=13)
    out = solver.solve(targets,batch_size=512,num_solutions=1,collision="hard")
    # Never pass NaN padding or a failed/inaccurate result back as a warm start.
    good = out.ok(5,.001)[:,0]
    warm = None
    if good.any():
        warm = solver.solve(targets[good],q_init=out.q[good],batch_size=512,
                            collision="hard",collision_margin_m=.01)
        print("warm-start accurate targets:",int(warm.ok(5,.001).any(axis=1).sum()))
    solver.set_world(spheres=[[0,0,0,100]])
    blocked = solver.solve(targets[0],batch_size=37,collision="hard")
    if not blocked[0].count:
        print("No qualifying solution found within the budget; no rows to index.")
    return solver,out,warm,blocked


if __name__ == "__main__":
    run()
