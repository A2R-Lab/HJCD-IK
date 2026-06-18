# Collision avoidance

With `collision_free=True`, candidate solutions are filtered against the environment using per-block pRRTC
collision checking. The robot is approximated by collision spheres defined per robot in
`src/robots/{panda,fetch}.cuh`; obstacles come from the problem set (cuboids / cylinders).

Collision geometry is currently hand-tuned for Panda and Fetch. Adding a new robot requires its collision
spheres and a matching kinematics header — see [custom robots](../tutorials/custom_robot.md). Without
collision data, run unconstrained (`collision_free=False`).
