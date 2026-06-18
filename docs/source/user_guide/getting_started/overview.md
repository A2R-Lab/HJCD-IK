# Library overview

HJCD-IK solves inverse kinematics by exploring many candidate joint configurations in parallel on the GPU and
refining the promising ones. The design is **one CUDA block per IK problem, warp-per-candidate**.

- **Kinematics** come from GRiD: a per-URDF generated `grid.cuh` provides the robot model and the warp-scoped
  forward-kinematics that produce world-frame joint transforms.
- **Linear algebra** (small warp-scoped reductions and SPD solves) comes from GLASS.
- **Collision** checking (optional) uses per-block pRRTC against robot collision spheres (Panda/Fetch).

See [the HJCD algorithm](../concepts/hjcd_algorithm.md) and [batched execution](../concepts/batch_execution.md)
for how the solver is structured, and [custom robots](../tutorials/custom_robot.md) for the codegen workflow.
