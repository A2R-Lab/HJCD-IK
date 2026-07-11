Python API
==========

The ``hjcdik`` package exposes the solver via pybind11.

``generate_solutions(target_pose, batch_size=2000, num_solutions=1, collision_free=False, ...)``
   Solve IK for a single 6-DOF target. ``target_pose`` is ``[x, y, z, qw, qx, qy, qz]`` (position +
   quaternion). Returns a dict ``{joint_config, pose, pos_errors, ori_errors, count}``.

``sample_targets(num_targets, seed=0)``
   Sample reachable random EE targets (list of 7-vectors), useful for benchmarking.

``num_joints()``
   The robot's joint count (``grid::NUM_JOINTS``).

.. code-block:: python

   from hjcdik import generate_solutions, sample_targets, num_joints

   targets = sample_targets(num_targets=10, seed=0)
   out = generate_solutions(targets[0], batch_size=2000, num_solutions=4)
   print(out["count"], out["pos_errors"].min())
