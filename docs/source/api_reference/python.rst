Python API
==========

The ``hjcdik`` package exposes the solver via pybind11. Calls use the calling thread's
current CUDA device. Sampling and solving release the GIL and serialize in the native layer.

Targets use meters and scalar-first quaternions (``wxyz``). Any finite nonzero quaternion
is normalized. Argument conversion errors raise ``TypeError`` and invalid values raise
``ValueError``; unusable scenes and CUDA runtime
failures raise exceptions. A returned candidate is not a success certificate: inspect both
position and orientation errors against your application's tolerances.

``generate_solutions(target_pose, batch_size=2000, num_solutions=1, collision_free=False, collision_mode="hard", ...)``
   Solve IK for a single 6-DOF target. ``target_pose`` is ``[x, y, z, qw, qx, qy, qz]`` (position +
   quaternion). Returns a dict ``{joint_config, pose, pos_errors, ori_errors, count}``. Hard collision mode
   is the default and may return fewer than ``num_solutions`` (including zero); ``soft`` ranks but does
   not guarantee collision freedom, and ``both`` ranks then filters.

``sample_targets(num_targets, seed=0)``
   Sample reachable EE targets from a seeded Halton sequence (list of 7-vectors).
   These targets are not filtered for self/environment collision.

``num_joints()``
   The robot's joint count (``grid::NUM_JOINTS``).

Result arrays are independent, owning NumPy arrays with dtype ``float64``:

.. list-table::
   :header-rows: 1

   * - Key
     - Shape
     - Units / meaning
   * - ``joint_config``
     - ``(count, num_joints())``
     - Joint angles in radians
   * - ``pose``
     - ``(count, 7)``
     - Position in meters, quaternion in ``wxyz`` order
   * - ``pos_errors``
     - ``(count,)``
     - Position error in millimeters
   * - ``ori_errors``
     - ``(count,)``
     - Orientation error in radians

``refine_fp64=-1`` selects fp64 refinement for one requested solution and fp32 for
multiple solutions; ``1`` and ``0`` force the precision. I/O stays float64 in both modes.
``write_stats=True`` writes diagnostic rows to ``ik_stats.csv`` in the current directory.

``collision_enabled()`` reports whether collision was compiled in; ``build_info()``
returns that flag, the joint count, ``grid_header_sha256``, and ``cuda_compiler_version``
without initializing CUDA. The header hash identifies the generated model actually compiled
into the loaded extension; compare it with your selected ``grid.cuh`` to detect stale installs
or a different robot wheel. The compiler version describes the build toolkit, not the driver.
Interactive ``help(hjcdik.generate_solutions)`` documents parameters, units, and caveats.

Collision scenes use
``problems_json_text``, ``problem_set_name``, and ``problem_idx``; see :doc:`collision`.
``collision_mode="auto"`` reads the legacy ``HJCD_CC_MODE`` environment variable;
the default ``"hard"`` is explicit and ignores that variable.

.. code-block:: python

   from hjcdik import generate_solutions, sample_targets, num_joints

   targets = sample_targets(num_targets=10, seed=0)
   out = generate_solutions(targets[0], batch_size=2000, num_solutions=4)
   print(out["count"], out["pos_errors"].min())
