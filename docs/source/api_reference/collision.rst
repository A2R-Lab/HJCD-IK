Collision environment
=====================

Collision is **URDF-driven**. The robot's covering spheres and self-collision ranges are baked
into the generated ``grid.cuh`` as GRiD's ``grid_collision`` namespace (pass ``--collision`` to
``scripts/codegen/generate_grid.py``); the kernel scores them *post-solve* via
``grid_collision::collision_distance`` (soft penetration cost) and, in hard mode, filters with
``grid_collision::config_free`` (see :doc:`kernel`). There is no hand-written per-robot collision
header — any URDF gets both FK and collision from one codegen step.

The obstacle set (spheres / cuboids / cylinders from a MotionBenchMaker-style problem JSON) is
parsed and uploaded to the device by ``grid_env.cuh``, then passed by value into the scoring
kernel as a ``grid_collision::Environment``.

.. doxygenfile:: grid_env.cuh
   :project: hjcdik
