Collision environment
=====================

Collision is **URDF-driven**. The robot's covering spheres and self-collision ranges are baked
into the generated ``grid.cuh`` as GRiD's ``grid_collision`` namespace (pass ``--collision`` to
``scripts/codegen/generate_grid.py``); the kernel scores them *post-solve* via
``grid_collision::collision_distance`` (soft penetration cost) and, in hard mode, filters with
``grid_collision::config_free`` (see :doc:`kernel`). There is no hand-written per-robot collision
header. Supported fixed-base serial arms get FK and collision from one codegen step;
see :doc:`../user_guide/tutorials/custom_robot` for the solver's model restrictions.

The obstacle set (spheres / cuboids / cylinders from a MotionBenchMaker-style problem JSON) is
parsed and uploaded to the device by ``grid_env.cuh``, then passed by value into the scoring
kernel as a ``grid_collision::Environment``.

The default ``collision_mode="hard"`` checks self-collision and the environment, then
excludes colliding candidates. It can return zero solutions. ``soft`` only ranks by
environment penetration cost and offers no collision-free guarantee; ``both`` ranks
and strictly filters. Collision checking is post-solve and does not certify a motion path
or guarantee that the returned candidate reaches the requested pose.

Each selected problem must explicitly describe its obstacles. Use ``"obstacles": {}``
for an empty environment (self-collision is still checked). Supported obstacle keys are
``sphere``, ``cuboid``, ``cylinder``, and legacy ``box``; unknown keys are rejected.
Dimensions, radii, and lengths must be positive finite numbers; poses use meters and
``[x,y,z,qw,qx,qy,qz]`` with a nonzero quaternion. Cylinders are conservatively represented
as capsules. Guarantees are relative to the generated sphere model, which excludes the
fixed Panda base geometry.

.. doxygenfile:: grid_env.cuh
   :project: hjcdik
