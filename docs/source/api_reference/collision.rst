Collision environment
=====================

Collision is **URDF-driven**. The robot's covering spheres and self-collision ranges are baked
into the generated ``grid.cuh`` as GRiD's ``grid_collision`` namespace (pass ``--collision`` to
``scripts/codegen/generate_grid.py``). Hard/both refinement uses the warp-scoped
``grid_collision::warp::config_free`` on existing joint transforms; the kernel also scores *post-solve* via
``grid_collision::collision_distance`` (soft penetration cost) and, in hard mode, filters with
``grid_collision::config_free`` (see :doc:`kernel`). There is no hand-written per-robot collision
header. Supported fixed-base serial arms get FK and collision from one codegen step;
see :doc:`../user_guide/tutorials/custom_robot` for the solver's model restrictions.

``hjcdik.worlds`` converts MotionBenchMaker JSON into arrays. ``Solver.set_world`` validates
spheres, boxes and capsules (cylinders become capsules) and creates an owned uploaded world.
The native ``CollisionWorld`` wraps GRiD's ``grid_collision::DeviceEnvironment``; its
``Environment`` view is passed by value into the solver kernels. See :doc:`python` for array shapes.

The default ``collision="hard"`` on a collision build checks self-collision and the environment, then
excludes colliding candidates. It can return zero solutions. ``soft`` only ranks by
environment penetration cost and offers no collision-free guarantee; ``both`` ranks
and strictly filters. The final filter is authoritative. These checks do not certify a motion path
or guarantee that the returned candidate reaches the requested pose.

Collision-aware early stopping, seed ranking and repair help search for a free candidate.
A free candidate within 5 mm / 0.05 rad can be retained as a fallback if this run finds
no exactly converged free candidate. Always inspect both reported errors for the same
candidate; an empty result is not a proof that the problem is infeasible.

Each selected problem must explicitly describe its obstacles. Use ``"obstacles": {}``
for an empty environment (self-collision is still checked). Supported obstacle keys are
``sphere``, ``cuboid`` and ``cylinder``; unknown keys are rejected. Cuboids and cylinders
take a ``pose``; spheres take a ``pose`` or a bare ``position``.
Dimensions, radii, and lengths must be positive finite numbers; poses use meters and
``[x,y,z,qw,qx,qy,qz]`` with a nonzero quaternion. Cylinders are conservatively represented
as capsules. Guarantees are relative to the generated sphere model, which excludes the
fixed Panda base geometry.

World ownership and replacement
-------------------------------

Each ``Solver`` owns one world bound to its CUDA device. Repeated solves reuse that
uploaded world without parsing or comparing JSON. ``set_world`` replaces it; ``clear_world``
removes the environment while hard mode still checks self-collision. Invalid input or a
failed upload leaves the previous world installed. Replacing a world waits for an in-flight
solve, so one world applies to its entire target batch; returned results own their host data.

Different solver instances have independent worlds. Native ownership releases device memory
on the device that allocated it. ``collision_margin_m`` widens environment checks only;
self pairs retain zero margin. ``self_collision=False`` skips only self pairs. Both options
apply to hard/both modes, not the soft penetration cost. This does not establish safe motion
between configurations or coverage of unmodeled robot geometry.

Default Panda geometry
----------------------

The compiled Panda retains fixed finger-joint origins at y = +/-40 mm in the hand frame.
Foam supplies sphere-local centers and radii; the kinematic URDF supplies fixed-link
transforms. The historical paper reference instead has +/-65 mm finger origins, shifting
four finger spheres by 25 mm. These are explicitly different models, not a GRiD FK defect.

``benchmark/panda_collision.py`` defaults to ``model="paper"`` for historical cross-solver
comparisons. Use ``model="hjcd"`` for the current URDF-derived geometry. Both CPU helpers
check environment collisions only; their boolean result does not certify self-collision
freedom or reproduce the compiled self-pair exclusion policy. The legacy reference and
production geometry are intentionally unchanged.

The native input structure and world-construction function are documented in :doc:`kernel`.
