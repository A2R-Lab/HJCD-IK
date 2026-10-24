Python API
==========

The ``hjcdik`` package wraps the CUDA solver in a small object API. A ``Solver`` binds to the
CUDA device current at its construction, owns one obstacle world, and solves one pose or a batch
of poses for the robot the installed extension was compiled for. Calls release the GIL and are
serialised in Python and in the native layer; calling from several threads is safe but not
concurrent on the GPU.

Targets use metres and scalar-first quaternions (``wxyz``). Any finite nonzero quaternion is
normalised. ``ValueError`` reports invalid shapes, values and collision requests before any GPU
work; ``SolveError`` (a ``RuntimeError``) reports CUDA failures. A returned candidate is not a
success certificate: inspect both errors against your application's tolerances.

.. code-block:: python

   import hjcdik

   solver = hjcdik.Solver()                      # the compiled robot; solver.info says which
   targets = solver.sample_targets(10, seed=0)   # (10, 7) reachable poses
   result = solver.solve(targets, batch_size=2000, num_solutions=4, collision=None)
   print(result.count, "candidates per target")
   best = result[0]                              # un-padded view of target 0
   if best.count:
       print("best position error (mm):", best.pos_err_mm.min())
       print("best orientation error (rad):", best.ori_err_rad.min())

Solver
------

``Solver(robot=None, *, ee_link=None, spheres=None, device=None, allow_build=True, cache_dir=None)``
   ``Solver()`` is the packaged robot (``"panda"`` in the shipped wheel; ``Solver("panda")`` is the
   same). Any other name is a prepared runtime build of that name (see *Runtime robot builds*). A
   URDF path with ``ee_link`` (the fixed joint naming the end-effector frame) and optionally
   ``spheres`` builds the robot on first use; ``allow_build=False`` only loads a prepared build and
   raises ``BuildError`` naming the command otherwise. ``device`` defaults to the current CUDA
   device. ``solver.info`` is an ``Info`` with ``robot``, ``dof``, ``ee_frame``,
   ``collision_enabled``, ``grid_header_sha256``, ``cuda_compiler_version``, ``device``,
   ``max_batch_size`` and ``artifact`` (the store path of a runtime build, ``None`` for the packaged
   robot).

``solver.solve(poses, *, num_solutions=1, batch_size=2000, collision=<omitted>, precision="auto", stats=False)``
   Solve one pose ``(7,)`` or a batch ``(T, 7)``; returns a ``Result`` with ``T`` targets (a single
   pose keeps the leading dimension; ``T == 0`` is legal). ``batch_size`` is the number of candidate
   configurations explored per target, not a target count; ``num_solutions`` is the fixed result
   capacity per target. The whole batch is validated before any GPU work. ``precision`` is
   ``"auto"`` (fp64 refinement for one solution, fp32 otherwise), ``"fp32"`` or ``"fp64"``; I/O is
   float64 either way. ``stats=True`` collects per-target diagnostics without changing the solution.

``solver.set_world(*, boxes=None, capsules=None, spheres=None, cylinders=None)``
   Replace the obstacle world. Arrays are float64, one row per primitive, in the robot base frame,
   metres, quaternions ``wxyz``. The new world is uploaded before the old one is released; a failed
   call leaves the previous world intact. Calling with no primitives installs the empty world.

``solver.clear_world()``
   Install the empty world. Self-collision is still checked under ``"hard"`` and ``"both"``.

``solver.sample_targets(num_targets, seed=0)``
   Reachable poses ``(num_targets, 7)`` from a seeded Halton sequence over the joint limits, through
   the compiled forward kinematics. They are not filtered for self or environment collision.

Collision policy
----------------

A build has collision geometry when its header was generated with ``--collision``
(``solver.info.collision_enabled``). An omitted ``collision`` argument is ``"hard"`` on a collision
build and ``None`` on an open-world build. "No world installed" and an explicitly empty world are the
same state.

.. list-table::
   :header-rows: 1

   * - ``collision=``
     - self-collision checked
     - environment
     - notes
   * - ``None`` (explicit)
     - no
     - no
     - open-world; the result is not a safety statement
   * - ``"hard"``
     - yes
     - filtered; an empty environment never hits
     - default on a collision build; ``clear_world()`` never disables self-collision
   * - ``"soft"``
     - no
     - penetration cost ranks candidates; raises if the environment is empty
     - environment-only ranking, not a collision-free solve
   * - ``"both"``
     - yes
     - ranks by environment cost and filters self and environment; an empty environment behaves as ``"hard"``
     -
   * - anything but ``None`` on an open-world build
     -
     -
     - raises

Collision checks describe configurations, never paths, and the compiled sphere model, never meshes.
In ``"hard"`` and ``"both"`` mode the solver's early stop and refinement are collision-aware: an
accurate but colliding candidate gets a repair round, and when no exactly converged free candidate
exists the best free one inside 5 mm / 0.05 rad is returned; read the errors.

Geometry
--------

.. list-table::
   :header-rows: 1

   * - array
     - shape
     - row
   * - ``boxes``
     - ``(M, 10)``
     - ``cx cy cz, qw qx qy qz, sx sy sz`` (full side lengths)
   * - ``capsules``
     - ``(M, 7)``
     - ``x0 y0 z0, x1 y1 z1, r`` (segment endpoints, radius)
   * - ``cylinders``
     - ``(M, 9)``
     - ``cx cy cz, qw qx qy qz, r, L`` (axis = local +z, full length)
   * - ``spheres``
     - ``(M, 4)``
     - ``cx cy cz, r``

Capsules are the native primitive. A cylinder is converted to the capsule with the same axis and
length, a conservative approximation whose rounded ends extend ``r`` beyond each flat face.
``hjcdik.world_arrays(...)`` performs the same validation and conversion without a solver.
``hjcdik.worlds`` loads MotionBenchMaker problem JSON: ``from_mbm_problem(problem)``,
``from_mbm_text(text, set_name, index)``, ``from_mbm_file(path, set_name, index)`` return the keyword
arguments for ``set_world``; ``select_problem`` and ``goal_pose`` pick an instance and its goal.

Results
-------

``Result`` holds fixed-capacity, NaN-padded, read-only host arrays for ``T`` targets and ``S =
num_solutions`` rows per target. They never reference device memory and survive later solves and
the solver's destruction.

.. list-table::
   :header-rows: 1

   * - Field
     - Shape
     - Units / meaning
   * - ``q``
     - ``(T, S, dof)``
     - Joint angles in radians; NaN beyond ``count``
   * - ``pose``
     - ``(T, S, 7)``
     - Reached pose: metres and ``wxyz`` quaternion
   * - ``pos_err_mm``
     - ``(T, S)``
     - Position error in millimetres
   * - ``ori_err_rad``
     - ``(T, S)``
     - Orientation error in radians
   * - ``count``
     - ``(T,)``
     - Valid rows per target, zero allowed
   * - ``valid``
     - ``(T, S)``
     - ``arange(S) < count[:, None]``
   * - ``elapsed_ms``
     - ``(T,)``
     - Host-observed per-target solve time; diagnostic only
   * - ``stats``
     - ``Stats`` or ``None``
     - Per-target diagnostics when ``stats=True``

``result.ok(pos_mm, ori_rad)`` marks the valid candidates inside both tolerances (pose accuracy
only, never physical safety). ``result[i]`` is a ``TargetResult`` with the un-padded ``q``,
``pose``, ``pos_err_mm``, ``ori_err_rad``, ``count``, ``elapsed_ms`` and ``stats`` of one target.

``solve`` is all-or-nothing: a CUDA failure at target ``i`` raises ``SolveError`` with
``target_index`` and ``completed``; no partial ``Result`` is returned. One world applies to the whole
call: ``set_world`` from another thread waits for a solve in flight.

Diagnostics
-----------

``Stats`` carries one array of shape ``(T,)`` per field: ``batch_size``, ``n_refined``,
``n_ik_accurate``, ``n_coll_free_refined``, ``n_feasible``, ``n_ik_lost``, ``n_coll_in_refined``,
``n_coll_in_coarse``, ``env_cost_min_mm``, ``env_cost_max_mm``, ``env_cost_mean_mm``,
``n_returned``, ``n_returned_ik_accurate``, ``n_returned_coll_free``, ``pct_returned_coll_free``,
``n_returned_feasible``; ``stats[i]`` is a dictionary of plain scalars for one target. The
accuracy tallies use position error < 5 mm **and** orientation error < 0.001 rad.

Collision and feasibility fields are **-1 when not measured**, including open-world calls. In
``"hard"`` and ``"both"`` mode the collision counts include self and environment checks; in
``"soft"`` mode they report environment penetration only. ``env_cost_*_mm`` summarise
refined-candidate penetration in ``"soft"`` or ``"both"`` mode; hard-only calls report -1. With
collision checking enabled but no returned solutions, ``pct_returned_coll_free`` is 0, not a success
percentage. The solver never opens files; the benchmark's ``benchmark/_results_io.py`` writes the
legacy ``ik_stats.csv`` rows and ``python scripts/ik_stats_summary.py path/to/ik_stats.csv``
summarises them.

Runtime robot builds
--------------------

The base package solves with the packaged robot and loads prepared builds with numpy only. With
the ``hjcdik[build]`` extra (GRiD's ``grid-rbd`` and ``pybind11``) and ``nvcc`` on the machine,
any fixed-base serial-chain URDF becomes a solver:

.. code-block:: text

   hjcdik build my_arm.urdf --ee-link tool_frame_joint                 # open-world
   hjcdik build panda.urdf --ee-link panda_grasptarget_hand --spheres panda_foam --name panda-cc
   hjcdik list

``hjcdik.build(urdf, *, ee_link, spheres=None, name=None, collision_res=0.05, cache_dir=None,
cuda_arch=None, force_rebuild=False, plan_only=False)`` is the same operation from Python and
returns an ``Artifact`` (``name``, ``robot``, ``ee_frame``, ``collision_enabled``, ``path``,
``header``, ``content_key``, ``cuda_arch``, ``compiled``). ``spheres`` is a preset name
(``hjcdik.sphere_presets()``; ``"panda_foam"`` ships) or a spherized URDF whose kinematic frames
match the robot URDF; GRiD validates it, and ``grid-spherize --rebind-to`` adapts a model made for
another URDF of the same robot. ``name`` defaults to ``<robot>-<ee_link>`` with ``-cc`` when
collision geometry is baked.

The build runs through ``grid_rbd.build_artifact``: GRiD generates ``grid.cuh`` for the URDF and
compiles HJCD-IK's two translation units against it under the store's rules (an identity over the
URDF, the sphere model, the codegen options and tree, GLASS, our sources and headers by content,
defines, flags, toolchain and CUDA architecture; per-unit object reuse; atomic publish). Artifacts
live under ``<cache_dir>/consumers/hjcdik/`` (default ``~/.cache/grid-rbd``, ``GRID_RBD_CACHE_DIR``).
``hjcdik.load(name)`` returns a prepared ``Artifact`` with no toolchain, checking only the artifact
schema and that the current GPU can run it; ``hjcdik.list_builds()`` lists the store. Each
artifact is its own extension module, so several robots can be solved in one process with
independent worlds and locks. A build that changes after it was imported into a process cannot be
re-imported there; restart the process.

Build metadata
--------------

``hjcdik.build_info()`` returns ``robot``, ``num_joints``, ``ee_target``, ``collision_enabled``,
``grid_header_sha256``, ``cuda_compiler_version`` and ``binding_abi`` for the packaged robot without
initialising CUDA. The header hash
identifies the generated model compiled into the loaded extension; compare it with your selected
``grid.cuh`` to detect stale installs or a different robot wheel. The compiler version describes the
build toolkit, not the driver. ``solver.info`` repeats these for the bound solver.
