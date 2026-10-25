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
       print("candidate position errors (mm):", best.pos_err_mm)
       print("candidate orientation errors (rad):", best.ori_err_rad)
       print("accurate candidates:", best.q[best.ok(pos_mm=5, ori_rad=0.001)])

Solver
------

``Solver(robot=None, *, ee_link=None, spheres=None, device=None, allow_build=True, cache_dir=None)``
   ``Solver()`` is the packaged robot (``"panda"`` in the shipped wheel; ``Solver("panda")`` is the
   same). Any other name is a shipped preset (``hjcdik.presets()``; built through the artifact store on
   first use, loaded afterwards) or a prepared runtime build of that name (see *Runtime robot builds*). A
   URDF path with ``ee_link`` (the fixed joint naming the end-effector frame) and optionally
   ``spheres`` builds the robot on first use; ``allow_build=False`` only loads a prepared build and
   raises ``BuildError`` naming the command otherwise. ``device`` defaults to the current CUDA
   device. ``solver.info`` is an ``Info`` with ``robot``, ``dof``, ``ee_frame``,
   ``collision_enabled``, ``grid_header_sha256``, ``cuda_compiler_version``, ``device``,
   ``max_batch_size`` and ``artifact`` (the store path of a runtime build, ``None`` for the packaged
   robot). Ordered ``joint_names``, ``joint_types``, ``joint_units`` and ``joint_limits`` are
   immutable tuples describing the coordinates of ``q`` and ``q_init``. Units are ``"rad"``
   for revolute/continuous joints and ``"m"`` for prismatic joints. Continuous-joint effective
   limits are [-pi, pi]. LM norms and warm-start proximity use raw coordinates, not a
   unit-invariant mixed-joint metric.

``solver.solve(poses, *, num_solutions=1, batch_size=2000, collision=<omitted>, precision="auto", stats=False, collision_margin_m=0.0, self_collision=True, q_init=None, seed=None)``
   Solve one pose ``(7,)`` or a batch ``(T, 7)``; returns a ``Result`` with ``T`` targets (a single
   pose keeps the leading dimension; ``T == 0`` is legal). ``batch_size`` is the number of candidate
   configurations explored per target, not a target count; ``num_solutions`` is the fixed result
   capacity per target. The whole batch is validated before any GPU work. ``precision`` is
   ``"auto"`` (fp64 refinement for one solution, fp32 otherwise), ``"fp32"`` or ``"fp64"``; I/O is
   float64 either way. ``stats=True`` collects per-target diagnostics without changing the solution.
   The remaining four knobs are described under *Clearance, self-collision, warm starts and seeds*.

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

Clearance, self-collision, warm starts and seeds
------------------------------------------------

``collision_margin_m`` (metres, default 0, ``"hard"`` and ``"both"`` only) is the clearance below
which a robot sphere and an obstacle count as a hit; zero is the contact-free rule. The margin applies
to environment pairs only. The baked self pairs are always checked at zero clearance: a sphere model
such as foam's Panda has neighbouring spheres a centimetre apart in every configuration, so a margin on
self pairs would reject everything. The margin enters every collision-aware step (early stop, repair,
fallback, final filter and the ``stats`` tallies); the ``"soft"`` penetration cost is unchanged.

``self_collision=False`` (``"hard"`` and ``"both"`` only) skips the self pairs so the verdict reflects
the environment alone.

``q_init`` warm-starts the coarse search with ``(K, N)`` configurations shared by every target or
``(T, K, N)`` per target, ``K <= batch_size``, clamped to the joint limits; the other ``batch_size - K``
candidates start from the seeded random initialisation. A warm start that already meets the coarse
tolerance skips coordinate descent and goes straight to refinement, and one that is already exact
(position error below 1 µm, orientation error below 1 µrad) is returned as it is, under fp32 refinement
too. With a warm start, the candidates
that converged (position error below 1 mm and orientation error below 1e-3 rad) are ranked by
joint-space distance to the nearest supplied configuration. If such candidates exist, the first
returned row is nearest among those found in that convergence band; this is not a global nearest-IK
guarantee. Candidates outside that band keep the accuracy ranking behind them. Check ``count`` and
``ok`` before indexing or forwarding a warm start; collision filtering can leave no rows.

``seed`` (an unsigned 32-bit integer; default 1337, the historical constant) shifts the random
initialisation. Seeds and warm starts change which valid candidates are found, never their validity;
with ``batch_size=1`` a given seed is reproducible, with larger batches the cross-block early stop makes
candidate identity run-to-run unstable as before.

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
``target_index`` and ``completed``; no partial ``Result`` is returned. The whole batch is one native
call: the host loop over targets, the NaN padding and the statistics arrays are built in the binding,
and the Python layer adds about 10 µs per target (measured October 2026; see the upgrading guide). One world applies to the whole
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

The base package solves with the packaged robot using numpy only. Loading prepared runtime
artifacts also requires ``grid-rbd``, but not nvcc or pybind11. With
the ``hjcdik[build]`` extra (GRiD's ``grid-rbd`` and ``pybind11``) and ``nvcc`` on the machine,
a supported fixed-base serial-chain URDF becomes a solver:

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
schema and that the current GPU can run it; ``hjcdik.list_builds()`` lists the store.

``hjcdik.presets()`` (``hjcdik presets`` on the command line) names the robots the package ships as a
URDF, an end-effector frame and a sphere model, each a fixed-base serial chain:

.. list-table::
   :header-rows: 1

   * - Preset
     - End-effector frame
     - Sphere model
     - Source
   * - ``panda``
     - ``panda_grasptarget_hand``
     - ``panda_foam`` (foam, rebound to our URDF)
     - franka_description; compiled into the wheel
   * - ``iiwa14``
     - ``iiwa_joint_ee``
     - ``iiwa14`` (GRiD spherizer on Drake's collision model)
     - Drake ``iiwa_description``; built on first use
   * - ``ur5e``
     - ``flange-tool0`` (the fixed joint whose child is ``tool0``)
     - ``ur5e`` (bounded-bulge fit to upstream collision meshes)
     - Universal Robots ROS2 description, nominal calibration; built on first use

The fidelity of each shipped sphere model (sampled bulge and uncovered surface per link, from
``grid-spherize --report``) is recorded at the top of the sphere URDF under ``csrc/collision/``.
UR5e's complete report and mesh hashes are in ``ur5e_provenance.json`` beside its sphere URDF.
Its sampled bulge is up to 14.8 mm against collision meshes and 60.2 mm against visual meshes;
zero uncovered surface samples are not a continuous-coverage certificate. Base-anchored geometry
is not checked by the generated sphere checker. This is not a safety-certified whole-robot model.
Custom models may have arbitrary-axis revolute, continuous and prismatic joints (1 to 32,
independent, finite ordered non-continuous limits). Floating bases, actuated branches and mimics
are rejected. GRiD 0.6.0 also requires at least one revolute or continuous joint; all-prismatic
chains are rejected before generation. Xacro-only descriptions need expansion to URDF first. Each
artifact is its own extension module, so several robots can be solved in one process with
independent worlds and locks; the modules are compiled with hidden symbol visibility so no static
state binds across them. A build that changes after it was imported into a process cannot be
re-imported there; restart the process.

Build metadata
--------------

``hjcdik.build_info()`` returns ``robot``, ``num_joints``, ``ee_target``, ``collision_enabled``,
``grid_header_sha256``, ``joint_model_sha256``, ``cuda_compiler_version`` and ``binding_abi`` for the packaged robot without
initialising CUDA. The header hash
identifies the generated model compiled into the loaded extension; compare it with your selected
``grid.cuh`` to detect stale installs or a different robot wheel. The compiler version describes the
build toolkit, not the driver. The joint-model hash identifies the ordered axes, types, limits,
tool frame and source URDF. ``solver.info`` exposes the bound solver's geometry hash and joint metadata.
