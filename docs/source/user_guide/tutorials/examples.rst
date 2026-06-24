Examples
========

Self-contained, runnable programs live in the repository's ``examples/`` directory. Each is included in
full below (so the docs never drift from the code). Run any of them with the project's virtual
environment active, e.g. ``python examples/01_open_world_solve.py``.

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Example
     - Shows
     - Needs
   * - ``01_open_world_solve``
     - Batch-solve one 6-DOF target; inspect the best returned solutions
     - built ``hjcdik``
   * - ``02_collision_free_solve``
     - Collision-free solve against a MotionBenchMaker scene (obstacles on GPU)
     - ``grasptarget`` build
   * - ``03_batch_sweep``
     - How the best-solution accuracy improves with batch size
     - built ``hjcdik``

01 — Open-world solve
---------------------

.. literalinclude:: ../../../../examples/01_open_world_solve.py
   :language: python

02 — Collision-free solve
-------------------------

The scene and goal come from ``tests/mb_problems.json``; the GPU filters candidates against the obstacles
in the chosen problem set. See :doc:`collision_free_benchmark` for the benchmark harness around this.

.. literalinclude:: ../../../../examples/02_collision_free_solve.py
   :language: python

03 — Batch-size sweep
---------------------

.. literalinclude:: ../../../../examples/03_batch_sweep.py
   :language: python
