Results
=======

HJCD-IK generates large batches of IK solutions in parallel and is competitive with — and often faster
than — GPU baselines (cuRobo, PyRoki) and learned solvers (IKFlow), while matching the solution diversity
of the TRAC-IK reference.

.. note::

   **Every figure and number on this page is taken verbatim from the camera-ready paper**
   (`arXiv:2510.07514 <https://arxiv.org/abs/2510.07514>`_, IROS 2026) — it is the single source of truth.
   Benchmarks you run locally (see :doc:`reproduce`) are for your own validation and may differ with
   hardware; they are **not** published here. For the methodology behind the comparison, see :doc:`baselines`.

.. The committed paper figures live in ``docs/source/_static/paper/`` (extracted from the camera-ready PDF)
   and are wired in below once provided. Until then these directives are placeholders.

Open-world IK (Table I)
-----------------------

Solve time and accuracy across batch sizes for the Panda and Fetch arms, against cuRobo, PyRoki, and IKFlow.

.. figure:: /_static/paper/table1_open.png
   :alt: Open-world IK — solve time vs. batch size, HJCD-IK vs. baselines
   :width: 100%

   *Placeholder — replace with the camera-ready Table I / Figure.*

Collision-free IK (Table II)
----------------------------

MotionBenchMaker bookshelf problems, with the in-optimizer collision constraint.

.. figure:: /_static/paper/table2_collfree.png
   :alt: Collision-free IK — solve time and success rate
   :width: 100%

   *Placeholder — replace with the camera-ready Table II / Figure.*

DoF scalability (Table III)
---------------------------

Solve time as the arm's degrees of freedom grow (7 → 24).

.. figure:: /_static/paper/table3_dof.png
   :alt: Solve time vs. degrees of freedom
   :width: 100%

   *Placeholder — replace with the camera-ready Table III / Figure.*

Solution diversity (Table IV)
-----------------------------

Maximum-mean-discrepancy (MMD / MMD²) of the returned solution set against the TRAC-IK reference
distribution — lower is a closer match to the full IK manifold.

.. figure:: /_static/paper/table4_mmd.png
   :alt: MMD / MMD² solution-diversity comparison
   :width: 100%

   *Placeholder — replace with the camera-ready Table IV.*
