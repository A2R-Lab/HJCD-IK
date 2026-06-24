Results
=======

HJCD-IK generates large batches of IK solutions in parallel and stays on or near the **accuracy–latency
Pareto frontier** across every batch size and degree-of-freedom count, with order-of-magnitude gains over
the GPU baselines cuRobo, PyRoki, and IKFlow, while returning the most diverse (lowest-MMD) solution set.

.. note::

   **All numbers below are from the camera-ready paper** (`arXiv:2510.07514
   <https://arxiv.org/abs/2510.07514>`_, IROS 2026) — the single source of truth. They were collected on
   an NVIDIA RTX 4060 (Intel i7-14700HX, WSL Ubuntu 24.04, CUDA 12.5) over 100 Halton open-world poses and
   the *bookshelf_thin_panda* MotionBenchMaker scene. Benchmarks you run locally (see :doc:`reproduce`) are
   for your own validation and will differ with hardware. Position error is in **mm**, orientation error in
   **rad**, time in **ms**; **bold** marks the best (HJCD-IK) value.

Open-world IK — Panda (Table I)
-------------------------------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Batch
     - HJCD-IK Time
     - HJCD-IK Pos
     - HJCD-IK Ori
     - PyRoki Time
     - PyRoki Pos
     - PyRoki Ori
     - cuRobo Time
     - cuRobo Pos
     - cuRobo Ori
     - IKFlow Time
     - IKFlow Pos
     - IKFlow Ori
   * - 1
     - **4.04**
     - 7.04e-2
     - 2.04e-3
     - 14.86
     - 1.39e-2
     - 1.12e-5
     - 5.33
     - 2.56e1
     - 1.11e-1
     - 18.48
     - 4.67e0
     - 2.28e-2
   * - 10
     - **3.82**
     - **1.21e-4**
     - **6.74e-7**
     - 14.62
     - 1.39e-2
     - 1.12e-5
     - 5.55
     - 2.49e-3
     - 3.95e-6
     - 18.95
     - 1.38e0
     - 6.21e-3
   * - 100
     - **4.07**
     - **2.25e-5**
     - **8.95e-8**
     - 14.20
     - 1.39e-2
     - 1.12e-5
     - 6.01
     - 9.16e-4
     - 2.83e-6
     - 22.29
     - 5.94e-1
     - 2.76e-3
   * - 1000
     - **4.22**
     - **1.60e-5**
     - **9.15e-8**
     - 13.96
     - 1.39e-2
     - 1.12e-5
     - 19.80
     - 3.67e-4
     - 1.68e-6
     - 49.78
     - 2.06e0
     - 5.43e-3
   * - 2000
     - **4.37**
     - **1.81e-5**
     - **5.15e-8**
     - 13.97
     - 1.39e-2
     - 1.12e-5
     - 30.30
     - 2.65e-4
     - 1.33e-6
     - 99.98
     - 1.92e0
     - 6.59e-3

Open-world IK — Fetch (Table I)
-------------------------------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Batch
     - HJCD-IK Time
     - HJCD-IK Pos
     - HJCD-IK Ori
     - PyRoki Time
     - PyRoki Pos
     - PyRoki Ori
     - cuRobo Time
     - cuRobo Pos
     - cuRobo Ori
     - IKFlow Time
     - IKFlow Pos
     - IKFlow Ori
   * - 1
     - **2.59**
     - 5.79e-1
     - 1.20e-3
     - 13.70
     - 2.10e-5
     - 3.12e-8
     - 5.30
     - 4.48e0
     - 3.70e-3
     - 17.40
     - 1.92e1
     - 6.67e-2
   * - 10
     - **2.41**
     - **1.40e-6**
     - **9.56e-9**
     - 13.48
     - 2.10e-5
     - 3.12e-8
     - 5.52
     - 6.74e-4
     - 1.08e-6
     - 16.36
     - 9.60e0
     - 3.66e-2
   * - 100
     - **2.52**
     - **1.67e-6**
     - **8.97e-9**
     - 13.16
     - 2.10e-5
     - 3.12e-8
     - 7.57
     - 1.61e-4
     - 8.87e-7
     - 19.75
     - 1.65e1
     - 7.24e-2
   * - 1000
     - **2.59**
     - **1.67e-6**
     - **6.10e-9**
     - 12.92
     - 2.10e-5
     - 3.12e-8
     - 11.32
     - 5.17e-5
     - 6.43e-7
     - 48.68
     - 2.05e1
     - 6.03e-2
   * - 2000
     - **2.73**
     - **1.66e-6**
     - **9.70e-9**
     - 13.37
     - 2.10e-5
     - 3.12e-8
     - 14.62
     - 3.96e-5
     - 5.94e-7
     - 87.89
     - 1.52e1
     - 4.87e-2

.. figure:: /_static/paper/pareto_batch.png
   :width: 100%
   :alt: Open-world accuracy–latency Pareto frontier across batch sizes

   Open-world accuracy–latency frontier (Table I) — HJCD-IK (orange), cuRobo (blue), PyRoki (green).

Collision-free IK — Panda, bookshelf_thin_panda (Table II)
----------------------------------------------------------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Batch
     - HJCD-IK Time
     - HJCD-IK Pos
     - HJCD-IK Ori
     - PyRoki Time
     - PyRoki Pos
     - PyRoki Ori
     - cuRobo Time
     - cuRobo Pos
     - cuRobo Ori
   * - 1
     - **4.69**
     - **3.41**
     - **6.02e-3**
     - 77.07
     - 6.09e2
     - 3.93e-1
     - 33.70
     - 8.03
     - 3.39e-2
   * - 10
     - **4.48**
     - **2.78e-1**
     - **9.57e-5**
     - 72.17
     - 3.56
     - 6.34e-3
     - 35.43
     - 8.03
     - 1.22e-3
   * - 100
     - **5.43**
     - **2.38e-1**
     - **3.53e-5**
     - 66.25
     - 2.72e-1
     - 1.03e-4
     - 34.85
     - 7.84
     - 2.05e-3
   * - 1000
     - **5.16**
     - **2.38e-1**
     - **3.50e-5**
     - 46.83
     - 2.46e-1
     - 1.13e-4
     - 32.05
     - 7.84
     - 2.05e-3
   * - 2000
     - **6.16**
     - **2.41e-1**
     - **2.09e-5**
     - 45.72
     - 2.46e-1
     - 1.14e-4
     - 70.55
     - 7.92
     - 2.94e-3

.. figure:: /_static/paper/pareto_collfree.png
   :width: 100%
   :alt: Collision-free accuracy–latency Pareto frontier

   Collision-free frontier on the *bookshelf_thin_panda* scene (Fig. 4, Table II).

DoF scalability — Panda variants, B = 1000 (Table III)
------------------------------------------------------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - DoF
     - HJCD-IK Time
     - HJCD-IK Pos
     - HJCD-IK Ori
     - PyRoki Time
     - PyRoki Pos
     - PyRoki Ori
     - cuRobo Time
     - cuRobo Pos
     - cuRobo Ori
   * - 7
     - **4.25**
     - **1.71e-5**
     - **4.11e-8**
     - 15.09
     - 2.63e-2
     - 3.70e-5
     - 9.11
     - 3.38e-4
     - 1.59e-6
   * - 12
     - **4.55**
     - **1.94e-5**
     - **6.91e-8**
     - 16.29
     - 1.99e-2
     - 1.86e-5
     - 12.66
     - 7.78e-1
     - 2.57e-2
   * - 18
     - **4.62**
     - **3.76e-5**
     - **6.95e-8**
     - 20.82
     - 2.15e-2
     - 2.14e-5
     - 16.26
     - 8.41e-1
     - 3.03e-2
   * - 24
     - **4.66**
     - **3.84e-5**
     - **7.32e-8**
     - 24.34
     - 1.84e-2
     - 1.99e-5
     - 19.55
     - 7.50e-1
     - 3.58e-2

.. figure:: /_static/paper/pareto_dof.png
   :width: 100%
   :alt: DoF-scaling accuracy–latency Pareto frontier

   DoF scaling, 7–24 DoF (Fig. 5, Table III) — HJCD-IK keeps the lowest error and latency at every DoF.

Solution diversity — MMD vs. TRAC-IK (Table IV)
-----------------------------------------------

Maximum Mean Discrepancy between each solver's 50 best configurations (of a batch of 2000) and 50
ground-truth samples, over 100 target poses — lower is a closer match to the full IK manifold.

.. figure:: /_static/paper/solution_distributions.png
   :width: 100%
   :alt: Distribution of collision-free IK solutions: cuRobo, PyRoki, HJCD-IK

   Distribution of collision-free IK solutions for a representative target — cuRobo (left), PyRoki
   (center), HJCD-IK (right). HJCD-IK returns a broader, more diverse spread of locally-optimal solutions.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Metric
     - HJCD-IK
     - PyRoki
     - cuRobo
     - IKFlow
   * - MMD ↓
     - **0.02261**
     - 0.04514
     - 0.05348
     - 0.03670
   * - MMD² ↓
     - **0.00051**
     - 0.00203
     - 0.00286
     - 0.00134
