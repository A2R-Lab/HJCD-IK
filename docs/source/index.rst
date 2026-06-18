HJCD-IK
=======

**Hybrid Jacobian Coordinate Descent Inverse Kinematics** — a GPU-accelerated, batched IK solver that
generates many candidate solutions in parallel for a 6-DOF end-effector target, with optional collision
avoidance. Built on `GRiD <https://github.com/A2R-Lab/GRiD>`_ (robot kinematics codegen) and
`GLASS <https://github.com/A2R-Lab/GLASS>`_ (single-block / warp CUDA linear algebra).

Paper: `arXiv:2510.07514 <https://arxiv.org/abs/2510.07514>`_.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/getting_started/installation
   user_guide/getting_started/quickstart
   user_guide/getting_started/overview
   user_guide/concepts/hjcd_algorithm
   user_guide/concepts/batch_execution
   user_guide/concepts/collision_avoidance
   user_guide/tutorials/custom_robot
   user_guide/tutorials/collision_free_benchmark

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/index
