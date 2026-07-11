HJCD-IK
=======

**Hybrid Jacobian Coordinate Descent Inverse Kinematics** — a GPU-accelerated, *batched* inverse
kinematics solver that generates many candidate solutions in parallel for a 6-DOF end-effector target,
with optional collision avoidance. One CUDA block per IK problem, one candidate per warp.

Built on `GRiD <https://github.com/A2R-Lab/GRiD>`_ (robot kinematics codegen) and
`GLASS <https://github.com/A2R-Lab/GLASS>`_ (single-block / warp-scoped CUDA linear algebra).

Paper: `arXiv:2510.07514 <https://arxiv.org/abs/2510.07514>`_ (IROS 2026).
Project page: `a2r-lab.org/HJCD-IK <https://a2r-lab.org/HJCD-IK/>`_.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Get started
      :link: user_guide/getting_started/installation
      :link-type: doc

      Install the extension and run your first batched IK solve.

   .. grid-item-card:: How it works
      :link: user_guide/concepts/hjcd_algorithm
      :link-type: doc

      The three-phase algorithm: coordinate-descent init, Jacobian polish, collision filter.

   .. grid-item-card:: Examples & results
      :link: user_guide/benchmarks/results
      :link-type: doc

      Runnable examples, benchmark results vs. cuRobo / PyRoki / IKFlow, and how to reproduce.

   .. grid-item-card:: Custom robots
      :link: user_guide/tutorials/custom_robot
      :link-type: doc

      Generate kinematics for a new URDF / end-effector frame.

   .. grid-item-card:: API reference
      :link: api_reference/index
      :link-type: doc

      The Python API and the Doxygen-generated C++/CUDA reference.

   .. grid-item-card:: Developer guide
      :link: developer_guide/contribution_guidelines
      :link-type: doc

      Contributing, the codegen/build discipline, and how to edit these docs.

Quick start
-----------

.. code-block:: python

   from hjcdik import generate_solutions, sample_targets, num_joints

   target = sample_targets(num_targets=1, seed=0)[0]   # [x, y, z, qw, qx, qy, qz]
   out = generate_solutions(target, batch_size=2000, num_solutions=4)
   print(out["count"], "solutions; best position error:", out["pos_errors"].min())

.. toctree::
   :hidden:
   :caption: Getting Started

   user_guide/getting_started/installation
   user_guide/tutorials/custom_robot

.. toctree::
   :hidden:
   :caption: Algorithm

   user_guide/concepts/hjcd_algorithm

.. toctree::
   :hidden:
   :caption: Examples & Results

   user_guide/benchmarks/results

.. toctree::
   :hidden:
   :caption: API Reference

   api_reference/index

.. toctree::
   :hidden:
   :caption: Developer Guide

   developer_guide/contribution_guidelines
