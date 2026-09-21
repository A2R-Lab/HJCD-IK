Solver kernel
=============

The batched HJCD solver: block-cooperative coarse search, then warp-per-candidate
Levenberg–Marquardt refinement.

The native host API lives in ``csrc/kernel/hjcd_kernel.h``. Both sampling and solving
serialize access to shared CUDA constants and stop flags. Python releases the GIL while
these operations run. Models and joint limits are cached per CUDA device; callers must
keep the CUDA context alive (calling ``cudaDeviceReset`` invalidates those caches).

``Result<T>`` owns its host arrays, releases them on destruction, and supports moves
but not copies. Temporary device allocations are released on normal return and exceptions.
HJCD CUDA runtime failures raise ``std::runtime_error`` (Python ``RuntimeError``);
GRiD's generated model initialization still uses its upstream error policy.

A null model argument selects the internally cached model for sampling. The solver selects
its own coarse/refine models by precision; its model argument is retained for source
compatibility. Position errors are in millimeters and orientation errors in radians.

.. doxygenfile:: hjcd_kernel.h
   :project: hjcdik
