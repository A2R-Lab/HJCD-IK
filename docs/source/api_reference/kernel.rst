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
HJCD CUDA runtime failures raise ``std::runtime_error`` (Python ``RuntimeError``),
including generated model/limit initialization. HJCD uses GRiD's checked APIs:
initialization failures report the operation, roll back partial allocations best-effort,
and never reset the CUDA context or terminate the process. This does not promise
recovery after a context-invalidating CUDA error.

A null model argument selects the internally cached model for sampling. The solver always
uses its own internally cached coarse/refine models (one per precision) and takes no model
argument. Position errors are in millimeters and orientation errors in radians.

.. doxygenfile:: hjcd_kernel.h
   :project: hjcdik
