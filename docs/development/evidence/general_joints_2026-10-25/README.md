# General-joint correctness gate, October 25, 2026

Shared-GPU correctness only. No latency or compile-speed claim is made here.

The declared scale matrix passed: 96 cells, 32 targets each, 3,072/3,072 targets with at
least one independently verified candidate below 5 mm position error and 0.001 rad
orientation error. The acceptance rule was at least 31/32 in every cell, not a pooled rate.
Travel was 0.01, 0.5 and 5 m, with the slider before and after a rotating joint, fp32/fp64,
one/four outputs, seeds 13/29, cold and perturbed warm starts, batch size 512. Half of each
target set used a common 1 cm slider workspace. Exact joint bounds were included.

`scale_cells.json` records every cell. Its maximum-error fields cover all returned
candidates, not just each target's accurate candidate. Multiple-output solves can also
return inaccurate candidates; callers must inspect errors or use `Result.ok`. All returned
coordinates were finite and within limits. This finite matrix supports retaining the
existing raw-coordinate LM/proximity metric for this wave, not universal scale invariance.

Additional checks cover independent FK and finite-difference Jacobians for rotated-frame,
negative-axis and slider-mounted Panda, collision verdict parity at mapped configurations,
100-target representation parity, and slider targets beyond a triangle-inequality reach
bound for fixed Panda. Mixed-chain collision tests use hard/both modes, both precisions,
zero/nonzero margins, self checking on/off, blocked warm starts and partial multi-warp blocks.
Fresh processes load same-DoF models in both orders with nvcc absent from PATH and building
disabled, checking independent FK after each load.

The original unchanged Panda differed from direct URDF trigonometry by about 6e-12 m in
fp64 due to GRiD fixed-transform simplification. The initial test's 3e-12 tolerance was
corrected to the plan's 1e-9 FK tolerance. No solver convergence threshold was changed.

Reproduce the numerical matrix and evidence:

```bash
.venv/bin/python -m pytest tests/test_general_joints.py -q -o junit_family=legacy --junitxml=/tmp/joints.xml
.venv/bin/python scripts/bench/summarize_joint_gate.py /tmp/joints.xml /tmp/scale_cells.json
```

Memcheck, racecheck and synccheck use `test_mixed_collision_partial_warps`; test-only diagnostics
call the production FK, Jacobian-column and collision helpers. The signed full-suite receipt
also exercises the shipping packaged build, which does not contain diagnostic entry points.
