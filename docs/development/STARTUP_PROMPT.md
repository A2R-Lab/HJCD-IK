# Startup prompt — joining HJCD-IK

A 90-second primer for picking up work on **HJCD-IK**.

**What it is:** a GPU-accelerated, batched inverse-kinematics solver — Hybrid Jacobian Coordinate Descent —
that produces many candidate IK solutions in parallel for a 6-DOF EE target, with optional collision
avoidance (Panda/Fetch). Coarse search uses one candidate per block; LM uses **one candidate per warp**.
Coarse block-shared state needs block barriers; independent LM math stays warp-scoped.
Built on GRiD (kinematics codegen) + GLASS (single-block/warp linear algebra).

**Read in order:**
1. [`CLAUDE.md`](../../CLAUDE.md) — mental model, key files, build commands, discipline.
2. [`agent_debugging_guide.md`](agent_debugging_guide.md) — recurring traps (stale `grid.cuh`, `FLANGE_IDX`,
   warp/block sync, robot constants, submodules).
3. `docs/open-tasks/TODO.md` — the running roadmap / open work list. **LOCAL and gitignored**
   (agent scratch, not a tracked artifact; a fresh clone will not have it). At the **start** of a
   session read it if present; at the **end**, create or update it so the next session has the
   current state. If it is missing, the "What next" section of `CLAUDE.md` is the tracked summary.
4. `README.md` — user-facing usage + benchmark guide; `docs/source/user_guide/upgrading.md` — the
   current API contract (hard collision default, counts may be zero, errors raise).

**Where things live:** solver `csrc/kernel/hjcd_kernel.cu`; generated kinematics **and collision**
`csrc/generated/grid.cuh` (from `external/GRiD`; `--collision` bakes `grid_collision`); obstacle-env
parser `csrc/kernel/grid_env.cuh`; warp linalg `external/GLASS`; Python API `hjcdik/__init__.py` +
`csrc/bindings/pybind_module.cpp`; benchmark `benchmark/hjcd_ik_bench.py`.

**Discipline:** never hand-edit `grid.cuh`; keep `FLANGE_IDX`/target consistent; keep math warp-scoped;
short single-line commits, no Co-Authored-By footer.

**Build:** `bash scripts/setup/bootstrap.sh && python -m pip install -e '.[dev,codegen]' --no-build-isolation`
(then `scripts/setup/rebuild.sh` after kernel edits — `ninja` alone does not update the imported `.so`).

**Verify:** `python -m pytest tests` on a GPU, then `scripts/setup/run_gpu_proof.sh` to re-sign
`gpu-proof.json`; any change under `csrc/`, `tests/`, `docs/source/` or the submodule pins
invalidates the committed receipt and the CPU-only CI check until it is re-recorded.

Use `git status --short --branch` and `docs/open-tasks/TODO.md` (if present) to establish the current working state.
