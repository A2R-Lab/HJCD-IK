# Startup prompt — joining HJCD-IK

A 90-second primer for picking up work on **HJCD-IK**.

**What it is:** a GPU-accelerated, batched inverse-kinematics solver — Hybrid Jacobian Coordinate Descent —
that produces many candidate IK solutions in parallel for a 6-DOF EE target, with optional collision
avoidance (Panda/Fetch). One CUDA block per problem; **warp-per-candidate**, warp-scoped math throughout.
Built on GRiD (kinematics codegen) + GLASS (single-block/warp linear algebra).

**Read in order:**
1. [`CLAUDE.md`](../CLAUDE.md) — mental model, key files, build commands, discipline.
2. [`agent_debugging_guide.md`](agent_debugging_guide.md) — recurring traps (stale `grid.cuh`, `FLANGE_IDX`,
   warp/block sync, robot constants, submodules).
3. `HANDOFF.md` — current status / open work. **This is a LOCAL, gitignored per-session handoff**
   (agent scratch, not a tracked artifact), alongside `docs/open-tasks/`. At the **start** of a session,
   read it if present; at the **end**, create or update it so the next session has the current state.
4. `README.md` — user-facing usage + benchmark guide.

**Where things live:** solver `csrc/kernel/hjcd_kernel.cu`; generated kinematics `csrc/generated/grid.cuh`
(from `external/GRiD`); warp linalg `external/GLASS`; robot collision `csrc/robots/{panda,fetch}.cuh`;
Python API `hjcdik/__init__.py` + `csrc/bindings/pybind_module.cpp`; benchmark `benchmark/hjcd_ik_bench.py`.

**Discipline:** never hand-edit `grid.cuh`; keep `FLANGE_IDX`/target consistent; keep math warp-scoped;
short single-line commits, no Co-Authored-By footer.

**Build:** `git submodule update --init --recursive && python -m pip install -e .`

**Now:** on branch `grid-glass-integration` re-basing onto latest GRiD/GLASS (see the local `HANDOFF.md` if present).
