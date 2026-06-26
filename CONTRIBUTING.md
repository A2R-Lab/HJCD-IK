# Contributing to HJCD-IK

Thanks for contributing! HJCD-IK is a batched GPU IK solver built on GRiD + GLASS.

## Before you start
- Read [`CLAUDE.md`](CLAUDE.md) and [`docs/development/agent_debugging_guide.md`](docs/development/agent_debugging_guide.md).
- Initialize submodules: `git submodule update --init --recursive`.

## Workflow
- Branch from `main`; keep PRs focused. Short, single-line commit messages; **no `Co-Authored-By` footer**.
- Changes that belong upstream (kinematics → GRiD, linear algebra → GLASS) should be PR'd to those repos
  rather than patched locally — keep HJCD-IK thin and the dependencies modular.

## Discipline
- **Never hand-edit `csrc/generated/grid.cuh`** — it is GRiD codegen output. Regenerate via
  `python scripts/codegen/generate_grid.py <urdf> -t <target>` and rebuild.
- **Keep math warp-scoped.** The solver is warp-per-candidate; use warp primitives, not block-scoped ones.
- **No regressions.** Run `python benchmark/hjcd_ik_bench.py --skip-grid-codegen` and compare to the committed
  baseline before/after kernel changes; isolate timing runs (no concurrent GPU load).

## Tests & docs
- `pytest tests/` — regression (solved-rate / pos-ori error vs baseline) + FK-equivalence.
- Docs: `cd docs && make all` (Sphinx + Doxygen/Breathe).
