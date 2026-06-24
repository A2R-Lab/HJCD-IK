# Contributing

HJCD-IK is a batched GPU IK solver built on [GRiD](https://github.com/A2R-Lab/GRiD) (kinematics codegen)
and [GLASS](https://github.com/A2R-Lab/GLASS) (single-block / warp-scoped CUDA linear algebra). The
canonical contributor entry points are the repository's `CLAUDE.md` (architecture + mental model) and
[the debugging guide](https://github.com/A2R-Lab/HJCD-IK/blob/main/docs/development/agent_debugging_guide.md)
(recurring traps + validation checklist).

## Before you start

- Initialize submodules: `git submodule update --init --recursive`.
- Set up a dev environment (venv, deps, codegen, build, **and the docs toolchain**) with
  `./scripts/setup/setup_dev.sh`.

## Workflow

- Branch from `main`; keep PRs focused. **Short, single-line commit messages; no `Co-Authored-By` footer.**
- Changes that belong upstream — kinematics in **GRiD**, linear algebra in **GLASS** — should be PR'd to
  those repos rather than patched locally. Keep HJCD-IK thin and the dependencies modular.

## Discipline

- **Never hand-edit `include/test_cuh/grid.cuh`.** It is GRiD codegen output. Regenerate it with
  `python scripts/codegen/generate_grid.py <urdf> -t <target>` and rebuild — see
  {doc}`../user_guide/tutorials/custom_robot`.
- **Keep the math warp-scoped.** The solver is warp-per-candidate; use warp primitives
  (`__shfl_*_sync` / `__syncwarp`, `grid::ee_pose_inner_warp`, `glass::warp::`), not block-scoped,
  cooperative-groups, or vendor paths. See {doc}`../user_guide/concepts/batch_execution`.
- **No regressions.** Run `python benchmark/hjcd_ik_bench.py --skip-grid-codegen` before/after kernel
  changes and compare to the committed baseline. Isolate timing runs (no concurrent GPU load).

## Tests & docs

- `pytest tests/` — regression (solved-rate / position–orientation error vs. the committed baseline) plus
  FK-equivalence checks.
- `cd docs && make all` — build the docs (Sphinx + Doxygen/Breathe). See {doc}`sphinx_edit_guide`.
