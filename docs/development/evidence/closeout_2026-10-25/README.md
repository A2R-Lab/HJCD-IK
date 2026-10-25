# Usability closeout validation

Starting revision: `e8cc1c3`; comparison base: locally recorded `origin/main`, `119cacc`.
The starting signed receipt verified all 317 tests. Dependencies remain GRiD `712df1c`
(grid-rbd 0.6.0), GLASS `9e57178`, foam `116928f`. No timing is authorized during
this shared-box implementation session. No solver policy or default-model change is planned.

## Coverage map

Existing tests cover runtime identity/load/damaged entries and module isolation (`test_build`),
independent FK/Jacobian and mixed-unit scale (`test_general_joints`), invalid inputs/world
replacement/result lifetime (`test_solver_contract`), and warm-start validation (`test_stage3`).
New coverage will focus on the realistic seven-joint assembly and one integrated installed
workflow, not repeat the entire matrix. The two-joint example remains the minimal introduction.

Intended changes: example and deterministic assembly; reuse/generalize the independent test
sphere-placement oracle; frozen CPU-derived feasible target witnesses; integrated wheel runner;
targeted failure diagnostics/tests and stale user/agent instructions. Kernel mathematics,
default robot assets, dependency pins and paper data are outside this change.

Validation outcomes and final review findings are pending.
