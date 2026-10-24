# Native RTT computational modules

These modules implement the tanh-diffusion experiment and its trajectory-tube
admission/checking path. Each computational object is defined in one module.
The top-level `run_rtt_iclr2027_arrhenius.py` entry point is a small compatibility
wrapper; it contains no duplicated implementation.

| Responsibility | Modules |
|---|---|
| Configuration and reproducibility | `metadata`, `config`, `errors`, `utils`, `artifacts` |
| Graphs and calls | `data`, `environment` |
| Incumbent and reference calculus | `models`, `incumbent_training`, `tails`, `complexity` |
| Internal adaptation | `proposals`, `solver`, `execution`, `bounds`, `checker` |
| Fitting and calibration | `training`, `placement`, `calibration`, `baselines` |
| Evaluation and assurance | `geometry`, `evaluation`, `statistics`, `diagnostics` |
| Experiment lifecycle | `loading`, `task_plan`, `tasks`, `gates`, `reporting`, `cli` |

## Stable API

Use `from rtt.core import api` when assembling an experiment from these
components. A component can also be imported directly from its defining module,
for example `from rtt.core.solver import solve_block_program`.

The explicit `_api.symbol` references in computational modules resolve through
a static symbol-ownership table. Resolution is lazy to avoid circular imports
between fitting, evaluation, and experiment assembly. It does not evaluate
generated source or dynamically execute files. Assignments through the public
API are forwarded to the owning module, preserving one object identity for
explicit adapters and test substitutions.

The package's other dataset interfaces should use their declared execution
backend. The native tanh tube assumptions must not silently be applied to a
different neural-network architecture.

## Numerical interpretation

The core retains the documented research arithmetic status and staged fitting
procedure. A successful numerical check or a passed regression test is not an
operation-level directed-rounding proof. Measured outputs and paper reference
values remain separate in the reporting module.
