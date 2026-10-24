# Contributing

Contributions should preserve the distinction between the method, its numerical
implementation, and measured evidence. Changes to a dataset, split, sampling
unit, learned component or checking backend should be visible in configuration
and documentation.

## Development setup

Use Python 3.11 or later in a separate environment. Install an appropriate
PyTorch wheel first; for CPU development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[data,dev]'
python scripts/check_repository.py
python -m pytest -q tests
```

The tests use small generated fixtures and mocked Slurm commands. They must not
download a benchmark, submit a cluster job or require a private checkpoint.

## Code organization

Dataset loading and validation belong under `src/rtt/datasets`. Model families,
adaptation, training and evaluation use shared package interfaces. Dataset
configurations and user entry points belong in their respective `experiments/`
directory. The detailed native tanh trajectory-tube implementation is organized
under `src/rtt/core`; retain its separate assumptions and execution accounting.

Avoid duplicating the full algorithm across dataset folders. A dataset wrapper
should select its configuration and call the common implementation. Preserve
the supported top-level native-arxiv entry points when changing internal module
boundaries.

## Changes requiring focused checks

- For mathematical changes, test a small independently calculable case and the
  relevant feasibility, label-independence or fallback invariant.
- For new data paths, validate shapes, masks, task type, train-only preprocessing,
  split separation and content fingerprints using local fixtures.
- For changes to learned proposals, retain the physical displacement semantics
  and make training approximations explicit.
- For audit changes, preserve independent draws, fixed policy decisions,
  complete record counts, declared denominators and conservative treatment of
  missing or failed records.
- For scheduler changes, test dependency ordering, source/configuration
  snapshots, shard ownership, and resume behavior with mocks.

Do not replace measured outputs with paper reference values, hide failed calls,
or interpret passing finite tests as a universal floating-point proof. Performance
claims must identify the hardware, backend, workload and timing boundaries.

## Experiments and artifacts

Use a new run directory when changing scientific settings. Resume is for the same
frozen experiment and must preserve task indices and seeds. Keep raw data,
licensed image archives, private checkpoints and large generated results outside
the source repository. Commit only selected, documented validation evidence.

A pull request should explain the concrete issue, changed behavior, affected
datasets, tests run and any remaining limitations. The repository's CI workflow
checks source interfaces, CPU tests and packaging; cluster performance still
requires a separate server run.
