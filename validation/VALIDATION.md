# Repository validation

Validation date: **2026-10-24**. This report describes checks actually run while preparing this repository. It does not report a full benchmark experiment or an Arrhenius execution.

## Result

The complete local CPU suite passed:

```text
218 passed, 2 warnings in 77.75s (0:01:17)
```

The unedited pytest result is in [tests.txt](tests.txt). The warnings concern PyTorch sparse invariant checking and sparse CSR support status; neither was a test failure. The local environment is recorded in [environment.json](environment.json): Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.3.5, SciPy 1.17.0, and pytest 9.1.1 on x86-64 Linux. CUDA was unavailable. The Arrhenius setup and CI have their own explicitly declared dependency profiles; they were not executed on their target infrastructure during this validation.

## Coverage

| Test group | Passed | What it establishes for the tested inputs |
|---|---:|---|
| Preserved native arxiv suite | 77 | Tube/contract calculus, terminal all-class prices, hull geometry, calibration, statistical bounds, continuous frontier, deployed no-extra-endpoint behavior, persisted pipeline, high-precision diagnostics, and original launcher behavior |
| Reference architectures | 22 | Actual model continuations, shapes, graph pooling, molecular virtual nodes, ResNet-50 structure, freezing, and parameter contracts |
| Tent proposal | 13 | Independent-call reset, last-four-block BN-affine adaptation, label-free entropy updates, and preservation of the reference |
| Dataset adapters | 24 | Validated source formats, masks, graph identity, temporal/source splits, feature vocabulary, image identities, and portable arrays |
| Dataset protocols/statistics | 22 | Chronology, disjoint label roles, grouped image/scaffold allocation, conservative conformal ranks, strict regression events, and incomplete-audit blocking |
| Shared reference-tail engine | 31 | Actual nonzero admissions, all-label floors, row/call limits, ReLU-kink finite remainders, molecular masking, zero-step equality, and forced endpoint fallback |
| Separate dataset pipelines | 11 | All ten train/calibrate/select/audit/report/predict paths on explicit synthetic fixtures, molecular pooled evaluation, and smoke profiles that never request pretrained downloads |
| Frozen experiment integrity | 5 | Two-shard audit completion, refusal to resample committed shards, checkpoint changes despite rewritten sidecars, and invalid calibration/selection binding |
| Shared Arrhenius scheduler | 13 | Mocked Slurm arguments, dependencies, frozen-source inventories, artifact hashes, stage completion, duplicate-job rejection, and stable shard resume |
| **Total** | **218** | |

The raw-adapter tests use local source-format fixtures or mocked upstream APIs. OGB and Folktables packages were not installed in the validation environment, and their full datasets were not downloaded. This does not constitute a download/throughput test against their public services.

## Nonzero adaptation and task semantics

Synthetic pipeline completion alone is weak evidence for an admission algorithm: the deliberately small calibration counts often produce infinite radii, so those runs may legitimately serve only the incumbent. Separate engine tests therefore exercise genuine nonzero interventions inside actual APPNP and molecular reference continuations.

The multiclass test independently enumerates every possible label assignment on its small call and checks both aggregate and row floors. The molecular test uses 128 independent binary assays with a missing-label mask, checks both Bernoulli outcomes, and introduces a large perturbation on an unobserved assay to expose accidental treatment as a multiclass softmax. Additional tests cross a ReLU kink, fabricate invalid local admission records, and verify exact incumbent fallback after the endpoint check fails.

These are finite numerical tests. They do not prove the manuscript's real-arithmetic theorems for every floating-point primitive or establish selected-member conformal coverage after data-dependent selection.

## Repository and package checks

[repository_checks.json](repository_checks.json) records:

- Ten complete dataset/shift interfaces.
- Seventeen complete configurations and 51 successful full/pilot/smoke resolutions.
- Python syntax validation for 94 files.
- Bash syntax validation for 15 files.
- No benchmark downloads and no Slurm submissions during these checks.

An isolated build through the `setuptools.build_meta` PEP 517 backend produced a wheel and a source distribution. Archive checks verified all package modules, dataset entry points, variant configurations, native compatibility files, source provenance, and method documents. They also rejected runtime caches, checkpoints, environments, or private run output inside the build archives. The installed wheel's actual `rtt --list-datasets` command was run from outside the checkout and listed all ten datasets; the installed native package API also imported successfully. [packaging_checks.json](packaging_checks.json) records these checks.

The public workflow in `.github/workflows/tests.yml` installs its declared CPU runtime, checks repository structure, runs the suite, and builds/verifies distributions for Python 3.11 and 3.12. The workflow is prepared for GitHub; no remote GitHub Actions run is claimed here.

The delivered source archive includes this report and its captured logs. Build products are intentionally excluded from the GitHub source tree. `SHA256SUMS` records the final source files so the delivered snapshot can be checked after extraction.

## What still requires real experiment execution

The following were not measured in this environment:

- Accuracy, AP, gain, regression risk, or acceptance rate on the full real benchmarks.
- Original paper checkpoint or table reproduction.
- GH200 memory use, accelerator timings, Slurm quota, current allocation access, or server module availability.
- Full 512-call-per-depth/member calibration and 2,048-call real-data audits.
- Formal directed-rounding validity of the installed neural-network arithmetic.
- Joint proposal/solver training using implicit KKT differentiation, which is outside the implemented staged fitting scheme.

Run the dataset's data preparation and pilot on the target machine before a full experiment. Preserve the resulting configuration, source snapshot, checkpoints, protocol and audit records. A successful process exit or `status=COMPLETE` does not imply `statistical_release=true`; the scientific decision is reported separately.

## Re-run the checks

From the repository root with development dependencies installed:

```bash
python scripts/check_repository.py
python -m pytest -q tests
python -m build --no-isolation
python scripts/check_repository.py --archives dist
```

Use the dataset-specific `smoke` commands in the root README for standalone execution checks. They deliberately mark their outputs as synthetic.
