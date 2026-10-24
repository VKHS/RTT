# Reference-Tail Trust (RTT)

Research code for **Reference-Tail Trust: Certified Probability Floors for Learned Updates Inside a Deployed Network**. RTT learns internal proposals, evaluates them against the frozen incumbent's remaining computation, and admits updates subject to uncertainty and probability-floor budgets. A final check determines whether to serve the adapted prediction or return the incumbent for the whole call.

The repository contains separate experiment interfaces for all nine benchmark families in the supplied manuscript. GOOD-Arxiv degree and time shifts have distinct interfaces, giving **ten dataset/shift directories**. The manuscript's Appendices A–H were included in the supplied PDF. Source provenance is recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

## Implementation scope

There are two explicitly named execution paths:

| Path | Entry point | Computation |
|---|---|---|
| Native arxiv trajectory tube | `experiments/ogbn_arxiv/run_native.py` and `run_native.sh` | The 32-step tanh-diffusion reference, interval factors, backward contracts, short windows, exact affine terminal price, independent trajectory checks, and the registered corruption-mixture protocol. Its implementation is organized under `src/rtt/core/`. |
| Shared exact reference tail | Each dataset's `run.py` and `run.sh` | The actual graph, convolutional, tabular, or molecular continuation is evaluated for candidate admission. Joint physical uncertainty, sequential calibration, call/row budgets, a finite-step remainder, and an independent endpoint replay are implemented in `src/rtt/`. |

The second path makes the different architectures executable without applying tanh-specific bounds to unrelated networks. It performs additional reference-tail evaluations and **does not establish the native path's fast-execution claims**. Its non-image profiles use the loaded deployment graph/domain; they do not silently inherit the native arxiv corruption generator.

Fitting uses proposal warm starts followed by admission-head fitting on executed-prefix trajectories. Implicit differentiation through the paper's KKT system is not implemented. Numerical checks use floating-point computation and conservative margins; they are research checks, not a verified directed-rounding proof. Original trained weights and original run records were not available, and no paper accuracy, gain, or latency is presented as a newly reproduced result. The method documents identify the exact implemented calculations and the choices left unspecified by the manuscript.

## Separate datasets

Every directory below contains its own `config.json`, `run.py`, `run.sh`, and `README.md`. Each configuration is complete and can be frozen independently. Loaders, task losses, masks, and split rules are dataset-specific; the RTT engine is shared.

| Directory under `experiments/` | Task and deployment setting | Default reference | Primary metric |
|---|---|---|---|
| [ogbn_arxiv](experiments/ogbn_arxiv/) | Citation-node classification; real publication-year allocation | Tanh diffusion, 32 steps | Accuracy |
| [good_arxiv_degree](experiments/good_arxiv_degree/) | GOOD degree/covariate shift | GCNII | Accuracy |
| [good_arxiv_time](experiments/good_arxiv_time/) | GOOD time/concept shift | Smooth GRAND diffusion | Accuracy |
| [twitch](experiments/twitch/) | Cross-graph binary classification | APPNP | ROC-AUC |
| [facebook100](experiments/facebook100/) | Cross-university node classification | GCNII | Accuracy |
| [elliptic](experiments/elliptic/) | Temporal illicit/licit transaction classification | Smooth GRAND diffusion | Illicit-positive F1 |
| [ogbn_products](experiments/ogbn_products/) | Product-node classification | Tanh diffusion | Accuracy |
| [imagenet_c](experiments/imagenet_c/) | Severity 3 across 15 corruptions | Actual ResNet-50; last four bottleneck blocks with Tent proposals | Top-1 accuracy |
| [acs_income](experiments/acs_income/) | California-to-five-state tabular transfer | Residual tanh MLP | Accuracy |
| [ogbg_molpcba](experiments/ogbg_molpcba/) | Molecular graphs; 128 masked binary assays | Bond-aware GIN | Macro average precision |

Seven additional complete configurations live in the relevant `variants/` directories: arxiv APPNP, GraphSAGE-2 and GCN-3; molecular contractive diffusion, centered GIN, GIN with a virtual node, and a scaffold-group protocol requiring a real scaffold manifest. There are **17 full configurations**, each resolvable as `full`, `pilot`, or `smoke`.

The configurations record reconstruction choices such as cross-graph domains, ACS year/states, broad-model widths, optimization settings, and molecular regression tolerance. These values were not uniquely recoverable from the supplied manuscript. Consult the dataset README before interpreting an experiment as a comparison with a paper table.

## Install and run a local execution check

Python 3.11 or newer is required. Run commands from the repository root. Select a PyTorch build appropriate to the machine before installing the package. A CPU development environment can be created as follows:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[data,dev]'

python experiments/ogbn_arxiv/run.py run --profile smoke \
  --run-dir outputs/ogbn_arxiv/smoke --device cpu
```

`smoke` executes real training, checkpointing, calibration, selection, auditing, and prediction logic on small generated fixtures. It does not download a benchmark or pretrained weights. Image smoke checks use the actual ResNet-50 structure on small images. A small conformal sample often produces an infinite uncertainty radius, which conservatively closes admission; `NO_MEMBER_SELECTED` and an incumbent-only audit can be valid completed outcomes.

To run another dataset's execution check, change the directory name:

```bash
python experiments/ogbg_molpcba/run.py run --profile smoke \
  --run-dir outputs/ogbg_molpcba/smoke --device cpu
```

The installed command exposes the same interface:

```bash
rtt --list-datasets
rtt inspect --config experiments/ogbn_arxiv/config.json --profile full
rtt --help
```

The source distribution and repository archive include experiment configurations and shell scripts. The wheel installs the Python package and `rtt` command; supply a configuration path when using the wheel independently of the repository.

## Prepare real data and run a pilot

Read [DATA_SOURCES.md](docs/DATA_SOURCES.md) and the dataset-specific README first. OGB and Folktables have supported upstream download paths; other datasets require documented local files. ImageNet data and pretrained weights are separate inputs. Missing real data produces an error; it is never silently replaced with a synthetic fixture.

For ogbn-arxiv:

```bash
python experiments/ogbn_arxiv/run.py prepare --download \
  --data-root data/ogbn_arxiv --run-dir outputs/ogbn_arxiv/prepared

python experiments/ogbn_arxiv/run.py run --profile pilot \
  --data-root data/ogbn_arxiv --run-dir outputs/ogbn_arxiv/pilot --device cuda
```

`prepare` validates and fingerprints data and creates the label-role protocol. The example uses a separate preparation run because the pilot and full profiles have different frozen configurations. `pilot` uses real data with reduced fitting and audit counts; it is a resource and execution check. `full` uses the complete declared configuration and a 2,048-call final audit. Full reference-tail work can be expensive, especially for large graphs and image batches; inspect the pilot's memory, timings, inactive depths, and statistical decisions before the full run.

Use a new run directory after changing scientific settings. Relative dataset option paths and model checkpoint paths resolve against the selected data root; see the input contract for the exact fields.

## Stages, predictions, and metrics

The shared pipeline is:

1. `prepare`: validate data, record its identity, and allocate disjoint label roles.
2. `train`: fit/freeze the incumbent, temperature, proposal, admission heads, and matched output corrector.
3. `calibrate`: process each registered member depth by depth with earlier radii frozen.
4. `select`: evaluate the fixed ordered menu on the separate selection allocation.
5. `audit`: evaluate the frozen policy using predetermined call indices, optionally in shards.
6. `report`: require the complete committed audit and generate summaries.

`run` performs these stages locally in order and skips completed stages only after their frozen inputs are checked. Individual stage commands accept the same configuration/profile/run arguments. Audit labels are read after label-free serving to calculate outcomes. Serving rejects a batch containing labels.

For an existing completed smoke run:

```bash
python experiments/ogbn_arxiv/run.py predict --profile smoke \
  --run-dir outputs/ogbn_arxiv/smoke \
  --output outputs/ogbn_arxiv/smoke_prediction
```

The prediction artifact contains item IDs, logits, and normalized log probabilities. For molecules, probabilities have shape `[graphs, 128, 2]`, representing two outcomes per independent assay; missing assays remain masked.

Audit summaries average per-episode metrics. A mean of batch AP values is not whole-population AP. Use the separate `evaluate` command to collect predictions across an official split and compute pooled metrics:

```bash
python experiments/ogbg_molpcba/run.py evaluate --profile full \
  --data-root data/ogbg_molpcba --run-dir outputs/ogbg_molpcba/full \
  --split test --batch-size 64 --device cuda \
  --output outputs/ogbg_molpcba/official_test
```

This evaluation is labeled separately and records its population and batch size. It is not an independent confirmatory sample if it overlaps a previously used custom audit population. The [protocol guide](docs/DATASET_PROTOCOLS.md) explains chronology, image identity grouping, missing labels, scaffold groups, and the statistical reporting rules.

## Run on Arrhenius

The launchers preserve the supplied settings:

| Property | Value |
|---|---|
| Compute project | `naiss2025-5-643-gpu` |
| Storage project | `naiss2025-6-428` |
| Base path | `/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls` |
| Virtual environment | `<base>/code/venv_arrhenius_gh200` |
| Build module | `GPU/buildenv-nvhpc/25.9-cu13.0` |
| Python module | `GPU/Python/3.13.5-bare-gcc-2025b-eb` |
| Default numerical worker | One GPU, 16 CPUs, `gpu` partition |

Place the repository in the code area, then run:

```bash
bash scripts/setup_arrhenius.sh
# Wait for the submitted environment check to finish successfully.
bash experiments/ogbn_arxiv/run.sh smoke
bash experiments/ogbn_arxiv/run.sh prepare
bash experiments/ogbn_arxiv/run.sh pilot
```

Each dataset uses its own cache/output namespace. The scheduler freezes source and configuration files, submits dependent jobs, commits stage receipts, and preserves original audit indices on infrastructure resume. It rejects changed inputs, active duplicate submissions, and automatic retries of recorded scientific failures. `status` and `resume` take the printed absolute run directory. See [ARRHENIUS.md](docs/ARRHENIUS.md) for resource overrides, real-data prerequisites, full runs, logs, and the separate native arxiv commands. Local validation uses mocked Slurm; GH200 execution and performance require a real server run.

## Repository organization

| Location | Responsibility |
|---|---|
| `experiments/<dataset>/` | Independent configuration, Python and shell entry points, data/run instructions |
| `src/rtt/datasets/` | Validated raw-data adapters, portable interchange format, dataset identity |
| `src/rtt/models.py`, `tent.py` | Actual reference architectures and independent-call Tent proposal |
| `src/rtt/adaptation.py`, `losses.py`, `frontier.py` | Proposals, physical geometry, admission, task losses, floor allocation |
| `src/rtt/training.py`, `protocol.py`, `experiment.py` | Fitting, label roles, calibration, frozen artifacts, selection and evaluation |
| `src/rtt/cli.py`, `hpc.py` | Local stage interface and Slurm orchestration |
| `src/rtt/core/` | Modular native arxiv trajectory-tube implementation |
| `scripts/`, `requirements/` | Setup, submission, repository checks, dependency profiles |
| `tests/`, `validation/` | Mathematical, model, data, lifecycle and launcher checks; captured validation evidence |
| `docs/` | Method, data, protocol, architecture and server documentation |
| `.github/workflows/tests.yml` | CPU test/build workflow for pushes and pull requests |

The top-level original Python and shell filenames remain compatibility entry points. They call the modular native implementation and retain its own configuration and output layout.

## Verification and development

```bash
python scripts/check_repository.py
python -m pytest -q tests
python -m build --no-isolation
python scripts/check_repository.py --archives dist
```

Tests exercise mathematical prices and fallbacks, all model families, raw-file adapters, label-role integrity, real nonzero internal admissions, ReLU-kink remainders, masked molecular tasks, all ten synthetic end-to-end pipelines, and both Slurm launch paths. They do not download the full benchmarks or submit server jobs. The exact local environment, results, and limits are recorded in [validation/VALIDATION.md](validation/VALIDATION.md).

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing frozen-run semantics. `SHA256SUMS` records the delivered source snapshot. Generated runs, data, environments, caches, checkpoints, and build output are excluded by `.gitignore`.

## Method and provenance

- [Shared reference-tail method and implementation map](docs/METHOD_SPECIFICATION.md)
- [Native arxiv trajectory-tube method](docs/NATIVE_ARXIV_METHOD.md)
- [Dataset sources and local file formats](docs/DATA_SOURCES.md)
- [Dataset protocols and statistical interpretation](docs/DATASET_PROTOCOLS.md)
- [Arrhenius setup and submission](docs/ARRHENIUS.md)
- [Citation metadata](CITATION.cff)

The supplied manuscript is the primary method source. Dataset and software documentation links are collected beside the corresponding input contracts. Cite the manuscript and identify the backend, configuration, data allocation, and actual software environment when reporting new experimental results.
