# Running the dataset experiments on Arrhenius

Each dataset has its own experiment directory, configuration, Python entry point
and shell wrapper. The shared scheduler freezes that experiment before submitting
its training, calibration, selection, audit shards and final report.

## Preserved server settings

| Setting | Default |
|---|---|
| Compute allocation | `naiss2025-5-643-gpu` |
| Storage allocation | `naiss2025-6-428` |
| Base directory | `/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls` |
| Virtual environment | `<base>/code/venv_arrhenius_gh200` |
| Python module | `GPU/Python/3.13.5-bare-gcc-2025b-eb` |
| Build module | `GPU/buildenv-nvhpc/25.9-cu13.0` |
| Partition | `gpu` |
| Numerical worker | One GPU and 16 CPUs |
| Maximum concurrent audit shards | Two |
| Dataset root | `<base>/datasets/rtt/<dataset>` |
| Run root | `<base>/outputs/RTT_<dataset>_<mode>_<timestamp>_<pid>` |

These are the project settings supplied for this work. Their current allocation,
quota and module availability are checked on the server. No remote jobs were
submitted while preparing the repository.

## 1. Place the repository and check the environment

For example, extract the repository into the existing code area:

```bash
cd /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/code/RTT
bash scripts/setup_arrhenius.sh
```

The setup command submits a short Slurm job. It creates a new environment at its
final path, or checks an existing environment without changing its packages.
It validates the Arm architecture, CUDA, and float32/float64 sparse and dense
operations. Setup logs are `<base>/logs/RTT_setup_<jobid>.out` and `.err`.

To explicitly synchronize an existing environment to the dependency requirements:

```bash
RTT_SETUP_UPDATE=1 bash scripts/setup_arrhenius.sh
```

To use another environment, export `RTT_VENV` before both setup and submission.
The default setup includes the dataset dependencies in `requirements/data.txt`.
Optional extras are comma-separated:

```bash
RTT_EXTRAS=data,image bash scripts/setup_arrhenius.sh
```

The image extra includes torchvision for users who need its image tooling. The
torch-only ResNet implementation does not require torchvision. The optional
CUDA pair is `torch==2.12.1+cu130` and `torchvision==0.27.1+cu130`; the official
PyTorch documentation pairs these versions, and the CUDA 13.0 index includes
CPython 3.13 aarch64 wheels. Dependencies are installed from the official wheel
index, with the remaining requirements installed from their listed packages.

Wait until setup finishes successfully before starting an experiment.

## 2. Choose a dataset

Use the corresponding directory directly:

| Experiment | Directory |
|---|---|
| ogbn-arxiv | `experiments/ogbn_arxiv/` |
| GOOD-Arxiv degree shift | `experiments/good_arxiv_degree/` |
| GOOD-Arxiv time shift | `experiments/good_arxiv_time/` |
| Twitch | `experiments/twitch/` |
| Facebook100 | `experiments/facebook100/` |
| Elliptic | `experiments/elliptic/` |
| ogbn-products | `experiments/ogbn_products/` |
| ImageNet-C | `experiments/imagenet_c/` |
| ACSIncome | `experiments/acs_income/` |
| ogbg-molpcba | `experiments/ogbg_molpcba/` |

Each directory's documentation describes its expected source data and split
policy. Dataset archives, licensed image data and trained checkpoints are not
bundled. A loader does not silently replace an unavailable benchmark with a
different dataset. Synthetic fixtures are used only when `smoke` is requested.

## 3. Run a smoke check, prepare data, then run a pilot

For ogbn-arxiv:

```bash
bash experiments/ogbn_arxiv/run.sh smoke
```

The smoke profile uses synthetic fixtures with small fitting and audit counts.
Its results establish execution behavior; they are not benchmark measurements
or evidence of statistical acceptance at paper sample sizes.

Prepare the real source data following the experiment's data instructions:

```bash
bash experiments/ogbn_arxiv/run.sh prepare
```

`prepare` permits downloads supported by that loader. Datasets requiring manual
files still require those files at the documented location. Other modes expect
prepared data by default. To explicitly permit a supported download while
starting training, set `RTT_DOWNLOAD=1`.

After preparation finishes:

```bash
bash experiments/ogbn_arxiv/run.sh pilot
```

Inspect the pilot's logs, artifacts, inactive depths, fallback rates, resource
usage and statistical decisions before running the full profile:

```bash
bash experiments/ogbn_arxiv/run.sh full
```

For any other dataset, replace `ogbn_arxiv` with its directory name. For example:

```bash
bash experiments/elliptic/run.sh smoke
bash experiments/acs_income/run.sh prepare
RTT_DOWNLOAD=1 bash experiments/imagenet_c/run.sh pilot
```

Every command prints its exact run directory. An optional final argument supplies
an explicit new directory:

```bash
bash experiments/ogbn_products/run.sh pilot /absolute/path/to/new_run
```

## 4. Execution and reporting

The scheduler submits this dependency chain:

1. Train the incumbent, proposal, admission heads and declared comparisons.
2. Calibrate the frozen fitting outputs.
3. Select the registered member.
4. Run independent audit shards with their original deterministic indices.
5. Generate the report after every audit shard succeeds.

Each stage runs through `srun`. A failed job prevents downstream jobs from
starting. The report uses the audit's actual records and separately reports its
statistical decision; a successful Slurm exit does not imply that a member or
statistical claim passed.

The main numerical artifacts include:

| Stage | Artifacts |
|---|---|
| Data preparation | `data_manifest.json` |
| Training | `manifest.json`, `protocol.json`, `training.json`, `checkpoints/model.pt`, `checkpoints/adaptation.pt` |
| Calibration | `calibration.json` |
| Selection | `selection.json` |
| Audit shard | `records/audit_shard_<index>.jsonl` and its `.meta.json` |
| Report | `summary.json`, `metrics.csv`, `REPORT.md` |

The scheduler stores its independent source/configuration snapshot and completion
receipts under `orchestration/`. Per-stage stdout/stderr files are under `logs/`.
The exact modules, package list and GPU information are recorded per job under
`environment/`.

## 5. Monitor and resume the same frozen experiment

```bash
bash experiments/ogbn_arxiv/run.sh status /absolute/path/to/run
squeue -u "$USER"
```

After earlier jobs have finished or been cancelled:

```bash
bash experiments/ogbn_arxiv/run.sh resume /absolute/path/to/run
```

Resume checks the source inventory, frozen configuration, plan and completed
artifact hashes. It skips completed stages and preserves unfinished audit shard
indices. It refuses to submit duplicates while this run has active jobs. Recorded
scientific failures are not automatically retried or resampled. Infrastructure
interruption can be resumed with the same frozen inputs.

A changed method, configuration or source requires a new run. Resume uses the
original run's settings even if the current shell has different resource
overrides or the working repository has changed.

## Resource settings

The default numerical request is one GPU and 16 CPUs. Most stages initially
request 110 GB host memory; ogbn-products requests 180 GB, and preparation/report
jobs request 64 GB. These are operational starting points rather than measured
requirements. Full-profile walltimes vary by stage; smoke and pilot requests are
shorter.

For example:

```bash
export RTT_CPUS=16
export RTT_MAX_CONCURRENT=2
export RTT_AUDIT_SHARDS=8
export RTT_MEM_TRAIN=110G
export RTT_TIME_TRAIN=20:00:00
export RTT_DATA_ROOT=/absolute/path/to/this_dataset
bash experiments/ogbn_arxiv/run.sh full
```

`RTT_MEM_<STAGE>` and `RTT_TIME_<STAGE>` accept `PREPARE`, `TRAIN`, `CALIBRATE`,
`SELECT`, `AUDIT` and `REPORT`. Resource choices are frozen at submission.
`RTT_CONFIG=/absolute/path/to/config.json` selects an explicit configuration for
the same dataset; the file is copied into the run before submission.

Do not add `-N 1` to the shared one-GPU requests. Arrhenius documentation states
that specifying `-N` reserves a whole node. Log directories are created before
the first `sbatch` call. There is no artificial GPU keepalive workload.

## Native arxiv trajectory-tube implementation

The detailed tanh-diffusion implementation remains available through the supplied
top-level filenames. It uses the modular `src/rtt/core` computation and retains
its separate native experiment plan:

```bash
bash run_rtt_iclr2027.sh smoke
bash run_rtt_iclr2027.sh pilot
bash run_rtt_iclr2027.sh full
```

The arxiv experiment folder exposes the same native route without changing its
arguments or exit codes:

```bash
python experiments/ogbn_arxiv/run_native.py --help
bash experiments/ogbn_arxiv/run_native.sh smoke
bash experiments/ogbn_arxiv/run_native.sh pilot
bash experiments/ogbn_arxiv/run_native.sh full
```

The native Python interface is flag based; the generic `run.py` uses subcommands.
`configs/arxiv.json` contains native overrides, while the dataset folder's
`config.json` configures the generic exact-tail pipeline. These schemas and run
directories remain distinct.

Its existing data paths remain `<base>/datasets/ogb_cache` and
`<base>/datasets/graph/rtt_ogbn_arxiv_graph.npz`; its outputs retain the
`000_MANIFEST`, numbered stage directories and `900_REDUCE` layout. Monitor and
resume this native route with the matching top-level script, not the generic
dataset wrapper:

```bash
bash run_rtt_iclr2027.sh status /absolute/path/to/native_run
bash run_rtt_iclr2027.sh resume /absolute/path/to/native_run
```

The two routes use different declared checking/execution backends. Their timings
must retain that distinction; an endpoint-verified cross-architecture run is not
evidence of the native fast path's performance.

## Validation boundary and official references

The repository's launcher tests mock Slurm and Lmod locally. They check arguments,
dependency ordering, quoted paths, log creation, source and artifact integrity,
audit shard ownership and resume behavior. They do not submit jobs or establish
GH200 performance.

- [Arrhenius scheduling and shared-node behavior](https://user.naiss.se/arrhenius_hpc/job_management/scheduling/)
- [Arrhenius quick start](https://user.naiss.se/arrhenius_hpc/quickstart/)
- [Official PyTorch version pairs](https://pytorch.org/get-started/previous-versions/)
- [Official CUDA 13.0 torchvision wheel index](https://download.pytorch.org/whl/cu130/torchvision/)
