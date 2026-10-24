# ogbn-arxiv

This directory owns this dataset's complete configuration and Python/Arrhenius entry points. It uses `tanh_diffusion` with the shared **exact-reference-tail** RTT executor. Internal residual proposals are evaluated through the actual frozen remaining network; calibrated uncertainty, per-row/call budgets and a final label-free endpoint check govern admission. This profile fits a proposal bank and signed-message proposal together with admission heads.

## Registered configuration

| Setting | Value |
|---|---|
| Task | Node multiclass classification |
| Primary metric | accuracy |
| Incumbent | `tanh_diffusion` |
| Continuation steps | 32 |
| Hidden width | 256 |
| First intervention depth (zero based) | 16 |
| Maximum scored items per call | 4096 |
| Loss-regression tolerance zeta | 0.002 nats |
| Main call budget H_plus | 0.05 nats (=25 × zeta) |
| Per-row budget H_row | 1.0 nats |
| Full calibration / selection / audit calls | 512 per depth per member / 512 per member / 2048 |
| Seed | 1 |

The complete optimizer, proposal, covariance, solver and member-selection settings are in [config.json](config.json). `pilot` reduces work on real supplied data. `smoke` uses generated fixtures and records that it is not a benchmark result. A small calibration budget may yield an infinite radius and no admitted updates; this is an explicit conservative outcome.

## Data and identity

The OGB adapter expects `data/ogbn_arxiv/ogbn_arxiv/` or a cache created with `--download`. The outer directory is `--data-root`; OGB owns the inner benchmark directory. It retains real publication years and the official train/validation/test masks. Features are standardized using the official training allocation only.

With real publication years, `official_disjoint` resolves to the chronological protocol: training through 2017, 2018 temperature/calibration/selection pools, a reserved set and separate development/audit halves from 2019, and an unscored 2020 transfer pool. All node IDs are written to `protocol.json`. Synthetic smoke fixtures are explicitly non-temporal.

The prepare command below explicitly permits the documented upstream downloader. Subsequent commands expect the prepared cache.

## Local commands

Run from the repository root after `python -m pip install -e '.[data,dev]'` and installing a suitable PyTorch build. Use a new output directory for each frozen experiment.

```bash
python experiments/ogbn_arxiv/run.py run --profile smoke --run-dir outputs/ogbn_arxiv/smoke --device cpu
python experiments/ogbn_arxiv/run.py prepare --data-root data/ogbn_arxiv --run-dir outputs/ogbn_arxiv/prepared --download
python experiments/ogbn_arxiv/run.py run --profile pilot --data-root data/ogbn_arxiv --run-dir outputs/ogbn_arxiv/pilot --device cuda
python experiments/ogbn_arxiv/run.py run --profile full --data-root data/ogbn_arxiv --run-dir outputs/ogbn_arxiv/full --device cuda
```

`prepare`, `train`, `calibrate`, `select`, `audit`, `report`, `predict`, and `inspect` are also separately callable. Use `--help` for the current command arguments. `--config PATH` overrides this folder's default configuration. Relative dataset option paths and model checkpoint paths resolve against `--data-root`.

## Arrhenius commands

```bash
bash scripts/setup_arrhenius.sh
bash experiments/ogbn_arxiv/run.sh prepare
bash experiments/ogbn_arxiv/run.sh smoke
bash experiments/ogbn_arxiv/run.sh pilot
bash experiments/ogbn_arxiv/run.sh full
bash experiments/ogbn_arxiv/run.sh status /absolute/path/to/run
bash experiments/ogbn_arxiv/run.sh resume /absolute/path/to/run
```

The wrapper selects the dataset-specific configuration, cache and output root. It preserves project `naiss2025-5-643-gpu`, storage allocation `naiss2025-6-428`, the provided base path and GH200 modules. Each run freezes source/configuration hashes and submits train → calibrate → select → audit shards → report with dependencies. Environment overrides and stage outputs are documented in [Arrhenius setup](../../docs/ARRHENIUS.md).

## Compute and interpretation

The full citation graph remains on the chosen device; a query selects scored nodes without removing graph context. The 256-channel, 32-step model and repeated tail differentiation require substantially more memory than a shallow classifier.

Exact remaining-tail evaluation trades compute for support across the registered architectures. Full 512/512/2048-call settings can be expensive and are not wall-time guarantees. Hardware timing and paper accuracy have to be measured on actual benchmark runs; no such results are bundled in this directory.

The supplied main backbone uses the manuscript's T=32, alpha=0.1, tau=0.9 and transport spectral norm 1.6. Width, optimizer settings and initialization are declared reconstruction choices. This unified entry point uses exact remaining-tail evaluation and a fixed loaded deployment graph. The paper's registered corruption mixture and specialized interval-tube/window executor are available through the separate native arxiv implementation; see the native commands in [Arrhenius instructions](../../docs/ARRHENIUS.md). Results from these paths must be identified by backend and protocol.

Shared input contracts: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md). Data-role rules and reporting distinctions: [DATASET_PROTOCOLS.md](../../docs/DATASET_PROTOCOLS.md).

## Native trajectory-tube experiment

The native arxiv implementation is directly exposed in this same dataset folder:

```bash
python experiments/ogbn_arxiv/run_native.py --help
python experiments/ogbn_arxiv/run_native.py --print_identity
python experiments/ogbn_arxiv/run_native.py --semantic_preflight_only --cpu --out_dir outputs/ogbn_arxiv/native_preflight
bash experiments/ogbn_arxiv/run_native.sh smoke
bash experiments/ogbn_arxiv/run_native.sh pilot
bash experiments/ogbn_arxiv/run_native.sh full
bash experiments/ogbn_arxiv/run_native.sh status /absolute/path/to/native_suite
bash experiments/ogbn_arxiv/run_native.sh resume /absolute/path/to/native_suite
```

`run_native.py` forwards the native flag-based interface and preserves its exit
code. It accepts `--scale full|pilot|preflight`, `--init_suite`, `--stage`,
`--stage_gate` and the native dataset arguments shown in `--help`. It does not use
the generic `run/prepare/train` subcommand schema. The local semantic preflight is
synthetic and exercises the native computation and checks.

`run_native.sh` forwards to the preserved native Arrhenius orchestrator. It uses
the registered corruption mixture, specialized trajectory-tube/window checks,
native correctors and diagnostics. Its data-cache paths and numbered stage
layout remain documented in [ARRHENIUS.md](../../docs/ARRHENIUS.md#native-arxiv-trajectory-tube-implementation).
The complete original configuration defaults are in `rtt.core.config.RTTConfig`;
[configs/arxiv.json](../../configs/arxiv.json) supplies the main explicit native
overrides. For example, on a new full native run:

```bash
RTT_CONFIG="$PWD/configs/arxiv.json" bash experiments/ogbn_arxiv/run_native.sh full
```

Keep native and generic runs in separate directories: they intentionally have
different frozen configurations, artifact layouts and compute profiles.

## Separate backbone variants

Every file below is a complete configuration, so it can be reviewed and frozen independently.

| File in `variants/` | Family | Steps | Width | First open depth |
|---|---|---:|---:|---:|
| `appnp.json` | `appnp` | 32 | 256 | 16 |
| `graphsage2.json` | `graphsage` | 2 | 256 | 0 |
| `gcn3.json` | `gcn` | 3 | 256 | 0 |

```bash
python experiments/ogbn_arxiv/run.py run --config experiments/ogbn_arxiv/variants/appnp.json --profile smoke --run-dir outputs/ogbn_arxiv/appnp_smoke --device cpu
RTT_CONFIG="$PWD/experiments/ogbn_arxiv/variants/appnp.json" bash experiments/ogbn_arxiv/run.sh pilot
```

Variant results belong to a fresh run directory. The full remaining network is retained for the shallow GCN/GraphSAGE families; molecular pooling and virtual-node states remain part of their actual model continuation.
