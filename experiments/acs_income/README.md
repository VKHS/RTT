# Folktables ACSIncome

This directory owns this dataset's complete configuration and Python/Arrhenius entry points. It uses `residual_tanh_mlp` with the shared **exact-reference-tail** RTT executor. Internal residual proposals are evaluated through the actual frozen remaining network; calibrated uncertainty, per-row/call budgets and a final label-free endpoint check govern admission. This profile fits a proposal bank and signed-message proposal together with admission heads.

## Registered configuration

| Setting | Value |
|---|---|
| Task | Cross-state tabular binary classification |
| Primary metric | accuracy |
| Incumbent | `residual_tanh_mlp` |
| Continuation steps | 8 |
| Hidden width | 64 |
| First intervention depth (zero based) | 4 |
| Maximum scored items per call | 512 |
| Loss-regression tolerance zeta | 0.002 nats |
| Main call budget H_plus | 0.05 nats (=25 × zeta) |
| Per-row budget H_row | 1.0 nats |
| Full calibration / selection / audit calls | 512 per depth per member / 512 per member / 2048 |
| Seed | 1 |

The complete optimizer, proposal, covariance, solver and member-selection settings are in [config.json](config.json). `pilot` reduces work on real supplied data. `smoke` uses generated fixtures and records that it is not a benchmark result. A small calibration budget may yield an infinite radius and no admitted updates; this is an explicit conservative outcome.

## Data and identity

The Folktables adapter creates an `ACSDataSource` rooted at `data/acs_income/` for the 2018 one-year person survey. Use explicit `--download` to populate its official Census cache. The official `ACSIncome.df_to_numpy` transformation supplies ten predictors and the `PINCP > 50000` target.

California supplies fitting and validation; New York, Texas, Florida, Illinois and Pennsylvania supply target audit/reserved pools. Source fitting statistics standardize the official numerical/categorical codes. Each row remains a tabular example; no graph is fabricated.

The prepare command below explicitly permits the documented upstream downloader. Subsequent commands expect the prepared cache.

## Local commands

Run from the repository root after `python -m pip install -e '.[data,dev]'` and installing a suitable PyTorch build. Use a new output directory for each frozen experiment.

```bash
python experiments/acs_income/run.py run --profile smoke --run-dir outputs/acs_income/smoke --device cpu
python experiments/acs_income/run.py prepare --data-root data/acs_income --run-dir outputs/acs_income/prepared --download
python experiments/acs_income/run.py run --profile pilot --data-root data/acs_income --run-dir outputs/acs_income/pilot --device cuda
python experiments/acs_income/run.py run --profile full --data-root data/acs_income --run-dir outputs/acs_income/full --device cuda
```

`prepare`, `train`, `calibrate`, `select`, `audit`, `report`, `predict`, and `inspect` are also separately callable. Use `--help` for the current command arguments. `--config PATH` overrides this folder's default configuration. Relative dataset option paths and model checkpoint paths resolve against `--data-root`.

## Arrhenius commands

```bash
bash scripts/setup_arrhenius.sh
bash experiments/acs_income/run.sh prepare
bash experiments/acs_income/run.sh smoke
bash experiments/acs_income/run.sh pilot
bash experiments/acs_income/run.sh full
bash experiments/acs_income/run.sh status /absolute/path/to/run
bash experiments/acs_income/run.sh resume /absolute/path/to/run
```

The wrapper selects the dataset-specific configuration, cache and output root. It preserves project `naiss2025-5-643-gpu`, storage allocation `naiss2025-6-428`, the provided base path and GH200 modules. Each run freezes source/configuration hashes and submits train → calibrate → select → audit shards → report with dependencies. Environment overrides and stage outputs are documented in [Arrhenius setup](../../docs/ARRHENIUS.md).

## Compute and interpretation

Identity transport treats each survey row independently. Minibatches train the incumbent, and deployment calls use 512 rows. CPU runs are supported; the Slurm wrapper requests one shared GPU for the complete training/admission pipeline.

Exact remaining-tail evaluation trades compute for support across the registered architectures. Full 512/512/2048-call settings can be expensive and are not wall-time guarantees. Hardware timing and paper accuracy have to be measured on actual benchmark runs; no such results are bundled in this directory.

California to five states is the manuscript's setting; the particular target states and 2018 survey year here are declared example choices because the paper omits them. T=8, width 64 and optimizer settings are reconstruction choices. The residual tanh MLP uses identity transport and its actual nonlinear continuation. This benchmark measures predictive accuracy and does not establish a fairness or causal claim.

Shared input contracts: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md). Data-role rules and reporting distinctions: [DATASET_PROTOCOLS.md](../../docs/DATASET_PROTOCOLS.md).
