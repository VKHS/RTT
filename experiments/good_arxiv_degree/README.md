# GOOD-Arxiv degree / covariate

This directory owns this dataset's complete configuration and Python/Arrhenius entry points. It uses `gcnii` with the shared **exact-reference-tail** RTT executor. Internal residual proposals are evaluated through the actual frozen remaining network; calibrated uncertainty, per-row/call budgets and a final label-free endpoint check govern admission. This profile fits a proposal bank and signed-message proposal together with admission heads.

## Registered configuration

| Setting | Value |
|---|---|
| Task | Node multiclass classification |
| Primary metric | accuracy |
| Incumbent | `gcnii` |
| Continuation steps | 16 |
| Hidden width | 128 |
| First intervention depth (zero based) | 8 |
| Maximum scored items per call | 512 |
| Loss-regression tolerance zeta | 0.0023 nats |
| Main call budget H_plus | 0.0575 nats (=25 × zeta) |
| Per-row budget H_row | 1.0 nats |
| Full calibration / selection / audit calls | 512 per depth per member / 512 per member / 2048 |
| Seed | 1 |

The complete optimizer, proposal, covariance, solver and member-selection settings are in [config.json](config.json). `pilot` reduces work on real supplied data. `smoke` uses generated fixtures and records that it is not a benchmark result. A small calibration budget may yield an infinite radius and no admitted updates; this is an explicit conservative outcome.

## Data and identity

Use the official GOOD cache under `data/good_arxiv_degree/`; the adapter calls `GOODArxiv.load(..., domain="degree", shift="covariate", generate=False)`. Install the official GOOD package and its PyG dependencies for this loading route. It is not part of the minimal RTT runtime. `train_mask`, released OOD `val_mask`, and `test_mask` are preserved; available in-distribution masks are retained separately.

An official export can instead be supplied through `dataset.options.npz`, following [the portable schema](../../docs/DATA_SOURCES.md#portable-npz-format). Export the actual graph and released masks. A trusted `.pt` export needs `data_pt` and explicit `trust_data_pickle=true`; arbitrary tensor paths are not assumed.

The prepare command below explicitly permits the documented upstream downloader. Subsequent commands expect the prepared cache. GOOD additionally needs its official loader dependencies or an exact portable export.

## Local commands

Run from the repository root after `python -m pip install -e '.[data,dev]'` and installing a suitable PyTorch build. Use a new output directory for each frozen experiment.

```bash
python experiments/good_arxiv_degree/run.py run --profile smoke --run-dir outputs/good_arxiv_degree/smoke --device cpu
python experiments/good_arxiv_degree/run.py prepare --data-root data/good_arxiv_degree --run-dir outputs/good_arxiv_degree/prepared --download
python experiments/good_arxiv_degree/run.py run --profile pilot --data-root data/good_arxiv_degree --run-dir outputs/good_arxiv_degree/pilot --device cuda
python experiments/good_arxiv_degree/run.py run --profile full --data-root data/good_arxiv_degree --run-dir outputs/good_arxiv_degree/full --device cuda
```

`prepare`, `train`, `calibrate`, `select`, `audit`, `report`, `predict`, and `inspect` are also separately callable. Use `--help` for the current command arguments. `--config PATH` overrides this folder's default configuration. Relative dataset option paths and model checkpoint paths resolve against `--data-root`.

## Arrhenius commands

```bash
bash scripts/setup_arrhenius.sh
bash experiments/good_arxiv_degree/run.sh prepare
bash experiments/good_arxiv_degree/run.sh smoke
bash experiments/good_arxiv_degree/run.sh pilot
bash experiments/good_arxiv_degree/run.sh full
bash experiments/good_arxiv_degree/run.sh status /absolute/path/to/run
bash experiments/good_arxiv_degree/run.sh resume /absolute/path/to/run
```

The wrapper selects the dataset-specific configuration, cache and output root. It preserves project `naiss2025-5-643-gpu`, storage allocation `naiss2025-6-428`, the provided base path and GH200 modules. Each run freezes source/configuration hashes and submits train → calibrate → select → audit shards → report with dependencies. Environment overrides and stage outputs are documented in [Arrhenius setup](../../docs/ARRHENIUS.md).

## Compute and interpretation

This profile operates on the released full GOOD citation graph. GCNII stores layer activations for remaining-tail derivatives, so a smaller query does not make graph propagation local.

Exact remaining-tail evaluation trades compute for support across the registered architectures. Full 512/512/2048-call settings can be expensive and are not wall-time guarantees. Hardware timing and paper accuracy have to be measured on actual benchmark runs; no such results are bundled in this directory.

The GCNII family and degree/covariate shift are specified by the manuscript. T=16, width 128, alpha, lambda, source-only scaling, optimization and within-mask subdivisions are explicit choices. The generic backend evaluates the actual ReLU continuation. It does not claim the specialized interval-Clarke-Jacobian runtime of the paper's breadth experiment.

Shared input contracts: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md). Data-role rules and reporting distinctions: [DATASET_PROTOCOLS.md](../../docs/DATASET_PROTOCOLS.md).
