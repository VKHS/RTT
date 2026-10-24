# ogbn-products

This directory owns this dataset's complete configuration and Python/Arrhenius entry points. It uses `tanh_diffusion` with the shared **exact-reference-tail** RTT executor. Internal residual proposals are evaluated through the actual frozen remaining network; calibrated uncertainty, per-row/call budgets and a final label-free endpoint check govern admission. This profile fits a proposal bank and signed-message proposal together with admission heads.

## Registered configuration

| Setting | Value |
|---|---|
| Task | Node multiclass classification |
| Primary metric | accuracy |
| Incumbent | `tanh_diffusion` |
| Continuation steps | 16 |
| Hidden width | 64 |
| First intervention depth (zero based) | 8 |
| Maximum scored items per call | 4096 |
| Loss-regression tolerance zeta | 0.0016 nats |
| Main call budget H_plus | 0.04 nats (=25 × zeta) |
| Per-row budget H_row | 1.0 nats |
| Full calibration / selection / audit calls | 512 per depth per member / 512 per member / 2048 |
| Seed | 1 |

The complete optimizer, proposal, covariance, solver and member-selection settings are in [config.json](config.json). `pilot` reduces work on real supplied data. `smoke` uses generated fixtures and records that it is not a benchmark result. A small calibration budget may yield an infinite radius and no admitted updates; this is an explicit conservative outcome.

## Data and identity

The OGB adapter uses `data/ogbn_products/ogbn_products/`, with the outer directory supplied as `--data-root`. Explicit `--download` permits OGB to populate this cache. The official product features, co-purchase edges, labels and sales-rank train/validation/test allocation are retained.

Training feature statistics are fitted only to the official training nodes. Validation is subdivided into temperature/calibration/selection roles; the official test allocation is divided into disjoint audit/reserved roles. This is a declared non-temporal protocol: products has no publication-year split.

The prepare command below explicitly permits the documented upstream downloader. Subsequent commands expect the prepared cache.

## Local commands

Run from the repository root after `python -m pip install -e '.[data,dev]'` and installing a suitable PyTorch build. Use a new output directory for each frozen experiment.

```bash
python experiments/ogbn_products/run.py run --profile smoke --run-dir outputs/ogbn_products/smoke --device cpu
python experiments/ogbn_products/run.py prepare --data-root data/ogbn_products --run-dir outputs/ogbn_products/prepared --download
python experiments/ogbn_products/run.py run --profile pilot --data-root data/ogbn_products --run-dir outputs/ogbn_products/pilot --device cuda
python experiments/ogbn_products/run.py run --profile full --data-root data/ogbn_products --run-dir outputs/ogbn_products/full --device cuda
```

`prepare`, `train`, `calibrate`, `select`, `audit`, `report`, `predict`, and `inspect` are also separately callable. Use `--help` for the current command arguments. `--config PATH` overrides this folder's default configuration. Relative dataset option paths and model checkpoint paths resolve against `--data-root`.

## Arrhenius commands

```bash
bash scripts/setup_arrhenius.sh
bash experiments/ogbn_products/run.sh prepare
bash experiments/ogbn_products/run.sh smoke
bash experiments/ogbn_products/run.sh pilot
bash experiments/ogbn_products/run.sh full
bash experiments/ogbn_products/run.sh status /absolute/path/to/run
bash experiments/ogbn_products/run.sh resume /absolute/path/to/run
```

The wrapper selects the dataset-specific configuration, cache and output root. It preserves project `naiss2025-5-643-gpu`, storage allocation `naiss2025-6-428`, the provided base path and GH200 modules. Each run freezes source/configuration hashes and submits train → calibrate → select → audit shards → report with dependencies. Environment overrides and stage outputs are documented in [Arrhenius setup](../../docs/ARRHENIUS.md).

## Compute and interpretation

This is a full-graph implementation of the multi-million-node co-purchase graph. It does not substitute an induced subgraph. Host RAM, edge storage, full hidden states and repeated autograd tails may exceed a single GPU allocation. The launcher starts with 180G host RAM for this profile; measure a pilot and override RTT_MEM_TRAIN/RTT_MEM_CALIBRATE/RTT_MEM_AUDIT and time limits as needed. The 64-channel setting reduces state storage but is not a hardware fit guarantee.

Exact remaining-tail evaluation trades compute for support across the registered architectures. Full 512/512/2048-call settings can be expensive and are not wall-time guarantees. Hardware timing and paper accuracy have to be measured on actual benchmark runs; no such results are bundled in this directory.

The paper specifies a tanh-diffusion incumbent but does not provide the products width, horizon or all deployment details. T=16, width 64, query size 4096 and within-split allocation are practical reconstruction choices. This generic profile uses the fixed loaded graph and exact remaining tails. It does not reproduce the paper's unspecified products corruption mixture or claim its fast tube timing.

Shared input contracts: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md). Data-role rules and reporting distinctions: [DATASET_PROTOCOLS.md](../../docs/DATASET_PROTOCOLS.md).
