# Elliptic

This directory owns this dataset's complete configuration and Python/Arrhenius entry points. It uses `grand` with the shared **exact-reference-tail** RTT executor. Internal residual proposals are evaluated through the actual frozen remaining network; calibrated uncertainty, per-row/call budgets and a final label-free endpoint check govern admission. This profile fits a proposal bank and signed-message proposal together with admission heads.

## Registered configuration

| Setting | Value |
|---|---|
| Task | Temporal transaction node binary classification |
| Primary metric | positive illicit-class F1 |
| Incumbent | `grand` |
| Continuation steps | 16 |
| Hidden width | 64 |
| First intervention depth (zero based) | 8 |
| Maximum scored items per call | 512 |
| Loss-regression tolerance zeta | 0.0028 nats |
| Main call budget H_plus | 0.07 nats (=25 × zeta) |
| Per-row budget H_row | 1.0 nats |
| Full calibration / selection / audit calls | 512 per depth per member / 512 per member / 2048 |
| Seed | 1 |

The complete optimizer, proposal, covariance, solver and member-selection settings are in [config.json](config.json). `pilot` reduces work on real supplied data. `smoke` uses generated fixtures and records that it is not a benchmark result. A small calibration budget may yield an infinite radius and no admitted updates; this is an explicit conservative outcome.

## Data and identity

Place the original `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, and `elliptic_txs_edgelist.csv` directly under `data/elliptic/`. The feature CSV has no header and starts with transaction ID and time step. Class code `1` becomes positive illicit, `2` becomes negative licit, and `unknown` remains unscored. Edges use `txId1,txId2` and are remapped by actual transaction ID.

The configured chronology uses steps <=34 for fitting, 35–42 for validation-derived roles, and >=43 for target audit/reserved pools. Time is kept as metadata and excluded from predictors. Scaling uses the fitting allocation.

The loader reads your staged files; this profile does not download raw benchmark archives automatically.

## Local commands

Run from the repository root after `python -m pip install -e '.[data,dev]'` and installing a suitable PyTorch build. Use a new output directory for each frozen experiment.

```bash
python experiments/elliptic/run.py run --profile smoke --run-dir outputs/elliptic/smoke --device cpu
python experiments/elliptic/run.py prepare --data-root data/elliptic --run-dir outputs/elliptic/prepared
python experiments/elliptic/run.py run --profile pilot --data-root data/elliptic --run-dir outputs/elliptic/pilot --device cuda
python experiments/elliptic/run.py run --profile full --data-root data/elliptic --run-dir outputs/elliptic/full --device cuda
```

`prepare`, `train`, `calibrate`, `select`, `audit`, `report`, `predict`, and `inspect` are also separately callable. Use `--help` for the current command arguments. `--config PATH` overrides this folder's default configuration. Relative dataset option paths and model checkpoint paths resolve against `--data-root`.

## Arrhenius commands

```bash
bash scripts/setup_arrhenius.sh
bash experiments/elliptic/run.sh prepare
bash experiments/elliptic/run.sh smoke
bash experiments/elliptic/run.sh pilot
bash experiments/elliptic/run.sh full
bash experiments/elliptic/run.sh status /absolute/path/to/run
bash experiments/elliptic/run.sh resume /absolute/path/to/run
```

The wrapper selects the dataset-specific configuration, cache and output root. It preserves project `naiss2025-5-643-gpu`, storage allocation `naiss2025-6-428`, the provided base path and GH200 modules. Each run freezes source/configuration hashes and submits train → calibrate → select → audit shards → report with dependencies. Environment overrides and stage outputs are documented in [Arrhenius setup](../../docs/ARRHENIUS.md).

## Compute and interpretation

All transactions, including unknown-label graph context, remain available to the model. The implementation validates that original edges stay within a time step. It uses width 64 and a fixed 16-step continuation.

Exact remaining-tail evaluation trades compute for support across the registered architectures. Full 512/512/2048-call settings can be expensive and are not wall-time guarantees. Hardware timing and paper accuracy have to be measured on actual benchmark runs; no such results are bundled in this directory.

The chronology boundaries 34 and 42 are explicit reconstruction choices; the paper does not uniquely identify the original cutoff. GRAND's forward-Euler discretization, T=16 and width 64 are also declared choices. The metric counts illicit transactions as the positive class. Unknown transactions are never converted to a third label or a negative example.

Shared input contracts: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md). Data-role rules and reporting distinctions: [DATASET_PROTOCOLS.md](../../docs/DATASET_PROTOCOLS.md).
