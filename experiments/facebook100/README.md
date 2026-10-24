# Facebook100

This directory owns this dataset's complete configuration and Python/Arrhenius entry points. It uses `gcnii` with the shared **exact-reference-tail** RTT executor. Internal residual proposals are evaluated through the actual frozen remaining network; calibrated uncertainty, per-row/call budgets and a final label-free endpoint check govern admission. This profile fits a proposal bank and signed-message proposal together with admission heads.

## Registered configuration

| Setting | Value |
|---|---|
| Task | Cross-university node binary classification |
| Primary metric | accuracy |
| Incumbent | `gcnii` |
| Continuation steps | 16 |
| Hidden width | 128 |
| First intervention depth (zero based) | 8 |
| Maximum scored items per call | 512 |
| Loss-regression tolerance zeta | 0.0034 nats |
| Main call budget H_plus | 0.085 nats (=25 × zeta) |
| Per-row budget H_row | 1.0 nats |
| Full calibration / selection / audit calls | 512 per depth per member / 512 per member / 2048 |
| Seed | 1 |

The complete optimizer, proposal, covariance, solver and member-selection settings are in [config.json](config.json). `pilot` reduces work on real supplied data. `smoke` uses generated fixtures and records that it is not a benchmark result. A small calibration budget may yield an infinite radius and no admitted updates; this is an explicit conservative outcome.

## Data and identity

Place `Penn94.mat` and `Caltech36.mat` directly under `data/facebook100/`. Each file must contain sparse adjacency `A` and node metadata `local_info`. The default target is metadata column 1 (gender): original values 1/2 become classes 0/1 and 0 remains unknown. The target column is excluded from the input features.

Source Penn94 labeled nodes supply fitting and validation; target Caltech36 labeled nodes supply audit/reserved pools. Unknown labels remain available as graph context and are never scored. Other categorical columns use a source-training vocabulary.

The loader reads your staged files; this profile does not download raw benchmark archives automatically.

## Local commands

Run from the repository root after `python -m pip install -e '.[data,dev]'` and installing a suitable PyTorch build. Use a new output directory for each frozen experiment.

```bash
python experiments/facebook100/run.py run --profile smoke --run-dir outputs/facebook100/smoke --device cpu
python experiments/facebook100/run.py prepare --data-root data/facebook100 --run-dir outputs/facebook100/prepared
python experiments/facebook100/run.py run --profile pilot --data-root data/facebook100 --run-dir outputs/facebook100/pilot --device cuda
python experiments/facebook100/run.py run --profile full --data-root data/facebook100 --run-dir outputs/facebook100/full --device cuda
```

`prepare`, `train`, `calibrate`, `select`, `audit`, `report`, `predict`, and `inspect` are also separately callable. Use `--help` for the current command arguments. `--config PATH` overrides this folder's default configuration. Relative dataset option paths and model checkpoint paths resolve against `--data-root`.

## Arrhenius commands

```bash
bash scripts/setup_arrhenius.sh
bash experiments/facebook100/run.sh prepare
bash experiments/facebook100/run.sh smoke
bash experiments/facebook100/run.sh pilot
bash experiments/facebook100/run.sh full
bash experiments/facebook100/run.sh status /absolute/path/to/run
bash experiments/facebook100/run.sh resume /absolute/path/to/run
```

The wrapper selects the dataset-specific configuration, cache and output root. It preserves project `naiss2025-5-643-gpu`, storage allocation `naiss2025-6-428`, the provided base path and GH200 modules. Each run freezes source/configuration hashes and submits train → calibrate → select → audit shards → report with dependencies. Environment overrides and stage outputs are documented in [Arrhenius setup](../../docs/ARRHENIUS.md).

## Compute and interpretation

Penn94 and Caltech36 remain separate graph components in one context. Categorical inputs are fitted to the source training vocabulary, with explicit unseen-category entries. Graph size and full-tail activations determine memory.

Exact remaining-tail evaluation trades compute for support across the registered architectures. Full 512/512/2048-call settings can be expensive and are not wall-time guarantees. Hardware timing and paper accuracy have to be measured on actual benchmark runs; no such results are bundled in this directory.

Penn94 to Caltech36 is a declared example assignment, not an author-recovered university pair. The manuscript names GCNII but omits these domains and detailed architecture. T=16, width 128, alpha, lambda and source split settings are explicit reconstruction choices. Changing the universities defines a new frozen experiment.

Shared input contracts: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md). Data-role rules and reporting distinctions: [DATASET_PROTOCOLS.md](../../docs/DATASET_PROTOCOLS.md).
