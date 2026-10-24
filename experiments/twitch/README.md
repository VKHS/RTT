# Twitch-explicit

This directory owns this dataset's complete configuration and Python/Arrhenius entry points. It uses `appnp` with the shared **exact-reference-tail** RTT executor. Internal residual proposals are evaluated through the actual frozen remaining network; calibrated uncertainty, per-row/call budgets and a final label-free endpoint check govern admission. This profile fits a proposal bank and signed-message proposal together with admission heads.

## Registered configuration

| Setting | Value |
|---|---|
| Task | Cross-language node binary classification |
| Primary metric | ROC-AUC |
| Incumbent | `appnp` |
| Continuation steps | 16 |
| Hidden width | 64 |
| First intervention depth (zero based) | 8 |
| Maximum scored items per call | 512 |
| Loss-regression tolerance zeta | 0.0031 nats |
| Main call budget H_plus | 0.0775 nats (=25 × zeta) |
| Per-row budget H_row | 1.0 nats |
| Full calibration / selection / audit calls | 512 per depth per member / 512 per member / 2048 |
| Seed | 1 |

The complete optimizer, proposal, covariance, solver and member-selection settings are in [config.json](config.json). `pilot` reduces work on real supplied data. `smoke` uses generated fixtures and records that it is not a benchmark result. A small calibration budget may yield an infinite radius and no admitted updates; this is an explicit conservative outcome.

## Data and identity

Extract the official Twitch-explicit release under `data/twitch/` with these files:

```text
DE/musae_DE_target.csv
DE/musae_DE_features.json
DE/musae_DE_edges.csv
ENGB/musae_ENGB_target.csv
ENGB/musae_ENGB_features.json
ENGB/musae_ENGB_edges.csv
```

The target CSV uses `new_id` and `mature`; JSON features and edge endpoints retain those node IDs. Feature indices have one shared meaning across languages. The source DE graph supplies fitting and validation; ENGB supplies target audit/reserved pools.

The loader reads your staged files; this profile does not download raw benchmark archives automatically.

## Local commands

Run from the repository root after `python -m pip install -e '.[data,dev]'` and installing a suitable PyTorch build. Use a new output directory for each frozen experiment.

```bash
python experiments/twitch/run.py run --profile smoke --run-dir outputs/twitch/smoke --device cpu
python experiments/twitch/run.py prepare --data-root data/twitch --run-dir outputs/twitch/prepared
python experiments/twitch/run.py run --profile pilot --data-root data/twitch --run-dir outputs/twitch/pilot --device cuda
python experiments/twitch/run.py run --profile full --data-root data/twitch --run-dir outputs/twitch/full --device cuda
```

`prepare`, `train`, `calibrate`, `select`, `audit`, `report`, `predict`, and `inspect` are also separately callable. Use `--help` for the current command arguments. `--config PATH` overrides this folder's default configuration. Relative dataset option paths and model checkpoint paths resolve against `--data-root`.

## Arrhenius commands

```bash
bash scripts/setup_arrhenius.sh
bash experiments/twitch/run.sh prepare
bash experiments/twitch/run.sh smoke
bash experiments/twitch/run.sh pilot
bash experiments/twitch/run.sh full
bash experiments/twitch/run.sh status /absolute/path/to/run
bash experiments/twitch/run.sh resume /absolute/path/to/run
```

The wrapper selects the dataset-specific configuration, cache and output root. It preserves project `naiss2025-5-643-gpu`, storage allocation `naiss2025-6-428`, the provided base path and GH200 modules. Each run freezes source/configuration hashes and submits train → calibrate → select → audit shards → report with dependencies. Environment overrides and stage outputs are documented in [Arrhenius setup](../../docs/ARRHENIUS.md).

## Compute and interpretation

Both language graphs are loaded as disconnected components with shared feature IDs. The dense multihot input can dominate memory; the hidden state has 64 channels. No cross-language edges are fabricated.

Exact remaining-tail evaluation trades compute for support across the registered architectures. Full 512/512/2048-call settings can be expensive and are not wall-time guarantees. Hardware timing and paper accuracy have to be measured on actual benchmark runs; no such results are bundled in this directory.

DE to ENGB is a declared example domain assignment because the RTT manuscript does not identify its exact language graphs. It is not a recovered author split. APPNP is the named model family; T=16, width 64, normalization, source split seed and optimization are explicit choices. This profile uses the exact linear continuation through the generic admission path.

Shared input contracts: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md). Data-role rules and reporting distinctions: [DATASET_PROTOCOLS.md](../../docs/DATASET_PROTOCOLS.md).
