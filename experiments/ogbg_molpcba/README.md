# ogbg-molpcba

This directory owns this dataset's complete configuration and Python/Arrhenius entry points. It uses `gin` with the shared **exact-reference-tail** RTT executor. Internal residual proposals are evaluated through the actual frozen remaining network; calibrated uncertainty, per-row/call budgets and a final label-free endpoint check govern admission. This profile fits a proposal bank and signed-message proposal together with admission heads.

## Registered configuration

| Setting | Value |
|---|---|
| Task | Graph-level 128-task binary prediction |
| Primary metric | mean average precision over eligible assay tasks |
| Incumbent | `gin` |
| Continuation steps | 5 |
| Hidden width | 128 |
| First intervention depth (zero based) | 2 |
| Maximum scored items per call | 64 |
| Loss-regression tolerance zeta | 0.0016 nats |
| Main call budget H_plus | 0.04 nats (=25 × zeta) |
| Per-row budget H_row | 1.0 nats |
| Full calibration / selection / audit calls | 512 per depth per member / 512 per member / 2048 |
| Seed | 1 |

The complete optimizer, proposal, covariance, solver and member-selection settings are in [config.json](config.json). `pilot` reduces work on real supplied data. `smoke` uses generated fixtures and records that it is not a benchmark result. A small calibration budget may yield an infinite radius and no admitted updates; this is an explicit conservative outcome.

## Data and identity

The OGB adapter expects `data/ogbg_molpcba/ogbg_molpcba/`, with the outer directory supplied as `--data-root`; explicit `--download` populates it. Each molecule retains nine categorical atom features, three categorical bond features and its original graph edges. All 128 assay columns remain present, and NaN means a missing assay.

The default starts from OGB's released scaffold split, then uses declared disjoint subdivisions within its train/valid/test allocations. Subdivision of the test set by molecule is a custom audit/reserved holdout, not the manuscript's reserved-scaffold halves and not a full official-test score. Use the `scaffold_groups` variant with actual scaffold identities if every subdivision must keep a chemical scaffold together.

The prepare command below explicitly permits the documented upstream downloader. Subsequent commands expect the prepared cache.

## Local commands

Run from the repository root after `python -m pip install -e '.[data,dev]'` and installing a suitable PyTorch build. Use a new output directory for each frozen experiment.

```bash
python experiments/ogbg_molpcba/run.py run --profile smoke --run-dir outputs/ogbg_molpcba/smoke --device cpu
python experiments/ogbg_molpcba/run.py prepare --data-root data/ogbg_molpcba --run-dir outputs/ogbg_molpcba/prepared --download
python experiments/ogbg_molpcba/run.py run --profile pilot --data-root data/ogbg_molpcba --run-dir outputs/ogbg_molpcba/pilot --device cuda
python experiments/ogbg_molpcba/run.py run --profile full --data-root data/ogbg_molpcba --run-dir outputs/ogbg_molpcba/full --device cuda
```

`prepare`, `train`, `calibrate`, `select`, `audit`, `report`, `predict`, and `inspect` are also separately callable. Use `--help` for the current command arguments. `--config PATH` overrides this folder's default configuration. Relative dataset option paths and model checkpoint paths resolve against `--data-root`.

## Arrhenius commands

```bash
bash scripts/setup_arrhenius.sh
bash experiments/ogbg_molpcba/run.sh prepare
bash experiments/ogbg_molpcba/run.sh smoke
bash experiments/ogbg_molpcba/run.sh pilot
bash experiments/ogbg_molpcba/run.sh full
bash experiments/ogbg_molpcba/run.sh status /absolute/path/to/run
bash experiments/ogbg_molpcba/run.sh resume /absolute/path/to/run
```

The wrapper selects the dataset-specific configuration, cache and output root. It preserves project `naiss2025-5-643-gpu`, storage allocation `naiss2025-6-428`, the provided base path and GH200 modules. Each run freezes source/configuration hashes and submits train → calibrate → select → audit shards → report with dependencies. Environment overrides and stage outputs are documented in [Arrhenius setup](../../docs/ARRHENIUS.md).

## Compute and interpretation

Minibatches combine whole molecules as disconnected graphs, preserving categorical atoms and bonds. The data adapter accesses OGB graphs lazily. A call scores 64 complete molecules; no assay is expanded into a fictitious independent graph.

Exact remaining-tail evaluation trades compute for support across the registered architectures. Full 512/512/2048-call settings can be expensive and are not wall-time guarantees. Hardware timing and paper accuracy have to be measured on actual benchmark runs; no such results are bundled in this directory.

The paper names four molecular families but omits their full architecture and the molecular loss-regression tolerance. This main bond-aware GIN uses five layers, width 128 and mean pooling. zeta=0.0016 nats is an explicit chosen value, not a recovered Table-16 value. Missing assays are masked in fitting and reporting. Pooled episodic AP and custom reserved-holdout results must not be labeled full official OGB test performance.

Shared input contracts: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md). Data-role rules and reporting distinctions: [DATASET_PROTOCOLS.md](../../docs/DATASET_PROTOCOLS.md).

## Separate backbone variants

Every file below is a complete configuration, so it can be reviewed and frozen independently.

| File in `variants/` | Family | Steps | Width | First open depth |
|---|---|---:|---:|---:|
| `contractive_diffusion.json` | `contractive_diffusion` | 5 | 128 | 2 |
| `centered_gin.json` | `centered_gin` | 5 | 128 | 2 |
| `gin_virtual.json` | `gin_virtual` | 5 | 128 | 2 |

```bash
python experiments/ogbg_molpcba/run.py run --config experiments/ogbg_molpcba/variants/contractive_diffusion.json --profile smoke --run-dir outputs/ogbg_molpcba/contractive_diffusion_smoke --device cpu
RTT_CONFIG="$PWD/experiments/ogbg_molpcba/variants/contractive_diffusion.json" bash experiments/ogbg_molpcba/run.sh pilot
```

Variant results belong to a fresh run directory. The full remaining network is retained for the shallow GCN/GraphSAGE families; molecular pooling and virtual-node states remain part of their actual model continuation.

### Chemical scaffold grouping

`variants/scaffold_groups.json` requires `<data-root>/scaffold_ids.csv` with `graph_id,scaffold_id`, covering every OGB molecule exactly once in original index order. JSON lists or index-to-ID objects are also accepted after changing the path. Supply real chemical scaffold identities derived from the matching molecules. Empty acyclic scaffolds need an explicit shared ID. Random groups or split labels are not chemical scaffolds.

This policy keeps identities together in fitting, temperature, calibration, selection, audit and reserved subdivisions, and rejects overlap already present in input splits. It still describes a new explicit protocol; the original paper's unreleased reserved-scaffold IDs cannot be reconstructed from the manuscript alone.

## Separate full-population evaluation

After fitting, calibrating and selecting the member, the `evaluate` command can score the complete released test allocation in batches:

```bash
python experiments/ogbg_molpcba/run.py evaluate --profile full --run-dir outputs/ogbg_molpcba/full --data-root data/ogbg_molpcba --split test --batch-size 64 --output outputs/ogbg_molpcba/full/population_evaluation/test
```

This reports population-level masked AP separately from the episodic audit. Because the default protocol already draws audit/reserved roles from the official test allocation, the complete-test evaluation overlaps that custom audit; it is not an additional untouched validation population. The run records this distinction. Keep the deployment batch size explicit because predictions may depend on the call composition.
