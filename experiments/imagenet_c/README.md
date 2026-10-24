# ImageNet-C

This directory owns this dataset's complete configuration and Python/Arrhenius entry points. It uses `resnet50` with the shared **exact-reference-tail** RTT executor. Internal residual proposals are evaluated through the actual frozen remaining network; calibrated uncertainty, per-row/call budgets and a final label-free endpoint check govern admission. This image profile uses Tent proposals reset for independent calls, together with fitted admission heads.

## Registered configuration

| Setting | Value |
|---|---|
| Task | Image multiclass classification |
| Primary metric | top-1 accuracy |
| Incumbent | `resnet50` |
| Continuation steps | 4 |
| Hidden width | 2048 output channels; spatial blocks retained |
| First intervention depth (zero based) | 0 |
| Maximum scored items per call | 32 |
| Loss-regression tolerance zeta | 0.01 nats |
| Main call budget H_plus | 0.25 nats (=25 × zeta) |
| Per-row budget H_row | 1.0 nats |
| Full calibration / selection / audit calls | 512 per depth per member / 512 per member / 2048 |
| Seed | 1 |

The complete optimizer, proposal, covariance, solver and member-selection settings are in [config.json](config.json). `pilot` reduces work on real supplied data. `smoke` uses generated fixtures and records that it is not a benchmark result. A small calibration budget may yield an infinite radius and no admitted updates; this is an explicit conservative outcome.

## Data and identity

Extract the released severity-3 corrupted JPEGs under `data/imagenet_c/<corruption>/3/<wnid>/<image>.JPEG`. Supply separate clean fitting images under `data/imagenet/train/<wnid>/`; `fit_root="../imagenet/train"` is resolved relative to this profile's data root. The 15 registered corruptions are explicit in the config. ImageNet data access and extraction are separate from Python dependency installation.

A full 1000-class directory vocabulary is required, or provide `dataset.options.class_map` matching the checkpoint. Alternatively use an explicit `manifest` CSV containing `path,label,split,image_id,domain`. Copies of one underlying image stay in the same role; overlapping clean/corrupted identities are excluded or rejected as documented in [input contracts](../../docs/DATA_SOURCES.md#imagenet-c).

The configured ResNet weights are official `IMAGENET1K_V2`. Cache them in Torch's hub cache with an explicitly authorized `--download` run, or set `model.checkpoint` to a matching local state dictionary. Relative checkpoint paths resolve against `--data-root`. No checkpoint is bundled. The pure-Torch model is compatible with the official TorchVision weights and does not require TorchVision at inference.

The loader reads your staged files; `--download` does not obtain this dataset. For this image profile, explicitly adding `--download` to the training/run command permits retrieval of the configured pretrained weights.

## Local commands

Run from the repository root after `python -m pip install -e '.[data,dev]'` and installing a suitable PyTorch build. Use a new output directory for each frozen experiment.

```bash
python experiments/imagenet_c/run.py run --profile smoke --run-dir outputs/imagenet_c/smoke --device cpu
python experiments/imagenet_c/run.py prepare --data-root data/imagenet_c --run-dir outputs/imagenet_c/prepared
python experiments/imagenet_c/run.py run --profile pilot --data-root data/imagenet_c --run-dir outputs/imagenet_c/pilot --device cuda --download
python experiments/imagenet_c/run.py run --profile full --data-root data/imagenet_c --run-dir outputs/imagenet_c/full --device cuda
```

`prepare`, `train`, `calibrate`, `select`, `audit`, `report`, `predict`, and `inspect` are also separately callable. Use `--help` for the current command arguments. `--config PATH` overrides this folder's default configuration. Relative dataset option paths and model checkpoint paths resolve against `--data-root`.

## Arrhenius commands

```bash
bash scripts/setup_arrhenius.sh
bash experiments/imagenet_c/run.sh prepare
bash experiments/imagenet_c/run.sh smoke
RTT_DOWNLOAD=1 bash experiments/imagenet_c/run.sh pilot
bash experiments/imagenet_c/run.sh full
bash experiments/imagenet_c/run.sh status /absolute/path/to/run
bash experiments/imagenet_c/run.sh resume /absolute/path/to/run
```

The first pilot explicitly permits downloading the configured official pretrained weights; raw ImageNet images must already be staged. Later runs can use the cached weights without this flag. The wrapper selects the dataset-specific configuration, cache and output root. It preserves project `naiss2025-5-643-gpu`, storage allocation `naiss2025-6-428`, the provided base path and GH200 modules. Each run freezes source/configuration hashes and submits train → calibrate → select → audit shards → report with dependencies. Environment overrides and stage outputs are documented in [Arrhenius setup](../../docs/ARRHENIUS.md).

## Compute and interpretation

The full ResNet-50 convolutional network is used; its last four bottleneck blocks are the intervention continuation. Image batches have 32 examples and actual spatial activations. Repeated remaining-tail derivatives can be costly; pilot runtime and GPU memory before the 2048-call audit.

Exact remaining-tail evaluation trades compute for support across the registered architectures. Full 512/512/2048-call settings can be expensive and are not wall-time guarantees. Hardware timing and paper accuracy have to be measured on actual benchmark runs; no such results are bundled in this directory.

Severity 3, the 15 corruptions, ResNet-50's last four bottleneck blocks and Tent proposals are specified in the paper. The weight revision, 32-image call, disjoint clean-fitting protocol and one-step Tent learning rate 0.001 are explicit choices. The incumbent is frozen; only proposal/admission components and temperature are fitted. Tent is reset for independent calls. P=I at the image-example level preserves the convolutional continuation.

Shared input contracts: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md). Data-role rules and reporting distinctions: [DATASET_PROTOCOLS.md](../../docs/DATASET_PROTOCOLS.md).
