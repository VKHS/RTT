# Dataset sources and input contracts

The repository has ten separately addressable profiles covering the nine dataset
families in the manuscript. GOOD-Arxiv degree/covariate and time/concept use
different released splits and therefore have separate profiles. The graph,
molecular, tabular and image loaders preserve their native task types. Products
does not acquire invented publication years, and missing molecular assays do not
become negative labels.

All adapters return `rtt.datasets.DatasetBundle`. Its `splits` map contains
disjoint `train`, `valid`, and `test` indices. For node tasks, indices refer to
nodes; for molecules, images and tabular records, they refer to whole examples.
The training pipeline may subdivide these allocations into fitting, calibration,
selection and audit pools without changing their original provenance.

Use the selected experiment's `dataset.root` and `dataset.options` fields to specify
locations. The CLI flag `--download` explicitly enables the upstream OGB, GOOD or
Folktables download API; direct Python calls use `load_dataset(..., download=True)`.
Relative file paths inside `dataset.options`, including `npz`, `manifest`,
`fit_root`, `class_map`, `data_pt` and `scaffold_path`, resolve against
`dataset.root` when the CLI resolves the configuration. The other adapters read
the official files already present on disk;
they do not obtain data through unofficial mirrors or silently substitute another
benchmark. Data files are not included in the Git repository.

## OGB node tasks

| Profile | Source | Preserved information |
|---|---|---|
| `ogbn_arxiv` | [Official OGB node-property documentation](https://ogb.stanford.edu/docs/nodeprop/) | Node features, citation edges, subject labels, publication years, official chronological split |
| `ogbn_products` | [Official OGB node-property documentation](https://ogb.stanford.edu/docs/nodeprop/) | Product features, co-purchase edges, categories and official sales-rank split |

Both use OGB's library-agnostic `NodePropPredDataset`; PyTorch Geometric is not
required. `dataset.root` is the parent of OGB's `ogbn_arxiv` or `ogbn_products`
directory. With downloads enabled, OGB retrieves and processes its own archive.
Feature mean and standard deviation are fitted only on the official training
indices. `options.standardize=false` disables that preprocessing explicitly.

The adapter preserves the raw directed endpoints. The selected graph model owns
its explicitly recorded symmetrization, self-loop and normalization choices.
Full-graph products is substantially larger than arxiv; its memory and runtime
must be checked on the selected server before a full experiment.

## GOOD-Arxiv

| Profile | Domain | Shift |
|---|---|---|
| `good_arxiv_degree` | `degree` | `covariate` |
| `good_arxiv_time` | `time` | `concept` |

Use the [official GOOD project](https://github.com/divelab/GOOD) and its
[dataset API](https://diveintographs.readthedocs.io/en/latest/oodgraph/good.html).
The released `train_mask`, `val_mask`, and `test_mask` define the primary splits.
The adapter also retains `id_val_mask` and `id_test_mask` when present. It never
approximates GOOD's masks by changing OGB years or degree thresholds.

There are three input routes:

1. Install the official GOOD package and its dependencies; enable downloads or
   point `dataset.root` at its existing cache. The loader calls
   `GOODArxiv.load(dataset_root=..., domain=..., shift=..., generate=False)`.
2. Supply a trusted official `data_pt` export and explicitly set
   `trust_data_pickle=true`. Python-object checkpoints require PyG-compatible
   classes in the environment. Unknown files are not unpickled by default.
3. Export the same arrays and original masks to the portable NPZ schema below.
   This avoids a runtime GOOD/PyG dependency without regenerating the data split.

`options.valid_mask="id_val_mask"` explicitly chooses in-distribution validation
instead of the default released OOD `val_mask`. Such a choice changes the
protocol and is recorded.

## Twitch-explicit

Source: [Twitch social networks, SNAP](https://snap.stanford.edu/data/twitch-social-networks.html)
and the [dataset authors' repository](https://github.com/benedekrozemberczki/datasets).
Use the language-specific Twitch-explicit data, not the distinct single-network
Twitch Gamers benchmark.

Extract each selected language under `dataset.root/<language>/`, preserving:

```text
DE/musae_DE_target.csv
DE/musae_DE_features.json
DE/musae_DE_edges.csv
ENGB/musae_ENGB_target.csv
ENGB/musae_ENGB_features.json
ENGB/musae_ENGB_edges.csv
```

Set `source_domain` and `target_domains`, for example `"DE"` and `["ENGB"]`.
This is an example configuration, not a recovered paper domain assignment. The
paper does not identify every source/target graph. The target CSV requires
`new_id` and `mature`; JSON keys and edge endpoints use those node IDs.

Original feature IDs retain their shared meaning across languages. Graphs are
combined as disconnected components with offset node IDs. Source labeled nodes
are split into fitting/validation using the registered seed; target labeled nodes
form the test allocation. The primary task metric is ROC-AUC.

## Facebook100

Read the original MATLAB network files, one per selected university, with keys
`A` and `local_info`. The conversion follows the gender-classification task in
the [published large-scale graph benchmark implementation](https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/dataset.py).
Place files such as `Penn94.mat` and `Reed98.mat` directly under `dataset.root`.

`source_domain` and `target_domains` are explicit experiment choices. The
manuscript does not identify all university assignments. Column 1 of
`local_info` is the default gender label: original zero remains unknown (`-1`),
and original codes 1/2 become classes 0/1. The gender column is removed before
constructing features. Other categorical columns use a shared source-training
vocabulary and an explicit unseen-category entry.

Unknown-label nodes remain available as graph context but never appear in scored
splits. Source fitting and validation are separated; target universities form the
test allocation. The primary metric is accuracy.

## Elliptic

Obtain the original [Elliptic transaction dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
Its format is also documented by the [IBM EvolveGCN implementation](https://github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md).
Place these files directly under `dataset.root`:

```text
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

The features CSV has no header: transaction ID, time step, then numerical
features. The label CSV uses `txId,class`; `1` is mapped to positive illicit,
`2` to negative licit, and `unknown` remains unscored. The edge CSV requires
`txId1,txId2`. Transaction IDs are mapped consistently across all three files.

Time steps are retained as `bundle.time`, not mislabeled publication years.
The time column is excluded from the default predictors. Original Elliptic has
no edges between time steps; a modified graph crossing times is rejected because
it needs a separate causal protocol. Explicit `train_end` and `valid_end` define
the chronological allocations. Defaults 34 and 42 are registered reconstruction
choices, leaving steps after 42 for the post-shutdown audit. The primary metric
is positive-class F1; labels are never changed to improve class balance.

## Folktables ACSIncome

Source: [official Folktables](https://github.com/socialfoundations/folktables).
The loader calls `ACSDataSource(..., root_dir=dataset.root)` and the official
`ACSIncome.df_to_numpy` transformation, preserving its ten predictor columns and
income-above-$50,000 target. `survey_year` defaults to `2018` and the horizon to
`1-Year`; both are explicit configuration choices.

Set `source_state="CA"` and an explicit `target_states` list. The paper says
California to five states but does not name those five, so the adapter refuses
to invent them. Source and target states must be distinct. Source rows define
fitting and validation; selected target states define the test allocation.
Numeric/categorical codes are standardized with source-training statistics.
The primary task metric is accuracy. This is a prediction benchmark, not a
fairness claim or a population causal estimate.

## ImageNet-C

Source: [official ImageNet-C release](https://github.com/hendrycks/robustness).
Use the released JPEG corruptions so the evaluated images match the benchmark.
By default, the profile uses severity 3 and the registered 15 corruptions.
The loader keeps paths and decodes images only when they are batched.

The directory input expects:

```text
<dataset.root>/gaussian_noise/3/<wnid>/<image>.JPEG
<dataset.root>/shot_noise/3/<wnid>/<image>.JPEG
...the other registered corruptions...
<fit_root>/<wnid>/<clean-image>.JPEG
```

Provide a separate clean `fit_root` with fitting data. A complete class
vocabulary is required; otherwise provide `class_map`, a JSON mapping from
ImageNet synset IDs to the deployed checkpoint's class indices. A subset of
class folders must not silently reindex a 1,000-way classifier. Corrupted copies
whose underlying class/filename occurs in clean fitting data are excluded from
the audit and counted in metadata. If every target image overlaps, loading
fails instead of evaluating fitted images.

For exact author-defined allocations, use `options.manifest` with CSV columns:

```csv
path,label,split,image_id,domain
images/train_1.JPEG,17,train,source_image_1,clean
images/valid_1.JPEG,42,valid,source_image_2,clean
images/snow_1.JPEG,17,test,audit_image_1,snow
images/fog_1.JPEG,17,test,audit_image_1,fog
```

Paths may be relative to the CSV. `image_id` groups all corruptions of the same
underlying image. A group cannot cross fitting/validation/test allocations or
change its class. If `image_id` is absent, the filename is used. This identity
check prevents treating multiple corruptions of one image as unrelated samples
when dividing the data. The primary metric is top-1 accuracy.

## ogbg-molpcba

Source: [official OGB graph-property documentation](https://ogb.stanford.edu/docs/graphprop/).
The adapter uses the library-agnostic `GraphPropPredDataset` and its official
scaffold split. Each `GraphSample` preserves nine categorical atom features,
three categorical bond features, and the bidirectional molecular edges. A lazy
sequence avoids duplicating OGB's full graph collection.

Atom cardinalities are `[119,5,12,12,10,6,6,2,2]` and bond cardinalities are
`[5,6,2]`, matching the model encoders and the
[official OGB feature vocabulary](https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py).
The second atom feature has five chirality categories, including `misc`.

Labels have shape `[molecules,128]`. Missing assays remain NaN; loss and metric
calculations must mask them. Molecules without any observed assay are removed
from scored splits, while the graph collection is retained. The task metric is
average precision averaged over eligible binary tasks. A custom reserved-scaffold
audit and the full official-test result are distinct evaluations; an official
test population reused to construct custom halves is not untouched independent
evidence afterward.

`dataset.options.scaffold_path` optionally attaches actual chemical scaffold
identities from a CSV (`graph_id,scaffold_id`) or ordered/mapped JSON manifest.
Every official graph index must appear exactly once. The adapter records the
manifest checksum. It does not invent scaffolds from split membership. See
[DATASET_PROTOCOLS.md](DATASET_PROTOCOLS.md) for the `scaffold_groups` policy,
the default item-level holdout and the distinction between episodic, pooled and
full official-test AP.

## Portable NPZ format

Every profile also accepts `options.npz` or an `.npz` path as `dataset.root`. Loading
uses `allow_pickle=False` and requires explicit dataset identity and split IDs.
The required fields are:

| Field | Meaning |
|---|---|
| `name` | Scalar Unicode canonical dataset name |
| `task` | `node`, `tabular`, `graph`, or `image` |
| `num_classes` | Classification output count; 128 for molpcba tasks |
| `y` | Integer class vector, or `[N,128]` binary/NaN assay matrix |
| `train_idx`, `valid_idx`, `test_idx` | Disjoint integer index vectors |
| `x` | Finite `[N,F]` features for node/tabular tasks |
| `edge_index` | `[2,E]` integer node endpoints for node tasks |
| `year` / `time` | Optional native time metadata; never synthesized |
| `metadata_json` | Optional scalar Unicode JSON object recording provenance |

Molecular NPZ files concatenate all atom features into `x`, all bond features
into `edge_attr`, and all edges into `edge_index`. `graph_ptr` and `edge_ptr`
contain cumulative offsets, starting at zero and ending at the corresponding
array lengths. Edge endpoints use global concatenated node IDs; the loader
converts them to local indices per molecule. For molpcba, 9 atom features, 3 bond
features and 128 tasks are checked explicitly.

Image NPZ files use a Unicode `image_paths` array instead of `x`; relative paths
resolve against the NPZ directory. A Unicode `image_ids` array can explicitly
group corrupted copies of each source image. If absent, the filename is used.
Image identity grouping is checked for native and portable image inputs, so a
custom archive must preserve a consistent underlying-image allocation.

The adapters reject overlapping splits, duplicate indices, unknown scored
labels, invalid endpoints, nonfinite features, mismatched identities, and invalid
task shapes. A complete numerical-content fingerprint includes arrays, graph
topology, labels and split IDs; image fingerprints read the actual file bytes.
These checks validate data plumbing. They do not establish that a new training
run reproduces the paper's measurements.
