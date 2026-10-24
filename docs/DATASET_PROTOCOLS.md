# Dataset roles, call sampling and statistical scope

`rtt.protocol` builds a reviewable JSON manifest before fitting or evaluation.
The manifest stores every item ID, the seed, the effective split policy, pool
sizes and any reduction needed for a small dataset. Locations belong in
`dataset.root` and `dataset.options`; experiment choices belong in `protocol`.
The loaders and source formats are documented in [DATA_SOURCES.md](DATA_SOURCES.md).

These are executable reconstruction choices. The supplied manuscript identifies
the datasets and broad evaluation settings but does not supply every domain
assignment, molecular scaffold manifest or per-example allocation. A generated
manifest makes those choices reproducible; it does not recover the missing
original experiment records.

## Public API

```python
from rtt.datasets import load_dataset
from rtt.protocol import (
    make_protocol, sample_ids, conformal_radius, members, summarize_records,
)

bundle = load_dataset("ogbn_arxiv", "./data/arxiv", download=False)
protocol = make_protocol(bundle, config)
ids = sample_ids(protocol, "audit", size=config["protocol"]["query_size"], seed=123)
menu = members(config)
radius = conformal_radius(calibration_scores, delta=0.1)
summary = summarize_records(audit_records, config, expected_count=2048)
```

`make_protocol` returns ordinary JSON-compatible dictionaries and lists.
`sample_ids` returns an integer NumPy array. The scorer, calibration runner and
deployment runner must all use the same frozen manifest and fitted artifacts.
Changing an allocation or preprocessing choice requires new downstream artifacts.

## Label roles

| Role | Purpose | Parent allocation |
|---|---|---|
| `fit` | Fit proposals and auxiliary heads | Original train split |
| `inc_train` | Fit the incumbent | 90% of train items or identity groups |
| `inc_valid` | Early stopping/model choice for the incumbent | Remaining 10% of train |
| `temperature` | Fit the temperature after incumbent fitting | 20% of original validation |
| `calibration` | Fit the conformal radii on fresh sequential calls | 40% of validation |
| `selection` | Compare the preregistered budget members | Remaining 40% of validation |
| `audit` | Evaluate the frozen selected policy | Disjoint test allocation |
| `reserved` | Reserved diagnostic population | Disjoint test allocation |
| `transfer` | Separate temporal evaluation when a genuine cohort exists | Actual 2020 arxiv nodes |
| `development_unused` | Retained IDs of the unused arxiv development allocation | Actual 2019 arxiv nodes |

`fit` is intentionally the union of `inc_train` and `inc_valid`. Every other role
is a disjoint leaf. The protocol checks this invariant and refuses unknown scored
labels, duplicate IDs, entirely unobserved molecular targets and insufficient
independent groups. The training allocation can be used jointly for the
incumbent and proposal fitting; this does not authorize any reuse of calibration,
selection or audit labels for model fitting.

The integer allocation uses the requested proportions with a minimum of one
independent group per required role. Fractions therefore round for small
fixtures. At least two train groups, three validation groups and two test groups
are needed for the ordinary policy. A missing optional transfer population is
recorded as empty; it is never replaced by unrelated test items.

## Chronological arxiv policy

Actual `ogbn_arxiv` data with publication years uses `arxiv_temporal`, including
when the configuration initially requests the default `official_disjoint`.
The returned manifest records both the requested and effective policies.

- Original training labels must have publication years at most 2017.
- Original validation labels must have publication year 2018. This cohort is
  split into temperature, calibration and selection roles at 20/40/40.
- The 2019 test cohort first reserves up to 8,192 nodes, capped at one third of
  the cohort. Half of the remaining nodes are `development_unused`; the other
  half are the final `audit` allocation.
- The actual 2020 cohort is `transfer`. Unexpected test years fail with an error.

The 8,192-node target and retained development/audit separation follow the native
arxiv reconstruction. The general protocol uses a 90/10 incumbent train/validation
subdivision; the older native arxiv runner retains its separately configured
incumbent holdout fraction. These two experiment manifests are not interchangeable.
If `query_size` is absent, this module retains the native value of 4,096.
Resolved dataset configurations may explicitly choose another size.

A portable arxiv fixture without publication years can use `official_disjoint`.
Its manifest explicitly says that this is a non-temporal fixture allocation.
Requesting `arxiv_temporal` without actual years is an error.

OGB's official arxiv chronology is documented in the
[node-property prediction documentation](https://ogb.stanford.edu/docs/nodeprop/).

## Other datasets and identity groups

For GOOD degree, GOOD time, Twitch, Facebook100, Elliptic, products, ACSIncome
and the ordinary molecular route, `official_disjoint` preserves the adapter's
train/validation/test allocations and subdivides their test allocation 80/20
into audit and reserved. The adapter's metadata states whether the outer splits
come from an official benchmark, a temporal boundary or an explicit source/target
domain selection. This policy never infers years from index order.

For **ImageNet-C**, every subdivision groups by `metadata.image_ids`. All
corruption variants of one base image stay in one leaf role. If explicit IDs are
missing, source filenames are used and the manifest records that assumption;
the caller must provide IDs when filenames do not uniquely identify a base image.
All fractions apply to identity groups, so row counts can differ from nominal
fractions when groups contain different numbers of variants. Sampling a call
draws rows from its assigned pool; different variants may appear within the same
call, but never cross calibration/selection/audit roles.

The directory loader additionally excludes corrupted versions of clean fitting
images from the target allocation. The manifest route enforces the same identity
separation. These checks address repeated-image leakage, not independence of all
pixels or semantic similarity among images. The provided corruptions and
severity are those in the [authors' ImageNet-C distribution](https://github.com/hendrycks/robustness).

## Molecular scaffold policy and AP

OGB's official `ogbg-molpcba` allocation is already scaffold based; its 128 binary
assay labels preserve missing targets as NaN. The official metric is average
precision averaged across eligible tasks, as described in the
[OGB graph-property documentation](https://ogb.stanford.edu/docs/graphprop/).

The default `official_disjoint` policy creates an **item-level custom 80/20
holdout inside the official test allocation**. It does not claim that this new
holdout is scaffold-disjoint or that it recovers Table 16's reserved-scaffold
halves. To keep whole chemical scaffolds together in every subdivision, set
`protocol.split_policy="scaffold_groups"` and supply actual chemical identities:

Add these fields to an existing molecular experiment configuration; this is a
configuration fragment, so retain its existing `name`, `model` and other sections.
The CLI resolves the manifest below to
`<dataset.root>/manifests/molpcba_scaffolds.csv`. An absolute path is also accepted.

```json
{
  "dataset": {
    "name": "ogbg_molpcba",
    "root": "./data/ogb",
    "options": {"scaffold_path": "manifests/molpcba_scaffolds.csv"}
  },
  "protocol": {"split_policy": "scaffold_groups"}
}
```

The CSV has `graph_id,scaffold_id` columns, with every zero-based official graph
index appearing exactly once. A JSON list in exact graph order or a JSON object
mapping decimal graph IDs to scaffold IDs is also accepted. Derive the identities
from the matching official molecule order, for example from a documented
Bemis–Murcko scaffold calculation; record the derivation version and stereochemistry
choice with the manifest. All acyclic molecules assigned an empty scaffold by
that calculation should share an explicit nonempty identity such as `ACYCLIC`.
The adapter validates coverage, stores a SHA-256 checksum and refuses duplicate
or missing graph indices. It cannot authenticate the chemical provenance of an
arbitrary supplied string.

Whole scaffold IDs must also be disjoint across the original train, validation
and test allocations. The group policy retains the explicit 80/20 test split
by scaffold count. It is a reproducible additional protocol, not the missing
paper manifest or a claim to reproduce its exact scaffold halves.

The summary's mean per-call AP is a descriptive episode metric. **It is not pooled
AP.** Pooled AP requires labels and predictions over a precisely defined
population and an explicit treatment of repeated molecules across calls. A
pooled evaluation of a custom audit pool remains distinct from a once-only
evaluation of the complete official test population. Evaluations using overlapping
parts of that population must not be presented as independent test sets.

## Calls and conformal calibration

`sample_ids` draws a uniform subset without replacement within one query.
Independent queries may reuse items from the same role. Each query uses
`min(query_size, pool_size)` items; the manifest records any undersized pool,
and the requested query size is not silently rewritten. An empty pool produces
an actionable error.

The conformal radius for nonnegative joint scores is the order statistic at

\[
k=\lceil(n+1)(1-\delta)\rceil.
\]

When `k>n`, including an empty calibration set, the radius is positive infinity.
An infinite individual score remains infinite. NaN, negative values and negative
infinity are invalid. Small smoke runs therefore often produce infinite radii
and return the incumbent; this is the expected finite-sample behavior.

The engine supplies the **joint** coefficient-space error score. This utility
does not replace that score by a per-coordinate quantile. Each registered budget
member needs its own sequential calibration rollout, with the earlier calibrated
decisions active when observing later depths. Calibration labels are offline
inputs; deployed admission receives unlabeled batches. Conditional/exchangeability
assumptions belong to the registered call generator and frozen artifacts, not
to the quantile function alone. The underlying rank construction is described
in the manuscript's calibration procedure and the
[split-conformal tutorial by Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

## Complete-call statistical summaries

Every attempted call must leave a record. Required fields are `call_id`,
`status`, `gain` and `down`, with `gain=-down`. `status="OK"` denotes a complete
finite result. The record's `phase` or `stage` retains its provenance, and
different phases or frozen policies cannot be pooled. The summary never relabels
selection calls as audit calls.

Supply `expected_count` from the frozen run manifest. A missing record, duplicate
ID, non-OK status, nonfinite loss, inconsistent sign or contradictory checked
endpoint blocks all confidence and tolerance bounds. Available valid records
may still have a descriptive mean, visibly marked `INCOMPLETE`; they do not
become a replacement inference sample. Without an external expected count,
the summary can only check completeness of the records that were supplied.

Calls receive equal weight regardless of their node count or number of observed
assay entries. For the strict regression event `down > zeta`, the one-sided
Clopper–Pearson upper bound at error level `delta` is

\[
u=\operatorname{Beta}^{-1}(1-\delta;k+1,n-k),
\]

with `u=1` when every call regresses. The output's `u` and `cp_upper` are this
same value. The beta quantile uses SciPy; see its
[exact binomial interval interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
The mean-gain paired-t interval is labeled **descriptive** and is not used as a
probability-floor proof. CP requires the registered independent call-sampling
model; the code does not establish independence merely by counting records.

For optional betting inference, all calls need final endpoint evidence. The
implementation reuses `rtt.core.statistics.betting_ucb` with horizon-free
predictable bets and a preregistered one-sided damage bound. It never estimates a
range from the audit minimum or variance. The returned `betting_lb` is the
negative of the upper bound on mean damage. The statistical assumptions remain
those of the manuscript's betting construction.

To record endpoint evidence, emit `endpoint_checked=true` only after checking the
**returned** output, including incumbent fallback. Also emit finite `H_plus`,
`H_row`, `call_damage_bound` and `row_damage_max`. A failed pre-fallback checker
can coexist with a valid checked incumbent fallback. A declared endpoint that
exceeds its budget invalidates the record. The small consistency tolerance
between two evaluated loss expressions does not enlarge `H_plus` or `zeta`.

`deterministic_tolerance=true` additionally requires every declared call budget
to satisfy `H_plus<=zeta`, every final endpoint to pass, and every observed damage
to satisfy `down<=zeta`. This statement inherits the implementation's recorded
floating-point status. It is not a validated-arithmetic proof or a replacement
for the theoretical all-label bound under exact arithmetic.

Budget members are the fixed ordered list `m1`, `m2`, … with budget
`member_multipliers[i] * zeta`. Summaries use the supplied `delta` **without an
automatic family correction**. Selection across a preregistered menu must use
its registered multiplicity correction, for example `delta / len(members)`.
Once selection freezes a policy, its independent audit uses the audit's own
registered error level. Passing a pointwise bound on the selection data does not
turn those data into an independent final audit.

## Numerical and backend scope

The general dataset runner uses `exact_reference_tail`: it evaluates the frozen
continuation and the actual candidate endpoint, and checks label-free all-class
probability damage in floating-point arithmetic. It incurs extra tail/reference
evaluations. Its metrics must not be used to claim the manuscript's no-second-pass
tube runtime or outward-rounded machine certificate. The native arxiv tube
implementation remains a separately named backend with its own configuration,
checks and documented numerical limitations.

An all-incumbent smoke result verifies plumbing and fail-closed behavior. It is
not evidence that RTT gains, manuscript accuracy/AP numbers, calibration coverage
or Arrhenius throughput have been reproduced on the full datasets.
