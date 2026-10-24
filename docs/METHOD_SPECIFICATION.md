# RTT: shared reference-tail implementation

## 1. Source and execution boundary

The primary source is the supplied manuscript **Reference-Tail Trust: Certified Probability Floors for Learned Updates Inside a Deployed Network**, including Algorithm 1 and Appendices A–H. The repository separates its native arxiv trajectory-tube implementation from an architecture-general backend named `exact_reference_tail`. This document specifies the latter. [NATIVE_ARXIV_METHOD.md](NATIVE_ARXIV_METHOD.md) gives the interval, window, solver and corruption-protocol details of the native implementation.

The shared backend implements internal residual proposals, physical displacement geometry, learned field/covariance estimates, sequential calibration, probability-floor budgets, robust admission, and whole-call fallback. Its all-class prices and finite-step remainders are computed through the **actual frozen remaining model**. It therefore applies to the supplied graph, image, molecular, and tabular architectures at the cost of additional continuation evaluations. It is not an implementation of the paper's efficient interval continuation kernels for every architecture.

All mathematical identities below describe real arithmetic. The executable uses the installed PyTorch primitives and float64 probability/geometry calculations, with explicit margins and independent output checks. Its machine-readable status is:

```text
research_float64_output_checks_not_formally_validated_arithmetic
```

That status is retained in each served call, training manifest, and report. Finite checks and unit tests do not prove operation-level enclosure soundness on every CUDA, sparse, convolution, or transcendental kernel.

## 2. Data units, reference state, and labels

Let a call contain scored task units indexed by `i`, with nonnegative weights `w_i` summing to one. For node classification a call scores selected nodes while retaining the full loaded graph as context. For tabular/image classification it contains rows/images. For molecular classification, one observed molecule/assay pair is a binary task unit; unknown assays have zero weight.

Multiclass logits are mapped with log-softmax. Each molecular logit is mapped to two log probabilities, `log sigmoid(-z)` and `log sigmoid(z)`. Molecular loss is masked binary cross-entropy averaged over observed assays. This is not a 128-class softmax. Both binary outcomes enter the all-label floor calculation. The known task-availability mask remains part of the unlabeled input; label values do not.

Every reference implements the following common interface:

| Method | Meaning |
|---|---|
| `encode(x, context)` | Establish the reference's initial state and fixed call context |
| `transition(state, depth, context)` | Apply one actual frozen reference update |
| `tail(state, next_depth, context)` | Execute all remaining reference updates and the readout |
| `embedding(state, context)` | Obtain the final example/node/graph representation |
| `logits(state, context)` | Apply the frozen, temperature-adjusted readout |
| `state_channels(depth)` | Describe the intervention state's channel dimension |

An intervention is applied after a transition. At depth `l`, write `Hbar = F_l(H_l)` and `Hnew = Hbar + v_l`. Its comparison is the incumbent continuation launched from these two states with **no later adapted interventions**. Later admitted steps are handled by their own comparisons. This reference-tail definition is essential to the telescoping identity.

Serving requires `batch.labels is None` and rejects labeled input. Fitting and calibration may compute supervised targets through the true reference tail. Audit labels are read only after the served prediction has been fixed.

## 3. Actual reference architectures

The reference implementations live in `src/rtt/models.py`. They preserve the operations relevant to each model's continuation:

| Family | State and continuation |
|---|---|
| Tanh diffusion | Graph transport, skip to encoded input, residual/tanh diffusion, frozen head |
| APPNP | Encoded initial state and linear personalized propagation |
| GCNII | Initial residual, identity mapping, graph propagation, ReLU |
| GRAND | Declared attention-free smooth reaction/diffusion discretization |
| GCN / GraphSAGE | Complete configured shallow graph layers, including nonlinearities |
| Residual tanh MLP | Independent tabular rows and nonlinear residual updates |
| Molecular diffusion / centered GIN / GIN | Categorical atom and bond embeddings, graph updates, graph pooling and 128 assay readouts |
| GIN with a virtual node | Real atom states plus explicit virtual states, their updates, and real-node pooling |
| ResNet-50 | Actual convolutional bottlenecks, shortcuts, BatchNorm, ReLU, global pooling and 1,000-class head |

Image adaptation opens the last four bottleneck blocks: the final block of layer 3 and all three blocks of layer 4. It does not replace those convolutions with an MLP on pooled image features. `P=I` refers to example-level transport, not removal of spatial computation.

The full configurations specify dimensions, horizons, initialization and optional checkpoint inputs. Choices absent from the manuscript are disclosed in each configuration's `notes`. In particular, the broad GRAND discretization and molecular widths/horizons are declared reconstructions, not recovered original checkpoints.

## 4. Candidate hull and physical uncertainty

Each open depth has a learned 16-vector bank by default, a state-conditioned bank mixture, and a signed one-hop proposal. Image proposals may use spatial neighbors; independent tabular rows have no invented graph. For each state row, two candidate displacement vectors are norm-trimmed and weighted by a learned node-weight head.

Candidate-dependent features are support values at fixed probes and their negatives:

\[
s_{ij}=\max\{0,\max_m\langle u_j,d_i^{(m)}\rangle\}.
\]

The features depend on the post-trim candidate hull. The shared backend assigns rows to fixed structural roles by row or graph identity modulo the configured role count. This differs from the native arxiv confidence/query role rule and is recorded as an implementation choice.

For two candidates per role, coefficients satisfy `t_km >= 0` and `sum_m t_km <= coefficient_cap`. With the role-restricted candidate operator `D`, the physical step is `v = D t`. The code computes:

\[
G=D^\top D,\qquad \hat a=D^\top\hat g,\qquad
\hat\Sigma=D^\top\operatorname{diag}(\hat\sigma_i^2 I)D.
\]

`G` and the covariance are assembled as role blocks. There is one joint uncertainty penalty `kappa * sqrt(t^T Sigma t)`, not a sum of independently calibrated role penalties. The score uses the covariance eigenspace corresponding to nonzero physical directions. An error outside that represented subspace produces infinity; a genuinely zero physical span has score zero. Numerical rank tolerances are explicit in `joint_score`. They do not establish exact invariance of every finite optimizer iterate under arbitrary redundant reparameterization.

The direction generator approximately minimizes the convex surrogate

\[
\hat a^\top t+\kappa\sqrt{t^\top\hat\Sigma t}
 +\frac{\rho_x}{2\omega}t^\top Gt
\]

using projected Adam-style updates. It keeps the best improving iterate. The result is marked `approximate_projected_surrogate`; global optimality is not claimed. An infinite calibration radius or a non-improving surrogate produces the zero step. Actual price and remainder checks follow this proposal solve.

## 5. Exact continuation price and finite remainder

For a trial displacement `v`, let `p_minus` be the actual remaining reference output from `Hbar` and `p_plus` the output from `Hbar + v`. Define the per-unit all-class price:

\[
c_i(v)=\max_y\{\log p^-_{iy}-\log p^+_{iy}\},\qquad
C(v)=\sum_i w_i c_i(v).
\]

For any labels, the call loss increase is at most `C(v)`. The implementation computes these quantities from stable log probabilities. Unobserved molecular tasks receive zero weight. A nonnegative clamp protects against a tiny numerical sign error; normalized distributions have a nonnegative maximum log ratio in real arithmetic.

Let `j_iy(v)` denote the Jacobian-vector product of the remaining log probability at `Hbar`. The finite remainder is:

\[
R(v)=\sum_iw_i\max\left\{0,\max_y\left[
\log p^-_{iy}-\log p^+_{iy}+j_{iy}(v)\right]\right\}.
\]

For the selected local derivative, this gives the all-label inequality

\[
J(p^+;y)-J(p^-;y)\le \langle g_y,v\rangle+R(v).
\]

The remainder uses the actual before/after tail outputs. It therefore accounts for a crossed ReLU kink instead of assuming that a local Hessian or center Jacobian bounds the entire segment. Tests include a zero local JVP with a strictly positive finite remainder.

For each proposed coefficient vector the engine tries successively halved scales. A trial is admitted only when all of the following hold:

1. Its call price fits the remaining call budget divided by the remaining open depths, after the numerical margin.
2. Every observed task unit fits its remaining row allowance, after the margin.
3. The robust model is sufficiently negative:

\[
\Phi(t)=\hat a^\top t+\kappa\sqrt{t^\top\hat\Sigma t}
 +\frac{R(Dt)+\rho_x\|Dt\|_2^2/2}{\omega}\le-\text{margin}.
\]

The direction search and backtracking are a feasible admission procedure, not the paper's exact conic/KKT optimization algorithm. An unsuccessful search leaves the current reference state unchanged.

The implementation evaluates the JVP using the displacement actually produced by floating-point addition, `proposed - state`, and records the difference from the intended `D t`. The learned linear term and calibration geometry still use the coefficient representation. That rounding discrepancy is measured, not formally enclosed. This is one reason the reported robust predicates remain research numerical evidence.

## 6. Composition, endpoint replay, and fallback

Each local comparison uses an incumbent tail from the executed pre-intervention state. The resulting loss differences telescope over the complete trajectory. In real arithmetic:

\[
J(p^{\rm adapted};y)-J(p^{\rm reference};y)
 =\sum_l\left[V_{l+1}(\bar H_{l+1}+v_l)-V_{l+1}(\bar H_{l+1})\right]
 \le\sum_l C_l(v_l).
\]

Per-unit prices compose in the same way. The final engine independently replays the complete unmodified incumbent and recomputes the endpoint all-class call and row damage. It checks finite logits, total local debits, row ledgers, and the saved robust-model predicates.

The served outcomes are:

| Mode | Returned prediction |
|---|---|
| `R1` | Adapted output after at least one admitted update and a passed final check |
| `NO_STEP` | The incumbent when no update is admitted |
| `F` | A fresh incumbent prediction for the entire call after an endpoint/ledger/predicate failure |

The checker independently replays the endpoint; it does not independently reconstruct every local uncertainty score or JVP. The native arxiv checker has a different trajectory-recomputation contract and can also report `R2`. These status vocabularies must not be merged into one unqualified performance result.

If the reference itself is nonfinite, serving fails and the audit records a failure rather than inventing a valid fallback. The report refuses an incomplete or invalid audit. A failed candidate never changes the frozen reference parameters.

All proposal preparation, trial-tail evaluations, JVP work, final checks, and the separate full reference replay are inside the measured serving timer. Corrector comparison timing charges its own reference pass and output allocation. Consequently, measured times cannot be substituted for the manuscript's efficient native-path latency.

## 7. Fitting and Tent

Incumbent fitting uses only its allocated training labels and early-stopping validation labels. Node references preserve full-graph context; other tasks use configured sampled batches. Preprocessing is fitted on the source training allocation. A separate permitted temperature allocation fits a scalar frozen temperature.

For native proposals, a raw-proposal warm start minimizes terminal supervised loss. Admission heads are then fitted on states reached by the currently executed prefix with a provisional training radius. The field target is the supervised gradient of the **incumbent-only remaining tail** from that current state, not a reverse pass through the final fully adapted trajectory. Field error and a variance objective train the covariance estimate. A separate output corrector uses the same fitting-call generator and episode indices; its hidden width approximately matches admission-head parameter capacity, and actual counts are saved.

This is staged fitting. The proposal/admission solver is not differentiated implicitly through its KKT system. Integer widths, different parameterizations and objectives prevent claiming that the corrector and RTT have identical optimization capacity.

ImageNet-C uses an independent-call Tent proposal. It copies the frozen reference, resets optimizer state per call, updates only BatchNorm affine parameters in the last four bottleneck blocks using prediction entropy, and exports the resulting intermediate states. Running statistics and the original reference parameters remain unchanged. An exported state defines a candidate displacement relative to the current executed state; RTT still performs its own admission checks. The other candidate remains the signed spatial proposal. Tent uses no labels, and a continual Tent stream is a separate unimplemented sampling protocol.

## 8. Calibration, member selection, and audit

For each fixed budget member, process open depths from earliest to latest. Earlier radii are frozen. Fresh calls are generated for each depth, the current pre-action state is reached using that executed prefix, and the supervised physical slope error is computed offline:

\[
s_l=\sqrt{(a_l-\hat a_l)^\top\hat\Sigma_l^+(a_l-\hat a_l)}.
\]

With `n` scores and per-depth miscoverage `delta_cov / number_of_open_depths`, take the order statistic at

\[
k=\left\lceil(n+1)(1-\delta_l)\right\rceil.
\]

If `k > n`, store infinity. There is no interpolated percentile or forced finite radius. Partial calibration checkpoints retain original episode seeds and bind the model, adaptation, protocol and configuration hashes.

The registered budget menu is `[1, 5, 10, 25, 50] * zeta` by default. Selection examines this fixed sequence from least to most aggressive, using `delta / number_of_members` for its member bounds, and stops at the first failure. A member must have a complete valid sample, a regression upper bound at most `p_star`, and a betting lower bound on mean gain strictly above `minimum_gain`. If none passes, the selected policy is incumbent-only. This is a recorded scientific outcome, not a pipeline error.

Sequential conformal coverage is stated for a fixed member. Selecting a member does not automatically give the same nominal selected-member trajectory coverage; this implementation does not claim that coverage without a simultaneous or fresh post-selection construction. Its independent final audit evaluates the already frozen served policy.

For each audit call, `damage = loss_served - loss_reference`, `gain = -damage`, and regression means `damage > zeta`. The one-sided Clopper–Pearson bound is computed from the actual count and denominator. A predictable betting construction supplies a lower confidence bound on mean gain. Uniform episode weights are retained; failed, missing, duplicate or mixed-phase records block release rather than disappearing from the denominator.

The statistical result is distinct from process completion and from floating-point assurance. A completed audit can have `statistical_release=false`. A positive `H_plus` permits regression; zero observed regressions do not establish zero population risk. Conditional on sound arithmetic, `H_plus <= zeta` rules out this strict regression event deterministically; that logical implication does not turn the installed arithmetic into a formal proof.

## 9. Data splits and metrics

[DATASET_PROTOCOLS.md](DATASET_PROTOCOLS.md) specifies the concrete pool construction. Real arxiv years retain their chronological interpretation. Products and yearless graphs never receive invented year masks. Cross-graph datasets retain source/target identity. Unknown Elliptic labels are excluded from fitting/scoring but their nodes can remain context. Image corruption variants are grouped by underlying image before splitting. Molecular scaffold grouping requires actual scaffold IDs.

The generic protocols reserve some data and record it even where no separate reserved-node statistical command is supplied. The native arxiv path contains the manuscript-oriented finite-population reserved-node audit and diagnostic tasks. The generic dataset interface is not a claim that every native ablation or auxiliary table has been ported to every dataset.

Episode audit metrics and whole-population metrics are separate outputs. Molecular AP is calculated per task with missing labels masked and then averaged over valid tasks. The `evaluate` command concatenates the full declared split's predictions before computing AP. Averaged mini-batch AP is never labeled official full-test AP. Reuse of underlying populations is disclosed rather than called independent replication.

## 10. File and function map

| Responsibility | Implementation |
|---|---|
| Raw data and portable validated arrays | `datasets/*.py`, `datasets/common.py` |
| Task batches and unlabeled serving inputs | `batching.Batch`, `BatchFactory` |
| Reference continuations | `models.build_model` and model subclasses |
| Learned candidates and field/covariance | `adaptation.DepthAdapter`, `AdaptationModules` |
| Physical geometry and calibration score | `coefficient_geometry`, `physical_step`, `joint_score` |
| Approximate direction and actual admission | `projected_direction`, `ReferenceTailEngine._step` |
| Endpoint check and whole-call fallback | `ReferenceTailEngine.serve` |
| Reference-tail supervised target | `true_tail_gradient` |
| Fitting | `training.train_incumbent`, `fit_proposal`, `fit_heads` |
| Independent Tent proposal | `tent.TentProposal` |
| Masked losses, probability prices, AP/AUC/F1 | `losses.py` |
| Pool rules and statistical summaries | `protocol.make_protocol`, `conformal_radius`, `summarize_records` |
| Frozen stage artifacts and prediction | `experiment.Experiment` |
| Continuous output floor comparator | `frontier.continuous_floor_frontier` |
| Configuration, checksums, runtime metadata | `configuration.py`, `runtime.py` |
| Local and Arrhenius interfaces | `cli.py`, `hpc.py`, `scripts/` |

The training manifest binds the configuration, complete data digest, label-role protocol, and both trained checkpoints. Calibration binds those frozen inputs; selection binds calibration; audit-shard metadata binds selection and record contents. Final reduction requires every predetermined call index exactly once. Slurm adds an immutable source/configuration snapshot and stage receipts. These controls preserve a run's identity; they do not supply unavailable original weights or replace evaluation on real benchmarks.
