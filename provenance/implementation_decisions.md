# Implementation decisions

The release candidate adapts existing code rather than introducing a new pruning or recovery method. The experimental source manifest identifies copied files. CPU checks establish software behavior only; robot behavior remains a separate release gate.

## Functional metric and search identity

The paper's Eq. 3 and the experimental `scripts/layer_analysis.py` both specify mean flattened-chunk L2. The older live kernel computes mean absolute deviation. The candidate defaults to the former and names the latter `legacy_l1_v1`; it never reuses an old probe table. This deliberately changes future live searches. It does not establish which implementation every historical run used.

The paper and existing driver option already support duplicate checks on deletion set plus recipe. The candidate selects that existing option for live runs. Tests falsify this behavior if recovery escalation is rejected or an executed duplicate is evaluated again.

## Model integration and probe extraction

The official fixed OFT source and the local training/evaluation import closure were inspected. The local custom trainer, teacher generator and pruning support are absent from the checked public base, so an upstream install alone cannot reproduce the call chain. The candidate provides a checksum-verified overlay on the fixed base, retaining local-source hashes. That base is a reproducible assembly choice, not a reconstructed historical commit.

The OFT probe helper depended on a full StarVLA development tree and a large experiment script. The candidate extracts its existing function bodies and private wrapper import closure. No replacement preprocessing model was designed. Invalid action shapes and failed calibration samples now raise errors instead of being padded or skipped. Namespacing and import checks cannot establish numerical equivalence; the next acceptance check must compare identical images, normalization, proprio, action spans and model outputs with the original runtime.

## Physical export

The existing `scripts/prune_layers_8_23.py` filters and reindexes tensors but targets a fixed contiguous range, can remove an existing destination, and does not identify arbitrary selected recovery components. The candidate adapts that established mechanism with explicit source, mask and recovery step, leaving the source untouched and refusing an existing destination. The existing evaluator is extended to install the same adapters on an already shortened backbone without deleting layers again.

The smallest unverified assumption is that this assembled physical checkpoint and adapter loader reproduce the selected recovered policy. The synthetic tensor test can falsify index remapping, vision-tensor preservation and file portability. Acceptance for a real model additionally requires a separate-process reload, matching action outputs on fixed inputs, and complete closed-loop evaluation. No paper SR is assigned to the new exporter before those checks.

## Evaluation and state

Per-shard exit status and exact requested episode counts are required before computing an SR. Trial metadata survives interface boundaries. Each run owns its copied configuration and experience, and the original checkpoint's editable metadata is staged locally because upstream loading can rewrite it. The candidate does not infer held-out initial states or historical training seeds from current defaults.

## Primary references

- [AutoPruner paper](https://arxiv.org/pdf/2610.23880)
- [Fixed OFT source](https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8)
- [Official OFT setup](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/SETUP.md)
- [Official LIBERO benchmark loader](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/benchmark/__init__.py)

## Second self-review

See [SELF_REVIEW.md](SELF_REVIEW.md) for reproducible engineering failures and their fixes. Suite selection, calibration language, absolute subprocess paths and evaluation validation now fail explicitly. Historical source hashes remain unchanged; current release hashes are refreshed. No teacher-cache method change or new robot-performance claim is introduced.

## Teacher cache identity v2

Adopt official TFDS record IDs and original frame indices through the pinned dlimp metadata broadcast. Preserve the existing teacher/student view convention. See [CACHE_V2_FIX.md](CACHE_V2_FIX.md) for primary references, snapshot/version assumptions, falsifiable checks and the separation between matching verification and robot behavior. Legacy output tensors are not relabeled into the new format.

## Whole-framework acceptance and explicit execution contracts

The real OFT entrypoint exposed a misplaced calibration-validation block in the model loader, an incomplete LIBERO resource tree, and dependence on a stale home-directory LIBERO configuration. Validation now runs in its owning calibration function; resources are checked before execution; each invocation writes its own LIBERO config. No source checkpoint or user's home config is rewritten.

The live agent returned prose in numeric fields and sometimes exhausted its response budget. All providers now share a strict JSON decision contract; validation feedback preserves the actual error. Provider budgets are explicit and recorded. The original user request reaches the agent. Unsupported recovery names are rejected rather than silently replaced by another experiment. Users can explicitly restrict the model's recovery menu. Unsupported multi-suite/wall-clock requests fail before execution.

A real 4096-wide probe exposed the cost of repeatedly solving feature-sized ridge systems with only eight calibration rows. We inspected [scikit-learn's established primal/dual ridge implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py) before changing the local solver. The existing NumPy dependency is sufficient; adding scikit-learn solely for this operation was unnecessary. For positive regularization and fewer samples than features, solve the sample-sized dual system, then recover feature weights. Centering preserves the original unregularized intercept. The smallest hypothesis is numerical equivalence, not a new probe: tests compare weights and held-out predictions against the original primal solve across three regularization values, with and without intercept, at 1e-8 tolerance. Scalar-target shape is also checked. Real probe completion checks practical execution; it does not prove robot performance.

The physical-export handoff now resolves the qualifying winner against its exact proposal and completed evaluation, requires the saved recovery step, and writes a receipt. Adapter metadata must agree with the selected mask and step. Separate-process native loading compares identical HDF5 inputs before and after export; checksums alone do not count as numerical acceptance. Actual bounded evidence and its limits are recorded in [FRAMEWORK_ACCEPTANCE.md](../docs/FRAMEWORK_ACCEPTANCE.md).

## Model-plugin configuration and native export

The existing ModelPlugin/GenericVLABackend separation was retained. The actual entrypoint rejected custom constructor configuration and the finalizer accepted only OFT. Fix those handoffs rather than introducing a second search or a universal serializer. An optional model-owned native exporter writes reloadable artifacts; the framework validates selection and file integrity. The established separation between configurable training lifecycle and model-specific saving was checked against the [official Transformers Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer). No Trainer dependency or new pruning algorithm is introduced.

The falsifiable engineering assumption is that shared search/configuration/selection/export plumbing works independently of OFT's dimensions and task vocabulary. Tests and an installed-wheel external-plugin run use six layers, a custom suite, three-step/four-dimensional actions, two candidate recipes, recorded feedback, and a clearly synthetic native artifact. Missing boundaries, invalid arrays, wrong recipes, absent exporters, empty outputs and linked files must fail explicitly. Real OFT capture/prediction is separately rechecked. These tests establish software contracts, not transfer to a second real robot model; see the [review](../docs/MODEL_PLUGIN_REVIEW.md).

## 2026-10-30: two reader workflows and checkpoint publication

Read the paper's model-interface/export contract, pinned OFT/StarVLA/FastWAM inference code, current upstream model cards, pip installation guidance, Flash Attention 2.5.5's wheel installer, and GitHub's sensitive-history guidance. Inspect actual checkpoint headers, configuration, normalization, script arguments and current hashes. No historical score is rebound to a newly trained smoke model.

Reuse the existing finalizer, checksum verifier and native evaluator for opt-in automatic delivery. A nonqualifying frontier must not be delivered as a passing model; native export failure must preserve the completed search records. A downloaded OFT model needs its own evaluation entrypoint because the author-side action comparison requires a source search/reference that readers do not possess. The new entrypoint reads the manifest and uses the existing physical-model evaluator, without calibration or training datasets. No new pruning or recovery algorithm is introduced.

The first clean environment failed on old pip metadata handling. A separate newly created environment exposed the Flash Attention wheel installer's cross-filesystem rename under the test's NFS cache. After package metadata checks passed, an actual model subprocess still failed because tensorflow-metadata 1.21.0 required a protobuf API absent from protobuf 4. Pin the known working metadata version, keep Flash Attention's temporary wheel cache local, upgrade pip, and test the real OFT import closure. Pin the physics runtime to the version used by the current validation, not to an invented historical paper version. Do not inject private userspace libraries or silently reuse the old environment.

The proposed reuse of pinned public StarVLA with the existing pruning components was tested before adopting it. Correct the documented action-horizon schema, then require strict parameter matching and identical-input numerical comparison. All parameters matched, but one normalized action input differed by about 0.002518 while the original implementation's repeat control was exactly stable. This falsifies the narrow claim that this assembled public path is already numerically equivalent. It does not establish which preprocessing/forward change caused the difference or whether robot SR is worse. Do not publish a replacement module from intuition. The next distinguishing experiment is to compare camera tensors, instruction/token IDs, action-token spans and action-head inputs before changing the loader; acceptance also needs native reload and closed-loop checks.

FastWAM's recovered file retains both experts' full block indices. Its public mask patch is execution skipping, not proof of a physically reduced checkpoint. A new converter would require model-specific cache/index and action-equivalence checks; none is invented here. The artifact index marks the native inference package incomplete rather than treating file existence as readiness.

The selected Git-history scan found a nonplaceholder credential-like literal in an ancestor of the release branch. The least invasive publication plan is a reviewed file-only archive without old Git history, preserving the research repository as private. It does not rotate the credential or certify unscanned old branches. Detailed scope, negative findings and behavioral evidence are in [OPEN_SOURCE_REVIEW.md](../docs/OPEN_SOURCE_REVIEW.md).
