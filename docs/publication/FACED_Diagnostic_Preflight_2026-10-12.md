# FACED development diagnostic: cache and access preflight

Date: 12 October 2026. This is a read-only preflight for a new, separately declared development diagnostic. It activates no fitting or outcome access and changes no previous experiment, manuscript or research question.

## What was checked

The existing runners, pure identity/context/QC helpers, public development manifests and actual ignored local cache locations were inspected. Private NPZ file bodies were hashed as opaque bytes; only the NumPy headers of one exemplar per cache were interpreted. No private feature-array values, original rating values, cached numeric targets, raw waveform values or protected outcomes were decoded. No original data file was opened. Importing the pure identity bridge and checking metadata produced no fitted state.

The authoritative metadata bridge passed for all 1,260 development identities. All 1,260 artifact-control feature caches and all 1,245 available-trial representation-v2 caches exist and match their published per-file SHA-256 bindings. This does not replace any new experiment's activation and numerical audit.

## Exact reusable inputs

Paths below are relative to the Git repository, except private caches, which are relative to its parent workspace.

| Input | Path and interpretation |
|---|---|
| Development programme | `results/development/faced_development_protocol_2026-10-11/plan.json`: 3 rotations, each 560 source-fit, 120 unfamiliar-title validation and 160 familiar-title validation pairs; five fixed source-participant OOF folds. |
| Participant roles | `results/development/faced_backup_2026-10-10/participant_reservation.json`: 70 source, 20 validation, 33 protected participants. |
| Recording provenance | `results/development/faced_artifact_control_2026-10-11/source_records.json` and `validation_records.json`: 980 and 280 records, respectively; identity, role, acquisition, native channels/rate, previous 32-second interval, processed-window and feature-cache hashes. |
| Frozen EEG availability | `results/development/faced_inference_quality_2026-10-11/signal_audit.json`: all 980 source trials available; 265 of 280 validation trials available. Fifteen validation failures retain the fixed context/global-median fallback. |
| Spectral/context private cache | Workspace `publication_runs/faced_artifact_control_2026-10-11/{subject}_clip{vid:02d}_features.npz`: all 1,260 files. Exemplar header: `context` shape `(13,)`, `spectral` shape `(150,)`, both float64. Bind file bytes to each original record's `feature_cache_sha256`; do not use unavailable-trial EEG merely because the file exists. |
| Covariance/private representation cache | Workspace `publication_runs/faced_representation_v2_2026-10-11/{subject}_clip{vid:02d}_features.npz`: 1,245 available trials only. Exemplar header additionally has `covariance` shape `(2330,)` and `band_covariances` shape `(5,30,30)`, float64. Bind to `private_features_sha256` in its public `feature_manifest.json`. |
| Canonical existing development targets | `results/development/faced_inference_quality_2026-10-11/predictions.csv`, SHA-256 `84805ab0c7c4076e9e95a926c8ff7e3afbeeb0455e08fac923e2a73a9e3724d0`. Numeric target values were not opened in this preflight. The declared new loader can reuse these targets after identity/cache/source gates. |
| Representation outputs | `results/development/faced_representation_v2_2026-10-11/predictions.csv`, SHA-256 `790ce1a56b691409992565808bed1e8e44fc7de70596a0226f70026c055f7a0b`. Contains deliberately permuted capacity targets; do not naively merge every `target` column. |

The development clip set is exactly `{2,3,4,6,9,11,12,17,18,20,23,25,27,28}`. A union cache grants no permission to fit a rotation on all 980 source trials: its scaler, prior, target vector and estimator must fetch only its own 560 `source_fit` pairs. The existing record files are ordered by subject and clip; use keyed joins, and preserve explicit rotation-arm order for prediction output.

## Safe helper reuse

Use `gruxnet.faced_eegnet_bridge.authoritative_bundle()` with the programme, reservation, clip catalogue (`faced_codebooks_2026-10-10/audit.json`), the two record lists and availability decisions. It enforces participant roles, exact clip rectangle, technical geometry and participant/title separation through `guard_bundle()`. It is safe to reuse the pure helpers without executing the old runners.

`scripts.run_faced_eegnet_diagnostic_v2.metadata_bundle()` is the existing orchestration example. Avoid constructing its `Execution`: construction creates previous-run output files, and preparation decodes original EEG. Existing frozen runner/module bytes must remain unchanged; implement the new bounded diagnostic in separate files.

Reconstruct context with `gruxnet.faced_classical_contract.context_features(metadata)` and check its `array_sha256` against `context_feature_sha256`. Its exact columns are:

- 0–7: Amusement, Anger, Disgust, Fear, Inspiration, Joy, Sadness, Tenderness (one-hot assigned category).
- 8–10: legacy_1000, legacy_250, modern_1000 (one-hot acquisition).
- 11: catalogue duration in seconds.
- 12: presentation trial ordinal.

All scaling/feature transformations used by fitting must be trained only on the current source set. Category-only and full-context comparisons can be built from these authenticated features without raw-data access.

For targets, guard subject/clip/family and acquisition before numerical conversion. Require exact repeated-value agreement and final coverage of the 1,260 identities. If representation predictions are used, call `gruxnet.faced_cached_targets.cached_original_target()` and skip the 3,360 explicitly shuffled capacity rows before converting their target or prediction. Real capacity rows must still agree with the canonical original targets. Alternatively, use only the quality predictions as the canonical target source.

## Pairing control issue to freeze before outcomes

Within-source same-video/same-acquisition donor strata have at least three available participants. The source roster has 3 legacy_1000, 30 legacy_250 and 37 modern_1000 participants.

Within-validation strict acquisition-matched derangement is impossible for legacy_1000: only one validation participant remains EEG-available for each of the 14 clips. The other validation acquisition strata have multiple available donors. Any new pairing control must declare singleton treatment before cached targets or feature arrays are interpreted. A defensible bounded treatment is to leave singleton EEG unchanged, flag these recipients as unperturbed, retain the fixed full evaluation rectangle, and report a separate switchable-recipient comparison. If a subset becomes nonrectangular, the existing crossed-bootstrap helper cannot be applied without an appropriate separate estimator. Do not silently drop recipients, cross acquisition groups without a stated calibration caveat, or borrow source participants' held-title EEG.

The 15 unavailable validation trials (14 for one participant and one for another) must keep their existing fallback. A donor assignment must use only available trials, keep the recipient context and target, avoid self-pairing where a swap is declared, and avoid all protected people/materials. This diagnostic is a specificity check, not a causal test or an independent confirmation experiment.

## What is not cached

The completed EEGNet runner explicitly persisted no raw or processed window arrays; its 1,245 processed window tensors occupied memory only. A whole-trial coverage comparison therefore needs a separate interval/preprocessing/access declaration and qualification before opening broader original EEG. Existing late-feature NPZs cannot answer whether whole-trial coverage improves retrospective rating prediction. Native trial joins exist in the raw qualification metadata, but this preflight did not process broader intervals.

## Metadata bindings observed

| File | SHA-256 |
|---|---|
| Development programme | `dbd5c356c1ace1eda72c5b02ea74cbe2ad1ac19cba2c3c271c919375151060ba` |
| Participant reservation | `f41b319f8d3c27db7966560f7a7bc0b6392eb5e1528892c8d4d445557014d087` |
| Artifact source records | `084be723139633048a6626dee0472f04468dbe8098149e32e94475ccb7264b72` |
| Artifact validation records | `0881355c0f1d0396fba3351bf6879ff405ad33eca3cdefc32b6e90bc04762baa` |
| Availability audit | `c5ba4e498749c1fe367661d7f3acae15dd66e633e2cd23cea2324239a677dccc` |
| Representation-v2 feature manifest | `9fe3d386bd5229dc4a95f84a5bc59810b4105b18ffde26c6eb0606d09a1eb8fc` |

Access status: private arrays decoded 0; cached numeric targets decoded 0; original data files opened 0; new fits 0; confirmation access false; research question unchanged.
