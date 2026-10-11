# FACED item-10 storage, selective decoder and bounded development declaration

11 October 2026. The requested target-storage and selective-decoder verification is complete, and the bounded context-versus-EEG programme is declared. All 1,260 inspected development score vectors are **1 × 12 doubles**, matching the codebook's twelve ordered items. Item 10 is valence. Twenty manufactured-data decoder tests and eight classical-contract tests pass. **No original rating value was decoded, no new numerical EEG sample was opened and no original-data model was fitted in this phase.** The programme is declared, not launched, and the paper question remains unchanged.

The [target evidence](../../results/development/faced_target_contract_2026-10-11/README.md), [bounded programme](../../results/development/faced_development_protocol_2026-10-11/README.md) and [structured declaration](../../results/development/faced_development_protocol_2026-10-11/plan.json) record identities, processing, model/tuning limits, quality handling and confirmation seals. This follows the [source-quality pilot](GRU-XNet_FACED_Source_Quality_Findings_2026-10-11.md) and [target/material proposal](GRU-XNet_FACED_Design_Proposal_2026-10-11.md).

## What was verified

The metadata-only declaration and decoder were committed as [d3dee16ff](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/commit/d3dee16ff) before inspecting original score headers. The audit reauthenticated only the original `After_remarks.mat` files for **90 development participants**. All 33 reserved confirmation participants were skipped before file access. The 14 permitted development clips are the union of source/validation clips across the already fixed three rotations.

Each file's authenticated trial/video identity crosswalk was checked before any selected score-header interpretation. Score fields for the other 14 clips were skipped at their outer matrix tags, without inspecting their inner dimensions or numerical contents. Accuracy and response-time matrices also stayed opaque. Across the audit:

- 1,260 development score headers were inspected, all with the same shape/type.
- 1,260 protected or excluded score headers were skipped.
- 5,040 accuracy/response-time matrices were skipped.
- **Zero original score numbers** were interpreted.

Each qualified score is a nonlogical real `mxDOUBLE`/`miDOUBLE`, stored as a 1×12 vector with eight bytes per item. The single permitted future rating occupies bytes **72..80**, half-open, within the numeric payload: zero-based position 9, MATLAB item 10. The source codebook's order is joy, tenderness, inspiration, amusement, anger, disgust, fear, sadness, arousal, **valence**, familiarity and liking. Scale and anchors remain 0–7, very negative to very positive. This is storage and documented-semantic qualification; actual missingness, range adherence, variance and rating reliability remain unknown.

The selective decoder validates the entire identity crosswalk before decoding a rating. It checks participant authorization before file access, the original-file digest before numerical interpretation, and the qualified storage shape/type before reading exactly one scalar. It skips protected films' score headers and the other eleven values in allowed score vectors. A missing/nonfinite/out-of-scale selected value stops the future run; it does not trigger a different item, silently removed row or imputed target. Endpoints 0/7 and midpoint 3.5 remain eligible.

Manufactured tests cover compressed/uncompressed vectors, row/column vectors, reordered fields, poisoned protected matrices, poisoned non-target items, changed source hashes, protected participants, incorrect identities and invalid selected values. A [SciPy `savemat`/`loadmat` roundtrip](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) independently agrees on selected values in manufactured structs. Ordinary `loadmat(variable_names=['After_remark'])` selects a top-level variable; it does not supply the required row/item boundary inside that struct. It was not used on original behavioral files. Compressed protected bytes can be inflated as opaque transport bytes, but are not numerically interpreted.

## Fixed programme and fitting boundaries

| Per rotation | People | Clips/families | Original trials | Purpose |
| --- | --- | --- | --- | --- |
| Source fit | 70 | 8 | 560 | Fit scaler, priors and models |
| Unseen-material validation | 20 | 6 | 120 | Select alpha and assess development transfer |
| Familiar-material validation | 20 | 8 | 160 | Diagnostic only |

All three source/validation material rotations are preserved. Their union contains 1,260 permitted development participant/clip pairs. A possible future union cache does not authorize pooled fitting: each rotation's scaler, targets, prior estimates and regression head may fetch only its own 560 source pairs. All confirmation films in every participant, and all 33 confirmation participants, remain sealed.

The same 20 validation people and overlapping clips recur across rotations. These are **not independent replications**, and their outcomes cannot be pooled as additional independent trials. Validation still lacks unseen fear/inspiration films. Six held validation families and eight later confirmation families limit precision. The experiment covers non-neutral elicitation, with neutral clips 13–16 and globally version-ambiguous clip 22 excluded from metadata before values; no rating-dependent eligibility is added.

The initial matched models are:

- Context Ridge: 13 experimenter-provided features—eight assigned-category indicators, three acquisition-group indicators, catalogue duration and presentation ordinal.
- EEG Ridge: 150 channel/band features, fixed Welch log-power over five half-open bands (1–4, 4–8, 8–13, 13–30, 30–45 Hz), averaged across all seven windows.
- Joint Ridge: concatenate the same 13 context and 150 EEG features. It receives no self-inclusive target-derived prior.

Each uses a source-only `StandardScaler`, intercept and SVD Ridge with the same five alphas: **0.1, 1, 10, 100, 1000**. Clip predictions to 0–7 before scoring. Select alpha separately for each family/rotation by unseen-validation participant/film-equal MAE, then unseen MSE, then declared first-setting order. Familiar validation never chooses alpha. All five candidates and all three families are retained; no globally winning rotation or target is selected.

Additional EEG-free checks are source global, coarse-valence and assigned-category medians. A source video median is legal for familiar videos only, with global fallback for unknown films. The primary matched comparator is context Ridge; a continuation also has to beat the strongest fixed unseen-context prior. These additional priors are a separate, explicitly larger context-control suite, not an extra matched EEG hyperparameter search.

Source learning capacity is checked with five fixed, participant-disjoint source OOF folds at each selected alpha. Each fold has 14 held people; its scalers/models/priors use the other 56 source people. OOF video priors exclude the held participants' labels. This diagnostic is conditioned on development-selected alpha and familiar source films; it is not independent confirmation or unseen-stimulus evidence.

The declared cap is **45 main Ridge candidates plus 45 selected-setting source OOF fits**, with 72 bounded median-prior estimations across the main/OOF panels. Maximum two hours, **no neural fits and no GPU use**. Synthetic learning tests exercised the numerical building blocks; no original-data fit occurred. Runtime: Python 3.11.13, NumPy 2.4.4, SciPy 1.17.1 and scikit-learn 1.8.0.

## Processing, failures and decision rule

The source-informed development declaration includes the fixed native-rate 50-Hz/Q30 notch, followed by the preserved 30-channel, 0.5–45 Hz trial-local processing and 128-Hz resampling. The late core remains end−30..end−2. No global ICA/interpolation, augmentation, target adaptation or source/validation-pooled scaler is introduced.

Before target access, the future runner must authenticate and check source EEG. Source numerical failures or engineering flags hold fitting for source-only diagnosis. The original flags plus a source-informed per-channel 49–51/1–45 Hz ratio above 0.10 are review rules, not validated artifact labels. Validation quality flags are reported without deleting rows or selecting new filters. A numerically unavailable evaluated EEG input retains its trial and coverage: joint uses the selected context prediction, EEG-only the source global median. Original targets are never imputed.

Score one prediction per original trial, averaging clips within participant/film, films within participant, then people. Report ordinary MAE/MSE, acquisition breakdowns and leave-one-film sensitivity. Fixed paired crossed-bootstrap intervals are exploratory with six validation films and do not account for hyperparameter selection. They are not guaranteed nominal-coverage or confirmation intervals.

The preexisting **0.15 rating-unit** continuation threshold remains an engineering requirement, not a psychometric minimum. A development lead requires a mean joint gain at least 0.15 over selected context Ridge, a positive gain in at least two of three rotations, improvement beyond the strongest fixed unseen-context prior, and interpretable source capacity/coverage. Weak or ambiguous results are retained and trigger reassessment rather than another encoder grid. Meeting this rule would still require a closest-method novelty comparison and a concrete contribution proposal before a paper pivot.

## Remaining activation and publication boundary

This completes the user's requested **verification and declaration**. The pure selective decoder and numerical/identity contracts are tested and bound. An original-data execution runner and its immutable activation receipt still need implementation/verification before any real rating or broader EEG access. Those are routine work within authorized exploration, not a new user-approval requirement. The metadata audit does not establish that actual rating values are valid or useful.

The next technical phase is to implement and freeze that runner, prepare and audit the declared source features, then activate only the permitted item-10/source/validation controls if the source quality gate passes. Retain every result, fallback and coverage record. Confirmation requires a later locked analysis. The current manuscript/archive remain unchanged; any adopted new paper question still requires findings, explicit author approval and a fresh archive immediately before revision.
