# Independent design review: FACED target, context and EEG pairing

12 October 2026, Asia/Karachi. **Design review only. No targets, waveform values, fitted states or predictions were opened for this review, and no analyses or fits were run.** This document recommends a bounded development diagnostic under the existing information contract. It does not adopt a research question or authorize protected confirmation access.

## Fixed scope and scientific purpose

Keep the existing 70 source and 20 validation participants, 14 development title families, three dependent rotations, and 1,260 previously permitted target pairs. Each rotation has 560 source, 120 unfamiliar-validation and 160 familiar-validation trials. Keep the 265 available validation EEG trials and all 15 fixed unavailable-EEG fallbacks. The same people and overlapping films appear in several rotations; these are sensitivity panels, not three independent replications.

Answer three limited questions: what variation is present in the retrospective target; which supplied contextual information supports prediction; and whether correctly matching EEG to a person or film matters for two existing fixed predictors. Do not turn a swap result into a causal emotion claim or a null test of all EEG information.

## Target description without reliability claims

Describe the complete 70-by-14 source rating rectangle and, separately, the 20-by-14 validation rectangle. Show the grand mean, participant means, film means, and the two-way additive residual. Retain a squared-deviation decomposition and its exact reconstruction check. These are descriptive participant, film and remaining variation, not identified psychometric variance components.

A single rating per person-film cell cannot separate interaction, memory effects and measurement error, or establish response-style stability, test-retest reliability, an irreducible prediction floor or momentary affect. Do not call the residual "noise". Do not combine the two roles to estimate a predictor or a transform.

## Bounded context qualification

Compare category-only Ridge using the eight existing category indicators with full-context Ridge using the existing 13 features. Keep category availability at inference; this is an ablation, not a category-unavailable deployment claim. Retain the source global, coarse-valence and category medians. Familiar-film medians remain a secondary familiar-arm comparator only.

For a source-only selection recipe, use the five existing source participant folds crossed with four disjoint two-title folds among each rotation's eight source titles. Construct title pairs deterministically from metadata before outcome access, preserve them in the declaration, and never change them after scores. Every fit trains on 56 participants and six titles and evaluates 14 participants and two held titles. Across the 20 cells, every source person-title target is evaluated exactly once per candidate. Fit all preprocessing and target offsets only on that cell's training pairs. Fixed metadata vocabularies may describe categories; numerical standardization may not use held rows.

Reuse the five existing alphas (.1, 1, 10, 100, 1000). Select separately for the two context matrices on accumulated participant/title-equal source MAE with the existing smaller-alpha tie rule. This yields **600 small closed-form selection fits plus six final source fits, cap 606**, with no neural fitting or validation-driven model selection. Compute source-fold priors within the same training cells; retain their separate construction count. Report both context matrices and all candidate source scores. A strongest context comparator may be selected among these two and the three fixed source priors using the same source-held-out predictions, with a frozen deterministic tie rule. Do not select it from development validation results.

These source scores are adaptive development evidence, not independent estimates: their targets have been used earlier in the programme. This recipe does not retroactively replace the old validation-selected context alphas, the EEGNet residual offsets, or the manuscript's historical results.

## Fixed EEG anchors

Use only the original selected spectral joint Ridge and the original selected two-seed EEGNet residual predictor for each rotation. Keep their features, fitted normalization, hyperparameters, checkpoint schedules, context branches, clipping and fallbacks unchanged. No anchor is selected by a swap result and no new encoder is fit here. Compare each with its original matched context and the newly source-selected context comparator; label the comparisons separately.

Before swapping, reproduce every original unmodified anchor prediction through the exact fitted state and input bridge, and compare against its retained row. The EEGNet branch needs a true state-to-window forward replay, which preceding integrity checks did not independently establish. Freeze numerical tolerances using an outcome-free same-device replay fixture; a mismatch holds that branch rather than allowing a correction selected by favorable outcomes. If the neural replay cannot be qualified, report the spectral diagnostic alone and the neural dependency explicitly.

## Pairing controls

Declare **32 fixed seeds, 20261210 through 20261241**, shared across both anchors. Use the same donor mappings for both anchors and record identities before evaluating predictions. All donor pools are within the current validation role and the same familiar/unfamiliar arm; never borrow source, protected, unavailable-EEG or cross-arm donors.

1. **Same film, other person:** derange available participants within exact film and acquisition-group cells. Keep the recipient's target and complete context. This preserves shared film and coarse acquisition information while disturbing individual matching. If a cell has fewer than two available participants, retain that recipient's original input and report it as unperturbed; do not remove it or widen the donor pool after viewing scores.
2. **Same person, other film:** derange available films within recipient and assigned coarse-valence cells, retaining the same familiar/unfamiliar arm. Keep recipient context and target. This preserves participant identity and broad assigned valence while disturbing exact film matching. Small cells remain unperturbed and are reported. Duration, category and acquisition-related film differences can still change the input distribution, so degradation is not specific evidence of emotion decoding.

The original 15 unavailable recipient trials use their exact context fallbacks for every repeat, even when a valid donor exists. Available recipients may use only available donors. Permute complete trial EEG blocks or spectral vectors; do not permute isolated channels/windows, fit a donor-derived transform, or alter the target.

Spectral joint prediction must concatenate **recipient context plus donor EEG features**. EEGNet residual prediction must attach **recipient context offset plus donor neural correction**, preserving seed ensemble-before-clipping. Reusing a donor's clipped joint prediction would also substitute its context and is invalid.

For each repeat compute absolute errors, then average those errors per recipient across repeats. Do not average swapped predictions first, which would create a new ensemble. Retain every repeat's donor, prediction, error, eligibility and fallback branch. Report mean paired MAE degradation versus correct pairing and the repeat distribution; 32 repeats provide numerical stability, not 32 independent experiments.

## Metrics, budgets and stopping rules

Primary descriptive endpoint is participant/title-equal MAE on unfamiliar validation titles. Familiar validation, source-held-out scores, per-title effects, acquisition strata and availability-conditioned sensitivities are secondary. Retain every outcome in the primary denominator. Use the existing 4,000 crossed participant/title resamples and fixed seed for paired differences, applied to recipient errors averaged across swap repeats. Keep their development and small-film uncertainty limits explicit. Do not use the swap repeat distribution as a permutation p-value: the regression residuals are not established exchangeable, and donor dependence remains.

Recommended hard bounds: 606 new tiny context Ridge fits; no new neural fits; 32 repeats for exactly two swaps and two anchors; at most 107,520 validation swap rows; 30 CPU minutes for target/context calculations after a verified bridge; two hours for any necessary authenticated signal preparation plus 30 GPU minutes for replay/inference. Declare preparation and inference time separately. Preserve partial output on a cap, schema, numerical or identity failure; no automatic retry, schedule extension or additional swap/model arm. A branch that cannot reproduce its unmodified predictions is held.

Freeze the executable declaration and its dependencies before analysis. Guard identity and role before converting target fields; read only the existing permitted target projection. Authenticate original/feature/state caches, replay the source quality gate before validation signal access, and reject protected identities before numerical decoding. Keep waveform/features/model states private; publish only the already authorized anonymous predictions, donor identities and technical receipts.

After this diagnostic, stop and interpret all declared conditions. A favorable panel alone does not activate an encoder grid, confirmation access or a new paper claim. Any bounded longer neural schedule belongs to a separate declaration informed by source/development learning behavior. Present the resulting claim and fixed confirmation proposal for author review before adopting a research-question change.

## Concrete dependencies before launch

- Exact frozen participant/title source-fold mapping, context selection tie rules and metadata feature schema.
- Verified fitted-state/feature bridge and original prediction forward replay, particularly for EEGNet residual correction before clipping.
- Availability-aware same-arm donor derangements with explicit singleton handling and recipient fallback invariance.
- Synthetic tests for source-only transforms, identity-before-target guards, donor constraints, complete denominators, error-averaging rather than prediction-averaging, and fixed context in both predictors.
- A declaration distinguishing the new source-selected contextual comparison from the earlier adaptively chosen anchors and documenting all fit/time/row caps.

