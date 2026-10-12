# Independent review: category-supported source context selection

12 October 2026, Asia/Karachi. **Metadata-only review. No target, prediction, waveform or model-state values were accessed, and no new fit or diagnostic was executed.** This review concerns a separately declared second context-qualification stage, not a research-question adoption or confirmation launch.

## Verified reason for the second stage

Independent reconstruction from the existing development programme, catalogue and technical role records confirms the following for **each of rotations 0, 1 and 2**:

- Eight source titles and eight assigned categories.
- Exactly one source title per assigned category.
- Exactly 70 source person-title rows per category.
- The categories are Amusement, Anger, Disgust, Fear, Inspiration, Joy, Sadness and Tenderness.

Consequently, the first diagnostic's crossed two-title source holdouts also remove two category levels from training. Those predictions evaluate transport to absent source categories as well as absent films. The actual held-title development validation task retains source support for the held films' assigned categories. This is a substantive difference in the selection task, not a numerical or identity implementation bug.

A category-preserving source participant-held-out selection is therefore a defensible qualification of the contextual comparator for the actual information contract. Preserve the first stage, describe why the second stage was declared after it, and do not relabel either stage as independent confirmation.

## Bounded proposed recipe

Retain each rotation's original 70-person/eight-title source rectangle and the five existing 14-person folds. Each source selection fit trains on **56 people/eight titles = 448 rows**, and predicts **14 people/eight titles = 112 rows**. All eight source categories remain represented in training and assessment. Fit scalers, target offsets and contextual parameters only on the 448 training rows; no held-person or development-validation target enters them.

For category-only and full-context Ridge, retain the five existing alphas (.1, 1, 10, 100, 1000). Use accumulated participant/title-equal MAE across the five person folds and a frozen smaller-alpha tie rule. This is **150 selection fits plus six final source fits = 156 Ridge fits**. Compute global, coarse-valence and category median controls within the same training folds, then once on each full source panel: **18 prior constructions**, each producing all three prior types.

Choose one source-selected comparator among the two qualified Ridge models and the three source priors, using only these held-person source predictions and a frozen method tie order. Report both context representations, all alpha scores and all prior controls. A five-method winner chosen from validation MAE would violate this selection contract.

Keep both existing EEG anchors fixed. Reuse the unchanged spectral-joint and two-seed EEGNet-residual predictions with their original full-context branches, seed averaging, clipping and acquisition-failure fallbacks. Do not substitute newly selected contextual predictions into a neural residual offset or joint fallback: that would define a new EEG-dependent predictor. The already completed pairing diagnostics need not be repeated when neither anchor nor donor mapping changes.

Assess all original 120 unfamiliar and 160 familiar validation trials per rotation. Keep every one of the 15 unavailable-EEG outcomes and both anchors' exact fallback predictions. Compare against the newly source-selected comparator and, separately, the original matched context. Retain the same participant/title-equal MAE and 4,000 crossed-resampling descriptive intervals; group any multiple clips by source-title family before aggregation.

Freeze a new activation and all dependencies before execution. Preserve the preceding diagnostic, current exploratory manuscript, submission note, original report and sealed confirmation roster. No new original-rating or waveform decoding is required for this cache-based stage. Identity/family/role guards must precede numeric conversion of cached outcomes.

## Interpretation hazards

1. **Participant-only source selection is not an unseen-film estimator.** The held people see the same eight source films as the fitted people. Call it source participant-held-out selection on familiar source films. Unfamiliar-film generalization is assessed only on the separately reported, already reused development validation panels.
2. **Category and film are indistinguishable within each source rotation.** Category medians also act as source-film medians there. A strong category prior does not establish a learned semantic emotion category or individual experience. The held-title task tests transport of that contextual prediction to other films within supported categories.
3. **Nominal duration is also confounded with category within source.** Each category has one source title and therefore one nominal duration. Full-context duration coefficients cannot identify an independent duration effect from this design, and new films can move duration beyond the observed category-specific combination. Category-only versus full-context comparisons reveal combined predictive differences, not causal contributions of duration, acquisition or order.
4. **The second stage is adaptive.** Its design responds to the first diagnostic, whose development outcomes were available. Source-only alpha/comparator selection prevents a new direct validation-tuning step; it does not undo earlier development reuse or anchor selection. The descriptive intervals omit the uncertainty of the overall adaptive research programme.
5. **Three rotations are dependent.** They reuse the same people and overlapping materials. Do not pool them as three independent experiments or narrow an uncertainty interval by treating them as independent seeds.
6. **The anchor comparison remains bounded.** Matching a stronger contextual comparator does not establish a new EEG method, optimized neural convergence, absence of useful EEG information, or clinical relevance. The 0.15 engineering screen is not a validated perceptual margin.

## What would justify the next step

Report the category-supported selection alongside the original crossed selection and explain any ranking changes through the declared support difference. Retain adverse or inconclusive anchor results. A favorable comparison can motivate a reviewed claim and a separately frozen confirmation proposal; it cannot itself open the protected roster. If the EEG anchors have no reproducible incremental advantage, this stage still improves comparator qualification and the honesty of an exploratory report, but it does not justify an encoder grid by itself.

