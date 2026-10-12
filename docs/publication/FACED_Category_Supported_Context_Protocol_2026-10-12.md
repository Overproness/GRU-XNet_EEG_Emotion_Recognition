# Bounded qualification of category-supported contextual controls

12 October 2026. This separate development declaration follows the completed target/context/pairing diagnosis. It qualifies the contextual baseline under known category metadata without new EEG fitting, new raw access, protected outcomes, paper edits or adoption of a new research question.

## Why this follow-up is needed

Each rotation has eight source titles and eight assigned categories, one title per category. The previous two-title source holdout consequently removes the held titles' categories from fitting. Category medians then fall back to a global median and a one-hot Ridge cannot estimate the missing category effect. That selection setting differs from predicting another film whose assigned category was observed in training. Preserve the previous diagnosis and all its outcomes; do not reinterpret its coarse-valence-selected comparator as the strongest qualified category-supported control.

This correction was chosen after inspecting development results and metadata. It is an adaptive development follow-up and cannot supply independent confirmation. It does not establish that category-supporting folds are intrinsically better for every deployment setting. With one source title per category, duration/title and category are also confounded; the full-versus-category ablation cannot identify a distinct causal duration effect.

## Fixed comparison and selection

Reuse the same 70 source and 20 validation people, three rotations, 14 development titles, categories, 13 context features, clipping and fixed unavailable-EEG fallbacks. Read cached ratings only from the authenticated completed diagnostic's `cv_predictions.csv` (source identities) and `predictions.csv` (validation identities), checking identity, family and acquisition before conversion and exact repeated agreement. No original rating files or private EEG arrays are accessed.

For each rotation, use the five existing source-person folds: fit on 56 people and all eight source titles (448 pairs), predict 14 held people on those same eight titles (112 pairs). Verify the complete source rectangle and that every training fold retains all eight categories. Fit numerical scaling and median priors on training pairs only. These folds test held-person prediction on familiar source titles; **they do not estimate unseen-title generalization**. That limitation is essential because there is only one source film per category.

Qualify category-only Ridge (eight indicators), full-context Ridge (13 columns), source global, coarse-valence and category medians. Use the existing five Ridge alphas .1,1,10,100,1000. Accumulate source held-person MAE across the five folds, select each Ridge by MAE then smaller alpha, and select the contextual comparator among five methods with tie order category-only, full-context, global median, coarse-valence median, category median. No validation score selects a comparator, alpha or epoch.

Fixed bounds: **150 source-person CV Ridge fits plus six final source fits =156**, **15 fold-prior plus three full-source constructions =18**, 21,840 CV rows including priors, and 5,880 ordinary validation rows for the five contexts/priors and two retained EEG anchors. CPU time cap:900 seconds. No new neural fits or altered EEG normalization/preprocessing. Persist partial ledgers on failure; no automatic retry or expansion.

Copy the original selected spectral-joint and EEGNet residual anchor predictions from the completed diagnostic; do not refit or reselect them. Compare both anchors with the source-selected contextual comparator on the fixed unfamiliar and familiar validation rectangles, preserving all targets and fallback rows. Report all five context conditions to expose comparator sensitivity. Use the existing 4,000 crossed participant/title resamples, seed425401, for descriptive paired gains. The EEG anchors were selected earlier using development validation; source-only context selection does not remove that historical adaptation.

## Preservation and interpretation

Freeze exact person-fold training/held identities, metadata, preceding numerical packet, code and original manuscript hashes in a separate activation before new fitting. The prior experiment remains immutable. Authentication of existing caches is sufficient here: no original waveform replay or new biological interpretation is activated.

Interpret this qualification alongside the original target decomposition and pairing controls. More convincing numerical gains over a poorly supported baseline are insufficient for a claim of emotion-specific EEG information. Present a proposed claim and independent confirmation design to the author before adopting a new research question. The 33 reserved participants and eight title families remain sealed throughout.
