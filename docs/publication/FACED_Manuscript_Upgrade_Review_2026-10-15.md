# Independent manuscript upgrade review

15 October 2026. Scope: the saved 12 October exploratory manuscript and supplement, public 13 October coverage findings, public 15 October EEGNet source findings and independent-audit description, the current claim decision, and HEART qualification. No raw EEG, private states, original ratings, validation/protected numerical data, or refitting were used. This is a writing and evidence-scope review, not independent waveform replay or scientific replication.

## Required structure for the new revision

Preserve the 12 October paper exactly and place the update in a new dated directory. Retain the approved question: whether particular EEG recipes improve individual retrospective item-10 valence prediction beyond supplied elicitation context. The latest evidence strengthens a bounded adverse-result interpretation; it does not establish a new architecture, absence of emotional EEG information, or an independently confirmed negative effect.

Present three evidence blocks with separate denominators, selection histories and captions:

| Evidence block | Training/evaluation scope | Scientific status |
| --- | --- | --- |
| Earlier development-validation rotations | Each rotation: 560 source trials; 120 unfamiliar-title and 160 familiar-title trials from the same 20 development-validation people | Three overlapping, adaptively reused rotations; existing predictions and descriptive intervals remain unchanged |
| Spectral coverage comparison | Ten crossed source cells; each 448 fit/84 held; 840 unique held participant/title pairs across 70 source people and 12 held titles | Source-only recipe/coverage selection after earlier programme adaptation; shared alpha 100 selects distributed coverage without a positive increment |
| Extended EEGNet learning/capacity diagnostic | One metadata-first quality-complete source cell; 448 fit from 56 people/eight titles; 84 held from 14 other people/six other titles | Reused source development; neither an all-fold neural result nor a new validation or reserved confirmation panel |

The spectral and neural source scores cannot be pooled or compared as if their evaluation panels were identical. The 360-epoch diagnostic does not update the earlier validation predictions. All three blocks lack held source material-transfer outcomes for Fear and Inspiration.

## New numerical results and safe interpretation

For the 840-pair shared-policy source comparison, report category median MAE 1.30397, matched context at alpha 100 1.36650, late/context 1.46462, and distributed/context 1.44554. Distributed improves on late by 0.01907 but trails matched context by 0.07904 and the category prior by 0.14157. Context's better standalone alpha 0.1 gives MAE 1.34496. Thus common-alpha matching is not independent optimization of the strongest context comparator, and the small coverage preference depends on regularization. No confirmatory interval should be invented for these reused source outcomes.

For the 84-pair neural diagnostic, report category median 1.09939, matched-context Ridge 1.29535, and the fixed two-seed 360-epoch EEGNet/context ensemble 1.40748. Its context-relative gain is -0.11214; category-relative gain is -0.30809. Fixed checkpoint ensemble MAEs at 120/240/360 epochs are 1.37689/1.37587/1.40748; every one is worse than both controls. Preserve 360 epochs as the declared primary endpoint rather than selecting the best intermediate score. The two final individual seed MAEs are 1.44259/1.41286.

Unbounded eval-mode training-offset MSE falls from the offset-only reference 3.03138 to 0.86470/0.87827 at epoch 360, reductions of approximately 71.5%/71.0%. Between epochs 240 and 360, the two trajectories fall from 0.99215/1.09557, respectively. Both continue improving by more than 10%, so optimization has not demonstrated a plateau. The distinct 32-trial real/shuffled controls end at MSE 0.10580/0.13969 versus 6.11297 offset-only. This verifies small-set fitting capacity for this implementation and optimizer; it does not show useful affective information or full 448-trial interpolation capacity.

Suggested results wording: "Longer fixed training substantially reduced source error but did not produce a favorable EEG increment on this reused source cell. Source error was still decreasing, so full convergence remains unproven. The pattern is compatible with overfitting or poor transfer; this diagnostic does not distinguish their causes."

Training MSE, clipped held MAE and the capacity-control MSE concern different targets/sets and must not share an unlabeled axis. A compact checkpoint table is clearer than suggesting training error predicts held generalization. Training minimizes MSE while primary utility uses MAE; retain this loss/endpoint mismatch as a limitation of the tested recipe.

## Methods and availability corrections

The existing statement that all 980 source trials pass is scoped to the late-only interval. In the shared three-region policy, one middle interval (sub-087, clip 20) fails the unchanged threshold; the entire trial is unavailable to both EEG representations. Shared-policy training retains all 980 outcomes/context rows with feature means estimated only from the 979 available EEG donors. The failed trial is training-only, so all 840 actual held pairs have available EEG; no empirical held failure result follows. The old 15/280 validation failure counts remain a different panel.

Early/middle/late 28-second cores overlap and do not span the full movie. Distributed features average the three already-logarithmic 150-dimensional vectors. Requiring all three regions for both EEG arms controls availability, but their contrast does not estimate recording cost or latency. Shared-policy matched context also consumes EEG-availability routing, so category median is the practical acquisition-free reference.

Disclose the extended neural adapter and source design: author-checked EEGNet-8,2 convolutional backbone; local 1,841-parameter scalar residual adaptation; seven late four-second windows with one loss per original trial; source-only channel scaling; thirteen context columns; alpha 100; four participant-excluded inner source folds for training offsets and one full-training context fit for held offsets; two 360-epoch main initializations at learning rate 0.001, dropout 0.5, AdamW weight decay 0.01, batch eight. Two additional 200-epoch tiny controls use real and one fixed shuffled target assignment, without held inference. These are four new neural trajectories, separate from the older twenty-fit programme.

Selection of a quality-complete source cell used existing metadata rather than labels, but it is not a random representative sample. Do not describe source-only fitting as removing programme-level adaptation.

## Audit, reserved cohort and comparator boundaries

The new independent audit checks 6,260 public prediction rows, 532 distinct source targets, five contextual normal equations, OOF identities/offsets, priors/permutation, ten tensor states/19,240 scalars, learning flags and reported scores. The producer additionally reloaded all ten checkpoints and compared forwards on fixed source subsets. Neither exercise independently reacquired all waveforms, reconstructed all channel-scaler moments, or replayed every neural forward. Describe those two checks separately. The older supplement's public-cache-only audit paragraph still applies to the earlier evidence activation, not to all subsequent source experiments.

The durable guard's 3,036 manufactured predictions are software qualification, not data or scientific findings. A short reproducibility/future-work mention is sufficient. The 33 reserved participants remain sealed, no result from them supports the paper, and the future four-recipe confirmation remains inactive. Do not claim independence, adequate future precision, or statistical equivalence from zero-crossing reused-development intervals.

EEGNet is an authenticated compact reference, with local regression/preprocessing/optimization adaptations, not the strongest available field method or a reproduced published FACED regression score. Stronger venue prospects still require credible comparator/loss qualification and participant/material replication fixed before further evaluation; they cannot be claimed from engineering safeguards alone.

## Closest-method addition

The new paper needs a HEART citation. The [authors' university record](https://recerca.udg.edu/ca/publications/heart-hybrid-embedding-for-affective-and-riemannian-trajectories/), rechecked 15 October, lists Jonah Fernandez, Bianca Innocenti and Beatriz López; *IEEE Transactions on Affective Computing*, 2026, accepted/in press, DOI [10.1109/TAFFC.2026.3729335](https://doi.org/10.1109/TAFFC.2026.3729335). It confirms continuous VAD regression and categorical prediction on FACED. The limited primary-method record does not authenticate original individual item-10 supervision, its mapping, reserved-material scope, or equivalence of the accessible preprint and accepted version.

Suggested text: "HEART combines continuous VAD regression with categorical prediction on FACED. Its exact correspondence to the individual item-10 target and catalogue-held evaluation used here remains unverified; we make no first-regression claim." Cite the article and record without comparing its classification accuracy with this paper's scalar MAE. Participant separation, out-of-fold fusion and temporal EEG representations are not inventions of this evaluation.

## Abstract, discussion and conclusion priorities

The abstract should include the new fixed-source adverse comparisons and capacity/convergence qualification, while clearly naming their source-development scope. Do not leave the favorable older spectral mean as the sole headline after the adverse coverage and neural source findings. It is acceptable to shorten old panel details in the abstract and keep them in the main results.

Keep the main claim exploratory and recipe-specific. Neither limited gains nor their intervals establish equivalence; shuffled-label fit cannot establish emotion specificity. Reserve causal language about emotional mechanisms and general negative conclusions about EEG. Material/annotation ancestry, catalogue rather than physical-footage grouping, one retrospective rating per cell, supplied category metadata, six held categories, adaptive reuse, narrow method coverage and continuing source learning remain substantive limitations. The current manuscript can be more complete and credible without claiming that these updates alone raise it to a stronger venue tier.

## Review of the actual 15 October revision

Read the new main manuscript, supplementary methods, source-followup table and bibliography after the revision was assembled. No scientific blocker or implicit independent-confirmation claim was found. The three evidence blocks are now explicit, the abstract retains the later adverse findings, the 360-epoch endpoint is preserved, category/availability limits are clear, HEART is appropriately qualified, and all 33 reserved people remain sealed. The displayed new MAEs, source MSEs, checkpoint reductions and capacity results agree with the reviewed public findings. Moving additive target decomposition to the supplement and replacing the historical selection display with prose retains its scientific boundary.

Three small precision edits were recommended before finalization:

1. Main Results: describe the 43.3%/42.3% additive title terms as shares of **squared** rating deviations, rather than rating deviations generally.
2. Supplement Audit boundaries: scope the zero-fit/public-cache-only evidence activation explicitly to the **earlier validation/pairing recomputation**, so it cannot be read as the access scope of the new source experiments.
3. In that same section, identify the 24-forward/six-state replay as qualification of the **earlier selected states**, distinguishing it from the later ten-checkpoint reload evidence.

Optional wording refinement: replace “All 980 training rows remain” in the main shared-availability paragraph with “All 980 original source trials remain eligible for fitting”; each CV fit actually contains 448 training rows, while the full-source state qualification contains 980. The supplement already specifies this correctly.

Disposition: suitable for the requested exploratory current-progress revision after those precision edits and the root agent's build/layout checks. This is manuscript review, not endorsement of venue suitability or independent replication of the empirical result.

