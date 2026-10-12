# Frozen, inactive FACED confirmation protocol

12 October 2026. **Preparation only. No confirmation signal or rating access, final source fit, manuscript edit or research-question adoption is authorized by this document.** The completed development diagnosis and category-supported baseline qualification remain unchanged.

## Selected present claim

In the reused FACED development panels, source-selected category medians are competitive with the two tested late-interval EEG predictors. Their gains and person/film pairing effects are inconsistent or uncertain. Category support materially changes which contextual baseline source selection qualifies.

This is the defensible current empirical claim. It is not independently confirmed superiority, equivalence, absence of EEG information, causal emotion decoding or architectural novelty. It is selected for author review; adopting it in a manuscript still follows the author's explicit approval/archive requirement.

## Candidate future question and information contract

On the reserved FACED catalogue roster, does a specified spectral-plus-context Ridge recipe improve retrospective valence prediction over source category medians, and does a matched context-only control support attributing a difference to adding the spectral representation?

The primary comparison measures **whole-recipe predictive utility**: joint Ridge receives category, acquisition, duration, order and EEG, while category medians use category alone. It does not isolate EEG information. A mandatory matched context-only Ridge contrast provides a separate check of adding the spectral features within the fixed regression recipe. Neither comparison identifies a causal affective mechanism.

Freeze the original 70 source participants and 14 development clips/title families, giving 980 clean source pairs. The 20 development-validation participants inform historical development choices and do not enter final source fitting. Source transforms, priors, intercepts and coefficients use exactly the source rectangle. The 14-title roster supports all eight categories, but Fear and Inspiration each have only one source title. Material-preserving category cross-validation is consequently unavailable for all categories.

Protected scope remains 33 participants: 297 unfamiliar clip trials across eight catalogue title families/nine clips; familiar secondary scope 462 trials across 14 development titles. Clips 7 and 8 share a family. Catalogue title separation is verified; independence of the original footage is not authenticated. Conditions concern this fixed roster, not an arbitrary movie population.

Target: item 10 retrospective whole-clip valence, finite range 0–7, no thresholding, target-based selection or imputation. Category is supplied elicitation metadata, not the individual's emotional label. Invalid targets stop the attempt and preserve its ledger.

Signal: existing 32-second native late interval with 28-second filtered/resampled core (end−30 to end−2 seconds), 30 common scalp channels, 128 Hz, seven 512-sample windows. Use the existing calibration, notch/bandpass, pre-CAR engineering artifact control and availability policy `faced-quality-context-fallback-v1`; at most three channels may be interpolated under the frozen engineering rules. No whole-video claim or neural encoder is included.

## Fixed future recipes and resource bounds

- Category comparator: median of the exact source ratings within assigned category, source global median fallback if a category is absent. No protected outcome informs it.
- Joint predictor: 13 context plus 150 spectral columns, source-only StandardScaler, Ridge with intercept, SVD solver, alpha 100, clipping to[0,7]. Alpha 100 was selected in all three historical spectral panels; it is a fixed development-informed recipe, not a demonstrated optimum.
- Matched control: the same 13 context columns, source identities, per-column source scaling, intercept convention and alpha 100. Its context scaling must agree with the context portion of the joint scaler.
- On unavailable protected EEG, joint and matched-control predictions both use the source category median. Retain every target in the denominator; paired matched gain on these trials is zero. This new final recipe is distinct from the historical anchors' full-context fallback, which remains unchanged in completed packets.

Prospective resource caps, not launched: two Ridge fits and one median-prior construction;60 minutes for source cache/identity qualification;15 minutes for the two final fits and state replay; four hours for permitted protected signal/QC preparation;15 minutes for inference;15 minutes for final scoring. No neural fitting, tuning grid, source-role reassignment or automatic retry. Any cap, annotation, identity, calibration, QC or state failure preserves partial evidence and stops the attempt. A changed budget requires a new recorded declaration before further work.

These bounds do not make a backend available: the current implementation is a pure guarded analysis/information contract. Original-file extraction, final source fitting and end-to-end protected inference are not implemented or launched by this freeze.

## Metric and interpretation fixed before access

Compute each clip's absolute error first. Average clip losses within person/catalogue family (including clips 7/8), then weight title families and people equally. Do not average predictions before computing losses. Primary gain is category-median MAE minus joint MAE on unfamiliar titles. Mandatory matched-context gain is context-only Ridge MAE minus joint MAE. Familiar outcomes are descriptive secondary.

Use 4,000 paired crossed person/title resamples, seed 425401, and percentile 2.5%/97.5% bounds. These are descriptive sensitivity bounds on a small, deliberately selected catalogue roster. They are not calibrated 95% confidence for arbitrary future films. A positive interval can support a direction under the declared resampling assumptions; it cannot establish a general EEG-information claim. An interval including zero is inconclusive and cannot establish equivalence. Do not select a different favorable contrast, interval, title subset, seed or target after confirmation.

There is one primary whole-recipe contrast. The matched-control contrast is mandatory but secondary and descriptive. If only the primary favors joint prediction, do not attribute the improvement specifically to EEG. Evidence favoring both remains conditional on these recipes, information inputs and roster. Retain adverse and null outcomes and the previously unfavorable neural development results.

Separate-axis resampling is an established approach to crossed data. Its theoretical properties rely on assumptions and do not certify small-roster percentile coverage: see [Owen 2007](https://arxiv.org/abs/0712.1111) and [Owen/Eckles 2012](https://arxiv.org/abs/1106.2125). The synthetic sensitivity study therefore checks declared scenarios rather than treating resampling as a guarantee.

## Protected access and prediction-lock gates

The metadata contract remains `frozen_not_activated`. Any future signal access needs a separate activation binding this exact contract, explicit author question approval and confirmation-access approval, an adequately reviewed precision receipt, source cache/QC qualification, an empirical continuation decision and qualified final model states. All current gates remain false. Existing source cache authentication must not be called a fresh all-source waveform replay.

Protected signals and protected ratings are separate access phases. Before any protected rating conversion, all three predictor exports for the complete 297+462 rectangles must be locked to file hashes, exact identities, final source states and prediction-only value projections. Rating access and scoring must reject changed predictions, missing trials, role/arm changes or missing locks before reading numeric targets. Protected predictions are committed before protected rating access.

The empirical continuation screen from the previous programme remains unmet. A synthetic precision result does not clear it. A new uncertain-result framing does not silently bypass it. If a future scientific purpose justifies a different continuation criterion, that requires an explicitly reviewed and author-approved new protocol; this frozen version stays inactive.

## Synthetic precision sensitivity authorized now

The independently specified study uses only reservation metadata and analyst-assumed independent Gaussian subject/title/interaction effects in paired loss differences. Four variance allocations,33×8 and three hypothetical larger layouts, true gains 0 / 0.045 / 0.15, 256 matrices per layout/scenario and 512 crossed draws yield 4,096 synthetic matrices,2,097,152 bootstrap means and 48 reporting rows. Code/design/plan are hashed before simulation. Five CPU/wall minutes are allowed; no empirical fit, target, prediction-array, private feature, waveform or protected-value read is permitted.

The 0.045 sensitivity value was supplied from published development summaries, so this is not a blinded design. Variance components are analyst assumptions rather than estimated FACED quantities. Unbounded Gaussian loss differences, exchangeable titles, omitted training/selection uncertainty and finite Monte Carlo resolution limit interpretation. Hypothetical 66-person/16-title layouts neither exist as additional FACED confirmation data nor authorize reassignment. The simulation never marks empirical precision or continuation gates passed.

## Preservation and output

Freeze exact identities, categories, grouping, code, metadata, prior audit receipts, manuscript hashes, budgets and gates in the public preparation packet before synthetic execution. Keep the existing manuscripts and historical report byte-identical. Publish the synthetic assumptions/results, guard tests and independent review, with a completion receipt. No researcher contact, paper submission, payment or new data collection is part of this work.

The final decision after preparation is whether the author wants the bounded exploratory evaluation framing or a better-supported public-data question. The original question is not changed by selecting a defensible current claim or freezing an inactive candidate protocol.
