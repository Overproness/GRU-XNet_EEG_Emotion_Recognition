# Matched full-fold DEAP stimulus, baseline and context findings

Completed 9 October 2026. **The high baseline-relative validation lead did not carry over to full-fold testing.** Baseline correction improves loss over absolute stimulus power in the unseen-video arm, but does not establish better prediction than relative stimulus power or baseline alone. EEG-plus-context does not improve on the contextual controls in these configurations. The manuscript and research question remain unchanged; the paper is still not submission-ready.

## Completed grid and evidence

The author approved this follow-up to the [small-panel diagnostic](GRU-XNet_Preprocessing_Diagnostic_Findings_2026-10-09.md). The [protocol](DEAP_Baseline_Folds_Protocol_2026-10-09.md) covers **160 matched cells**: two existing participant/video groupings, five video rotations, eight participant folds and both video-exposure arms. Every model/arm/group covers all **1,264 retained original trials from 32 people and 40 videos exactly once**. Sixteen valence-five trials remain excluded. No labels were changed.

Four native 32-electrode representations each have EEG-only and EEG-plus-context logistic controls: absolute stimulus log band power, within-channel relative stimulus power, pre-stimulus baseline alone, and stimulus-minus-baseline log power. Raw and calibrated context alone are retained. The first forty stimulus seconds and measured three-second baseline use the preceding diagnostic's separately filtered Welch features. This is established preprocessing under a declared local adaptation, not an exact published-score reproduction or novel method.

All **5,760 candidates** independently refit, selecting **1,440 logistic heads plus 160 fixed raw priors**. Scalers use training rows only. Every training prior excludes the receiving participant's entire label set. Equal familiar/unseen source-validation balanced log loss selects C; each choice is sealed before its outer-test feature/prior query. Complete test predictions contain **50,560 rows**. All 32 earlier source-candidate/scaler/coefficient comparisons and 320 earlier raw/calibrated context test comparisons pass.

The [full fit verification](../../results/development/deap_baseline_folds_v2_2026-10-09/verification.json) independently checks **23,680 probability metric sets**, with exact coefficient/scaler refits and zero direct probability replay error. The [analysis verification](../../results/development/deap_baseline_folds_v2_2026-10-09/analysis_verification.json) verifies all **120 regular metric points, 240 paired contrasts and both participant/crossed percentile endpoint sets** using a different flat-trial weighting calculation; largest discrepancy is 4.39e-16. The 40 dyadic within-video contrasts use the previously audited pairing design, with an additional synthetic comparison against its separate pairwise implementation and required context invariance. This is not a second full-data independent alignment-bootstrap implementation.

Twenty-seven relevant tests passed before the corrected grid; three additional schema, regular-interval and dyadic checks pass. Every common-electrode prefix regenerates exactly from the mirror, as do all 740 native/baseline overlaps with the preceding diagnostic. Raw EEG, per-trial features and coefficient arrays remain local. Verified evidence was committed/pushed every ten new cells and on completion; the completed experiment milestone is `4b68337a3`.

## Complete combined results

These are outer-fold **development** test scores on previously inspected people/videos. Combined estimates average correctness/loss within observed participant/video across groupings, not probabilities or unequal fold accuracies. Higher balanced accuracy and lower balanced log loss are better.

| Model | Familiar-video BA | Unseen-video BA | Familiar loss | Unseen loss |
| --- | ---: | ---: | ---: | ---: |
| Absolute stimulus EEG | 48.85% | 47.31% | 0.7226 | 0.7324 |
| Relative stimulus EEG | 52.31% | 51.31% | 0.7041 | 0.7058 |
| Baseline EEG alone | 49.26% | 49.01% | 0.7122 | 0.7086 |
| Stimulus-minus-baseline EEG | 51.51% | 51.15% | 0.7055 | 0.7022 |
| Absolute stimulus + context | 74.24% | 48.39% | 0.5657 | 0.7472 |
| Relative stimulus + context | 76.07% | 50.89% | 0.5474 | 0.7255 |
| Baseline alone + context | 75.30% | 49.05% | 0.5577 | 0.7186 |
| Stimulus-minus-baseline + context | 76.59% | 50.87% | 0.5479 | 0.7148 |
| Calibrated context, without EEG | 78.05% | 47.50% | 0.5004 | 0.6953 |
| Raw context, without EEG | 77.54% | 49.67% | 0.4991 | 0.7062 |

The [complete comparison](../../results/development/deap_baseline_folds_v2_2026-10-09/comparison.json) preserves every grouping, point, interval and contrast. The [figure](../../results/development/deap_baseline_folds_v2_2026-10-09/plots/matched_deap_controls.png) was visually checked. Its blue intervals cross participants and videos; orange/green points show each grouping separately.

## What survived the earlier lead

Stimulus-minus-baseline unseen-video BA is **51.15% with crossed 95% percentile interval [47.47%, 54.82%]**. The two grouping points are **50.16% and 52.14%**, versus earlier selected source-validation panels' 72.35% and 57.29%. These are different evaluation populations/roles, not a recalculation or replacement of those validation scores. The complete comparison gives no reliable above-chance accuracy advantage under this fixed-fit uncertainty. Its point loss 0.7022 is also higher than uniform binary prediction's approximately 0.6931; the crossed loss interval [0.6879, 0.7172] includes that reference.

In the unseen arm, baseline-relative versus absolute stimulus power improves loss by **-0.0302 [-0.0550, -0.0083]**. Its accuracy difference is +3.84 percentage points [-1.60, +9.06]. Relative to stimulus-relative power, loss is -0.0037 [-0.0253, +0.0156]; relative to baseline alone, -0.0065 [-0.0220, +0.0092]. Those latter intervals span zero. Thus correction helps one weaker representation's probabilistic performance; it has not established a distinct useful recognition method or an advantage beyond the relevant alternative feature controls.

On familiar videos, baseline-relative EEG-plus-context is **76.59% versus 78.05% for calibrated context alone**, paired BA difference **-1.46 pp [-3.61, +0.81]**. Its loss is worse by **+0.0476 [+0.0165, +0.0717]**. All four EEG-plus-context familiar-video configurations have worse loss than both raw/calibrated context under the declared crossed intervals. Their high accuracy is insufficient evidence of useful incremental EEG prediction. Unseen-video baseline-relative EEG-plus-context is 50.87%; its loss differences versus raw and calibrated context do not support an improvement either.

The [within-video study](../../results/development/deap_baseline_folds_v2_2026-10-09/alignment.json) includes all 2,528 repeated trial rows per model and 7,488 recipient/donor pairs, with no ineligible rows or nonestimable draws. All forty crossed aligned-minus-exchanged intervals span zero. For unseen-video baseline-relative EEG, alignment changes BA by -0.07 pp [-4.26, +3.89] and loss by -0.0022 [-0.0169, +0.0117]. The raw/calibrated context controls are invariant as required. This does not establish equivalence, general absence of EEG information or a causal explanation.

## Preserved implementation failures and scope

The initial preparation stopped before fitting because exact binary-float equality rejected three CSV/workbook rating differences of 4.44e-16. Revision 2 restores the previously verified absolute 1e-12 tolerance, with no class-label changes or scientific-factor changes. The original source/declaration/failure remains preserved and excluded.

The ninth first-cell verifier then encountered older probability column names (`p_0/p_1` versus `p0/p1`). A separately declared supplementary verifier preserves every fitted selection/record and uses the correct columns. The older bootstrap receives explicit column renaming; its coverage checker receives fixed random-state42 metadata in a temporary frame. Those schema adapters preserve all frozen fitting/selection/analysis sources and every declared mathematical comparison. No aggregate score preceded the analysis adapters. Recovery declarations and original failures are retained in the public study folder.

Intervals use 10,000 paired fixed-fit draws over the same observed participants/videos, remain unadjusted and exploratory, and do not include refitting uncertainty or new independent humans. The linear feature family does not test all EEG representations, native neural temporal dynamics, arousal targets, personalized recognition or joint three-corpus transfer. First-party DEAP recording authentication remains outstanding; mirror/metadata/cache agreement has its narrower scope. Offline full-trial filtering and these local baseline conventions must not be presented as causal early emotion detection.

## Research decision

**Do not escalate this baseline-relative feature lead into a large neural grid or a new paper question on these results.** The selected-panel accuracy lead failed to establish dependable full-fold recognition or improvement beyond context. A loss improvement over absolute power alone is insufficient novelty, and a generic context-dominance paper already faces close prior work.

The next useful decision should focus on a trustworthy EEG representation/task rather than another search over successful seeds or layers: assess one strong modern EEG representation with its documented preprocessing and target-pretraining-overlap limits, and review which existing corpus/native target can support a clear physiological prediction question. Use small source-only diagnostics before any broad new grid. That is a proposed direction for authorized exploration, not a contribution already established or an adopted research-question change.

Keep the current paper archived and unchanged. A concrete alternative contribution needs a demonstrated gap against closest prior work, a credible positive baseline and locked independent confirmation. Present that evidence/proposal to the author and obtain explicit approval before adopting a different paper question; archive the then-current manuscript immediately before the approved change. Historical-score/figure/reference repairs, first-party authentication and the remaining cross-corpus/ablation questions are still unresolved.
