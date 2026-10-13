# Shared missing-EEG handling: qualified comparison, no incremental source lead

## Actual result

The new shared, label-blind policy qualified and the full bounded comparison finished: **90 CV Ridge fits, three source-state qualification fits and eleven category-prior bundles**. All 980 original source rows, context values and self-reports were retained. The QC thresholds, preceding failed attempt, saved manuscripts and reserved outcomes remain unchanged. No validation or protected numeric input was interpreted.

**Distributed coverage slightly improves on late-only EEG at alpha 100, but both joint models are worse than matched context and category medians on primary MAE.** The common-alpha rule selects 100 and its coverage nomination selects distributed EEG. That is an algorithmic choice between EEG recipes, not a qualifying incremental EEG-utility lead.

The [completion receipt](../../results/development/faced_coverage_missing_2026-10-13/completion.json), [full comparison](../../results/development/faced_coverage_missing_2026-10-13/comparison.json) and [independent audit](../../results/development/faced_coverage_missing_2026-10-13/independent_audit.json) retain every score, prediction, recipe and binding.

## All declared MAE results

Errors are on the original 0–7 valence scale; lower is better. Every candidate predicts the same 840 held source participant/title pairs across ten crossed person/material cells. Category-median MAE is **1.3039702691**; MSE is **2.8763248563**.

| Ridge alpha | Matched context | Context + late EEG | Context + distributed EEG |
|---:|---:|---:|---:|
| 0.1 | 1.3449642498 | 1.7914571701 | 1.8748040066 |
| 10 | 1.3460040594 | 1.5761589332 | 1.5800775022 |
| **100: fixed primary and selected shared alpha** | **1.3665040593** | **1.4646169160** | **1.4455432096** |

At fixed alpha 100, late-minus-distributed MAE is **+0.0190737064**. Matched-context-minus-distributed MAE is **−0.0790391503**; category-median-minus-distributed MAE is **−0.1415729405**. Late EEG trails those references by 0.0981128567 and 0.1606466469. Every joint candidate loses to its matched-alpha context control and category medians on MAE. Distributed coverage does not beat late-only at the other alphas, so its small improvement depends on regularization.

Context has lower MSE than the category prior, while both EEG models at alpha 100 have higher MSE than both references (late 3.2704980442, distributed 3.2092463335). Primary MAE and secondary MSE must not be interchanged after results. Shared-alpha selection is symmetric across regressors; it does not independently optimize context, whose lowest MAE is at alpha 0.1.

## Qualified handling and evidence

The strict source gate remains false: `sub-087`, clip 20, middle retains its Fp2 excursion flag. The new policy marks that entire trial unavailable to both joint representations. All three of its numerical EEG arrays were skipped, including the otherwise passing late region. The remaining 979 trials supplied 2,937 authenticated regional vectors, or 440,550 scalar features. No raw signal was reread and no threshold, channel repair, label or endpoint changed.

The representation-specific means used only available training donors: 447 in eight cells and 448 in two. All models retained 448 original training rows per cell. Full-source fitting used 979 EEG donors while retaining 980 context/target rows, including the unavailable trial's valid self-report. Common category fallback preserves held denominators, but the actual failed trial is a train-only Inspiration singleton: **none of the 840 actual held trials is unavailable**. These outcomes do not empirically validate real held failure handling; manufactured fixtures exercise it.

The matched context regressor has thirteen non-EEG columns, but its procedure also uses shared EEG-availability routing. It is a controlled comparison, not an autonomous acquisition-free baseline. The always-category prior provides the practical reference. Both EEG arms require all three regions for common QC; their contrast concerns representations, not recording duration, cost or latency.

The protocol/code were pushed at `63011548a` before numerical qualification. The [independent pre-label audit](../../results/development/faced_coverage_missing_2026-10-13/qualification_independent_audit.json) checked all 2,937 regional vector identities and independently recomputed twenty fold means and two full-source means with exact hash agreement. It read only the guarded available snapshot: no failed-trial arrays, raw signals or targets. Those receipts were pushed at `1ca3ca5a0` before fitting.

The result audit checked all 8,400 emitted source predictions, nine candidates, 100 cell metrics, 840 category medians, 93 fit recipes, 62 mean/donor manifests and 31 context-scaler triplets. It used public source predictions and receipts, without private states or independent refitting. The worker's saved-state algebra agrees with estimators on all 980 source rows with maximum discrepancy 0.0; this is state qualification, not held-out performance or protected export qualification. All 56 focused synthetic tests passed, including role/target rejection, donor leakage, callback traps, fatal technical/numerical errors, retained rows, fallback, SVD and budgets.

## Scientific limits and next decision

This reuses adaptive source development. Overlapping-training cells are not independent replications; no confirmatory interval, general absence-of-EEG-information statement or equivalence claim follows. It concerns fixed spectral/readout recipes, six paired held categories and catalogue grouping. Fear and Inspiration have no source material-transfer held outcomes. Distributed cores overlap and do not cover every clip second. Brain-specific mechanisms, optimized neural performance and architectural novelty remain unestablished.

The missing-input dependency is resolved for this comparison. **These results provide no reason to expand the encoder grid or call the distributed nominee confirmed useful EEG.** Review contribution and confirmation purpose using both favorable and adverse contrasts. A one-time independent study could estimate this fixed recipe's performance, but consuming the reservation requires an explicit rationale, precision scope and state/fallback/export qualification. Those dependencies remain open. The [new inactive proposal](FACED_Coverage_Missing_EEG_Confirmation_Proposal_2026-10-13.md) records the actual nominee and absent incremental lead; earlier papers and the held proposal stay intact.
