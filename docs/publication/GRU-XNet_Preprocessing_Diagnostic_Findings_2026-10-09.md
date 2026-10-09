# EEG preprocessing diagnostic and next research decision

Completed 9 October 2026. **This phase is complete; the paper is still not submission-ready.** The useful lead is DEAP native-channel, baseline-relative band power. This is established preprocessing with encouraging small-panel validation, not a newly demonstrated conference contribution. The manuscript and research question remain unchanged.

## Completed work and verification

The [revision-2 protocol](Preprocessing_Diagnostic_Protocol_2026-10-09.md) completed **72 neural trajectories, 36 recipe/panel comparisons, 288 neural candidate state replays and 80 independently refitted classical candidates**. It crosses common/native electrodes, source-channel/per-trial normalization and four/forty-second EEGNet inputs. DEAP adds measured-baseline subtraction. The original context-API error was detected before EEG training; that first attempt's code, declaration and artifacts remain preserved and excluded. No experimental factor, seed, budget or selection rule was changed in revision 2.

Every input is restricted to the declared first-fold/session/material-rotation source training and validation panels in the two existing groupings. Native source waveforms regenerate **740 DEAP and 270 SEED-IV common-electrode prefixes exactly**. All eight forty-second common14/source-channel reference trajectories reproduce their older 200-update model states exactly. Thirteen relevant tests pass. The largest measured neural peak CUDA tensor allocation was approximately **0.96 GiB**, excluding driver/context memory. No outer-test prediction was generated.

The [independent public analysis](../../results/development/preprocessing_diagnostic_v2_2026-10-09/postfit_analysis/verification.json) recomputes **3,924 probability metric sets**, checks 3,996 retained artifact hashes, and verifies every source selection rule. Maximum metric discrepancy is 1.63e-8. Complete [recipe metrics](../../results/development/preprocessing_diagnostic_v2_2026-10-09/postfit_analysis/selected_recipe_metrics.csv), all **312 descriptive paired contrasts** and [learning curves](../../results/development/preprocessing_diagnostic_v2_2026-10-09/postfit_analysis/source_learning_curves.png) are exported. The curves were visually checked; their fixed-LR EMA traces are distinct from the source-selected population-normalized states.

After the complete diagnostic, a [separately declared baseline-alone control](../../results/development/prestimulus_control_2026-10-09/plan.json) fit and independently refit **16 further classical candidates**, with four selected heads. All 48 probability metric sets reproduce, saved coefficients/scalers refit exactly, and direct logistic-link replay differs by at most 1.11e-16. This control uses only the three-second pre-stimulus baseline for its features. It is explicitly postdiagnostic development evidence, not prospective confirmation.

## Most useful lead: baseline-relative features on DEAP

These are **source-validation balanced accuracy / balanced log loss**, not held-out test scores. Each unseen-video panel has **32 trials from four validation participants**. Hyperparameters were selected using equal familiar/unseen source-validation loss; both panels are therefore used for selection. Each row uses all 32 DEAP electrodes and the same declared source partitions and corrected individual-valence labels.

| Representation | Grouping 1, unseen validation | Grouping 2, unseen validation |
| --- | --- | --- |
| Stimulus absolute log band power | 46.47% / 0.8505 | 44.53% / 0.7479 |
| Stimulus within-channel relative log band power | 54.90% / 0.7075 | 57.09% / 0.6919 |
| Pre-stimulus baseline alone | 43.14% / 0.7390 | 51.01% / 0.6931 |
| Stimulus minus baseline log band power | **72.35% / 0.6943** | **57.29% / 0.6661** |

The baseline-relative representation improves both validation loss and accuracy over absolute stimulus power and the baseline-alone classifier in these two panels. Its gain over the stimulus-relative representation is much smaller in grouping 2. Grouping 1's 0.6943 loss is close to uniform binary prediction (approximately 0.6931), so its 72.35% accuracy should not be presented as strong calibrated predictive evidence. Familiar-video baseline-relative EEG accuracy is 54.55% and 58.37%; the corresponding EEG-free video priors give 80.76% and 73.23%. Useful incremental EEG prediction beyond context remains unestablished.

The baseline-alone control does not reproduce the baseline-relative accuracy in these panels. It rules out a particular simple explanation within this linear control family; it does not establish an emotion mechanism, causal stimulus response or general absence of predictive pre-stimulus information. The observed contrasts include representation and each recipe's own source-only regularization selection.

## Neural and SEED-IV findings

For DEAP's source-channel-normalized four-second EEGNet, native electrodes give **59.41% and 53.85% unseen-validation BA**, versus **51.96% and 50.00%** for common14. Validation loss improves by approximately 0.0307 and 0.0006. Native forty-second EEGNet also raises unseen-validation accuracy in both groupings, but slightly worsens log loss; electrode expansion is therefore not a uniformly better prediction recipe.

Per-trial normalization, short inputs and waveform baseline subtraction do not produce a consistent improvement across both corpora/groupings. Raw-waveform baseline subtraction reduces native four-second DEAP accuracy from 59.41% to 58.63% in grouping 1 and from 53.85% to 42.91% in grouping 2. The useful feature-space subtraction lead cannot be extended automatically to waveform subtraction.

SEED-IV remains unstable: native/source-channel/four-second EEGNet gives 33.33% and 61.11% unseen-validation accuracy; native/trial-normalized/forty-second gives 27.78% and 44.44%. Each unseen panel has just **12 trials from three people**. A high selected point from one grouping is not independent replication. Forty-second models can fit training examples well while validation remains weak. Some EMA curves also retain the previously identified running-moment problem; exact source-population candidate replays distinguish that issue from lack of model capacity. The 600-update budget does not prove convergence.

## Reference preprocessing and source authentication

The inspected pinned LibEER DEAP loader subtracts the mean of three one-second baseline blocks before downstream preprocessing. Our local diagnostic separately filters baseline and stimulus before subtraction, uses the first forty seconds and the corrected rating policy. These are declared adaptations, not exact reproduction of LibEER's optimizer, protocol or published scores. [Pinned author loader](https://github.com/XJTU-EEG/LibEER/blob/dddff9776dbdae21195fe320dff0a5ba61628a18/LibEER/data_utils/load_data.py).

The [fresh first-party access audit](DEAP_Authentication_Recheck_2026-10-09.md) still receives HTTP 503 from official DEAP dataset/metadata/Python archive routes, with an access-host timeout. The non-www route redirects to a generic department page. The institutional EPFL page points to Queen Mary, and the targeted checksum/relocation searches did not find an independently verified first-party recording copy. **First-party signal-content authentication remains outstanding.** Mirror/cache/metadata agreement and corrected labels retain their narrower scopes; no new label changes are needed.

## Recommendation before a larger programme

**Retain the current research question while investigating this representation lead. No paper pivot is supported yet.** Native baseline-relative band power is an existing method, so merely adding it, a transformer or another recurrent layer is not a novelty claim.

The next concrete scientific hypothesis is: *under participant- and video-disjoint DEAP evaluation, baseline-relative stimulus features improve prediction over stimulus-only and pre-stimulus-only features, and retain useful information after accounting for source-only video priors.* It is a baseline/measurement hypothesis within authorized exploration, not an adopted replacement paper question.

Before a large joint/neural grid, prepare a separately declared, matched full-fold DEAP comparison of stimulus-only, baseline-only and baseline-relative features, including EEG-plus-context against raw/calibrated context controls. Use fold-local training scalers and source selection, identical original trial cohorts, all existing groupings/material rotations, per-participant/video paired uncertainty, and complete outcomes. The current cohorts remain development data. Balanced log loss should be primary; accuracy alone, one favourable grouping or an advantage only on familiar videos is insufficient. Failure to improve beyond these controls should stop escalation of this lead rather than trigger repeated successful-seed searches.

If a reproducible advantage survives, assess a narrowly defined method or transfer question against the closest prior work and present the evidence-backed contribution proposal to the author. A final submission then needs locked independent confirmation and the remaining manuscript/reference/figure repairs. Full joint/LODO and GAMEEMO extension work should follow a credible signal/contribution decision rather than expand indiscriminately. Any actual research-question change still requires explicit author approval and an immediate fresh archive of the then-current paper.
