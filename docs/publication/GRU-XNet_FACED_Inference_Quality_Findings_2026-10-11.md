# FACED acquisition-failure policy and cached sensitivity

Completed: 11 October 2026. **The fifteen flagged validation trials now have an implemented, frozen handling policy. All participants, trials and outcomes remain in evaluation.** The policy uses context prediction when EEG is unavailable, or a source-only median for an EEG-only model. Its cached sensitivity leaves the exploratory EEG gain small and uncertain. The manuscript and research question remain unchanged; confirmation participants and films remain sealed.

The [fixed policy](FACED_Inference_Quality_Policy_2026-10-11.md), [execution packet](../../results/development/faced_inference_quality_2026-10-11/README.md), [signal audit](../../results/development/faced_inference_quality_2026-10-11/signal_audit.json) and [independent audit](../../results/development/faced_inference_quality_2026-10-11/independent_analysis/summary.json) preserve the evidence. The [preceding artifact-control experiment](GRU-XNet_FACED_Artifact_Control_Findings_2026-10-11.md) remains intact, including its fitted states, selections and predictions.

## What caused the flags

The signal-only audit used already authenticated technical caches from the preceding bounded recording reads. It opened no original BDF/MAT file, waveform-feature cache, model state or new rating value. The audited intervals are the previously authorized 32-second late-viewing intervals; findings do not extend to whole recordings.

All fourteen development trials for anonymous participant sub-030 have F8 fixed at the negative ADC rail: rail fraction 1.0, identical-adjacent fraction 1.0, raw SD zero and calibrated raw median -375,000 microvolts. Sub-100/clip-4 has the same failure at PO3. The earlier amplitude-only interpolation detector misses these flat channels: filtering reduces their pre-reference SD to approximately 9e-10 and 1e-8 microvolts. Common-average referencing then gives them apparent variation, respectively 4.17–20.46 and 6.18 microvolts of SD. This apparent variation is not recovered electrode information.

Three sub-030 trials also have partial Oz/PO4 rail occupancy. Across the fifteen flagged trials there are nineteen channel/trial rail events and fifteen stationarity events. Some other channels received the preceding amplitude interpolation, but that does not erase the original raw rail evidence. The exact physical acquisition cause is unknown. A failed electrode does not establish that all other electrodes are physically damaged.

The manufactured negative-rail tests reproduce the filtering/reference behavior at both native sampling rates. This supports checking raw quality before reference transforms. MNE likewise recommends inspecting unprojected raw data and marking bad channels early; its documentation does not prescribe this fallback policy or our thresholds. [MNE bad-channel tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html).

## Fixed handling policy

The policy reuses the existing engineering limits rather than selecting new cutoffs from prediction errors. A trial's EEG is unavailable if any common channel has raw ADC rail fraction above 0.0001, identical-adjacent fraction above 0.95, processed SD below 0.1 or above 100 microvolts, processed peak-to-peak above 1,000 microvolts, or processed 49–51/1–45 Hz power ratio above 0.10. Numerical failure and exceeded original repair capacity also make EEG unavailable. Exact boundary behavior, common-channel order and malformed-record rejection are covered by tests. These are operational quality limits, not clinically validated diagnostic cutoffs.

For an available trial, the existing prediction is unchanged. For an unavailable trial:

| Model family | Fixed prediction branch |
| --- | --- |
| Context only | Existing source-fitted context model, unchanged |
| EEG plus context | Existing selected source-fitted context model; the EEG predictor is bypassed |
| EEG only | Global median from the exact source participant/material set; EEG is bypassed |

No participant, material, trial, rating or metric denominator is removed. The joint fallback assumes its declared context metadata are available at inference. EEG-only fallback needs only a source-fitted prior. A whole-trial fallback deliberately forgoes potentially useful remaining channels; a channel-repair or masked-channel alternative would require a separate source-qualified declaration. It is not chosen retrospectively for better validation error.

Future source fitting has a stricter gate: exact permitted source identities are checked before quality reads, then every source trial must pass the same policy. An unavailable source trial holds fitting rather than silently dropping it or substituting context. All 980 current source trials pass. The implemented API and replay runner establish this gate for their declared paths; they do not retrofit the historical course-project training scripts.

The policy and signal-only audit were committed and pushed as [655316fc0](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/commit/655316fc0) before this policy's cached prediction replay. However, the initial development outcomes had already been examined in the previous experiment. **This is a development-informed prospective engineering policy and a post-hoc sensitivity check, not independent preregistration or confirmation.**

## Cached sensitivity with no refitting or reselection

The replay preserves all 27,720 original candidate/prior/source-OOF prediction rows, original alphas and fitted states. It replaces 450 candidate predictions: three rotations times fifteen unavailable trials times five alphas times two EEG-dependent families. At the original selected alphas there are forty-five joint and forty-five EEG-only fallback rows across rotations and both validation arms. Source OOF predictions, all context predictions and all source-prior predictions remain unchanged. There were zero new fits, original-recording reads, waveform-feature reads, model-state reads or rating decodes.

All twenty validation participants and fourteen development clips remain evaluated. EEG is available in 265 of 280 trials (94.64%). Nineteen participants have at least one available EEG trial; sub-030 contributes fourteen fallback predictions and remains in every relevant outcome denominator. Each rotation retains 120 unseen-film and 160 familiar-film trials. Unseen/familiar unavailable counts are respectively 6/9, 7/8 and 7/8.

Below are participant/film-equal MAEs for individual continuous valence on the original 0–7 scale; lower error is better. Gain is context MAE minus joint MAE. Values use the original selected alphas, even when post-policy candidate ordering changes.

| Development rotation | Context MAE | EEG-only MAE | Joint MAE | Joint gain over context | Descriptive 95% interval for gain |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 1.25265 | 1.53194 | 1.25885 | -0.00620 | [-0.18368, 0.15525] |
| 1 | 1.20988 | 1.46914 | 1.17667 | 0.03321 | [-0.09504, 0.15959] |
| 2 | 1.31060 | 1.47394 | 1.21114 | 0.09946 | [-0.03514, 0.24758] |

Mean joint gain is **0.042155 MAE**, compared with 0.041689 before fallback, and 0.044620 over the originally chosen strongest fixed source prior. This is below the preceding 0.15 engineering lead threshold. Only two rotations now have positive gains; all three intervals include zero. The first rotation's fixed category prior still beats the joint model. The [complete comparison](../../results/development/faced_inference_quality_2026-10-11/comparison.json) and [predictions](../../results/development/faced_inference_quality_2026-10-11/predictions.csv) retain every candidate, familiar-film result, coverage count and source OOF result.

Intervals use the original 4,000 crossed participant/film bootstrap draws and seed. The same twenty validation people and overlapping films recur across rotations, with only six unseen films per rotation and development-selected alphas. These are descriptive conditional intervals on reused development panels. The near-identical mean gain is a sensitivity finding; it does not prove general robustness or a new useful EEG effect. A threshold crossing in this post-hoc check would not itself qualify as independent evidence either.

## Verification and remaining work

Eighty-eight focused tests pass across the new policy and the existing preprocessing, selective-target and classical contracts. Thirteen new tests include strict cutoff boundaries, malformed/nonfinite metrics, identity-before-access guards, source failure holding fitting, poisoned unavailable-EEG callbacks and the manufactured stuck channels at both native rates.

The runner reproduces every cached row. An independent script, without calling the new policy or replay implementation, recomputes 226,800 channel-rule checks from 2,520 technical caches, verifies original/new row identities and targets, independently reconstructs eighteen exact source/fold medians from previously published prediction targets, and checks all fallback values, candidate/selected/OOF/prior metrics and descriptive bootstrap calculations. Maximum prediction and metric discrepancies are both zero. This independent cached audit involves no new selective-rating decoding. [Independent verification receipt](../../results/development/faced_inference_quality_2026-10-11/independent_analysis/verification.json).

**The requested acquisition-failure handling phase is complete. It does not establish a conference-ready contribution.** The implementation/source-quality issues at this scope are resolved; the small EEG benefit, representation suitability, independent confirmation, exact physical stimulus timing/content and pretraining overlap, author processed-data execution provenance and closest-method novelty remain scientific limitations or dependencies. Previously recorded DEAP provenance and EmoEEG-MC codebook limitations also remain open at their stated scope.

The next appropriate experiment would be a small, separately declared source learning-capacity and EEG-representation control with matched context/EEG/joint comparisons, this fixed quality policy, explicit training budgets and source-only preprocessing. It should diagnose whether the current spectral readout limits useful prediction before another encoder search. The reused-development status must remain explicit. **That experiment is not launched by this report.** Larger encoder expansion and opening confirmation remain unsupported by the present lead. An alternative research question still requires a concrete evidence-backed proposal, the author's explicit approval and a fresh archive of the then-current manuscript.
