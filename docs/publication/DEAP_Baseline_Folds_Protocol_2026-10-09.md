# Matched full-fold DEAP baseline and context controls

Declared 9 October 2026 as authorized exploration. The manuscript and research question remain unchanged. The native baseline-relative feature lead was already observed on two small validation panels; all people/videos in this new grid have been examined previously. These are development repetitions, not untouched confirmation.

The hypothesis is that measured-baseline-relative stimulus features improve individual-valence prediction over stimulus-only and pre-stimulus-only controls, and add useful prediction beyond source-only video context. This is a measurement/control hypothesis, not a novel method or an adopted replacement paper question.

Use all 1,264 retained corrected DEAP valence trials, with sixteen rating-five exclusions, all 32 EEG electrodes and the same first forty seconds. Separately filter the measured three-second baseline and full stimulus at 4–40 Hz. The representations are absolute stimulus log band power, within-channel relative stimulus log band power, baseline-only log power, and stimulus-minus-baseline log power. Welch bands and all window/rounding conventions exactly retain the preceding diagnostic; this is a declared local adaptation, not exact LibEER or author-score reproduction.

Every representation has an EEG-only and EEG-plus-context classifier. Retain raw and calibrated context without EEG. Training context excludes every label from the receiving participant before computing the Laplace-smoothed video/global prior. Validation/test priors use source training labels only. All learned feature/context scaling uses training rows alone.

There are **160 cells: two fixed participant/video groupings × five video rotations × eight participant folds × two exposure arms**. Source training, validation and test participants are disjoint. Arms share identical test and unseen-validation trials and match training counts within participant/class. Each model/arm/group covers every retained original trial exactly once. Unexposed source panels contain no outer-test video; familiar validation uses the validation people watching actual source-training videos. Exposure comparisons also change the familiar-validation regime and do not causally isolate clip identity.

Fit **5,760 candidates**, selecting 1,440 logistic heads, plus 160 fixed raw priors. Balanced logistic regression uses C = 0.01/0.1/1/10, maximum 4,000 iterations, tolerance 1e-6 and deterministic random state 42. Equal familiar/unseen source-validation balanced log loss selects C; mean panel balanced accuracy breaks ties, then first declared C. Preserve all train/validation candidate probabilities. Seal each source selection and coefficient hashes before retrieving that model's outer-test features or prior. Fixed label-free features for all recordings may be prepared before fitting; no learned global scaler or test-informed choice is allowed.

Independently refit every candidate with fresh training-only scaling and require exact coefficients plus direct logistic-link replay. Independently recompute probability metrics and original participant/trial boundaries. Require 32 exact earlier candidate/scaler/coefficient sentinels and agreement with all 320 previous raw/calibrated context test controls. Raw EEG, per-trial feature arrays and coefficient arrays remain local; verified derived evidence is exported.

Only after all cells verify, report all 50,560 outer-test probability rows, combined and grouping-specific metrics, and all 240 declared regular contrasts. Primary paired balanced log loss and secondary balanced accuracy use 10,000 existing person-only and crossed participant/video percentile draws. Combined results average correctness/loss per observed person/video across groupings, not probabilities or unequal fold scores. An independent flat-trial weighting calculation rechecks all regular points and percentile endpoints. Intervals are exploratory, unadjusted and conditional on fixed fits and reused people/videos.

The 40 within-video aligned-minus-exchanged contrasts swap every other held-out participant's whole feature prediction within the same selected cell/video, keeping recipient labels/context. Bootstrap weights apply to recipient, donor and video. Raw/calibrated context must be invariant. This tests conditional association; baseline/stimulus results can reflect traits, context or carryover and do not identify causal emotion physiology.

Preserve weak outcomes, failed checks and every declared comparison. Fail on changed sources/inputs, nonconvergence, missing classes/coverage, refit/replay mismatch or publication failure. An exclusive worker lock protects resumable sealed cases. Verified evidence is committed and pushed every ten newly completed cells and at completion. No force push or unrelated staged changes are allowed.

Loss improvement across groupings and relevant stimulus/baseline/context controls should govern escalation. A favourable accuracy point, successful seed or established preprocessing advantage alone is insufficient for novelty or a paper pivot. First-party DEAP signal authentication remains outstanding. Any changed paper question still requires presenting findings/proposal, explicit author approval and an immediate fresh archive.

```powershell
conda activate pytorch
python scripts/deap_baseline_folds.py plan
python scripts/deap_baseline_folds.py prepare
python scripts/deap_baseline_folds.py run --push
```

The machine-readable [plan](../../results/development/deap_baseline_folds_2026-10-09/plan.json) and [progress](../../results/development/deap_baseline_folds_2026-10-09/progress.json) distinguish declaration, running and completion. No outcome is implied by this protocol.
