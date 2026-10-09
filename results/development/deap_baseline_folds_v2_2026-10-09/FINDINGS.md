# Matched full-fold DEAP baseline/context controls

**Completed development study; no manuscript or research-question change.**

160 matched cells, 5,760 independently refitted candidates, 1,440 selected logistic heads and 160 raw priors. Each model/arm/group covers all 1,264 retained original trials exactly once; 50,560 total test prediction rows. Existing people/videos were previously examined. These are complete development repetitions, not new independent confirmation.

| Model | Familiar-video BA | Unseen-video BA | Familiar loss | Unseen loss |
| --- | ---: | ---: | ---: | ---: |
| stimulus_absolute | 48.85% | 47.31% | 0.7226 | 0.7324 |
| stimulus_relative | 52.31% | 51.31% | 0.7041 | 0.7058 |
| baseline_only | 49.26% | 49.01% | 0.7122 | 0.7086 |
| baseline_relative | 51.51% | 51.15% | 0.7055 | 0.7022 |
| stimulus_absolute_context | 74.24% | 48.39% | 0.5657 | 0.7472 |
| stimulus_relative_context | 76.07% | 50.89% | 0.5474 | 0.7255 |
| baseline_only_context | 75.30% | 49.05% | 0.5577 | 0.7186 |
| baseline_relative_context | 76.59% | 50.87% | 0.5479 | 0.7148 |
| context_logistic | 78.05% | 47.50% | 0.5004 | 0.6953 |
| prior | 77.54% | 49.67% | 0.4991 | 0.7062 |

All 240 regular contrasts and 40 within-video contrasts are reported in `comparison.json` and `alignment.json`. Negative loss differences favour the first model; positive accuracy differences favour it. Crossed participant/video uncertainty is primary; intervals are unadjusted and conditional on fixed fits. Baseline associations can reflect person/context/carryover or preprocessing. These analyses do not identify causal emotion physiology.

Features use all 32 electrodes, separate 4–40 Hz stimulus/baseline filtering, first 40 seconds, ten four-second Welch windows and measured three-second baseline. Labels retain the corrected individual-valence policy with sixteen midpoint exclusions. Every learned scaler and C choice uses training/source validation only. Training priors exclude the recipient participant entirely. Raw EEG/features and coefficients remain local. First-party DEAP signal authentication remains outstanding.

The finite linear control family and reused dataset do not establish a new method, EEG absence, joint negative transfer, or conference readiness. An adopted research-question change requires author approval and a fresh manuscript archive.
