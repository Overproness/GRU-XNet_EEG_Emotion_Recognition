# Cross-target extension of the matched neural controls

This exploratory extension follows the SEED-IV findings; it does not adopt a new paper question. The [plan](../../results/development/multitarget_transfer_plan_2026-10-05.json) is saved before fitting either new target. Ninety additional neural runs compare target-only, all-three shared head, and all-three separate binary heads on DEAP and GAMEEMO, using three initializations and five participant rotations. Every eligible target participant is tested once per initialization.

All 2,167 trials are re-extracted from the frozen common-14 waveform cache into the same 56-dimensional trial representation. Demand exact equality with every one of the preceding 1,746 target/source feature rows. Sources use only the original seed-42 split's training participants from the other datasets. Held-out target statistics or labels are not used for normalization, thresholds, calibration, or source inclusion.

The existing hashed `transfer_controls.py` is preserved. A sequential context maps the current target to dataset/head slot zero and the two other datasets to slots one/two, calls the same trainer, and restores the original mappings even on an exception. The mappings are recorded in each run's config; this is not safe to run concurrently within one Python process. All earlier scaler, sampling, initialization, optimizer, validation selection, and 600/1,800-update budget rules remain fixed. Linear target-only/joint controls fix effective training weight to the target-training trial count.

Seed-42 participant permutations form five nearly equal groups (DEAP: 7/7/6/6/6; GAMEEMO: 6/6/6/5/5). Each group is tested and the next validated in turn. The remaining groups train. If a declared partition lacks a class, fail rather than choose a more favorable grouping. Trial-level pooled balanced accuracy is the primary outcome. Participant bootstrap intervals use pooled class-conditional counts, preserving different per-participant trial counts and the possibility of a single-class participant.

Run from the repository using the author's `pytorch` environment:

```powershell
python scripts/extend_transfer_targets.py prepare --common-cache ../publication_runs/cache_common14 --previous-pack ../publication_runs/joint_seediv_diagnostic --output ../publication_runs/cache_common14_trial_features
python scripts/extend_transfer_targets.py run --feature-cache ../publication_runs/cache_common14_trial_features --common-cache ../publication_runs/cache_common14 --target DEAP --plan ../publication_runs/multitarget_transfer_plan_2026-10-05.json --output ../publication_runs/negative_transfer_neural_deap
python scripts/extend_transfer_targets.py run --feature-cache ../publication_runs/cache_common14_trial_features --common-cache ../publication_runs/cache_common14 --target GAMEEMO --plan ../publication_runs/multitarget_transfer_plan_2026-10-05.json --output ../publication_runs/negative_transfer_neural_gameemo
python scripts/verify_transfer_investigation.py --output ../publication_runs/negative_transfer_neural_deap
python scripts/verify_transfer_linear.py --output ../publication_runs/negative_transfer_neural_deap
# Repeat verification for negative_transfer_neural_gameemo.
```

Use fresh output directories. Raw EEG, feature arrays, detailed histories and checkpoints remain local; bounded summaries are exported after verification. Existing test cohorts are development evidence. This does not complete GRU/BiLSTM/attention ablations, native four-class objectives, multiple fold groupings, or unseen-dataset transfer. It tests whether the first target's result extends to the other tasks before selecting a contribution.

The extension completed and its checkpoints/candidate fits reproduced. [All-target findings](GRU-XNet_Multitarget_Transfer_Findings_2026-10-05.md) record every planned condition and remaining uncertainty. After both target verifications, generate the paired summary and standalone figure using a fresh output directory:

```powershell
python scripts/analyze_transfer_targets.py --runs-root ../publication_runs --output ../publication_runs/multitarget_transfer_analysis
```

The summary combines three separately trained/validated target studies, not one checkpoint tested on three datasets. No question change is adopted.

Derived summary files can be regenerated in their own analysis directory; the script refuses directories containing unrelated files. Training outputs, feature caches and linear replay directories still require fresh paths. Regenerate the summary after refreshing verification records so its input hashes remain current.
