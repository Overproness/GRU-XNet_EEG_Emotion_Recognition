# Complete held-out repetitions with fold-local tuning

Declared 6 October 2026. This is further development on the already examined SEED-IV and DEAP cohorts. The manuscript and historical research question remain unchanged. A new contribution still needs a concrete findings review and the author's explicit approval.

## What this phase resolves

The previous full-width study used one grouping and a 200-update budget. The subsequent learning study used two groupings and two initializations, but only one source fold per corpus. This phase completes the full held-out grid, independently selecting learning rate, training duration and BatchNorm treatment inside every fold. It then measures whether EEGNet plus context improves beyond EEGNet and strong context-only controls.

There are **2,040 selected neural cases**, **4,080 uninterrupted training trajectories**, and **24,480 source-validation candidates**. Models are full GRU-XNet, the numerically checked author-architecture EEGNet port, and that EEGNet with a fixed source-only log video prior. Each model covers both corpora, both pre-existing fixed participant/video groupings, both base initializations (42 and 91), every session/video rotation/participant fold and both video-exposure arms. Context-only raw priors and logistic calibrators cover 340 unique split/arm cells, with four C candidates each. Deterministic contextual predictions are repeated logically across the two initializations; this does not double their independent fits.

## Source validation and selection

Outer test participants are excluded from training and validation. For the fixed validation participants, each fold has two disjoint clip panels: the existing unseen-video panel, and videos actually present in source training. The mean of the two panels' class-balanced log losses, weighted equally, selects a candidate. Mean panel balanced accuracy breaks ties; exact ties keep the first declared candidate. The same rule applies to every neural model and contextual calibrator. Both panels' outcomes are saved separately. A panel missing a target class makes the split infeasible; it is not replaced by a successful seed.

The familiar-video panel makes the context calibrator's selection meaningful. Its video set differs across training exposure arms. Consequently, the exposure contrast includes this source-validation/selection procedure as well as training clip composition. It is not an isolated causal test of video identity. In the unseen arm, both validation panels exclude outer-test videos. Within each arm, all models use identical source/validation/test partitions.

Each learning-rate trajectory uses 1,200 balanced twelve-trial AdamW updates, constant learning rate 0.001 or 0.0003, weight decay 0.01, gradient clip 1 and dropout 0.5. Duration candidates 200, 600 and 1,200 are points on the same uninterrupted trajectory. Each point has original exponential-moving BatchNorm statistics and a source-population alternative. The latter collects float64 moments in three sequential evaluation-mode passes over actual source-training trials only, with dropout disabled. Learned parameters, optimizer state and ongoing trajectory statistics/RNG are unchanged. No target adaptation, augmentation, scheduler or AMP is introduced.

Training-only mean/std preprocessing and verified common fourteen electrodes/first forty seconds remain fixed. EEGNet uses raw waveforms; GRU uses the previously verified STFT. Their comparison therefore does not isolate an architectural effect. Longer training and recipe selection do not by themselves demonstrate convergence.

## Context and test access

The video prior uses Laplace-one class counts among source-training participants, falling back to source global counts for unseen videos. Every training row's prior excludes all labels from its own participant, including fallback counts. Validation/test priors never use validation/test ratings. EEGNet-plus-context adds log prior to EEG logits, introducing no extra learned parameters. Logistic context-only calibration uses training-only scaling, balanced classes, C = 0.01/0.1/1/10 and independently verified refitting.

All twelve candidate source-validation probability sets and metrics, selected-state/draw/split hashes and checkpoint SHA are sealed **before test inference**. Each selected model is loaded fresh for scoring. The independent auditor reconstructs partitions, scalers, priors, initialization and sampling hashes, recomputes every candidate/selected metric, and replays selected train/validation/test predictions. Selected population-normalization moments are reconstructed exactly. Sixty-four applicable 200-update source states must exactly match the completed learning study. These checks verify computation and state output, rather than replaying the entire optimization trajectory.

## Storage, resumption and Git checkpoints

All 1,360 selected EEGNet/EEGNet-plus-context states remain local. Thirty-two GRU sentinels, fixed at rotation zero/fold zero in every session/grouping/initialization/arm, also remain local. Every other selected GRU checkpoint is stored temporarily, independently replayed, and only then deleted by its exact verified path. Its SHA, state digest, source probabilities and immediate replay certificate remain. Later raw state replay of deleted checkpoints requires refitting; the archive does not retain those weights.

The worker has an exclusive operating-system lock, hash-bound resumption and automatic failure recording. A sealed case can resume scoring without changing selection. A resumed completed case must pass its artifact/certificate checks. Failure stops the job and preserves its evidence; a weak result does not stop or remove a candidate.

The public export includes only bounded scientific JSON/CSV/Markdown artifacts. It excludes EEG and model tensors. Every forty newly verified selected cases, the authorized publisher stages only this study's exported evidence and status note, commits and pushes to the existing origin/main. Unrelated staged changes, a changed branch/remote or push rejection stop publication rather than causing a force push. The complete-grid analysis/report is generated and published after all cases pass verification.

## Analysis and limits

Each original trial appears exactly once per model/arm/grouping/initialization: 1,080 SEED-IV coarse-three primary trials, 810 conditional-binary secondary trials and 1,264 DEAP individual-valence trials. Report all four grouping/initialization combinations and their combined mean correctness/loss within observed person/video cells. Do not ensemble probabilities, average unequal fold accuracies, or choose the best initialization.

Primary paired contrasts include every model's unseen-minus-familiar exposure difference, EEGNet-minus-GRU, and EEGNet-plus-context minus EEGNet/raw prior/calibrated context in both arms and all tasks. Ten thousand fixed paired participant-only and crossed participant/video percentile draws characterize these fitted models on the existing observed cohorts. These are exploratory, unadjusted conditional ranges; repeated groupings add no independent people or videos.

The separate within-video diagnostic exchanges each recipient's EEG with every other test participant watching the same video in the identical selected fold/grouping/initialization. It averages correctness/loss over donors, holding recipient context fixed. Dyadic participant/participant/video uncertainty respects shared donors. Context-only predictions must remain invariant. Assigned SEED video labels imply zero aggregate alignment difference and serve as a sanity check. DEAP individual rating alignment can be investigated, but cannot identify a causal physiological mechanism. Ineligible trials and nonestimable bootstrap draws are disclosed.

Original first-party DEAP waveform authentication remains outstanding. GAMEEMO's four interactive conditions need a different extension protocol. This phase neither completes joint/leave-one-dataset-out validation nor establishes submission readiness or a new method's novelty.

## Run and inspect

With the verified existing local caches and PyTorch environment:

```powershell
conda activate pytorch
# Declare once; commit/push the protocol before first fitting.
python scripts/heldout_tuning.py plan
python scripts/export_heldout_tuning.py
# Resume the declared study; milestone publishing is explicitly authorized by the author.
python scripts/heldout_tuning.py run --push-milestones
python scripts/audit_heldout_tuning.py
```

On a public checkout, verify the exported accepted candidates and selection certificates without raw EEG/checkpoints:

```powershell
python scripts/audit_heldout_tuning.py --output results/development/heldout_tuning_2026-10-06 --public --partial
```

Use the [status note](GRU-XNet_Heldout_Tuning_Status_2026-10-06.md) and [machine-readable declaration](../../results/development/heldout_tuning_2026-10-06/plan.json) to distinguish started work from complete findings.
