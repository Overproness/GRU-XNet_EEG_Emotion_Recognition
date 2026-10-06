# Matched material exposure within one session

This development experiment follows the participant/session controls without adopting a new paper question. The author has authorized exploration; a paper pivot still requires showing the evidence, explicit approval and a fresh archive of the then-current manuscript.

## Question and matched comparison

Does performance on the same held-out people's recordings change when their videos are included in training recordings from other people, versus excluded, while the session and training size stay fixed?

Use the existing verified 1,080-trial SEED-IV cohort and the same five participant folds: nine training, three validation and three test people. Fit on one session only. Participant sets remain disjoint in both arms. A material is a published-design (session,trial-position) key, not an independently hashed original video. [SEED-IV documentation](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) describes 72 films, three sessions and 24 trials per session. The six films of each native emotion within each session define the material strata.

Within each of twelve session/native-emotion strata, sort six material keys and permute using one RNG seeded 20261006 in sorted stratum order. Fix this assignment before training, without reading model outcomes. Three rotations cover every clip exactly once in test. For rotation r, use cyclic ranks `(2*r+i)%6` for i=0..5:

| Role | Material ranks in cyclic order, per native emotion |
| --- | --- |
| Test, both arms | First two |
| Validation, both arms | Third |
| Unexposed training | Fourth, fifth, sixth |
| Exposed training | First, second, fourth |

Both arms train on **108 trials** (9 people × 12 clips), validate on the **same 12 trials** (3 different people × 4 clips), and test on the **same 24 trials** (3 held-out people × 8 clips). One training clip per native emotion is shared; two are replaced. Native training emotion counts match. Validation clips are excluded from training and test in both arms. The exposed arm deliberately includes test material from training people; no recording of a test person is fitted. The unexposed arm excludes test material from fitting and selection. Other sessions and unused clips provide no population statistics or calibration.

## Models, input and budgets

Keep the authored absolute-bandpower mean MLP and temporal transformer from the preceding protocol unchanged: 19,079/19,075 parameters, first ten four-second windows, the same 14 physical electrodes, and the same coarse neutral/negative/positive task. Full source-trial offline preprocessing precedes cropping, so this is not a strictly causal 40-second recording protocol.

Two architectures × two arms × three sessions × three material rotations × three seeds (42/43/44) × five participant folds produce **540 neural fits**. Fit StandardScaler only to selected training windows. Complete 600 AdamW updates, batch 60 (20 per coarse class), learning rate 0.001, weight decay 0.01, gradient clip 1, validation every 25 steps. Select common three-class balanced accuracy, then balanced log loss, then the earliest exact tie. No AMP, augmentation, scheduler or early stopping. Pair participant/native-emotion/within-selected-emotion-trial-rank draws across arms and architectures. Initialize at seed+1000*fold+10000*rotation, shared across sessions/arms within an architecture; reset the stochastic training RNG by adding 1000000. Selection steps can differ despite identical available exposure.

Fit four multinomial logistic controls: mean prefix bandpower, full-trial duration only, frozen pretrained REVE, and its same-architecture frozen random42 control. Reuse the preceding verified 512-dimensional REVE feature packs with exact cohort/feature/source checks; neither encoder is refitted. Its stateless per-observation adapter, 37 seconds of direct patch coverage within a 40-second normalized observation, one random encoder initialization and limited public checkpoint-overlap audit remain explicit. Duration receives no EEG and is a diagnostic.

Four representations × two arms × three sessions × three rotations × five folds give **360 selected heads and 1,440 candidate fits**. Use training-only trial-feature scalers, balanced class weights, C=0.01/0.1/1/10, max_iter=2000 and tol=1e-6; select with the same validation three-class BA/log-loss rule. No test threshold or hyperparameter selection.

## Evaluation and interpretation

Each model/arm/seed tests all **1,080 original trials once**, across participant folds, sessions and material rotations. Neural arms yield 12,960 test rows; linear arms 8,640; total **21,600**. Primary is common three-class trial balanced accuracy; secondary conditions negative/positive probabilities on the same 810 nonneutral trials, retaining the preceding 1e-12 probability-mass floor. Average neural seed scores; initialization seeds and clip rotations are not independent participants. Report per-seed and per-session/rotation scores plus validation-selection variability.

Predeclare all twenty contrasts: unexposed-minus-exposed for all six models, transformer-minus-MLP in each arm, pretrained-minus-random in each arm, each on both tasks. Report 10,000 paired participant-only and crossed participant/material percentile draws at seed 20261006. Crossed draws resample 15 people and independently six keys in each session/native-emotion stratum. Average seed correctness within each person/material cell before resampling; pair identical weights across arms/models. Check point-score agreement with the original metrics, including probability floors and ties. Intervals are exploratory, unadjusted and conditional on fixed training folds/checkpoints, these sessions and observed material keys.

This controls the recording-session index and training size while changing material exposure. Different training clips also change EEG content, difficulty, order and duration, so it does not identify a pure causal video-identity effect. Training halves and validation shrinks relative to the previous session study; cross-study score differences are not causal estimates. The cohort has already been inspected. This is neither unseen-corpus evaluation nor pooled three-dataset training.

Stimulus-independent evaluation already has prior work: [Hu et al., EMBC 2021](https://www.paperhost.org/proceedings/embs/EMBC21/files/1565.pdf) uses session partitions for SEED-IV. Our within-session matched comparison reduces that recording-session confound while preserving separate validation. It is an empirical diagnostic, not a novelty claim or an exact reproduction of that SVM/KPCA study. Any paper contribution needs a narrower gap and evidence beyond this development corpus.

## Reproduction and verification

```powershell
conda activate pytorch
python scripts/within_session_material_controls.py plan --plan ../publication_runs/within_session_material_plan_2026-10-06.json
# Commit and push the declaration before fitting.
python scripts/within_session_material_controls.py run --cache ../publication_runs/cache_temporal_native_seediv --reve-run ../publication_runs/reve_frozen_seediv --output ../publication_runs/within_session_material_seediv --plan ../publication_runs/within_session_material_plan_2026-10-06.json --device cuda
python scripts/within_session_material_controls.py verify --cache ../publication_runs/cache_temporal_native_seediv --reve-run ../publication_runs/reve_frozen_seediv --output ../publication_runs/within_session_material_seediv --device cuda
python scripts/report_within_session_material.py --run ../publication_runs/within_session_material_seediv --destination ../GRU-XNet_Within_Session_Material_Findings_2026-10-06.md
```

Use the preserved temporal and frozen-feature preparation protocols for a fresh workspace. The new module is separate; previously bound experiment source bytes remain unchanged. Verify material and participant access, complete rotation/OOF coverage, initial states and canonical draw signatures, all 540 selected checkpoints, training-only scalers, validation selection and train/validation/test metrics. Independently refit all 1,440 classical candidates and compare selected coefficients and all 21,600 test rows. Use stable softmax as in sklearn; preserve strict log-loss tolerance. Recompute all aggregate/bootstrap contrasts. Checkpoints and full feature packs remain local; only bounded derived evidence is exported.
