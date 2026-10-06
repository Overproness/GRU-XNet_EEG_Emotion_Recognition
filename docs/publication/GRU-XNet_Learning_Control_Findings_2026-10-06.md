# Source learning, baseline authentication and normalization findings — 6 October 2026

All **96 declared fits** are complete: sixteen tiny-batch memorization checks and eighty source-training/validation trajectories through 1,200 updates. GRU and author-checked EEGNet include both declared groupings and initializations 42/91. Twelve older full-model 200-update prefixes reproduce their exact selected state and complete source-validation history. Final and selected checkpoints are independently replayed. Separately declared **post-hoc** BatchNorm diagnostics cover all 160 selected/final source states and all sixteen tiny-batch final states; every learned parameter is unchanged and every source moment/probability is independently replayed. The scientific-control suite has **71 passing tests**.

- SEEDIV, GRU at LR 0.0003: mean source training BA 76.85% at 200 updates → 94.37% at 1,200; validation 34.72% → 29.17%, balanced validation loss 1.838 → 9.360. All eight source panels are retained.
- DEAP, GRU at LR 0.0003: mean source training BA 59.66% at 200 updates → 89.27% at 1,200; validation 49.64% → 46.39%, balanced validation loss 0.729 → 2.954. All eight source panels are retained.
- Tiny-batch strict capacity checks pass 14/16 original states and 16/16 after source-only BatchNorm recalibration, without changing learned weights. Real and shuffled labels are both retained; this establishes capacity/evaluation-state behavior only.

These are source-only development diagnostics. They **do not evaluate outer-test performance** and do not change the manuscript or main research question. SEED-IV uses session 1 / rotation 0 / fold 0, with 108 source-training and 12 validation trials. DEAP uses rotation 0 / fold 0, with 358 (group 1) or 344 (group 2) training and 32 validation trials. Groupings can change validation people/videos; initializations are compared on the same population within each grouping/arm. These are repeated source panels, not new participants or complete OOF confirmation. Previously inspected cohorts remain development cohorts.

## Fixed-budget source results

All learning rates are retained. BA is a percentage; balanced log loss is lower-is-better. Uniform-probability loss is 1.099 for SEED-IV and 0.693 for DEAP. A large loss with near-chance accuracy can indicate confident mistakes without identifying their cause. Each entry is an unweighted descriptive panel mean. GRU/EEGNet have eight panels per rate, the local references two; widths, parameter counts and raw versus STFT representations differ. Scores do not form an architecture leaderboard. The 200/600/1,200 points come from the same uninterrupted trajectory, not separate restarts.

| Dataset | Model | LR | Panels | Train BA 200 | Train BA 600 | Train BA 1,200 | Val BA 200 | Val BA 600 | Val BA 1,200 | Val loss 200 | Val loss 1,200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SEEDIV | GRU-XNet | 0.001 | 8 | 53.40 | 73.15 | 80.02 | 39.58 | 37.50 | 34.03 | 1.56 | 9.39 |
| SEEDIV | GRU-XNet | 0.0003 | 8 | 76.85 | 92.44 | 94.37 | 34.72 | 38.19 | 29.17 | 1.84 | 9.36 |
| SEEDIV | Matched BiLSTM | 0.001 | 2 | 48.77 | 70.99 | 80.25 | 41.67 | 27.78 | 27.78 | 1.41 | 9.32 |
| SEEDIV | Matched BiLSTM | 0.0003 | 2 | 62.04 | 95.37 | 99.69 | 47.22 | 50.00 | 38.89 | 1.74 | 4.74 |
| SEEDIV | Local CBSAtt | 0.001 | 2 | 37.04 | 54.32 | 91.36 | 33.33 | 33.33 | 44.44 | 2.74 | 5.15 |
| SEEDIV | Local CBSAtt | 0.0003 | 2 | 62.35 | 86.73 | 73.77 | 36.11 | 33.33 | 41.67 | 1.45 | 6.52 |
| SEEDIV | Author-checked EEGNet | 0.001 | 8 | 62.89 | 66.13 | 66.44 | 31.94 | 33.33 | 36.11 | 1.56 | 3.52 |
| SEEDIV | Author-checked EEGNet | 0.0003 | 8 | 55.02 | 66.28 | 66.82 | 32.64 | 30.56 | 31.94 | 1.17 | 2.91 |
| DEAP | GRU-XNet | 0.001 | 8 | 53.19 | 54.40 | 60.58 | 48.85 | 47.37 | 47.08 | 0.72 | 1.01 |
| DEAP | GRU-XNet | 0.0003 | 8 | 59.66 | 74.21 | 89.27 | 49.64 | 47.70 | 46.39 | 0.73 | 2.95 |
| DEAP | Matched BiLSTM | 0.001 | 2 | 56.41 | 57.54 | 56.93 | 46.18 | 45.29 | 48.24 | 0.71 | 1.14 |
| DEAP | Matched BiLSTM | 0.0003 | 2 | 59.79 | 68.29 | 92.65 | 46.37 | 50.98 | 50.78 | 0.71 | 2.72 |
| DEAP | Local CBSAtt | 0.001 | 2 | 55.72 | 59.66 | 66.23 | 51.76 | 46.67 | 53.33 | 0.69 | 2.05 |
| DEAP | Local CBSAtt | 0.0003 | 2 | 58.22 | 64.24 | 82.65 | 52.16 | 47.16 | 47.25 | 0.69 | 0.94 |
| DEAP | Author-checked EEGNet | 0.001 | 8 | 69.54 | 76.31 | 80.03 | 55.95 | 52.80 | 54.21 | 0.75 | 2.20 |
| DEAP | Author-checked EEGNet | 0.0003 | 8 | 66.96 | 78.60 | 80.81 | 59.06 | 58.32 | 57.93 | 0.68 | 1.56 |

The validation panels are small, and expanding the checkpoint search increases selection opportunity. A larger source-selected validation score is not evidence of better test generalization. Fixed-step validation loss and accuracy are therefore retained alongside selection positions. A checkpoint chosen early is not proof of optimization convergence; a final checkpoint is not proof that a longer budget would help.

| Dataset | Model | LR | Median selected update | Selected after 200 | Selected at 1,200 |
|---|---|---:|---:|---:|---:|
| SEEDIV | GRU-XNet | 0.001 | 215 | 4/8 | 0/8 |
| SEEDIV | GRU-XNet | 0.0003 | 180 | 3/8 | 0/8 |
| SEEDIV | Matched BiLSTM | 0.001 | 175 | 0/2 | 0/2 |
| SEEDIV | Matched BiLSTM | 0.0003 | 430 | 1/2 | 0/2 |
| SEEDIV | Local CBSAtt | 0.001 | 70 | 0/2 | 0/2 |
| SEEDIV | Local CBSAtt | 0.0003 | 65 | 0/2 | 0/2 |
| SEEDIV | Author-checked EEGNet | 0.001 | 10 | 3/8 | 0/8 |
| SEEDIV | Author-checked EEGNet | 0.0003 | 30 | 3/8 | 0/8 |
| DEAP | GRU-XNet | 0.001 | 550 | 5/8 | 0/8 |
| DEAP | GRU-XNet | 0.0003 | 460 | 4/8 | 0/8 |
| DEAP | Matched BiLSTM | 0.001 | 430 | 1/2 | 0/2 |
| DEAP | Matched BiLSTM | 0.0003 | 875 | 2/2 | 0/2 |
| DEAP | Local CBSAtt | 0.001 | 555 | 1/2 | 0/2 |
| DEAP | Local CBSAtt | 0.0003 | 65 | 0/2 | 0/2 |
| DEAP | Author-checked EEGNet | 0.001 | 325 | 7/8 | 0/8 |
| DEAP | Author-checked EEGNet | 0.0003 | 525 | 5/8 | 0/8 |

## Can the implementation memorize training trials?

Twelve distinct source training trials, class-balanced, are fit for 400 updates with the declared dropout and constraints. The shuffled targets preserve class counts. The descriptive criterion requires BA at least 95% **and** balanced log loss below 0.15; failure can reflect limited confidence even when classification is perfect. All outcomes are retained. Success establishes capacity on a tiny batch, not physiological emotion information or generalization.

| Dataset | Model | Targets | Train BA (%) | Train balanced loss | Strict criterion met |
|---|---|---|---:|---:|---|
| SEEDIV | GRU-XNet | original | 100.00 | 0 | yes |
| SEEDIV | GRU-XNet | permuted | 100.00 | 0 | yes |
| SEEDIV | Matched BiLSTM | original | 100.00 | 0 | yes |
| SEEDIV | Matched BiLSTM | permuted | 100.00 | 0 | yes |
| SEEDIV | Local CBSAtt | original | 100.00 | 7.153e-07 | yes |
| SEEDIV | Local CBSAtt | permuted | 100.00 | 3.676e-07 | yes |
| SEEDIV | Author-checked EEGNet | original | 33.33 | 6.972 | no |
| SEEDIV | Author-checked EEGNet | permuted | 33.33 | 3.6 | no |
| DEAP | GRU-XNet | original | 100.00 | 0 | yes |
| DEAP | GRU-XNet | permuted | 100.00 | 0 | yes |
| DEAP | Matched BiLSTM | original | 100.00 | 0 | yes |
| DEAP | Matched BiLSTM | permuted | 100.00 | 0 | yes |
| DEAP | Local CBSAtt | original | 100.00 | 2.881e-07 | yes |
| DEAP | Local CBSAtt | permuted | 100.00 | 3.576e-07 | yes |
| DEAP | Author-checked EEGNet | original | 100.00 | 0.00368 | yes |
| DEAP | Author-checked EEGNet | permuted | 100.00 | 0.006284 | yes |

## Balanced tiny-batch normalization check

The EEGNet SEED tiny-batch sampled training-mode losses reached 0.0031/0.0108 while ordinary inference losses were 6.97/3.60. A separate supplement was declared after fifty source fits had completed, explicitly retaining this post-hoc observation. It uses the same twelve class-balanced trials, fixed labels, input scaling and final learned weights as each original tiny fit. Three-pass source moments are calculated with dropout off. Unlike full-population recalibration below, the tiny-batch check does not change the class prior; it still cannot separate moving-average lag from dropout/inference activation-distribution differences. Every architecture, corpus and real/shuffled target case is retained.

| Dataset | Model | Targets | Original BA (%) | Recalibrated BA (%) | Original loss | Recalibrated loss | Recalibrated strict criterion |
|---|---|---|---:|---:|---:|---:|---|
| SEEDIV | GRU-XNet | original | 100.00 | 100.00 | 0 | 0 | yes |
| SEEDIV | GRU-XNet | permuted | 100.00 | 100.00 | 0 | 0 | yes |
| SEEDIV | Matched BiLSTM | original | 100.00 | 100.00 | 0 | 0 | yes |
| SEEDIV | Matched BiLSTM | permuted | 100.00 | 100.00 | 0 | 0 | yes |
| SEEDIV | Local CBSAtt | original | 100.00 | 100.00 | 7.153e-07 | 6.258e-07 | yes |
| SEEDIV | Local CBSAtt | permuted | 100.00 | 100.00 | 3.676e-07 | 3.676e-07 | yes |
| SEEDIV | Author-checked EEGNet | original | 33.33 | 100.00 | 6.972 | 0.0006924 | yes |
| SEEDIV | Author-checked EEGNet | permuted | 33.33 | 100.00 | 3.6 | 0.002309 | yes |
| DEAP | GRU-XNet | original | 100.00 | 100.00 | 0 | 0 | yes |
| DEAP | GRU-XNet | permuted | 100.00 | 100.00 | 0 | 0 | yes |
| DEAP | Matched BiLSTM | original | 100.00 | 100.00 | 0 | 0 | yes |
| DEAP | Matched BiLSTM | permuted | 100.00 | 100.00 | 0 | 0 | yes |
| DEAP | Local CBSAtt | original | 100.00 | 100.00 | 2.881e-07 | 2.98e-07 | yes |
| DEAP | Local CBSAtt | permuted | 100.00 | 100.00 | 3.576e-07 | 3.576e-07 | yes |
| DEAP | Author-checked EEGNet | original | 100.00 | 100.00 | 0.00368 | 8e-05 | yes |
| DEAP | Author-checked EEGNet | permuted | 100.00 | 100.00 | 0.006284 | 1.939e-05 | yes |

## Running normalization statistics versus learned weights

This supplement was declared after nine completed source fits were available and some source curves had been inspected; it is explicitly post-hoc. It changes only BatchNorm running means/variances. Dropout is off, all actual source training trials receive equal weight, and moment calculation uses float64 sums/squares in three feed-forward passes. It differs from balanced mini-batch statistics used during optimization. Every learned parameter, original normalizer and originally selected optimizer checkpoint is retained. Validation/test inputs do not enter moments and no new checkpoint is selected after recalibration.

Mean after-minus-before changes are descriptive. Positive BA changes are percentage points; negative loss changes favor recalibration. Both original selected and final states and every model/rate/arm/group/initialization are preserved; a helpful training change need not improve validation. The diagnostic does not make population-generalization or BN-method-novelty claims.

| Dataset | Model | LR | State | Source role | Cases | BA change (pp) | Loss change |
|---|---|---:|---|---|---:|---:|---:|
| SEEDIV | GRU-XNet | 0.001 | selected | train | 8 | +8.56 | -0.519 |
| SEEDIV | GRU-XNet | 0.001 | selected | validation | 8 | -13.19 | +0.923 |
| SEEDIV | GRU-XNet | 0.001 | final | train | 8 | +18.75 | -1.683 |
| SEEDIV | GRU-XNet | 0.001 | final | validation | 8 | +3.47 | +0.570 |
| SEEDIV | GRU-XNet | 0.0003 | selected | train | 8 | +2.16 | -0.179 |
| SEEDIV | GRU-XNet | 0.0003 | selected | validation | 8 | -7.64 | +0.492 |
| SEEDIV | GRU-XNet | 0.0003 | final | train | 8 | +5.48 | -0.443 |
| SEEDIV | GRU-XNet | 0.0003 | final | validation | 8 | +6.25 | -0.610 |
| SEEDIV | Matched BiLSTM | 0.001 | selected | train | 2 | +5.25 | -0.117 |
| SEEDIV | Matched BiLSTM | 0.001 | selected | validation | 2 | -19.44 | -0.036 |
| SEEDIV | Matched BiLSTM | 0.001 | final | train | 2 | +19.14 | -2.018 |
| SEEDIV | Matched BiLSTM | 0.001 | final | validation | 2 | +8.33 | +0.446 |
| SEEDIV | Matched BiLSTM | 0.0003 | selected | train | 2 | +0.62 | -0.026 |
| SEEDIV | Matched BiLSTM | 0.0003 | selected | validation | 2 | -19.44 | +0.832 |
| SEEDIV | Matched BiLSTM | 0.0003 | final | train | 2 | +0.31 | -0.033 |
| SEEDIV | Matched BiLSTM | 0.0003 | final | validation | 2 | +13.89 | +1.562 |
| SEEDIV | Local CBSAtt | 0.001 | selected | train | 2 | +3.40 | -0.019 |
| SEEDIV | Local CBSAtt | 0.001 | selected | validation | 2 | -5.56 | +0.003 |
| SEEDIV | Local CBSAtt | 0.001 | final | train | 2 | +6.79 | -0.457 |
| SEEDIV | Local CBSAtt | 0.001 | final | validation | 2 | -19.44 | +0.767 |
| SEEDIV | Local CBSAtt | 0.0003 | selected | train | 2 | +0.00 | -0.009 |
| SEEDIV | Local CBSAtt | 0.0003 | selected | validation | 2 | -11.11 | +0.008 |
| SEEDIV | Local CBSAtt | 0.0003 | final | train | 2 | +26.23 | -2.337 |
| SEEDIV | Local CBSAtt | 0.0003 | final | validation | 2 | +0.00 | -0.723 |
| SEEDIV | Author-checked EEGNet | 0.001 | selected | train | 8 | +8.26 | -0.029 |
| SEEDIV | Author-checked EEGNet | 0.001 | selected | validation | 8 | -3.47 | +0.070 |
| SEEDIV | Author-checked EEGNet | 0.001 | final | train | 8 | +32.56 | -1.508 |
| SEEDIV | Author-checked EEGNet | 0.001 | final | validation | 8 | -2.08 | -0.531 |
| SEEDIV | Author-checked EEGNet | 0.0003 | selected | train | 8 | +10.88 | -0.071 |
| SEEDIV | Author-checked EEGNet | 0.0003 | selected | validation | 8 | -2.78 | +0.211 |
| SEEDIV | Author-checked EEGNet | 0.0003 | final | train | 8 | +31.79 | -1.212 |
| SEEDIV | Author-checked EEGNet | 0.0003 | final | validation | 8 | -0.69 | -0.394 |
| DEAP | GRU-XNet | 0.001 | selected | train | 8 | +0.82 | -0.004 |
| DEAP | GRU-XNet | 0.001 | selected | validation | 8 | -3.20 | +0.004 |
| DEAP | GRU-XNet | 0.001 | final | train | 8 | +0.29 | +0.011 |
| DEAP | GRU-XNet | 0.001 | final | validation | 8 | +0.28 | -0.001 |
| DEAP | GRU-XNet | 0.0003 | selected | train | 8 | +1.43 | -0.020 |
| DEAP | GRU-XNet | 0.0003 | selected | validation | 8 | -6.34 | -0.002 |
| DEAP | GRU-XNet | 0.0003 | final | train | 8 | +4.82 | -0.284 |
| DEAP | GRU-XNet | 0.0003 | final | validation | 8 | +0.65 | -0.190 |
| DEAP | Matched BiLSTM | 0.001 | selected | train | 2 | +0.31 | -0.004 |
| DEAP | Matched BiLSTM | 0.001 | selected | validation | 2 | -5.88 | +0.005 |
| DEAP | Matched BiLSTM | 0.001 | final | train | 2 | +0.78 | -0.002 |
| DEAP | Matched BiLSTM | 0.001 | final | validation | 2 | -4.41 | -0.105 |
| DEAP | Matched BiLSTM | 0.0003 | selected | train | 2 | -0.55 | +0.030 |
| DEAP | Matched BiLSTM | 0.0003 | selected | validation | 2 | -6.96 | +0.119 |
| DEAP | Matched BiLSTM | 0.0003 | final | train | 2 | +1.54 | -0.014 |
| DEAP | Matched BiLSTM | 0.0003 | final | validation | 2 | -2.94 | -0.109 |
| DEAP | Local CBSAtt | 0.001 | selected | train | 2 | -1.44 | +0.001 |
| DEAP | Local CBSAtt | 0.001 | selected | validation | 2 | -0.69 | -0.018 |
| DEAP | Local CBSAtt | 0.001 | final | train | 2 | +25.16 | -1.143 |
| DEAP | Local CBSAtt | 0.001 | final | validation | 2 | -1.86 | -0.544 |
| DEAP | Local CBSAtt | 0.0003 | selected | train | 2 | +1.72 | -0.005 |
| DEAP | Local CBSAtt | 0.0003 | selected | validation | 2 | -2.75 | +0.000 |
| DEAP | Local CBSAtt | 0.0003 | final | train | 2 | +3.00 | -0.067 |
| DEAP | Local CBSAtt | 0.0003 | final | validation | 2 | +0.00 | -0.020 |
| DEAP | Author-checked EEGNet | 0.001 | selected | train | 8 | +0.43 | -0.036 |
| DEAP | Author-checked EEGNet | 0.001 | selected | validation | 8 | -2.91 | +0.275 |
| DEAP | Author-checked EEGNet | 0.001 | final | train | 8 | +0.08 | -0.024 |
| DEAP | Author-checked EEGNet | 0.001 | final | validation | 8 | -0.70 | +0.388 |
| DEAP | Author-checked EEGNet | 0.0003 | selected | train | 8 | +1.99 | -0.039 |
| DEAP | Author-checked EEGNet | 0.0003 | selected | validation | 8 | -3.00 | +0.137 |
| DEAP | Author-checked EEGNet | 0.0003 | final | train | 8 | +0.18 | -0.035 |
| DEAP | Author-checked EEGNet | 0.0003 | final | validation | 8 | -0.91 | +0.334 |

## Initialization and grouping sensitivity

Final-update source-validation BA. Both initializations are shown; do not select a best seed or grouping. These reused source cohorts and partial validation panels cannot establish optimizer robustness for the complete test task.

| Dataset | Model | Group | Arm | LR | Init 42 BA (%) | Init 91 BA (%) |
|---|---|---:|---|---:|---:|---:|
| SEEDIV | GRU-XNet | 1 | exposed | 0.001 | 38.89 | 38.89 |
| SEEDIV | GRU-XNet | 1 | exposed | 0.0003 | 33.33 | 44.44 |
| SEEDIV | GRU-XNet | 1 | unexposed | 0.001 | 33.33 | 44.44 |
| SEEDIV | GRU-XNet | 1 | unexposed | 0.0003 | 33.33 | 22.22 |
| SEEDIV | GRU-XNet | 2 | exposed | 0.001 | 33.33 | 16.67 |
| SEEDIV | GRU-XNet | 2 | exposed | 0.0003 | 16.67 | 16.67 |
| SEEDIV | GRU-XNet | 2 | unexposed | 0.001 | 44.44 | 22.22 |
| SEEDIV | GRU-XNet | 2 | unexposed | 0.0003 | 27.78 | 38.89 |
| SEEDIV | Author-checked EEGNet | 1 | exposed | 0.001 | 33.33 | 33.33 |
| SEEDIV | Author-checked EEGNet | 1 | exposed | 0.0003 | 33.33 | 33.33 |
| SEEDIV | Author-checked EEGNet | 1 | unexposed | 0.001 | 33.33 | 33.33 |
| SEEDIV | Author-checked EEGNet | 1 | unexposed | 0.0003 | 33.33 | 33.33 |
| SEEDIV | Author-checked EEGNet | 2 | exposed | 0.001 | 33.33 | 44.44 |
| SEEDIV | Author-checked EEGNet | 2 | exposed | 0.0003 | 27.78 | 33.33 |
| SEEDIV | Author-checked EEGNet | 2 | unexposed | 0.001 | 38.89 | 38.89 |
| SEEDIV | Author-checked EEGNet | 2 | unexposed | 0.0003 | 27.78 | 33.33 |
| DEAP | GRU-XNet | 1 | exposed | 0.001 | 50.00 | 43.53 |
| DEAP | GRU-XNet | 1 | exposed | 0.0003 | 50.00 | 45.29 |
| DEAP | GRU-XNet | 1 | unexposed | 0.001 | 48.43 | 36.47 |
| DEAP | GRU-XNet | 1 | unexposed | 0.0003 | 45.69 | 39.02 |
| DEAP | GRU-XNet | 2 | exposed | 0.001 | 50.00 | 50.00 |
| DEAP | GRU-XNet | 2 | exposed | 0.0003 | 53.24 | 41.90 |
| DEAP | GRU-XNet | 2 | unexposed | 0.001 | 48.18 | 50.00 |
| DEAP | GRU-XNet | 2 | unexposed | 0.0003 | 50.00 | 45.95 |
| DEAP | Author-checked EEGNet | 1 | exposed | 0.001 | 55.29 | 49.80 |
| DEAP | Author-checked EEGNet | 1 | exposed | 0.0003 | 54.51 | 64.90 |
| DEAP | Author-checked EEGNet | 1 | unexposed | 0.001 | 61.18 | 44.71 |
| DEAP | Author-checked EEGNet | 1 | unexposed | 0.0003 | 61.18 | 40.98 |
| DEAP | Author-checked EEGNet | 2 | exposed | 0.001 | 53.64 | 66.19 |
| DEAP | Author-checked EEGNet | 2 | exposed | 0.0003 | 56.07 | 66.19 |
| DEAP | Author-checked EEGNet | 2 | unexposed | 0.001 | 49.39 | 53.44 |
| DEAP | Author-checked EEGNet | 2 | unexposed | 0.0003 | 63.56 | 56.07 |

## Verification and practical scope

The author-code EEGNet check executes the pinned original TensorFlow function and checks the PyTorch port's binary/three-class outputs, deterministic dropout-zero training, moving BatchNorm states, max-norm constraints and parameter counts. Dropout 0.5 is retained during real fitting. EEGNet has 6,450 binary or 9,011 three-class trainable parameters. Inputs and balanced AdamW optimization differ from the authors' original applications; published scores and training trajectories are not reproduced. The local CBSAtt reference remains unauthenticated author code.

Independent source replay checks all source/data/config/split/scaler/draw/initial/checkpoint bindings, the final and source-selected predictions and selection rule, and source/test disjointness. It checks 688 exported probability metric sets and performs 192 final/selected state replays, including repeats when states coincide. The largest neural probability error is 2.98e-08. All twelve old prefixes pass without reading previous test probabilities. The post-hoc normalization replay reconstructs all float32 moment arrays exactly and reaches maximum probability error 2.98e-08. Replay is not a rerun of all optimizer updates or authentication of first-party EEG.

The sixteen tiny-batch recalibrations also reconstruct source moments exactly and replay original/after probabilities with maximum error 2.97e-08; the strict capacity criterion passes 14/16 original states and 16/16 recalibrated states. Public probability checks independently recompute all 640 full-population before/after metric sets and all 32 tiny-batch metric sets, with no raw EEG or checkpoint access.

Recorded original fitting time is 90.53 minutes; maximum PyTorch-allocated CUDA memory is 674.52 MiB. These exclude preparation, driver/background allocations, software tests and independent replay/normalization diagnostics. Checkpoints, waveform caches, isolated compatibility dependencies and downloaded third-party source remain local; bounded probabilities, source records and charts are exported.

## Publication consequence and next valid experiments

This phase distinguishes limited early learning, later training fit, stochastic/normalization sensitivity and source-validation behavior; it does not demonstrate a publishable model advantage or useful incremental EEG prediction on an independently evaluated test population. Simply increasing the budget or selecting the most favorable source panel is insufficient. No global recipe is selected from these partial panels.

The next broad controls require a new declaration with per-outer-fold source validation for learning rate, training duration and any source-only normalization variant, preserving every comparison and independent initialization/grouping repeats. EEGNet plus a participant-excluded video prior should then be compared with both context-only controls. Full joint all-corpus/LODO studies, historical provenance and first-party signal authentication still remain. Our [latest prior-work audit](GRU-XNet_Learning_Control_Research_Update_2026-10-06.md) includes direct familiar-video prior/physiology work published as a public preprint on 2 October; the general contextual-prior idea alone is not an established new contribution.

Changing the paper's main question requires a concrete evidence-backed proposal, author approval and an immediate fresh archive of the then-current paper. No pivot has been adopted.

## Evidence and reproduction

- [Source protocol](Learning_Control_Protocol_2026-10-06.md), [plan](../../results/development/learning_controls_2026-10-06/plan.json), [complete source replay](../../results/development/learning_controls_2026-10-06/verification.json), [public probability check](../../results/development/learning_controls_2026-10-06/public_verification.json)
- [Complete source comparison](../../results/development/learning_controls_2026-10-06/comparison.json), [per-fit summary](../../results/development/learning_controls_2026-10-06/summary.csv), [SEED-IV curves](../../results/development/learning_controls_2026-10-06/learning_curves_seediv.png), [DEAP curves](../../results/development/learning_controls_2026-10-06/learning_curves_deap.png)
- [Post-hoc normalization protocol](Source_BN_Diagnostic_Protocol_2026-10-06.md), [declaration](../../results/development/source_bn_diagnostic_2026-10-06/plan.json), [replay](../../results/development/source_bn_diagnostic_2026-10-06/verification.json), [all differences](../../results/development/source_bn_diagnostic_2026-10-06/comparison.json)
- [SEED-IV normalization plot](../../results/development/source_bn_diagnostic_2026-10-06/calibration_seediv.png), [DEAP normalization plot](../../results/development/source_bn_diagnostic_2026-10-06/calibration_deap.png)
- [Tiny-batch post-hoc declaration](../../results/development/memo_bn_check_2026-10-06/plan.json), [local replay](../../results/development/memo_bn_check_2026-10-06/verification.json), [all tiny contrasts](../../results/development/memo_bn_check_2026-10-06/comparison.json), [public check](../../results/development/memo_bn_check_2026-10-06/public_verification.json)

On a fresh local study with the previous verified caches and author audit, declare with `python scripts/learning_controls.py plan` before `run`. After completion run `python scripts/audit_learning_controls.py`, then the separately declared `source_bn_diagnostic.py run`/`verify` and `memo_bn_check.py run`/`verify`. Run `python scripts/verify_source_bn_export.py --root ../publication_runs` and `python scripts/verify_memo_bn_export.py --root ../publication_runs` before `python scripts/report_learning_controls.py`, then `python scripts/verify_learning_report.py --root ../publication_runs`. Preserve the post-hoc timing in any reproduction. A public checkout can use `python scripts/audit_learning_controls.py --export-only`, `python scripts/verify_source_bn_export.py`, `python scripts/verify_memo_bn_export.py` and `python scripts/verify_learning_report.py` without raw EEG or neural checkpoints. On Windows, Git may require repository-local `core.longpaths=true` for this existing long workspace path.
