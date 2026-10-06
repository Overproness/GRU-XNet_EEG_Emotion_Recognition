# Full-width GRU-XNet and contextual-prior findings — 6 October 2026

All 680 full-model fits are complete and independently replayed: 360 SEED-IV and 320 DEAP. There are also 170 source-only context calibration cells, with all 680 regularization candidates independently refitted. The study changes neither the manuscript nor the main research question. It is one predeclared participant/video grouping with a fixed 200-update training budget, not a historical-score reproduction or a submission-ready result.

The primary matched reference comparisons establish no consistent GRU-XNet advantage: all sixteen reference-versus-GRU accuracy/log-loss intervals include zero. On shared DEAP videos, EEG-plus-context gives 78.10% balanced accuracy versus 77.36% for the raw video prior; its +0.74 percentage-point interval is [-0.94,+3.11]. All eight DEAP EEG-plus-context contrasts against the two context-only comparators, and all sixteen neural within-video alignment intervals, include zero. These fits do not establish useful incremental EEG prediction. The fixed budget, adapted inputs and one base initialization scheme leave better optimization and representations open; this is not a conclusion that EEG lacks information.

## Matched primary results

Balanced accuracy is a percentage; balanced log loss is in natural-log units (lower is better). Shared-video and unseen-video arms have identical held-out people, validation/test trials and matched training person/class counts. The study uses all 1080 SEED-IV trials with neutral/negative/positive classes, and all 1264 retained DEAP trials with individual binary valence. These different targets should not be compared as a common leaderboard. The earlier feature controls used the same first 40 seconds, but more updates, larger batches and much smaller models, so their scores are not architecture-only comparisons.

| Dataset | Model | Shared-video BA | Unseen-video BA | Shared log loss | Unseen log loss |
|---|---|---:|---:|---:|---:|
| SEEDIV | Full GRU-XNet | 39.32 | 36.11 | 1.20 | 1.28 |
| SEEDIV | Matched BiLSTM | 39.38 | 36.91 | 1.17 | 1.21 |
| SEEDIV | Local CBSAtt | 38.33 | 38.77 | 1.22 | 1.33 |
| SEEDIV | GRU + context | 97.72 | 34.32 | 0.22 | 1.20 |
| SEEDIV | Video prior (no EEG) | 100.00 | 33.33 | 0.18 | 1.15 |
| SEEDIV | Calibrated context (no EEG) | 100.00 | 33.33 | 0.50 | 1.11 |
| DEAP | Full GRU-XNet | 49.28 | 46.54 | 0.70 | 0.70 |
| DEAP | Matched BiLSTM | 48.21 | 48.34 | 0.70 | 0.70 |
| DEAP | Local CBSAtt | 49.05 | 51.10 | 0.71 | 0.71 |
| DEAP | GRU + context | 78.10 | 47.65 | 0.50 | 0.70 |
| DEAP | Video prior (no EEG) | 77.36 | 50.00 | 0.50 | 0.71 |
| DEAP | Calibrated context (no EEG) | 77.78 | 48.78 | 0.52 | 0.69 |

## Matched reference-model contrasts

Each reference minus EEG-only GRU-XNet, using paired crossed participant/video percentile ranges. Positive balanced-accuracy differences (percentage points) or negative balanced-log-loss differences favor the reference. An interval spanning zero does not establish equivalence. The single base initialization scheme and fixed training budget leave optimization and initialization robustness unresolved; CNN/recurrent widths are matched for the BiLSTM swap, but parameter counts differ.

| Dataset | Reference | Arm | Statistic | Reference minus GRU [95% interval] |
|---|---|---|---|---|
| SEEDIV | Matched BiLSTM | exposed | BA | +0.06 [-6.11, +6.36] |
| SEEDIV | Matched BiLSTM | unexposed | BA | +0.80 [-4.57, +6.23] |
| SEEDIV | Local CBSAtt | exposed | BA | -0.99 [-6.54, +4.51] |
| SEEDIV | Local CBSAtt | unexposed | BA | +2.65 [-3.27, +8.89] |
| SEEDIV | Matched BiLSTM | exposed | logloss | -0.03 [-0.10, +0.05] |
| SEEDIV | Matched BiLSTM | unexposed | logloss | -0.08 [-0.18, +0.02] |
| SEEDIV | Local CBSAtt | exposed | logloss | +0.02 [-0.06, +0.12] |
| SEEDIV | Local CBSAtt | unexposed | logloss | +0.05 [-0.08, +0.18] |
| DEAP | Matched BiLSTM | exposed | BA | -1.08 [-7.00, +5.16] |
| DEAP | Matched BiLSTM | unexposed | BA | +1.80 [-2.69, +6.44] |
| DEAP | Local CBSAtt | exposed | BA | -0.23 [-6.93, +6.92] |
| DEAP | Local CBSAtt | unexposed | BA | +4.57 [-1.22, +10.45] |
| DEAP | Matched BiLSTM | exposed | logloss | +0.00 [-0.00, +0.01] |
| DEAP | Matched BiLSTM | unexposed | logloss | +0.00 [-0.01, +0.01] |
| DEAP | Local CBSAtt | exposed | logloss | +0.01 [-0.01, +0.03] |
| DEAP | Local CBSAtt | unexposed | logloss | +0.01 [-0.01, +0.03] |

## Does EEG improve on context alone?

Training context excludes every label from the participant receiving that feature. Validation/test context comes only from outer training labels. A video receives Laplace-one per-class counts; an unknown video receives the global training distribution. The EEG-plus-context model adds the fixed log prior to full GRU logits and learns the residual; it has exactly the same EEG weights at initialization and parameter count as EEG-only GRU. Both the raw prior and source-validation-selected logistic calibration are retained as context-only controls.

These are EEG-plus-context minus each context-only comparator, with paired crossed participant/video 95% percentile intervals. Positive BA differences (percentage points) and negative log-loss differences favor EEG-plus-context. All intervals are conditional on these fixed fits/cohorts/grouping and unadjusted exploratory comparisons; improvement in one statistic does not imply general information gains or equivalence on the other.

| Dataset | Arm | Context-only comparator | Statistic | Difference [95% interval] |
|---|---|---|---|---|
| SEEDIV | exposed | Video prior (no EEG) | BA | -2.28 [-4.63, -0.68] |
| SEEDIV | unexposed | Video prior (no EEG) | BA | +0.99 [-1.91, +4.14] |
| SEEDIV | exposed | Calibrated context (no EEG) | BA | -2.28 [-4.63, -0.68] |
| SEEDIV | unexposed | Calibrated context (no EEG) | BA | +0.99 [-1.91, +4.14] |
| SEEDIV | exposed | Video prior (no EEG) | logloss | +0.04 [+0.01, +0.08] |
| SEEDIV | unexposed | Video prior (no EEG) | logloss | +0.05 [-0.01, +0.11] |
| SEEDIV | exposed | Calibrated context (no EEG) | logloss | -0.28 [-0.31, -0.24] |
| SEEDIV | unexposed | Calibrated context (no EEG) | logloss | +0.09 [+0.04, +0.16] |
| DEAP | exposed | Video prior (no EEG) | BA | +0.74 [-0.94, +3.11] |
| DEAP | unexposed | Video prior (no EEG) | BA | -2.35 [-6.78, +2.21] |
| DEAP | exposed | Calibrated context (no EEG) | BA | +0.32 [-0.99, +1.90] |
| DEAP | unexposed | Calibrated context (no EEG) | BA | -1.14 [-7.50, +4.94] |
| DEAP | exposed | Video prior (no EEG) | logloss | -0.00 [-0.02, +0.02] |
| DEAP | unexposed | Video prior (no EEG) | logloss | -0.01 [-0.02, +0.00] |
| DEAP | exposed | Calibrated context (no EEG) | logloss | -0.02 [-0.04, +0.00] |
| DEAP | unexposed | Calibrated context (no EEG) | logloss | +0.00 [-0.00, +0.01] |

SEED-IV's emotional labels are assigned to the videos and shared across participants: a known video can identify its target using training labels, without EEG. This is an intended contextual diagnostic, not a deployable EEG classifier or evidence that an EEG model used video identity. DEAP videos can receive both individual valence classes, so normative context remains imperfect. Neither dataset supplies test ratings to the models. Cross-fitting removes direct own-label leakage, but training videos have known priors while validation videos are unseen; context features consequently shift in distribution across roles. A residual model may fail to improve because of optimization, budget or that shift, even if EEG contains information.

## Correct individual EEG versus exchanged EEG within the same video

This no-refit supplement was declared while the full-model batch was running, before aggregate primary results. For every trial, compare its selected prediction with the scores obtained from **every other held-out participant who watched the same video in the same model/cell**. The video's source-only context and normalizer are identical, so exchanging verified donor probabilities exactly represents EEG exchange with recipient labels/context retained. Average correctness/log loss across donors, without ensembling probabilities. All 2344 trials have at least one other held-out donor, so the diagnostic excludes none.

These DEAP differences are correctly aligned minus exchanged EEG. Positive BA (percentage points) and negative log loss favor the correctly paired recording. Intervals use paired dyadic recipient-and-donor participant weights and video weights, with identical observed-class denominators. Nonestimable draws are disclosed rather than replaced. All six models, both arms and both SEED tasks are retained in the machine-readable comparisons.

| Model | Arm | Statistic | Aligned minus exchanged [crossed dyadic 95% interval] | Nonestimable draws |
|---|---|---|---:|---:|
| Full GRU-XNet | exposed | BA | -0.43 [-4.10, +3.13] | 0 |
| Full GRU-XNet | exposed | logloss | +0.00 [-0.00, +0.01] | 0 |
| Full GRU-XNet | unexposed | BA | -1.93 [-6.70, +1.60] | 0 |
| Full GRU-XNet | unexposed | logloss | +0.00 [-0.00, +0.01] | 0 |
| Matched BiLSTM | exposed | BA | -0.19 [-4.26, +3.42] | 0 |
| Matched BiLSTM | exposed | logloss | +0.00 [-0.01, +0.01] | 0 |
| Matched BiLSTM | unexposed | BA | -0.83 [-5.51, +3.55] | 0 |
| Matched BiLSTM | unexposed | logloss | +0.00 [-0.01, +0.01] | 0 |
| Local CBSAtt | exposed | BA | -1.34 [-6.53, +3.54] | 0 |
| Local CBSAtt | exposed | logloss | -0.00 [-0.01, +0.01] | 0 |
| Local CBSAtt | unexposed | BA | -0.08 [-5.86, +5.09] | 0 |
| Local CBSAtt | unexposed | logloss | +0.00 [-0.02, +0.02] | 0 |
| GRU + context | exposed | BA | -0.16 [-1.52, +1.07] | 0 |
| GRU + context | exposed | logloss | +0.00 [-0.01, +0.02] | 0 |
| GRU + context | unexposed | BA | -1.82 [-6.19, +1.93] | 0 |
| GRU + context | unexposed | logloss | +0.00 [-0.01, +0.01] | 0 |

Both source-only context controls are invariant to exchange. SEED-IV's assigned video labels imply zero aggregate differences for all models, including the symmetric pair-weighted bootstrap; that mathematical sanity check passes. This does not rule out useful EEG for classifying *unseen* SEED-IV videos. On DEAP, an alignment effect would support recording/rating association conditional on video in these fits. It could also involve stable participant traits, demographics or artifacts, and would not identify a causal physiological emotion mechanism. It uses a limited donor cohort and one fixed development partition.

## Material sensitivity

Unseen-video minus shared-video balanced accuracy; all six models retained. The arm change replaces training recordings/content and can change difficulty, order or subject responses. It does not isolate a causal effect of identity and is not evidence of joint multi-corpus negative transfer.

| Dataset | Model | Difference in percentage points [crossed 95% interval] |
|---|---|---:|
| SEEDIV | Full GRU-XNet | -3.21 [-9.57, +2.96] |
| SEEDIV | Matched BiLSTM | -2.47 [-8.40, +3.27] |
| SEEDIV | Local CBSAtt | +0.43 [-5.49, +6.11] |
| SEEDIV | GRU + context | -63.40 [-66.42, -60.31] |
| SEEDIV | Video prior (no EEG) | -66.67 [-66.67, -66.67] |
| SEEDIV | Calibrated context (no EEG) | -66.67 [-66.67, -66.67] |
| DEAP | Full GRU-XNet | -2.75 [-9.00, +3.51] |
| DEAP | Matched BiLSTM | +0.13 [-6.11, +6.27] |
| DEAP | Local CBSAtt | +2.05 [-2.88, +6.96] |
| DEAP | GRU + context | -30.45 [-37.13, -22.08] |
| DEAP | Video prior (no EEG) | -27.36 [-33.49, -19.87] |
| DEAP | Calibrated context (no EEG) | -29.00 [-35.83, -20.85] |

All architecture contrasts, both primary and secondary SEED-IV tasks, and both uncertainty calculations are preserved in the machine-readable comparisons, including nonsignificant results. SEED-IV secondary binary results condition positive probability on positive/negative mass and exclude all 270 neutral trials. The resampling ranges are exploratory: nominal coverage has not been demonstrated for these small fixed partitions and dyadic dependence, and no uniformly valid bootstrap claim is made.

## Budget and training behavior

| Dataset | Model | Arm | Parameters | Median selected update | Selected at last update | Mean selected train BA (%) | Mean selected validation BA (%) |
|---|---|---|---:|---:|---:|---:|---:|
| SEEDIV | Full GRU-XNet | exposed | 7,567,747 | 130 | 6/45 | 48.94 | 50.12 |
| SEEDIV | Full GRU-XNet | unexposed | 7,567,747 | 100 | 5/45 | 47.43 | 52.72 |
| SEEDIV | Matched BiLSTM | exposed | 9,534,851 | 100 | 6/45 | 47.59 | 50.62 |
| SEEDIV | Matched BiLSTM | unexposed | 9,534,851 | 150 | 5/45 | 47.30 | 51.11 |
| SEEDIV | Local CBSAtt | exposed | 3,568,771 | 90 | 0/45 | 51.32 | 50.86 |
| SEEDIV | Local CBSAtt | unexposed | 3,568,771 | 120 | 0/45 | 53.59 | 53.46 |
| SEEDIV | GRU + context | exposed | 7,567,747 | 150 | 6/45 | 99.15 | 39.26 |
| SEEDIV | GRU + context | unexposed | 7,567,747 | 130 | 9/45 | 99.84 | 39.75 |
| DEAP | Full GRU-XNet | exposed | 7,567,618 | 55 | 1/40 | 52.44 | 55.45 |
| DEAP | Full GRU-XNet | unexposed | 7,567,618 | 70 | 0/40 | 51.66 | 55.42 |
| DEAP | Matched BiLSTM | exposed | 9,534,722 | 65 | 2/40 | 52.29 | 55.90 |
| DEAP | Matched BiLSTM | unexposed | 9,534,722 | 60 | 1/40 | 51.81 | 56.40 |
| DEAP | Local CBSAtt | exposed | 3,568,642 | 110 | 0/40 | 55.21 | 60.95 |
| DEAP | Local CBSAtt | unexposed | 3,568,642 | 55 | 0/40 | 54.96 | 61.32 |
| DEAP | GRU + context | exposed | 7,567,618 | 115 | 4/40 | 78.22 | 54.15 |
| DEAP | GRU + context | unexposed | 7,567,618 | 95 | 5/40 | 78.24 | 53.93 |

Recorded neural fitting time is 149.75 minutes; maximum PyTorch-allocated CUDA memory is 810.93 MiB. These exclude data preparation, environment/driver memory, reporting, independent replay and context fitting. Per-cell training metrics, complete 20-point validation histories, resource counts and selected checkpoint positions are retained. A checkpoint chosen at the budget limit is a reason to consider a separately predeclared longer experiment; a checkpoint selected earlier is not proof of convergence. Small validation populations and repeated checkpoint comparisons introduce selection uncertainty.

## Implementation fidelity and verification

The dynamic model retains the full original 32/64/128 independent electrode CNN widths, two-layer bidirectional 128-unit GRU, four-head attention with residual normalization and the 256/128 classifier. The matched BiLSTM swaps only recurrence; same hidden width does not mean equal parameter count. Both use the same initialization seed and common CNN initialization, but different recurrent parameter sizes consume different random draws before attention/head initialization; those later weights are not identical. GRU versus GRU-plus-context starts with the entire EEG state identical. The second reference reproduces the local CBSAtt implementation, including global pooling and its one-step, single-layer LSTM. It is not certified author code or a reproduction of published performance.

Inputs are adapted to the same physical common14 channels and 40-second offline prefix, with real 4–40 Hz bins and 79 STFT frames, producing nine ordered recurrent steps after three pools. This differs from the historical interpolated 129×126/max-channel/augmented pooled configuration. Grouped convolutions preserve distinct weights and BatchNorm statistics per electrode. Weight-mapped forward outputs for both binary and three-class heads match the original local implementations within the predeclared tolerance. All 2344 input spectrograms independently match SciPy exactly after double-precision transform calculation and float32 storage; the original tolerance was retained through the preflight precision correction.

Every fit binds its input and source hashes, original trial split, canonical training draws, initial state, training-only scaler, participant-excluded context, history, selected checkpoint and predictions. Independent verification reconstructs all these, replays selected training/validation/test predictions and selection rules, refits every context candidate and checks once-per-trial out-of-fold coverage. This is selected-checkpoint replay, not rerunning every neural optimization trajectory. Largest neural probability error: SEED-IV 2.98e-08; DEAP 2.98e-08. Analysis was independently recomputed exactly before reporting. The scientific-control suite has 64 passing tests (`python -m pytest -q`); software tests alone do not establish scientific validity or novelty.

The first partial batch was stopped after 15 completed fits when a training-mode CBSAtt dropout/pooling discrepancy was found. Revision 2 corrects the ordering and adds training-mode checks. The partial fits and original source hashes are preserved as excluded preflight evidence; all 680 reported fits were restarted under the corrected declaration without changing splits, budgets or hyperparameters.

## Remaining publication work

These development results do not settle full-model convergence, independent optimizer/grouping robustness, unseen-corpus transfer, all-corpus joint learning, original first-party DEAP signal authentication, or historical 95.91% provenance. The manuscript still needs an approved contribution, revised claims, matched final experiments and compilable source figures. No new research question has been adopted. A concrete proposal requires the author's approval, and the then-current paper must be archived immediately before an approved pivot.

## Evidence and reproduction

- [Predeclared protocol](Full_Context_Control_Protocol_2026-10-06.md)
- [Additional primary prior work and inference limits](GRU-XNet_Context_Alignment_Research_Update_2026-10-06.md)
- [Public-probability reanalysis check](../../results/development/full_context_public_reanalysis_2026-10-06.json)
- [Machine-readable plan](../../results/development/full_context_v2_plan_2026-10-06.json)
- [Within-video declaration](../../results/development/within_video_alignment_plan_2026-10-06.json), [DEAP alignment contrasts](../../results/development/within_video_alignment/comparison_deap.json), [SEED-IV sanity controls](../../results/development/within_video_alignment/comparison_seediv.json)
- [SEED-IV complete contrasts](../../results/development/full_context_v2_seediv/comparison.json), [verification](../../results/development/full_context_v2_seediv/verification.json), [diagnostics](../../results/development/full_context_v2_seediv/diagnostics.csv), [scores](../../results/development/full_context_v2_seediv/comparison.png), [validation curves](../../results/development/full_context_v2_seediv/validation_curves.png)
- [DEAP complete contrasts](../../results/development/full_context_v2_deap/comparison.json), [verification](../../results/development/full_context_v2_deap/verification.json), [diagnostics](../../results/development/full_context_v2_deap/diagnostics.csv), [scores](../../results/development/full_context_v2_deap/comparison.png), [validation curves](../../results/development/full_context_v2_deap/validation_curves.png)

With the already prepared local datasets/caches, run `python scripts/full_context_controls_v2.py plan --root ../publication_runs` once on a fresh study, prepare each dataset, run `python scripts/audit_full_context_inputs.py` and `python scripts/audit_full_context_models_v2.py`. Declare the supplement with `python scripts/within_video_alignment.py plan` before running `python scripts/full_context_controls_v2.py batch --root ../publication_runs`; after both primary studies pass replay, run `python scripts/within_video_alignment.py analyze` and `python scripts/report_full_context.py`. Existing declarations/caches are protected; the batch resumes only hash-matching fitted records. A public checkout can run `python scripts/verify_publication_export.py --export-only` and `python scripts/verify_full_context_export.py` to check artifact integrity and recompute all reported probability-based numbers without raw EEG or checkpoints. Selected-model replay requires those local files.
