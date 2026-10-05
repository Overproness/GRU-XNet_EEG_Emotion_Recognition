# GRU-XNet implementation and experiment status

Date: 5 October 2026. This records the work after the extracted data were supplied. The initial literature and publication review remains in [GRU-XNet_Publication_Review_2026-10-05.md](GRU-XNet_Publication_Review_2026-10-05.md).

**Outcome:** an audited, standalone PyTorch pipeline now runs on the RTX 3050. The corrected development experiments do not support the old 95.91% generalization claim. Further model development and matched evaluations are necessary before publication.

**Current author instruction:** joint training on DEAP, GAMEEMO and SEED-IV is the historical question, but a strong publishable contribution takes priority. Exploratory alternatives are authorized; adopting a changed question requires showing the findings and receiving explicit approval first. The historical manuscript is preserved. For a joint-training claim, reference architectures still need matched data/labels/montage, participant splits and budget. Single-dataset controls diagnose learning; LODO tests an unseen dataset. Low transfer scores alone cannot establish that reference methods overfit their own datasets.

## 1. Critical new finding: modified DEAP labels

The included DEAP pickle files contain **439 altered valence and 439 altered arousal ratings**. Every altered value equals `9 - original` when compared with the included participant-rating spreadsheets. All 1,280 dominance and liking pairs match the spreadsheet when joined by participant and **Experiment_id**, which identifies the video order in the Python files. Presentation `Trial` is a different ordering.

The maintained loader uses the spreadsheet valence ratings, checks the video correspondence, and records both the original pickle rating and metadata hash. It refuses unexplained rating differences. The downloaded data were not edited. The two included spreadsheet copies agree.

The author has now identified their own Kaggle bundle and supplied the three upstream Kaggle sources. **The discrepancy is inherited from the supplied DEAP source:** independently downloaded upstream `s01.dat` and participant-rating spreadsheet match the local SHA-256 hashes and reproduce 13 changed trials. Across the local bundle, all 439 affected trials originally have valence and arousal above 5, and all their binary valence labels flip.

All **304 training input/metadata files** match the three upstream version-1 archives in size and CRC32; six representative downloads additionally match SHA-256. The [provenance report](GRU-XNet_Dataset_Provenance_2026-10-05.md) records the URLs, versions, scope, and evidence. This establishes the supplied mirror-to-bundle correspondence; it does not independently authenticate the EEG against a first-party DEAP release. The corrected runs already use spreadsheet labels, so this finding requires no changes to those completed runs.

A [direct first-party access attempt](GRU-XNet_First_Party_DEAP_Check_2026-10-05.md) subsequently reached HTTP 503 on both official metadata and EEG archives with verified TLS; the author access-request server times out. Archived official format documentation independently confirms the trial join, rating order, array dimensions, and all 32 channel names/order. A full-signal `compare-deap` command is ready for an author-issued archive; no such archive has been obtained in this session.

## 2. What is implemented

The source of truth for new experiments is [gruxnet/](../../gruxnet), with commands in [PUBLICATION.md](../../PUBLICATION.md). The historical working/public scripts and outputs are retained for interpretation of past results. The repository initializer now uses lazy imports, avoiding an eager dependency on the missing historical parent-directory loader. New training is independent of that loader.

| Concern | Implemented resolution |
| --- | --- |
| SEED-IV sadness mapped to positive | Sadness/fear → negative; happiness → positive; neutral excluded. |
| SEED trial keys ordered lexicographically | Numeric `_eeg1`…`_eeg24` ordering with exact-key validation; session labels checked against ReadMe.txt. |
| Only three SEED-IV subjects used previously | All 15 participants, all three sessions, 1,080 source trials verified. |
| GAMEEMO game IDs used as arousal labels in a valence task | Real participant SAM valence extracted from the supplied graphical forms. All 112 valence choices visually checked. |
| Incompatible neutral/positive definitions | Explicit strict valence policy; neutral or midpoint trials excluded. Semantic differences between subjective ratings and film categories are disclosed. |
| Random windows/trials shared subjects across partitions | Subject-disjoint splits; all windows, trials, and sessions of a subject stay together. |
| Dataset-local subject IDs collided | Dataset-qualified IDs, such as `DEAP:S01`. |
| Augmentation before splitting and invented parent IDs | Original data only; augmentation operates on training batches after subject partitioning. |
| GAMEEMO overlapping-window leakage | Fixed four-second windows with no overlap. |
| “Cross-dataset” scores from a pooled random test | Distinct pooled subject, true leave-one-dataset-out, and individual pooled LOSO-fold protocols. |
| Channel-count padding mixed anatomical locations | Common 14 electrodes by name; optional canonical 62-electrode cache with explicit masks. |
| Padded channel CNN biases could affect predictions | Masking both CNN inputs and CNN outputs after convolution/normalization. |
| Sampling rates and physical durations differed | Fixed 128 Hz; proper SEED-IV 200→128 polyphase resampling; four-second windows, no duration interpolation. |
| GRU received a repeated static vector in one legacy variant | Thirteen actual STFT time positions preserved through frequency-only CNN pooling and recurrence. |
| Large recurrent input on 6 GB hardware | Explicit compact variant with a 128-dimensional feature projection before the BiGRU. |
| GAMEEMO/window-duration dominance | Equal dataset/class/trial training contribution, inverse window count per original trial. |
| Inconsistent results/checkpoint/figure counts | Run-specific manifests, checkpoint hashes, window and trial predictions, reproducible metrics, and generated figures. |
| Unsupported augmentation/ablation claims | Actual three-transform training recipe documented; CLI controls provided for matched baselines and ablations. Historical tables are not reused as new evidence. |

## 3. Verified data inventory

| Dataset | Subjects | Source trials | Excluded neutral trials | Eligible trials | Windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEAP | 32 | 1,280 | 16 | 1,264 | 18,960 |
| GAMEEMO | 28 | 112 | 19 | 93 | 6,882 |
| SEED-IV | 15 | 1,080 | 270 | 810 | 27,405 |
| Total | 75 | 2,472 | 305 | 2,167 | 53,247 |

The prepared cache contains **2,167 source-trial files**, approximately **1.42 GiB** of EEG windows. No duplicate cached-trial SHA-256 hashes were found. Source hashes and trial/window lineage are retained. Old augmented arrays and the duplicated nested SEED-IV directory were not included.

Useful artifacts:

- Dataset audit (local workspace evidence: `publication_runs/cache_common14/audit/dataset_audit.json`)
- GAMEEMO ratings (local workspace evidence: `publication_runs/cache_common14/audit/gameemo_sam_ratings.csv`)
- SAM visual review, subjects 1–14 (local workspace evidence: `publication_runs/cache_common14/audit/sam_valence_review_1.png`) and subjects 15–28 (local workspace evidence: `publication_runs/cache_common14/audit/sam_valence_review_2.png`)
- Window manifest (local workspace evidence: `publication_runs/cache_common14/manifest.csv`), source lineage (local workspace evidence: `publication_runs/cache_common14/lineage.json`), and [preprocessing configuration](../../results/development/cache_common14/prepared.json)
- [Exact independent subject partitions](../../results/development/full_subject_seed42/split_audit.json)

## 4. Completed experiments

These are **development results**. The v2 pilots retain up to four uniformly spaced windows per original trial in every partition. They are not full benchmark evaluations. All percentages below refer to trial-level balanced accuracy, averaged across datasets where applicable.

| Run | Model | Protocol | Epochs completed | Test trials | Test macro balanced accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| `pilot_subject_seed42` | Compact v1, frequency averaging | Pooled, independent subjects; pilot | 3 | 319 | 50.00% |
| `full_subject_seed42` | Compact v1, frequency averaging | Pooled, independent subjects; all windows | 6 | 319 | 50.00% |
| `pilot_v2_subject_seed42` | Compact v2, retained frequency positions | Pooled, independent subjects; pilot | 6 | 319 | 50.00% |
| `pilot_v2_lodo_DEAP_seed42` | Compact v2 | Train GAMEEMO+SEED-IV; test DEAP; pilot | 3 | 1,264 | 50.00% |
| `pilot_v2_lodo_GAMEEMO_seed42` | Compact v2 | Train DEAP+SEED-IV; test GAMEEMO; pilot | 3 | 93 | 50.00% |
| `pilot_v2_lodo_SEEDIV_seed42` | Compact v2 | Train DEAP+GAMEEMO; test SEED-IV; pilot | 3 | 810 | 50.00% |
| `bandpower_subject_seed42` | Log-bandpower + logistic regression | Same full subject split as the neural run | Not applicable | 319 | **50.97%** |

The full neural run used 38,326 training, 7,453 validation, and 7,468 test windows. Early stopping ended the run after six epochs, with epoch 1 selected by validation balanced accuracy. The selected checkpoint predicted every test trial as positive. Its pooled trial accuracy is 50.47%; per-dataset accuracies differ because label proportions differ. The 50% balanced-accuracy result correctly exposes this trivial predictor.

The bandpower reference chooses regularization on validation only and fits normalization to weighted training data only. Its test trial balanced accuracies are **51.25% DEAP, 43.33% GAMEEMO, and 58.33% SEED-IV**. Pooled trial accuracy is 57.37%, but averaging per-dataset balanced accuracy gives 50.97%. This is an exploratory representation baseline, not proof of superiority to the neural architecture. It retains amplitude information that per-window z-scoring removes in the neural inputs.

The v1 run took approximately **506 seconds**, with **149 MiB** peak CUDA tensor allocation. V2 pilots used approximately **154 MiB**. These allocation measurements exclude the CUDA context, driver allocations, and other applications. The confirmed hardware is an RTX 3050 Laptop GPU with 6,144 MiB VRAM.

The maintained v2 architecture retains four reduced frequency positions before the feature projection, instead of the v1 frequency mean. It has **537,442 parameters**, compared with 365,410 in v1. Both preserve 13 real time positions. Neither is the historical model that generated the course-project accuracy.

The training-only capacity diagnostic (local workspace evidence: `publication_runs/training_capacity_check.json`) reaches 100% accuracy on 24 fixed training windows from six trials with dropout and augmentation disabled. This verifies that the model/gradients can memorize a small batch. It demonstrates no performance on unseen data and does not resolve underfitting/generalization in ordinary training.

Key run outputs:

- [Full neural metrics](../../results/development/full_subject_seed42/test_metrics.json), [training history](../../results/development/full_subject_seed42/history.json), and [test confusion matrices](../../results/development/full_subject_seed42/test_confusion.png)
- [V2 subject pilot](../../results/development/pilot_v2_subject_seed42/test_metrics.json)
- LODO pilots: [DEAP](../../results/development/pilot_v2_lodo_DEAP_seed42/test_metrics.json), [GAMEEMO](../../results/development/pilot_v2_lodo_GAMEEMO_seed42/test_metrics.json), [SEED-IV](../../results/development/pilot_v2_lodo_SEEDIV_seed42/test_metrics.json)
- [Bandpower baseline](../../results/development/bandpower_subject_seed42/test_metrics.json)

The subsequently authorized [DEAP-only EEGNet control](GRU-XNet_DEAP_Control_2026-10-05.md) is complete. It uses all 32 DEAP electrodes, raw filtered EEG, normalization fitted only to 22 training participants, five separate validation and five test participants, no augmentation, and a fixed-rate Adam optimizer. Its 100-epoch maximum / 25-epoch minimum / patience-15 plan stopped after **28 epochs** and selected **epoch 13**. The selected checkpoint's trial balanced accuracies are **66.91% training, 56.07% validation, and 46.25% test**; test AUROC is 0.5007. The subject-bootstrap interval, 40.28–52.10%, includes chance. Four of five test participants receive a constant class prediction despite both classes appearing in pooled predictions.

The new canonical32 cache contains 18,960 windows. Every original label/sample identity and all 14 shared-electrode waveform values exactly match the earlier DEAP cache, and the participant assignment matches the previous development split. The run takes 874 seconds, with 117.3 MiB peak allocated CUDA tensors. All 3,000 test probabilities, metrics, validation selection, participant splits, and training-only scaler values reproduce. This control demonstrates learning on training data without useful held-out generalization. It is not a matched architecture/normalization ablation and does not establish superiority of joint training. [Metrics](../../results/development/deap_eegnet32_trainchannel_seed42/test_metrics.json), [verification](../../results/development/deap_eegnet32_trainchannel_seed42/verification.json), [diagnostic curves](../../results/development/deap_eegnet32_trainchannel_seed42/development_diagnostics.png).

The subsequent [exploratory controls](GRU-XNet_Exploration_Findings_2026-10-05.md) are also complete. A matched DEAP per-window normalization control gives 46.67% test trial balanced accuracy and AUROC 0.4433; its 3,000 test probabilities and selection reproduce. Five SEED-IV participant folds give 67.13% binary balanced accuracy with common-14 classical trial features and 63.89% with native 62 electrodes. Native four-class recognition gives 36.85% and 36.11%, respectively, with 25% chance. Adding DEAP/GAMEEMO training trials under matched SEED-IV evaluation lowers binary balanced accuracy to 60.93% (global scaler) or 60.56% (separate training-fitted scalers). All validation candidate fits and held-out probabilities reproduce. These are development diagnostics, not a completed neural comparison or approved question change.

## 5. Verification completed

The [completed all-target extension](GRU-XNet_Multitarget_Transfer_Findings_2026-10-05.md) brings this feature-level investigation to **165 neural runs, 285 replayed selected checkpoints, 42,225 neural probability rows, 160 refitted linear validation candidates, and 5,954 linear probability rows**. All shared-joint versus target-only neural participant intervals include zero; simple separate heads do not consistently help. Exact waveform feature re-extraction reproduces every previous feature row. The current suite passes **32 tests**. These target-centered controls are distinct from a single jointly selected model evaluated on all three datasets; full architecture/LOSO/LODO work remains incomplete.

The first [neural pooling phase](GRU-XNet_Neural_Transfer_Investigation_2026-10-05.md) completed 75 feature-MLP runs. Target-only SEED-IV averaged 63.30% BA, shared joint 61.27%, and exposure-matched shared joint 62.38%. Participant intervals for those neural drops include zero. All 135 selected checkpoints and 21,870 probabilities reproduced, as did 80 anchored linear candidates and 3,240 probabilities. That milestone passed **27 tests**, including paired head initialization, correct head-gradient routing, and exact class/dataset quotas with matching target draw prefixes. The later extension above completes the remaining feature-level targets; full GRU-XNet ablations remain unfinished.

- **The initial pipeline phase passed 24 tests**, covering genuine label handling/recovery, SAM geometry, numeric SEED ordering, named electrode alignment, physical resampling and bandpower, subject/trial disjointness for all three protocol types, balanced sampling, channel-mask invariance, measured STFT time, trial-level metric aggregation, source catalog completeness/hash binding, independent detection of EEG-sample changes versus label-only changes, training-only channel scaling/amplitude preservation, registered EEGNet classifier gradients/updates, max-norm constraints, validation tie selection, native feature frequencies/amplitude/tail handling, participant-fold coverage, native/coarse label views, and equal dataset/class pooling weights. Additional transfer-control tests bring the current total to 32.
- Six neural runs were independently re-evaluated from their selected checkpoint. Saved test probabilities and reported window/trial/per-dataset metrics, including bootstrap intervals, reproduce. Verification rehashes the cached EEG and checks the saved partition and checkpoint hashes.
- The classical baseline's probabilities reproduce from exported scaler statistics and classifier coefficients without its live sklearn model.
- Python source compilation and Git whitespace checks pass.
- Source EEG files, previous model outputs, and `report.tex` were not overwritten.

## 6. Work still needed for publication

The [GitHub configuration review](GRU-XNet_GitHub_Configuration_Review_2026-10-05.md) compares five public repositories at fixed commits. It confirms precedents for 128 Hz, four-second windows, and learning rate 0.001, and identifies useful normalization/training controls plus differences in label boundaries and validation splits. All 51 saved source files passed hash checks. The authorized DEAP-only control has subsequently run as described above; repository code still does not authenticate the first-party DEAP release.

1. **Complete first-party DEAP authentication.** The supplied Kaggle download chain and upstream integrity checks are now established. Obtain/compare a first-party distribution before final confirmatory experiments, or disclose the remaining mirror limitation. The upstream label modification and spreadsheet-based recovery need disclosure.
2. **Stabilize the corrected neural baseline using training/validation only.** The memorization check succeeds while ordinary joint GRU-XNet runs remain single-class predictors. The DEAP-only raw EEGNet with training-fitted normalization learns training patterns but still fails on held-out participants; longer training and amplitude preservation were not sufficient in that configuration. Examine subject effects/calibration and matched normalization, regularization, augmentation, and architecture controls. Do not use held-out labels or target/test normalization statistics. Further tuning this already-inspected test split produces development evidence only.
3. **Run matched full evaluations.** Paired predeclared subject folds, every LODO target, GRU vs LSTM, attention vs no attention, CNN only, no augmentation, and common vs canonical electrodes. Document subject/fold variation and training budgets. No complete LOSO study or final architecture ablation suite has run yet.
4. **Revise the manuscript around the final measured protocol.** The old 95.91% result, comparison tables, claims of 11 active augmentation methods, three-subject SEED subset, and architectural description need replacement or explicit historical qualification. Novelty should focus on the auditable heterogeneous-data study and measured findings rather than claiming CNN–BiGRU–attention itself is new.
5. **Incorporate verified current literature and calibrate the claim to the venue.** The existing literature review identifies the close 2024/2025 precedents and 2026 cross-dataset work. Do not turn the present development runs into a claim of state-of-the-art accuracy or deployment robustness.

publication_methods.tex (local workspace evidence: `publication_methods.tex`) contains a draft replacement methods section matching the v2 pipeline. It is not automatically inserted into the historical manuscript. No local `pdflatex` or `latexmk` executable was found, so this fragment has not been compiled. A final publication manuscript and submission have not been produced.
