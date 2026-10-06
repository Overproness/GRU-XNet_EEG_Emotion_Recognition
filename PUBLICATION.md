# Reproducible publication experiments

This is the maintained experimental path. Root-level `train.py`, `config.py`, `model.py`, augmentation modules, notebook, and saved results describe the historical course project. Do not mix their checkpoints or metrics with `gruxnet/` runs.

Joint training on all three datasets is the historical research question. The author now prioritizes a strong publishable contribution and authorizes experiments to assess alternatives. **Show the findings and receive the author's explicit approval before adopting a different research question.** Preserve the then-current manuscript before an approved change. For the historical question, compare architectures trained jointly on the same inputs, labels, participant partitions, and budget, and report each dataset separately. DEAP-only controls diagnose learning; leave-one-dataset-out tests unseen-dataset transfer. A reference model's lower score or weak transfer alone does not establish overfitting.

The [publication readiness record](docs/publication/GRU-XNet_Publication_Readiness_2026-10-05.md) distinguishes repaired implementation problems from outstanding scientific evidence. Review reports and selected development artifacts are preserved in `docs/publication/` and `results/development/`. These are development results, not a completed conference study. Raw EEG and full local run directories remain outside Git.

## Local setup

Run from this repository directory in PowerShell:

```powershell
conda activate pytorch
python -m gruxnet --help
python -m pytest -q
```

Tested with `D:\DL_Frameworks\envs\pytorch\python.exe`, PyTorch 2.5.1, CUDA, and an RTX 3050 Laptop GPU with 6 GB VRAM. `requirements-publication.txt` lists dependencies. The existing environment already had everything except `xlrd`; it was installed into the ignored `.publication_deps/` directory because the environment is read-only to this session. On a fresh checkout, install the requirements in your environment, or add only the missing spreadsheet reader locally:

```powershell
python -m pip install --target .publication_deps xlrd
```

## Audited input data

```powershell
python -m gruxnet audit --data-root ../emotion-recognition-eeg-datasets --output ../publication_runs/audit_with_sources --provenance dataset_sources.json
python -m gruxnet prepare --data-root ../emotion-recognition-eeg-datasets --output ../publication_runs/cache_common14 --provenance dataset_sources.json
python -m gruxnet review-sam --audit-dir ../publication_runs/cache_common14/audit
```

Preparation refuses to overwrite an existing manifest. To change preprocessing, choose a new cache directory. The prepared cache used in this workspace already exists; reuse it instead of preparing again.

| Dataset | Subjects | Source trials | Neutral excluded | Eligible trials | Four-second windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEAP | 32 | 1,280 | 16 | 1,264 | 18,960 |
| GAMEEMO | 28 | 112 | 19 | 93 | 6,882 |
| SEED-IV | 15 | 1,080 | 270 | 810 | 27,405 |
| Total | 75 | 2,472 | 305 | 2,167 | 53,247 |

The EEG window cache occupies approximately 1.42 GiB. Source files, metadata hashes, excluded trials, channel orders, and discarded tails are recorded. One participant/session file is loaded at a time. The old augmented arrays and duplicate nested `sead-4/seed_iv` directory are ignored.

### Labels and provenance

The author's assembled bundle is [emotion-recognition-eeg-datasets](https://www.kaggle.com/datasets/fdskjlajlkfdsa/emotion-recognition-eeg-datasets). The supplied download sources are [DEAP](https://www.kaggle.com/datasets/manh123df/deap-dataset), [GAMEEMO](https://www.kaggle.com/datasets/sigfest/database-for-emotion-recognition-system-gameemo), and [SEED-IV](https://www.kaggle.com/datasets/phhasian0710/seed-iv), all checked at Kaggle version 1 on 5 October 2026. [dataset_sources.json](dataset_sources.json) records the download chain, versions, research dataset URLs, and verification scope. `--provenance` embeds its contents and SHA-256 into new audits; newly prepared caches also bind it into their fingerprint. This snapshots the declaration without automatically repeating external integrity checks. Existing prepared caches and completed runs are unchanged.

All 304 training input/metadata files match their supplied upstream archives in size and CRC32; six representative downloads additionally match SHA-256. The independently downloaded DEAP `s01.dat` and rating spreadsheet reproduce the label discrepancy, establishing that it is already present in the supplied upstream source. All 439 affected local trials originally have valence and arousal above 5, and all their binary valence labels flip. Detailed evidence is in the workspace's `publication_runs/provenance/` directory and `GRU-XNet_Dataset_Provenance_2026-10-05.md`. CRC32 is not a cryptographic hash, and these comparisons do not establish correspondence with a first-party DEAP release.

- DEAP: included spreadsheet valence `<5` is negative, `>5` positive, and exactly `5` excluded. Join `Participant_id` and `Experiment_id`, not presentation `Trial`. All 1,280 dominance/liking pairs match this ordering. In this download, 439 valence and 439 arousal values in the pickles differ from the spreadsheets and exactly match `9 - original`; no other rating differences were found. Use spreadsheet ratings. Retain the original pickle ratings and both source hashes in lineage. Before submission, verify this mirror against an official DEAP download; recovering metadata labels does not independently authenticate the EEG recordings.
- GAMEEMO: participant SAM valence, extracted from vector circles, crosses, or underlines in each PDF. The five manikin images and four intermediate dots form nine positions, negative to positive. Values `1..4` are negative, `6..9` positive, and `5` excluded. The parser rejects missing/conflicting marks and supports long ellipses selecting the same position in both rows. All 112 valence selections were visually checked using the rendered review sheets; difficult full-page forms were inspected separately. Arousal is retained as metadata and is not the classification target. The official [GAMEEMO repository](https://data.mendeley.com/datasets/b3pn4kwpmn/1) confirms participant SAM forms accompany the recordings.
- SEED-IV: numeric `_eeg1` through `_eeg24` ordering, all 15 participants and all three sessions. Session labels are checked against the supplied `ReadMe.txt`. Sadness and fear are negative, happiness positive; neutral is excluded. Channel order comes from the supplied spreadsheet.

This is an explicitly defined coarse binary valence task. Self-ratings and stimulus emotion categories remain different forms of supervision; the mapping does not prove psychological equivalence between datasets.

### First-party DEAP comparison

On 5 October 2026, the exact original metadata and EEG archive routes returned HTTP 503 with verified TLS, and the author access-request server timed out. Archived official documentation confirms the video-order join, rating order, array dimensions, and all 32 electrode names/order. No first-party signal file was obtained. The workspace's `GRU-XNet_First_Party_DEAP_Check_2026-10-05.md` and `publication_runs/provenance/first_party_deap/` retain the attempt and its evidence.

Once an author-downloaded copy is available, compare it without extracting a large ZIP or rewriting the existing data:

```powershell
python -m gruxnet compare-deap --official ../official_deap/data_preprocessed_python.zip --data-root ../emotion-recognition-eeg-datasets --output ../publication_runs/provenance/deap_official_comparison.json
```

`--official` also accepts a directory containing all 32 subject files, or its parent `data_preprocessed_python` directory. The example archive path is a placeholder. The comparator checks every signal value across all 40 channels, including baseline/peripheral channels, retains file and decoded-array hashes, and compares reference labels against the included spreadsheet. It requires a fresh output file and records that the reference's download origin must be established separately. Equal EEG with corrected labels can validate the present cache; different EEG requires a new source audit and cache before final experiments.

### Preprocessing and electrodes

All eligible trials use a fourth-order, zero-phase 4–40 Hz Butterworth filter. DEAP's initial 3-second baseline is removed. SEED-IV is resampled from 200 to 128 Hz using polyphase resampling (16/25). Signals are segmented into **non-overlapping four-second windows**, with partial tails discarded. Filtering and resampling are contained within the source trial, with no fitted population statistics.

Default inputs use the 14 physically matching electrodes shared by all three datasets, in the order `AF3 AF4 F3 F4 F7 F8 FC5 FC6 O1 O2 P7 P8 T7 T8`. DEAP's input order is checked against [TorchEEG's source constants](https://github.com/torcheeg/torcheeg/blob/main/torcheeg/datasets/constants/emotion_recognition/deap.py). To study all available electrodes, prepare a separate `--montage canonical62` cache. Names, rather than channel positions, determine alignment. The canonical mode zero-pads missing electrodes and masks their inputs and their CNN outputs after convolution and normalization.

The complete DEAP 32-channel order also matches the channel table in the [archived official README](https://web.archive.org/web/20231030222856/https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html), checked programmatically on 5 October 2026.

Per-window, per-observed-channel z-score normalization uses that window alone. It removes absolute amplitude and is a methodological choice that needs comparison with training-fitted normalization if power features are important. A periodic Hann STFT uses 128 samples, a 32-sample hop, no boundary padding, and bins 4–40 Hz: **37 frequency bins × 13 real time positions**. Magnitudes are divided by the Hann-window sum and transformed with `log1p`. Different-duration trials are never stretched into the same time axis.

## Model and hardware settings

`GRU-XNet-Compact-v2` preserves independent electrode CNN parameters, genuine temporal features, bidirectional recurrence, and four-head temporal self-attention. It is **not the model that produced the historical 95.91% score**.

Three grouped convolution blocks use 8/16/32 filters per electrode, GroupNorm, GELU, and frequency-only pooling. The four reduced frequency positions are retained and concatenated with electrode/filter features at each of the 13 temporal positions, then projected to 128 dimensions. A two-layer BiGRU with 64 hidden units per direction produces 128 features per time position. Four-head self-attention has a residual connection and LayerNorm. Time averaging feeds a 128→64→2 classifier. Dropout is 0.3. With 14 electrodes, the model has **537,442 trainable parameters**.

The exploratory v1 model averaged over frequency and had 365,410 parameters. Its completed full-window run is retained separately; it predicted one class and reached 50% balanced accuracy. `--frequency-pooling mean` reproduces its architecture. The default v2 avoids this loss of frequency identity; its limited-window checks are explicitly pilots. Neither variant establishes a performance gain without matched final experiments.

Defaults are batch size 16, accumulation 4, AdamW at 0.001 with weight decay 0.0001, cosine scheduling, AMP on CUDA, and gradient clipping at 1.0. Gradient accumulation is normalized by the actual number of examples, including the final short group. The pilot used about 149 MiB of peak CUDA tensor allocation; this excludes the CUDA context, other programs, and the driver's total memory accounting. Full-run memory/timing are saved separately.

## Evaluation protocols

```powershell
# Independent subjects within each of the three pooled datasets.
python -m gruxnet train --cache ../publication_runs/cache_common14 --output ../publication_runs/subject_seed42 --protocol subject --seed 42 --device cuda

# True transfer: GAMEEMO never enters training or validation.
python -m gruxnet train --cache ../publication_runs/cache_common14 --output ../publication_runs/lodo_GAMEEMO_seed42 --protocol lodo --target GAMEEMO --seed 42 --device cuda

# One pooled LOSO fold. All windows/sessions of that participant are held out.
python -m gruxnet train --cache ../publication_runs/cache_common14 --output ../publication_runs/loso_DEAP_S01 --protocol loso --test-subject DEAP:S01 --device cuda

# Bounded pilot, explicitly marked as limited-window evaluation.
python -m gruxnet train --cache ../publication_runs/cache_common14 --output ../publication_runs/my_pilot --epochs 3 --max-windows-per-trial 4 --device cuda

# Rehash the cache and reproduce every test prediction and metric.
python -m gruxnet verify --run ../publication_runs/subject_seed42
```

Subject partitions are created before sampling/augmentation. IDs are dataset-qualified, e.g. `DEAP:S01`, so subject 1 in different datasets cannot collide. The ordinary subject protocol allocates approximately 70/15/15% **of subjects**, rounding within each dataset. These are not claimed to be label-stratified sample splits. Validation checks reject class-deficient partitions instead of changing folds after inspecting test performance. LODO assigns the entire target dataset to testing and validates on held-out source subjects. Pooled LOSO holds out one dataset-qualified subject; complete LOSO requires running every planned fold and reporting the aggregation.

Sampling balances datasets and classes, then gives equal weight to original trials within each dataset/class; windows are inversely weighted by their trial's length. Gaussian noise (SD 0.03 after normalization), gain 0.9–1.1, and channel dropout (p=0.05) run on **training batches only**. A dropout mask always retains at least one actual electrode. Validation/test inputs are unaugmented. Synthetic examples are not stored or assigned invented subject IDs. MixUp, CutMix, SMOTE, GAN, and VAE are absent from this pipeline.

Checkpoint selection uses mean per-dataset **trial balanced accuracy on validation subjects**. Test is evaluated once after loading the selected checkpoint. Window probabilities are averaged within their original trial, then thresholded at 0.5. Report window metrics as secondary descriptive results. Subject-cluster bootstrap intervals resample subjects, never independent windows. A pooled subject test is not a cross-dataset transfer result.

Every run has fresh output paths, full/selected split manifests, split audit, source-code snapshot and hashes, cache fingerprint, configuration, history, best checkpoint, validation predictions, test window/trial predictions, consistent confusion matrices, and figures. The `verify` command rehashes cached EEG and reconstructs all test metrics from the selected checkpoint.

Two additional diagnostics help interpret a neural model stuck at chance:

```powershell
# Fit a fixed small batch of training data with dropout/augmentation disabled.
python -m gruxnet overfit --cache ../publication_runs/cache_common14 --split ../publication_runs/full_subject_seed42/selected_split.csv --output ../publication_runs/my_capacity_check.json

# A cheap reference on exactly the same full subject split.
python -m gruxnet baseline --cache ../publication_runs/cache_common14 --split ../publication_runs/full_subject_seed42/selected_split.csv --output ../publication_runs/my_bandpower_baseline
```

The capacity check fits 24 training windows from six trials and proves only that the model can memorize them. The baseline uses Welch log-bandpower in four ranges (4–<8, 8–<14, 14–<31, 31–<41 Hz). It retains amplitude information from the filtered windows. Its StandardScaler is fitted to weighted training data only, and logistic-regression regularization is selected from C=0.01/0.1/1/10 using validation trial balanced accuracy. Test probabilities are evaluated after selection and reconstructed independently from exported coefficients. This is a representation/control baseline, not an exact architectural ablation of normalized STFT inputs.

## DEAP-only learning control

This separate raw-EEG control uses all 32 canonical DEAP EEG electrodes, the same recovered valence labels, the existing 4–40 Hz trial filter, and four-second non-overlapping windows. It does not overwrite the common-electrode cache or previous runs. One frozen mean/std per electrode is fitted after splitting, using only training participants. The raw EEGNet-8,2 control has 2,130 parameters; all layers are registered before optimizer construction and it returns logits directly to cross-entropy. This is a PyTorch architecture implementation, not an exact published DEAP reproduction.

```powershell
python -m gruxnet prepare-deap --data-root ../emotion-recognition-eeg-datasets --output ../publication_runs/cache_deap32 --provenance dataset_sources.json
python -m gruxnet train-deap-control --cache ../publication_runs/cache_deap32 --output ../publication_runs/deap_eegnet32_trainchannel_seed42 --seed 42 --epochs 100 --minimum-epochs 25 --patience 15 --batch-size 32 --learning-rate 0.001 --normalization train-channel --device cuda
python -m gruxnet verify-deap-control --run ../publication_runs/deap_eegnet32_trainchannel_seed42
```

Existing output paths are immutable; choose fresh names to run another experiment. The predeclared control uses 22/5/5 disjoint participants, Adam at a fixed learning rate of 0.001, no augmentation, class/trial-balanced sampling, and a maximum of 100 epochs. Early stopping cannot occur before epoch 25. Checkpoints maximize validation trial balanced accuracy; ties select lower class-balanced validation trial log loss. Test inference follows final checkpoint selection. History includes training-only probe results, gradients, and validation probability spread/class counts. Verification refits the training-only scaler, checks the exact partition and cached arrays, reconstructs checkpoint selection, and reproduces test predictions/metrics.

The workspace run completed **28 epochs**, selected **epoch 13**, and obtained **66.91% training / 56.07% validation / 46.25% test trial balanced accuracy**. Test AUROC is 0.5007, and the subject-bootstrap interval includes chance. Four of five test participants receive a constant class prediction. The training-only scaler, exact participant split, validation checkpoint selection, all 3,000 test probabilities, and all metrics reproduce. See [the control report](docs/publication/GRU-XNet_DEAP_Control_2026-10-05.md) for the evidence, plots, and interpretation.

`--normalization window` supports a separately declared, same-model normalization comparison. That additional comparison is not part of the initial one-run plan. The DEAP-only control changes architecture, montage, representation, normalization, and budget together, so its outcome alone cannot attribute a difference specifically to normalization. Its held-out cohort was already inspected in earlier development work; keep its results labeled development evidence.

## Completed exploratory controls

The separately declared matched normalization run completes 25 epochs, selects epoch 4, and gives 46.67% test trial balanced accuracy / AUROC 0.4433. All 3,000 probabilities and validation selection reproduce. The matched change is +0.42 percentage points; its paired subject interval includes zero. This does not resolve DEAP generalization.

```powershell
python -m gruxnet train-deap-control --cache ../publication_runs/cache_deap32 --output ../publication_runs/deap_eegnet32_windownorm_seed42 --normalization window --epochs 100 --minimum-epochs 25 --patience 15 --batch-size 32 --seed 42 --learning-rate 0.001 --device cuda
python -m gruxnet verify-deap-control --run ../publication_runs/deap_eegnet32_windownorm_seed42
python scripts/describe_deap_control.py --run ../publication_runs/deap_eegnet32_windownorm_seed42
python scripts/compare_deap_normalization.py --reference ../publication_runs/deap_eegnet32_trainchannel_seed42 --alternative ../publication_runs/deap_eegnet32_windownorm_seed42
```

The native SEED-IV diagnostic uses all 62 electrodes, all four classes, and five fixed 9/3/3 participant rotations. It also evaluates the existing binary label policy and named common-14 electrodes on those folds. Training-only standardized mean-window-log-bandpower features and validation-selected logistic regression give 36.11% / 36.85% four-class balanced accuracy with 62 / 14 electrodes (25% chance); binary gives 63.89% / 67.13% (50% chance). Targets have different trial populations and cannot be compared through raw accuracy.

```powershell
python scripts/native_seediv_diagnostic.py prepare --data-root ../emotion-recognition-eeg-datasets --cache ../publication_runs/cache_native_seediv_features
python scripts/native_seediv_diagnostic.py run --cache ../publication_runs/cache_native_seediv_features --output ../publication_runs/native_seediv_diagnostic
python scripts/verify_native_seediv_diagnostic.py --cache ../publication_runs/cache_native_seediv_features --common-cache ../publication_runs/cache_common14 --output ../publication_runs/native_seediv_diagnostic
python scripts/joint_seediv_diagnostic.py --native-cache ../publication_runs/cache_native_seediv_features --common-cache ../publication_runs/cache_common14 --output ../publication_runs/joint_seediv_diagnostic
python scripts/verify_joint_seediv_diagnostic.py --output ../publication_runs/joint_seediv_diagnostic
python scripts/plot_exploration.py
```

The pooling control adds only original DEAP/GAMEEMO training participants while retaining the same SEED-IV features, target folds, C candidates and validation subjects. Joint training gives 60.93% (global training scaler) or 60.56% (separate training-only dataset scalers), versus 67.13% target-only. All 140 validation candidates across native/pooling controls and all 6,210 held-out prediction rows reproduce. Common-14 feature values agree exactly with the independently retained waveform cache for all 810 nonneutral SEED-IV trials. [Detailed findings and limitations](docs/publication/GRU-XNet_Exploration_Findings_2026-10-05.md).

These remain exploratory classical controls. They do not establish neural negative transfer, its cause, unseen-dataset performance, or a novel method. Checkpoint selection uses validation; previously inspected test cohorts remain development evidence. The author has approved exploration, **not adoption of a new question**.

## Planned comparisons

The [matched neural investigation](docs/publication/GRU-XNet_Neural_Transfer_Investigation_2026-10-05.md) completes 75 SEED-IV feature-MLP runs, including individual source additions, separate binary heads, and equal-compute/equal-available-target-exposure controls. The [cross-target extension](docs/publication/GRU-XNet_Multitarget_Transfer_Findings_2026-10-05.md) adds 90 DEAP/GAMEEMO runs. All 165 runs, 285 selected checkpoints and 42,225 neural probabilities are checked; 160 anchored linear candidates and 5,954 probabilities also reproduce. See the [initial protocol](docs/publication/NEURAL_TRANSFER_PROTOCOL.md) and [extension protocol](docs/publication/MULTITARGET_TRANSFER_PROTOCOL.md) for commands. Shared-joint versus target-only neural intervals include zero on every target/budget; simple heads do not consistently help. These separate target-centered studies do not establish a single joint checkpoint's performance, a novel mitigation, or full GRU-XNet ablations. mdJPT (NeurIPS 2025) is now an essential prior-work comparator.

The [GitHub configuration review](docs/publication/GRU-XNet_GitHub_Configuration_Review_2026-10-05.md) inspects TSception, LibEER, TorchEEG, EEGain, and DeepVANet at fixed commits. It documents preprocessing, optimizer settings, training budgets, label boundaries, and actual split implementations. The DEAP-only EEGNet control and matched normalization diagnostic are complete; matched neural architecture ablations remain outstanding. Keep existing recovered labels and completed run artifacts unchanged; repository configurations do not replace first-party DEAP authentication.

Use the same predeclared subject folds and training budget for every variant:

| Comparison | CLI settings | Question |
| --- | --- | --- |
| CNN–BiGRU–attention | Defaults | Reference compact architecture |
| CNN–BiLSTM–attention | `--recurrent lstm` | What does GRU substitution change? |
| CNN–BiGRU | `--no-attention` | Does attention help? |
| CNN only | `--recurrent none --no-attention` | Does recurrence help? |
| No augmentation | `--no-augment` | Do the three training transforms help? |
| Canonical 62-channel input | Separate `canonical62` cache | Does using additional masked electrodes help? |

Start with paired subject folds at seeds 42/43/44, then all three LODO targets. Choose hyperparameters on validation data and freeze the protocol before the final comparison. Report parameter count, time, peak allocation, per-dataset metrics, fold distributions, and confidence intervals. Foundation-model or graph baselines can follow after the corrected baseline is stable. Historical tables are not evidence that these ablations have already run.

The initial pilot is a software check, not publication evidence. Its 50% balanced accuracy does not support the old performance claim. See the workspace implementation report for the completed longer run, v2 pilots, and remaining work. These development runs have inspected test results and should not be treated as a pristine confirmatory study after further tuning.

## Completed transformer and native-label controls (6 October 2026)

The [separate protocol](docs/publication/Temporal_Native_Control_Protocol_2026-10-06.md) completes 120 SEED-IV runs across architecture, representation and label granularity, five participant rotations and three initialization seeds. Both objectives train on all 1,080 original trials with identical batch streams, including neutral, and use fixed first-40-second inputs. Three-class common valence is primary; conditional binary valence is secondary. The tiny bandpower transformer is independently written and is not a raw-waveform Conformer or pretrained EEG-model reproduction.

The [verified findings](docs/publication/GRU-XNet_Transformer_Native_Label_Findings_2026-10-06.md) show 44.24% versus 42.00% common three-class BA for absolute/coarse transformer versus MLP, with an exploratory unadjusted participant interval of +0.14 to +4.47 percentage points. All native-label objective intervals include zero. The absolute logistic binary point estimate (64.63%) is higher than the neural binary means. Existing binary/full-trial experiments change several factors relative to this phase and are not an exact architecture comparison. This does not establish negative-transfer mitigation or a new contribution.

```powershell
python scripts/temporal_native_controls.py run --cache ../publication_runs/cache_temporal_native_seediv --output ../publication_runs/temporal_native_seediv --plan ../publication_runs/temporal_native_plan_2026-10-06.json --device cuda
python scripts/temporal_native_controls.py verify --cache ../publication_runs/cache_temporal_native_seediv --output ../publication_runs/temporal_native_seediv --device cuda
python scripts/plot_temporal_controls.py --output ../publication_runs/temporal_native_seediv
python scripts/describe_temporal_controls.py --output ../publication_runs/temporal_native_seediv --cache ../publication_runs/cache_temporal_native_seediv --destination ../GRU-XNet_Transformer_Native_Label_Findings_2026-10-06.md
```

Use the protocol's preparation commands before a fresh run; completed outputs refuse refitting. Verification replays all 120 checkpoints and 15 linear coefficient models, checks training-only scalers, all 25,920 neural and 3,240 linear probabilities, paired draw streams, validation selection and OOF/bootstrap aggregates. It does not independently refit the 60 linear candidates. Source preprocessing/checkpoint versions are bound, and earlier experiment modules remain unchanged. The [additional research review](docs/publication/GRU-XNet_Transformer_Research_Update_2026-10-06.md) includes explicit prior work on heterogeneous transformer pretraining and shared stimulus material. Research-question adoption still requires the author's approval and a then-current manuscript archive.

## Completed participant/session and frozen-pretraining controls

The [session protocol](docs/publication/Session_Stimulus_Control_Protocol_2026-10-06.md) fits and selects using a single source session's training/validation people, then tests all three sessions of held-out people. Ninety matched MLP/transformer fits and 30 selected linear controls are complete. The [REVE protocol](docs/publication/Frozen_REVE_Control_Protocol_2026-10-06.md) compares the pinned official frozen encoder with its frozen random architecture, using the same raw prefixes and physical electrodes, a declared stateless adapter and 40 validation-selected linear heads. Downloaded assets and embeddings remain local. Its limited public-provenance audit is not independent certification of target-corpus exclusion.

The [findings](docs/publication/GRU-XNet_Session_Pretraining_Findings_2026-10-06.md) show transformer three-class BA 45.64% familiar versus 39.90% unseen sessions; its MLP advantage disappears on unseen sessions. REVE's pretrained/random unseen-session BAs are 39.14%/33.33%. Both participant-only and crossed participant/material intervals are preserved; the latter was explicitly declared as an exploratory addendum after initial session results, before REVE head outcomes. This is SEED-IV development evidence, with sessions changing recording conditions and material together. It is not pooled three-corpus training or evidence for a novel transfer mitigation.

```powershell
python scripts/session_stimulus_controls.py verify --cache ../publication_runs/cache_temporal_native_seediv --output ../publication_runs/session_stimulus_seediv --device cuda
python scripts/reve_frozen_probe.py verify --assets ../publication_runs/reve_audit_2026-10-06 --cache ../publication_runs/cache_reve_input_seediv --output ../publication_runs/reve_frozen_seediv
python scripts/analyze_session_material_sensitivity.py verify --session-run ../publication_runs/session_stimulus_seediv --reve-run ../publication_runs/reve_frozen_seediv --output ../publication_runs/session_material_sensitivity --plan ../publication_runs/session_material_sensitivity_plan_2026-10-06.json
python scripts/report_session_pretraining.py --runs ../publication_runs --destination ../GRU-XNet_Session_Pretraining_Findings_2026-10-06.md
```

Follow the two protocols for downloading/reviewing assets, declaring plans and preparing inputs before fresh fitting. The maintained REVE verifier independently refits all 160 candidate heads and replays all 40 selected heads, exact frozen state hashes and 20 sampled embeddings. Matching sklearn's softmax resolves the original verifier's amplified rounding difference without altering fitted data/metrics or widening its tolerance. Across both new phases 34,560 test probability rows replay, and the crossed bootstrap deterministically recomputes. Original experiment modules remain bound and unchanged. That phase completed with 43 passing tests; a research-question pivot still requires the previously recorded evidence/approval gate.

## Completed matched within-session material controls

The [predeclared protocol](docs/publication/Within_Session_Material_Control_Protocol_2026-10-06.md) holds session, participant groups, training size and exact validation/test trials fixed while changing test-material exposure among training people. Three material rotations and five participant rotations cover every original trial once per model/arm/seed. Both arms use 108 training, twelve validation and 24 identical test trials per fit. Selection can be noisy with these small validation sets; the cohort remains development evidence.

All 540 MLP/transformer fits and 360 selected classical heads are complete. The verifier replays every selected neural checkpoint and train/validation/test metric, independently refits all 1,440 classical candidates, exactly reproduces selected coefficients, and checks all 21,600 probabilities and the complete paired bootstrap. [All findings and twenty contrasts](docs/publication/GRU-XNet_Within_Session_Material_Findings_2026-10-06.md). Transformer three-class BA changes from 43.35% to 39.92%, with crossed interval [-7.88,+0.56] pp for unseen-minus-shared. All twelve exposure intervals span zero; this does not establish equivalence. There is no clear unseen-material transformer advantage. No new paper question is adopted.

```powershell
python scripts/within_session_material_controls.py verify --cache ../publication_runs/cache_temporal_native_seediv --reve-run ../publication_runs/reve_frozen_seediv --output ../publication_runs/within_session_material_seediv --device cuda
python scripts/report_within_session_material.py --run ../publication_runs/within_session_material_seediv --destination ../GRU-XNet_Within_Session_Material_Findings_2026-10-06.md
python -m scripts.audit_material_populations --cache ../publication_runs/cache_common14_trial_features --output ../publication_runs/material_population_audit
python scripts/export_publication_evidence.py
python scripts/verify_publication_export.py
# For a public checkout without the original local workspace:
python scripts/verify_publication_export.py --export-only
```

Follow the matched protocol for fresh fitting and commit its plan first. Completed paths refuse refitting; old experiment sources remain unchanged. The [focused research update](docs/publication/GRU-XNet_Material_Generalization_Research_Update_2026-10-06.md) adds stimulus-aware SSL, an audited ACM Multimedia alignment method, efficient EEG pretraining and explicit target-access qualifications. The metadata audit fits no model: DEAP has forty video IDs and individual labels, whereas GAMEEMO has four interactive conditions with sparse class coverage. These require different extension protocols. All 47 scientific-control tests pass. Repeated groupings/LODO, matched full GRU-XNet ablations, first-party source authentication and a revised paper remain unfinished.

## Repeated groupings and DEAP control in progress

The [new protocol](docs/publication/Repeated_Material_Control_Protocol_2026-10-06.md) and [plan](results/development/repeated_material_plan_2026-10-06.json) were pushed before fitting in commit f5bbdfa. All ninety SEED-IV and eighty DEAP split pairs passed feasibility checks. Two additional SEED participant/material groupings and two DEAP groupings are fixed before results. DEAP uses corrected individual spreadsheet ratings, exact participant/class training-count matching, identical held-out trials and validation videos absent from both training arms. Repeated partitions reuse the same people/videos; they are sensitivity checks, not new independent replications. One fixed initialization holds optimizer randomness constant; original SEED initialization42 is reused separately from its earlier three-seed mean.

The batch schedules 680 neural fits, 880 selected linear heads, 3,520 linear candidates and 160 training-label-only DEAP video-prior diagnostics. Full checkpoint/candidate replay precedes conclusions. The maintained scientific-control suite now has 53 passing tests. The manuscript and research question remain unchanged. No completed results from this batch are claimed here yet.

```powershell
# Existing verified common14, native temporal and frozen SEED features are prerequisites.
python scripts/repeated_material_controls.py prepare-deap --common-cache ../publication_runs/cache_common14 --cache ../publication_runs/cache_temporal_deap
python scripts/repeated_material_controls.py plan --plan ../publication_runs/repeated_material_plan_2026-10-06.json
python scripts/repeated_material_controls.py freeze --workspace .. --plan ../publication_runs/repeated_material_plan_2026-10-06.json
# Review/commit/push the plan and splits before fitting; use fresh local output directories.
python scripts/repeated_material_controls.py batch --workspace .. --plan ../publication_runs/repeated_material_plan_2026-10-06.json
python scripts/report_repeated_material.py --workspace ..
```
