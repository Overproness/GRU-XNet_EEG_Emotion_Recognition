# Transformer, representation and native-label findings

Date: 6 October 2026. **120 neural runs and 15 selected logistic models are complete and verified. The paper is still not ready for submission, and no new research question has been adopted.**

The highest observed neural three-class mean is **44.61%** (Transformer / absolute / native4); the highest observed conditional binary mean is **63.80%** (Transformer / absolute / coarse3). These are descriptive development results, not validation-selected winners for a new contribution. Of the 24 predeclared factor contrasts, **4** have unadjusted participant percentile intervals excluding zero. All contrasts and seeds are reported below; selecting a favourable contrast after inspection would require further independent confirmation.

The straightforward absolute-power/coarse-label comparison gives **44.24% transformer versus 42.00% MLP**, a **+2.24 percentage-point** primary-task difference with unadjusted interval **[+0.14, +4.47]**. This supports further temporal-model investigation, not a transformer-novelty or broad superiority claim. **8 of 8** native-minus-coarse objective intervals include zero, so preserving sadness/fear as separate training targets has not shown a clear benefit here. The absolute-power logistic control's **64.63%** binary point estimate exceeds every neural binary mean in this phase. Representation and task matter; the primary-task gain should not be presented as an improvement on all emotion targets.

## What was tested

The [committed protocol](Temporal_Native_Control_Protocol_2026-10-06.md) and [machine declaration](../../results/development/temporal_native_plan_2026-10-06.json) were saved before fitting. The full factorial compares temporal transformer versus mean MLP, absolute versus relative log-bandpower, and native four-emotion versus coarse three-valence supervision. Both label objectives see the **same 1,080 original trials**, including neutral, the same sampling stream and the same common evaluation tasks. Native supervision keeps sadness and fear separate; coarse supervision combines them as negative. This within-SEED-IV study does not test joint training with DEAP/GAMEEMO, so it cannot determine whether native supervision fixes cross-dataset negative transfer.

All 15 SEED-IV participants and three sessions are included. Five existing participant rotations use nine train, three validation and three test participants; seeds 42/43/44 initialize each model. All trials of a participant stay together. The common three-class task uses all trials, with 33.33% chance balanced accuracy; the conditional binary task excludes the same 270 neutral trials, with 50% chance. Native-head probabilities are grouped for common-task evaluation; binary probabilities are conditioned on negative plus positive. Native four-class accuracy is a distinct secondary measure and cannot be compared directly to binary accuracy.

Each trial contributes the first ten non-overlapping four-second bandpower windows: a fixed **40-second input** using 14 named electrodes and four bands. The transformer has two independently initialized four-head encoder layers, width 32, feedforward width 64, fixed positions in seconds and mean token pooling. The MLP averages the same ten windows before its 56 → 128 → 86 encoder. Coarse heads give 19,075 and 19,079 parameters respectively. The networks have nearly identical parameter counts and differ in latent width and computation, so the comparison cannot isolate attention's contribution. The transformer is independently written and is not a raw-waveform EEG-Conformer or pretrained foundation-model reproduction.

Relative features are log power fractions computed within each channel/window. Training-only scalers are frozen for validation/test and paired across objectives and architectures. Every arm receives the identical 600 batches for a fold/seed, each with 20 samples from each coarse class. The four native classes are not equally sampled: the two negative subtypes share the negative quota. Native heads begin with identical grouped coarse probabilities. Every run completes 600 AdamW updates with identical optimizer settings; validation every 25 updates selects common three-class BA, then balanced log loss, then the earliest exact tie. Test is evaluated afterward. Different selected checkpoints can have different actual training exposures despite the identical available budget.

## Results on common tasks

Neural values are out-of-fold trial BA averaged over the three initialization results. Training values average the 15 selected checkpoint training-fold BAs; they describe fit on known participants and are not directly a paired population comparison with the OOF score.

| Model / representation / supervision | Three-class BA | Conditional binary BA | Three-class training BA |
| --- | ---: | ---: | ---: |
| MLP / absolute / coarse3 | 42.00% | 63.40% | 74.70% |
| MLP / absolute / native4 | 41.11% | 63.12% | 80.72% |
| MLP / relative / coarse3 | 41.40% | 61.54% | 77.30% |
| MLP / relative / native4 | 41.54% | 61.33% | 81.10% |
| Transformer / absolute / coarse3 | 44.24% | 63.80% | 80.86% |
| Transformer / absolute / native4 | 44.61% | 63.77% | 78.54% |
| Transformer / relative / coarse3 | 43.31% | 60.43% | 82.72% |
| Transformer / relative / native4 | 42.98% | 60.56% | 75.72% |

| Logistic input | Three-class BA | Conditional binary BA |
| --- | ---: | ---: |
| absolute | 40.68% | 64.63% |
| duration | 39.81% | 56.94% |
| relative | 41.05% | 61.20% |

The duration diagnostic receives only the original full-trial window count, with a training-only scaler and validation-selected multinomial logistic classifier. It receives no EEG features. It quantifies label-duration association in this corpus and is excluded from claims about fixed-duration EEG recognition. It does **not** prove that the historical model exploited duration. Fixed prefix length removes direct sequence-length variation; the participants still watch familiar corpus stimuli. Zero-phase filtering and resampling use the full trial before cropping, so this is an offline protocol rather than strictly causal acquisition limited to 40 seconds.

The earlier 67.13% binary logistic control averages every available window and excludes neutral during training. The current experiment changes observation duration, training task, neutral inclusion, model width, and selection metric. Differences from that historical control cannot be attributed solely to architecture or label granularity. The factorial contrasts below use matched current inputs and trial populations.

## Paired effects and uncertainty

| Comparison | Common task | Difference (pp) | Paired participant 95% interval (pp) |
| --- | --- | ---: | ---: |
| Transformer / absolute / coarse3 minus MLP / absolute / coarse3 | coarse3 | +2.24 | [+0.14, +4.47] |
| Transformer / absolute / native4 minus MLP / absolute / native4 | coarse3 | +3.50 | [+1.32, +6.05] |
| Transformer / relative / coarse3 minus MLP / relative / coarse3 | coarse3 | +1.91 | [-0.29, +4.18] |
| Transformer / relative / native4 minus MLP / relative / native4 | coarse3 | +1.44 | [-1.07, +3.97] |
| MLP / relative / coarse3 minus MLP / absolute / coarse3 | coarse3 | -0.60 | [-2.86, +1.58] |
| MLP / relative / native4 minus MLP / absolute / native4 | coarse3 | +0.43 | [-1.93, +2.61] |
| MLP / absolute / native4 minus MLP / absolute / coarse3 | coarse3 | -0.88 | [-1.93, +0.00] |
| MLP / relative / native4 minus MLP / relative / coarse3 | coarse3 | +0.14 | [-0.88, +1.07] |
| Transformer / relative / coarse3 minus Transformer / absolute / coarse3 | coarse3 | -0.93 | [-3.02, +1.17] |
| Transformer / relative / native4 minus Transformer / absolute / native4 | coarse3 | -1.63 | [-3.89, +0.70] |
| Transformer / absolute / native4 minus Transformer / absolute / coarse3 | coarse3 | +0.37 | [-0.56, +1.34] |
| Transformer / relative / native4 minus Transformer / relative / coarse3 | coarse3 | -0.33 | [-1.23, +0.62] |
| Transformer / absolute / coarse3 minus MLP / absolute / coarse3 | binary | +0.40 | [-1.51, +2.22] |
| Transformer / absolute / native4 minus MLP / absolute / native4 | binary | +0.65 | [-1.17, +2.38] |
| Transformer / relative / coarse3 minus MLP / relative / coarse3 | binary | -1.11 | [-2.75, +0.52] |
| Transformer / relative / native4 minus MLP / relative / native4 | binary | -0.77 | [-2.31, +0.74] |
| MLP / relative / coarse3 minus MLP / absolute / coarse3 | binary | -1.85 | [-4.51, +0.59] |
| MLP / relative / native4 minus MLP / absolute / native4 | binary | -1.79 | [-4.57, +0.74] |
| MLP / absolute / native4 minus MLP / absolute / coarse3 | binary | -0.28 | [-1.11, +0.65] |
| MLP / relative / native4 minus MLP / relative / coarse3 | binary | -0.22 | [-1.42, +1.11] |
| Transformer / relative / coarse3 minus Transformer / absolute / coarse3 | binary | -3.36 | [-6.14, -0.52] |
| Transformer / relative / native4 minus Transformer / absolute / native4 | binary | -3.21 | [-5.96, -0.43] |
| Transformer / absolute / native4 minus Transformer / absolute / coarse3 | binary | -0.03 | [-0.83, +0.77] |
| Transformer / relative / native4 minus Transformer / relative / coarse3 | binary | +0.12 | [-0.71, +0.99] |

The 10,000 paired participant bootstrap draws use seed 20261006 and average the three initialization results within each resampled participant cohort. Seeds are not independent participants. The intervals condition on one participant grouping and these trained models; training folds overlap, and these participants were previously inspected in development. There is no multiplicity correction for the 24 exploratory contrasts and no new confirmatory cohort.

## All initialization results

- MLP / absolute / coarse3: three-class seed 42: 43.33%, seed 43: 41.30%, seed 44: 41.36%; binary seed 42: 64.44%, seed 43: 62.78%, seed 44: 62.96%.
- MLP / absolute / native4: three-class seed 42: 41.54%, seed 43: 40.06%, seed 44: 41.73%; binary seed 42: 62.78%, seed 43: 60.74%, seed 44: 65.83%.
- MLP / relative / coarse3: three-class seed 42: 40.80%, seed 43: 41.67%, seed 44: 41.73%; binary seed 42: 62.41%, seed 43: 61.20%, seed 44: 61.02%.
- MLP / relative / native4: three-class seed 42: 40.62%, seed 43: 42.47%, seed 44: 41.54%; binary seed 42: 61.11%, seed 43: 61.76%, seed 44: 61.11%.
- Transformer / absolute / coarse3: three-class seed 42: 46.11%, seed 43: 43.52%, seed 44: 43.09%; binary seed 42: 65.46%, seed 43: 61.30%, seed 44: 64.63%.
- Transformer / absolute / native4: three-class seed 42: 44.88%, seed 43: 45.19%, seed 44: 43.77%; binary seed 42: 63.70%, seed 43: 62.22%, seed 44: 65.37%.
- Transformer / relative / coarse3: three-class seed 42: 42.47%, seed 43: 43.09%, seed 44: 44.38%; binary seed 42: 61.39%, seed 43: 60.93%, seed 44: 58.98%.
- Transformer / relative / native4: three-class seed 42: 41.48%, seed 43: 43.02%, seed 44: 44.44%; binary seed 42: 61.48%, seed 43: 60.93%, seed 44: 59.26%.

Secondary native four-class results, reported only within the native objective:

- MLP / absolute / native4: 31.30% native four-class balanced accuracy (25% chance).
- MLP / relative / native4: 30.40% native four-class balanced accuracy (25% chance).
- Transformer / absolute / native4: 33.95% native four-class balanced accuracy (25% chance).
- Transformer / relative / native4: 33.09% native four-class balanced accuracy (25% chance).

## Integrity, resources and saved evidence

- Re-extracted features from all 45 hashed raw SEED-IV source files agree exactly with the prior full-trial mean features for **all 1,080 trials**. Full waveforms agree exactly with the retained common-montage cache for all **810** overlapping nonneutral trials. Source labels, metadata and physical electrode ordering are checked. New sequence-cache fingerprint: `a3d623b784b2f2855a84589db929d08a9f49b90b0631966acaa363764a62af76`.
- Verification replays **120 selected neural checkpoints**, **25,920 neural trial probability rows**, **15 linear coefficient models** and **3,240 linear probability rows**. It checks source/cache hashes, full participant coverage, validation selection, training-only scalers, identical 600-batch streams, checkpoint training/validation/test metrics and aggregate/bootstrap recomputation. Maximum replay probability error is `2.98e-08`. Linear verification replays the saved coefficients; it does not independently refit every C candidate.
- Fitting time summed across neural runs: **536.4 seconds** (8.94 minutes), excluding preparation/replay/export. Maximum PyTorch allocated CUDA memory: **74.64 MiB**; CUDA context/driver and other processes are excluded. The runs used the author's RTX 3050 environment. Selected steps span **25–600**, median **300.0**, from a fixed 600-update available budget.
- [Environment snapshot](../../results/development/temporal_native_seediv/environment.json) records package/runtime versions after replay. It is a post-run snapshot from the same active environment; dependencies were unchanged during this phase.
- [Verification](../../results/development/temporal_native_seediv/verification.json), [all model metrics](../../results/development/temporal_native_seediv/model_metrics.json), [aggregate comparisons](../../results/development/temporal_native_seediv/comparison.json), [linear models and validation candidates](../../results/development/temporal_native_seediv/linear_models.json). Individual checkpoints and histories remain local and are excluded from GitHub with raw EEG and feature arrays.
- [Common-task figure](../../results/development/temporal_native_seediv/comparison.png), [paired-contrast figure](../../results/development/temporal_native_seediv/paired_comparison.png). Probabilities are exported separately for each of the eight conditions to keep bounded review artifacts.

## Meaning of a negative-transfer contribution

Negative transfer means **adding a source dataset worsens prediction on a target dataset**, compared with training on that target alone under a fair protocol. A contribution in this area would establish a repeatable pattern, isolate an explanation through controlled interventions, or introduce and validate a method that prevents the harm. It does not mean merely publishing a low score. In our earlier study, linear pooling penalties were clearer, while every neural shared-joint versus target-only participant interval included zero. Causes such as label mismatch, electrode differences and optimization remain hypotheses, not demonstrated explanations.

The present transformer/label experiment broadens model exploration. It does not change that pooling conclusion, because it trains on SEED-IV alone. Keeping original emotions separate is itself an established idea; a model or loss change needs evidence and a gap from the closest work before it can be a strong-conference contribution.

## Remaining work and decision

The [new primary-source review](GRU-XNet_Transformer_Research_Update_2026-10-06.md) adds EEG-Conformer, One Model for All, PESD, an EMBC 2021 shared-stimulus study, and a July 2026 evaluation preprint. Broad claims about unified EEG models, transformer novelty, or the discovery that evaluation protocols matter already have substantial precedents. [Prior cross-target findings](GRU-XNet_Multitarget_Transfer_Findings_2026-10-05.md) and the [readiness checklist](GRU-XNet_Publication_Readiness_2026-10-05.md) still apply.

Before choosing a new contribution, the next useful validation is to separate both participants and stimulus/session material, and audit a modern pretrained baseline for checkpoint overlap, target-data access and hardware feasibility. SEED-IV session separation changes recording conditions and clips together; it is a robustness test and cannot isolate a video's causal influence. Current results do not establish performance on entirely unseen datasets or a single jointly selected checkpoint for all three corpora.

Still outstanding: convincing protocol-compatible reference comparisons; full GRU/LSTM/attention/montage/augmentation controls where relevant to the selected question; repeated participant groupings or final evaluation; full LODO evidence; first-party DEAP signal authentication; final novelty assessment; an updated manuscript, references, consistent figures and compiled PDF. Historical 95.91% is not replaced by these different tasks. Corrected existing labels remain unchanged.

**The manuscript source and historical PDF remain preserved, and the research question remains unchanged.** Show a concrete evidence-based proposal and obtain the author's explicit approval before adopting a changed question; archive the then-current source immediately beforehand. Completing this exploratory phase is not completing the publication project.
