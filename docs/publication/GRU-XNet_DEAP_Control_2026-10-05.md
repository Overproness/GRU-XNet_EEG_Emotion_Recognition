# DEAP-only learning control

Date: 5 October 2026. **Completed and independently verified.** The run follows the [predeclared plan](../../results/development/deap_control_plan_2026-10-05.json). The control learns training patterns, but its **46.25% test trial balanced accuracy provides no evidence of useful participant-independent generalization**.

The author clarified that the paper's central objective is **one model trained jointly on DEAP, GAMEEMO, and SEED-IV**, evaluated separately on each dataset. This DEAP-only experiment diagnoses whether a small, conventional neural model can learn under participant-independent evaluation. It does not replace that joint study.

## Configuration and evidence

The new [EEGNet control](../../gruxnet/eegnet.py) follows the EEGNet-8,2 architecture: eight temporal filters, depth multiplier two, 16 separable-convolution output filters, 64/16-sample temporal kernels, pooling by 4/8, and dropout 0.5. It has **2,130 trainable parameters**. It is an independent PyTorch implementation using raw EEG and logits, with all layers registered before optimizer construction and max-norm constraints applied after updates. This is an architectural control rather than a reproduction of a published DEAP emotion score. [Original EEGNet implementation](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py), [Lawhern et al. paper](https://arxiv.org/abs/1611.08024).

Inputs use all 32 canonical DEAP electrodes, 128 Hz sampling, four-second non-overlapping windows, and the same recovered valence labels and 4–40 Hz trial filter as the maintained pipeline. The first three baseline seconds are discarded. No baseline waveform/feature subtraction or augmentation is added. Exactly-five valence ratings remain excluded.

The [new cache](../../results/development/cache_deap32/prepared.json) contains **1,264 eligible trials / 18,960 windows** from 32 participants. Cache fingerprint: `13c1e7922a7c1cf7583f34225a34ef1d9f68831e680e2fcb9a0cb42f04d90ab9`. [Consistency checks](../../results/development/deap32_cache_consistency.json) confirm identical labels/sample lineage, exactly identical values on all 14 shared electrodes in every eligible DEAP trial, and the same participant assignments as the earlier development split. Earlier caches and completed runs are retained unchanged.

| Partition | Participants | Trials | Windows | Negative / positive trials |
| --- | ---: | ---: | ---: | ---: |
| Training | 22 | 865 | 12,975 | 403 / 462 |
| Validation | 5 | 199 | 2,985 | 73 / 126 |
| Test | 5 | 200 | 3,000 | 80 / 120 |

Training-only per-electrode means and population standard deviations are fitted to **6,643,200 values per electrode**, covering the 12,975 training windows. The same frozen affine transform is used for validation and test. It preserves relative window amplitude rather than forcing every channel/window to unit RMS. Fitted statistics and their training subjects/sample IDs are saved in [normalizer_fit.json](../../results/development/deap_eegnet32_trainchannel_seed42/normalizer_fit.json).

The budget is **100 epochs maximum, 25 minimum, patience 15**. Training uses batch 32, fixed-rate Adam at 0.001, zero weight decay, full precision, no augmentation, and the existing class/trial-balanced training sampler. The checkpoint maximizes validation trial balanced accuracy; ties use lower class-balanced validation trial log loss. Test inference occurs after final selection. Two fixed windows per training trial form a training-only diagnostic probe; this subset measures learning, not generalization. History records losses, gradients, prediction class counts, and probability spread.

## Outcome

Training stopped after **28 epochs**, after 15 epochs without improving on the selected **epoch-13** checkpoint. The fixed 25-epoch minimum was respected. The selected checkpoint's results are:

| Partition | Trial balanced accuracy | Trial accuracy | Trial AUROC |
| --- | ---: | ---: | ---: |
| Training, all 865 original trials | 66.91% | 66.24% | 0.7445 |
| Validation, 199 trials | 56.07% | 52.76% | 0.5709 |
| Test, 200 trials | **46.25%** | **44.50%** | **0.5007** |

The subject-cluster bootstrap 95% interval for test balanced accuracy is **40.28–52.10%**, based on 1,000 replicates and only five test participants. It includes chance. Test window balanced accuracy is 47.36%; trial aggregation does not rescue performance. The model predicts 119 negative and 81 positive test trials overall, but **four of the five test participants receive a constant class prediction**. The remaining participant receives 39 negative and one positive prediction. Thus predicting both classes in the pooled cohort does not imply useful within-participant discrimination at the fixed threshold. [Per-participant results](../../results/development/deap_eegnet32_trainchannel_seed42/test_subject_metrics.csv).

Training cross-entropy falls from 0.6916 to 0.6221. Validation cross-entropy rises later, and training-probe balanced accuracy reaches approximately 69% without a consistent validation gain. This is an observed generalization gap in this control, rather than evidence that all reference methods overfit. The selected checkpoint was additionally evaluated on **all training windows after selection** to make the training row above use the same 15-window trial aggregation as validation/test. Those descriptive training results did not influence checkpoint selection.

The run took **874 seconds (14.6 minutes)** and used **117.3 MiB peak allocated CUDA tensors / 160 MiB reserved**. These figures exclude the CUDA context and other applications. It completed on the supplied RTX 3050 with batch 32, so VRAM exhaustion was not the limiting factor.

| Earlier development result on the same DEAP test participants | DEAP trial balanced accuracy |
| --- | ---: |
| Joint compact GRU-XNet v1, common14 STFT | 50.00% |
| Joint common14 log-bandpower logistic regression | 51.25% |
| Current DEAP-only canonical32 raw EEGNet | 46.25% |

This table is descriptive: the models differ in training sources, montage, representation, normalization, and budget. It is not a matched architecture comparison or a statistical superiority claim. The control does not supply a reliable improvement over the earlier chance-level experiments. [Saved comparison](../../results/development/deap_eegnet32_trainchannel_seed42/development_comparison.json).

Artifacts: training log (local workspace evidence: `publication_runs/deap_eegnet32_trainchannel_seed42.log`), [configuration](../../results/development/deap_eegnet32_trainchannel_seed42/config.json), [history](../../results/development/deap_eegnet32_trainchannel_seed42/history.json), [training metrics](../../results/development/deap_eegnet32_trainchannel_seed42/training_metrics.json), [validation metrics](../../results/development/deap_eegnet32_trainchannel_seed42/best_validation_metrics.json), [test metrics](../../results/development/deap_eegnet32_trainchannel_seed42/test_metrics.json), [diagnostic curves](../../results/development/deap_eegnet32_trainchannel_seed42/development_diagnostics.png), [test confusion matrix](../../results/development/deap_eegnet32_trainchannel_seed42/test_confusion.png).

## Interpretation and next comparison

This run changes architecture, electrode count, raw-versus-STFT representation, normalization, and budget together. Its result cannot isolate normalization as the cause of any improvement or degradation. A same-model comparison using the supported `--normalization window` mode would address that narrower question in a separately declared run.

The global training-fitted scaler preserves amplitude, but that change plus a longer budget and a simpler model did not produce useful held-out performance here. Per-participant prediction offsets and varied AUROC suggest inspecting subject effects/calibration and a matched normalization comparison before blaming the GRU alone. That is an inference from the predictions, not a verified cause. No thresholds or settings were changed in response to this test result, and no second control was trained.

The test cohort was already inspected during earlier development, so this result remains **development evidence**. First-party DEAP authentication remains unresolved. The mirror's 439 altered valence and arousal ratings are recovered from the matching spreadsheets exactly as before; this experiment changes no source ratings or label policy.

For the central publication question, train GRU-XNet and every comparator **jointly on the same three training datasets**, using common named electrodes or an explicitly matched montage, independent validation/test participants, equal budgets, and identical labels/aggregation. Evaluate each model's single selected checkpoint separately on DEAP, GAMEEMO, and SEED-IV. Report per-dataset and macro/worst-dataset balanced accuracy rather than relying on one pooled accuracy. Separately compare joint versus single-dataset training on matched partitions. Use LODO if claiming transfer to an unseen source. A reference model's lower accuracy or weak cross-dataset transfer alone cannot establish overfitting.

## Verification

**20 tests pass**, including training-only normalization, amplitude preservation, registered classifier/gradient checks, max-norm constraints, and validation tie selection/trial aggregation, alongside the previous pipeline tests. Local output: test log (local workspace evidence: `publication_runs/deap_control_tests.log`).

`verify-deap-control` completed successfully: it rehashed the full cache, recreated the participant split, refitted the scaler from training windows only, reconstructed checkpoint selection from validation history, and reproduced **all 3,000 test probabilities and all reported metrics**, including the bootstrap interval. [Verification record](../../results/development/deap_eegnet32_trainchannel_seed42/verification.json), verification log (local workspace evidence: `publication_runs/deap_eegnet32_verification.log`). Source compilation and Git whitespace checks also pass.
