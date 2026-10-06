# Fixed-duration transformer and native-label development protocol

These experiments explore better representations before choosing the paper's contribution. They do not adopt a new research question. The author must approve a proposed change after seeing evidence; archive the then-current manuscript immediately before that change. The historical manuscript remains preserved in [the archive](../paper_archive/2026-10-05-pre-exploration/README.md).

## Questions and comparisons

Use all 1,080 original SEED-IV trials (15 participants, three sessions, 270 trials of each original emotion). Keep five existing participant rotations: nine training, three validation, three test; each participant is tested once for each initialization. All trials of a participant remain together. This is the previously inspected development cohort.

Run the complete 2 × 2 × 2 factorial, five folds, three initialization seeds (42, 43, 44): **120 neural runs**. Factors are mean-feature MLP versus temporal transformer; absolute versus relative log-bandpower; coarse three-class valence supervision versus original four-emotion supervision. No arm changes the training or evaluation trial population.

The MLP averages input tokens, then uses 56 → 128 → 86 with LayerNorm, GELU and dropout 0.2. The transformer projects 56 → 32, uses two independently initialized four-head encoder layers with feedforward width 64, GELU, dropout 0.2 and pre-normalization, fixed sinusoidal positions in seconds, mean token pooling and final LayerNorm. With three output classes these have **19,079 and 19,075 parameters**, respectively. This is an authored bandpower transformer baseline; it does not reproduce raw-waveform [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer), which already establishes EEG convolution/transformer prior work. A transformer is not itself a novel contribution.

## Input and label controls

Take the first ten non-overlapping four-second windows from every trial: a constant **40-second observation**. Original full-trial window counts vary by emotion (neutral median 38, sadness 42, fear 34, happiness 28). Every trial has at least ten windows, so this prefix retains the entire cohort and removes direct sequence-length variation. Filtering and resampling still use the full source trial before taking the prefix; the experiment is offline and cannot claim strictly causal 40-second acquisition.

Use the same 14 physically named common electrodes. Each token holds natural log Welch power in four half-open bands: 4–8, 8–14, 14–31, 31–40 Hz. Relative representation subtracts each channel's four-band log-sum-exp **within each window**, giving log power fractions; transform before temporal averaging. Fit each representation's per-feature scaler only to training participants' windows, and reuse it across both architectures and label objectives. Regenerate full-trial features from the 45 hashed raw source files, demand exact agreement with all 1,080 preceding native-cache mean features, and check exact full waveform agreement against the retained nonneutral cache for all 810 overlapping trials.

Coarse supervision predicts neutral / negative (sadness + fear) / positive (happiness). Native supervision predicts neutral / sadness / fear / happiness. Sample the **same original trials** using an identical 600-batch draw stream in every factorial arm for a given fold/seed: 20 draws with replacement from each coarse class per batch. Native classes are therefore intentionally not equally sampled. Hold feature normalization, total trial exposure, budget, and sampling fixed so objective granularity is the controlled difference. Initialize the native negative rows by duplicating the coarse negative row with bias reduced by ln(2); the backbones and initially grouped probabilities are identical within an architecture. Reset the stochastic training RNG after construction.

## Selection and analysis

Train 600 AdamW updates, batch 60, learning rate 0.001, weight decay 0.01, clip gradient norm 1. Use deterministic CUDA math scaled-dot-product attention, no AMP, scheduler, augmentation or early stopping. Evaluate validation every 25 updates; select maximum **common three-class validation trial balanced accuracy**, breaking ties by common three-class balanced log loss, then earliest checkpoint. Run test evaluation after selection. No test normalization, threshold, or checkpoint selection.

Primary evaluation is common three-class valence on all 1,080 trials (chance balanced accuracy 1/3). For native heads combine sadness and fear probabilities. Secondary binary evaluation excludes the same 270 neutral trials and conditions positive probability on negative + positive; both objectives are assessed on the same 810 trials (chance 1/2). Native four-class metrics are secondary and have chance 1/4: raw accuracies across different tasks are not an improvement estimate.

Report each initialization, all twelve paired factor contrasts on each common task, and participant-block percentile intervals from 10,000 fixed draws with seed 20261006. Average the three initializations within each resampled participant cohort; do not count seeds as independent people. These 24 comparisons are exploratory and not corrected for multiplicity. One participant grouping, overlapping training folds, inspected cohorts and a single corpus limit interpretation.

Also fit coarse three-class multinomial logistic controls on absolute and relative prefix-mean features and on the single full-trial window-count feature. Use identical folds, training-only scalers, class weighting, C = 0.01 / 0.1 / 1 / 10, validation BA then balanced log loss selection: 60 candidate fits and 15 selected models. The length-only control measures label-duration association; it cannot prove that earlier models exploited duration.

## Reproduction

Run in the author's PyTorch environment. The JSON plan is saved before fitting and committed with this protocol. Feature caches and checkpoints remain local; only bounded derived evidence is exported.

```powershell
conda activate pytorch
python scripts/temporal_native_controls.py plan --plan ../publication_runs/temporal_native_plan_2026-10-06.json
python scripts/temporal_native_controls.py prepare --data-root ../emotion-recognition-eeg-datasets --native-cache ../publication_runs/cache_native_seediv_features --common-cache ../publication_runs/cache_common14 --cache ../publication_runs/cache_temporal_native_seediv
python scripts/temporal_native_controls.py run --cache ../publication_runs/cache_temporal_native_seediv --output ../publication_runs/temporal_native_seediv --plan ../publication_runs/temporal_native_plan_2026-10-06.json --device cuda
python scripts/temporal_native_controls.py verify --cache ../publication_runs/cache_temporal_native_seediv --output ../publication_runs/temporal_native_seediv --device cuda
```

`plan` and `prepare` refuse existing paths; use the saved declaration/cache for replay. Verification checks bound sources and caches, paired draws, training-only normalization, validation selection, checkpoint replay, linear coefficients, coverage, and aggregate/bootstrap recomputation. This work is not pooled three-dataset training, unseen-dataset transfer, a foundation-model evaluation, or a submission-ready result. Existing corrected labels remain unchanged. Direct first-party DEAP recording verification and historical training provenance remain unresolved.
