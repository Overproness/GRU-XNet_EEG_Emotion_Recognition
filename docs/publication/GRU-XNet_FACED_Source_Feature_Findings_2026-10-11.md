# FACED source features, quality hold and reference diagnosis

11 October 2026. The original-data runner is implemented, tested and frozen. All **980 declared source trials from 70 participants** were authenticated, decoded and converted into the declared context/spectral features. Numerical processing passed on every trial, but **eight trials in four participants exceeded the frozen amplitude checks**. The source gate correctly held the conditional controls: **zero original ratings decoded, zero original-data models fitted and zero validation EEG trials opened**. All confirmation people/materials and the historical manuscript/question remain unchanged.

The [execution evidence](../../results/development/faced_classical_execution_2026-10-11/README.md), [immutable gate](../../results/development/faced_classical_execution_2026-10-11/source_gate.json) and [independent source analysis](../../results/development/faced_classical_execution_2026-10-11/source_analysis/summary.json) preserve the full result. This completes the requested runner/source audit. The declared supervised controls were conditional on source quality passing; they were **not executed**.

## What was implemented and checked

The exact runner/activation manifest was pushed in [3bbab7f07](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/commit/3bbab7f07) before broader original EEG access. It implements all three fixed rotations, context/EEG/joint Ridge families, five alphas, source-only scalers, four source-fitted priors, five held-participant source OOF folds, validation-input fallbacks, candidate retention and paired crossed-bootstrap diagnostics. Its maximum remains 90 original Ridge fits and 72 prior estimations, with a two-hour wall cap. No neural model or GPU is needed for these controls.

Eight manufactured runner tests cover a full bounded rehearsal, outcome-access denial, exact budgets, held-participant exclusion, prior scope, crossed-bootstrap pairing and validation fallback. Twenty-eight existing target/classical-contract tests, thirteen decoder/notch tests, three independent analysis checks and two reference-decomposition checks also pass: **54 checks in total**. These numerical fitting rehearsals use manufactured features/ratings; they are not original-data learning evidence.

The real source phase took **406.25 seconds**. It preserved all 70 people and 14 development clips, interpreted only the declared 30 scalp channels in each late 32-second interval, and retained **6,860 windows**. Every one of **638.4 million selected signed counts** was independently scalar-checked. Original technical headers, complete data/companion hashes and post-access data hashes passed. Header calibration agreed with an independently expressed gain/offset formula to **5.83e-11 µV** or better.

Each private feature row contains 13 metadata context values and 150 fixed Welch log-power values. No raw waveform array was saved. Public/local receipt replay checks their hashes; independent analysis reconstructs **970,110 exact byte-range entries**, recalculates all 176,400 channel/rule decisions from cached numerical metrics, checks marker-derived intervals/channel identities, and verifies all 2,940 private feature/QC/access objects without rereading original ratings or recordings. It reconstructs context features from frozen metadata; byte fingerprints and synthetic operator checks support spectral preparation, rather than a second independent spectral extractor on all originals.

## Source quality findings

| Acquisition group | Participants | Source trials | Flagged trials |
| --- | ---: | ---: | ---: |
| Legacy, 1,000 Hz | 3 | 42 | 0 |
| Legacy, 250 Hz | 30 | 420 | 5 |
| Modern, 1,000 Hz | 37 | 518 | 3 |
| Total | 70 | 980 | 8 |

The eight flags concern **0.82% of source trials**, across sub-004, sub-085, sub-087 and sub-091. They are amplitude flags: processed channel SD above 100 µV and/or peak-to-peak above 1,000 µV. Six trials exceed the SD rule and all eight exceed the excursion rule. There are no ADC-rail, stationary-count, low-variation or residual-line rule failures. The maximum processed 49–51/1–45 Hz power ratio is **0.01232**, below the fixed 0.10 threshold. Engineering thresholds are review checks, not validated artifact diagnoses or a reason to discard these people.

The complete [per-trial table](../../results/development/faced_classical_execution_2026-10-11/source_analysis/trial_quality.csv) and [channel flags](../../results/development/faced_classical_execution_2026-10-11/source_analysis/flagged_channels.csv) retain every trial. No threshold was raised, no flagged trial was removed, and no label was opened to decide eligibility.

## Separately frozen source diagnosis

The completed audit and a three-trial diagnostic declaration were pushed in [2a64ca5d4](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/commit/2a64ca5d4) before further source decoding. Select the two flagged trials with the highest processed excursion, plus the lowest-index unflagged trial in the same participant. This selects **sub-004 clips 18, 11 and 2**. Dominant native-SD channels, selected from existing source QC, are Oz, Oz and F8 respectively. The additional 720,000 decoded counts are repetitions of already permitted intervals, not additional participants/materials.

Counts and processed-window byte fingerprints reproduce the initial audit exactly. The diagnostic decomposes the unchanged linear pipeline into the dominant channel's contribution and all other channels' contribution. It checks their sum against the original output; maximum discrepancy is **0.00529 µV**, reflecting float32 output rounding. This is a mathematical counterfactual, **not a cleaned trial or a proposed zero-channel repair**.

| Source trial | Median SD in the other 29 channels: observed | Dominant-channel contribution | Remaining contribution |
| --- | ---: | ---: | ---: |
| sub-004 / clip 18 / Oz | 342.97 µV | 343.11 µV | 7.41 µV |
| sub-004 / clip 11 / Oz | 28.44 µV | 26.50 µV | 10.32 µV |
| sub-004 / clip 2 / F8, unflagged control | 7.52 µV | 0.56 µV | 7.43 µV |

The first trial establishes that the extreme Oz signal spreads through common-average referencing into the other channels. The window aggregates locate its strongest effect in the final two retained windows. The second trial shows a similar reference contribution concentrated in the first retained window; the matched control has a small contribution. The source of the extreme signal—motion, electrode contact or another process—is still not verified. Standard deviations of components are not additive, and the remaining contribution is not certified clean EEG.

The [PNG](../../results/development/faced_classical_execution_2026-10-11/reference_diagnosis/window_sd_diagnosis.png) and [SVG](../../results/development/faced_classical_execution_2026-10-11/reference_diagnosis/window_sd_diagnosis.svg) plot seven-window **SD aggregates**, not EEG waveforms. Numerical records and figure hashes are bound in the [diagnosis packet](../../results/development/faced_classical_execution_2026-10-11/reference_diagnosis/verification.json); the PNG was visually checked. The original gate remains held.

## Decision and next dependency

Prepare a **bounded source-only artifact-handling comparison**, focusing on identifying/treating extreme channels before common-average referencing. First qualify the authors' actual preprocessing code and montage/interpolation rules. The [FACED source paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10600242/) describes trial-level outlier detection, channel interpolation and ocular ICA before common-average referencing, and places its execution code in the Synapse release. The description alone does not authenticate that code or specify every implementation choice. Our minimal fixed pipeline omits those steps; the current diagnosis supports investigating that difference without claiming it explains prediction performance.

Freeze any continuation separately, retain the original failed quality result, apply fixed signal-only handling without opening ratings, and recheck the entire source rectangle. A passing source-quality result would then permit the already declared context/EEG/joint controls under an explicit updated processing declaration. Do not bypass the source gate, drop flagged people, use zeroed diagnostic channels as modeling inputs, or search artifacts from validation/confirmation outcomes.

Actual rating validity/variance, useful incremental EEG prediction, learning capacity, independent confirmation, physical calibration/timing and closest-method novelty remain unresolved. These implementation and QC findings establish neither a new contribution nor conference readiness. No alternative paper question has been adopted; present predictive findings and obtain the author's approval before any manuscript pivot.
