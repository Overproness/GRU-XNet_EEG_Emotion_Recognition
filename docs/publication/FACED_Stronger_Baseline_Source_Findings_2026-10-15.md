# FACED EEGNet source learning and held-source findings

Completed and independently audited 15 October 2026. **The authenticated trainable reference learns source ratings, including shuffled tiny-set assignments, but its fixed 360-epoch ensemble does not improve held-source prediction beyond either declared context comparator on this one cell.** The complete four-fit panel, 6,260 predictions and ten saved-state reload/forward checks are finished. The [frozen independent numerical audit passes](../../results/development/faced_baseline_independent_audit_2026-10-15/completion_v2.json); the manuscript, research question, confirmation candidates and protected-access gates remain unchanged.

## Fixed experiment and scope

The [original qualification](FACED_Stronger_Baseline_Qualification_2026-10-14.md), [preserved implementation hold](FACED_Stronger_Baseline_Held_Attempt_2026-10-15.md), [continuation](FACED_Stronger_Baseline_Continuation_2026-10-15.md) and [representative CUDA qualification](FACED_Representative_Batch_Smoke_2026-10-15.md) preserve the sequence and fixed scientific scope. The operative [continuation packet](../../results/development/faced_stronger_baseline_continuation_2026-10-15/README.md) retains every failed and successful stage. Code/protocol/plan were frozen before each access stage; passing representative-smoke evidence and the independent checker were pushed at `2a711be26` before source activation.

Use the existing metadata-first complete source cell6, participant fold3/material assignment0: **448 training trials from56 people/eight titles and84 held trials from14 different people/six disjoint catalogue-title families**. No availability decision used a target/score; choosing a complete cell still limits representativeness. All532 selected pairs are existing source-development observations. They are neither the20-person validation panel nor the33-person protected cohort. Held categories are Anger, Disgust, Sadness, Amusement, Joy and Tenderness; Fear/Inspiration are training-only. A single reused source cell is not an all-fold estimate, independent replication or eight-category transfer result.

The local EEGNet-8,2 scalar residual adapter preserves the author-checked convolutional backbone and has1,841 parameters. It is not a reproduction of a published FACED regression result or the strongest field decoder. Use the exact qualified late28-second EEG core represented by seven four-second windows; one rating and one loss per original trial. Source channel scaling uses only448 training trials. Context uses the same13 supplied category/acquisition/duration/ordinal features and fixed alpha100, with four participant-excluded inner source folds for training residual offsets. A full448-training-trial context fit supplies held offsets. This matched context is not presented as independently optimized, acquisition-free context.

Two main initializations complete360epochs at the sole fixed0.001 rate, AdamW weight-decay0.01, dropout0.5, original-trial batch8 and the prescribed constraints. Earlier120/240 checkpoints diagnose learning; **360 remains the primary endpoint** and is not replaced by a better-looking intermediate score. Two additional200epoch controls fit the same32 metadata-first training trials with real or one fixed permutation of targets, without held inference. No new model grid, rate search, full980-source finalist fit or confirmation candidate is produced.

## Learning and held-source utility

All errors use the original0–7 rating scale. Source MSE is the **unbounded eval-mode training-offset** error; held MAE clips the two-seed mean only after ensembling. Source training and held prediction are different outcomes.

| Epoch | Source MSE, seed1 | Source MSE, seed2 | Held ensemble MAE | Gain over matched context | Gain over category median |
|---:|---:|---:|---:|---:|---:|
|120|1.341591|1.396618|1.376887|−0.081542|−0.277495|
|240|0.992146|1.095568|1.375874|−0.080529|−0.276482|
|360, fixed primary|0.864700|0.878265|1.407483|−0.112138|−0.308091|

The offset-only source MSE is **3.031377**. Both main trajectories learn substantially relative to that starting offset: their final source MSE reductions are approximately71.5% and71.0%. Both also improve more than10% in source MSE from240 to360, triggering the predeclared **budget_limited_convergence** flag. Thus this is not a demonstrated optimization plateau or fully optimized neural benchmark. The insufficient-source-learning and source/held seed-divergence flags do not trigger under their declared thresholds.

The84-pair matched-context MAE is **1.295346** and category-median MAE **1.099392**. The fixed primary EEG ensemble is worse by0.112138/0.308091 respectively; every earlier ensemble is also worse than both controls. The individual360epoch seed MAEs are1.442593/1.412859; averaging predictions lowers ensemble error but does not reverse this comparison. These are participant/title-family-equal descriptive source-cell errors, not intervals or new-person/new-film population inference.

| Tiny32-trial control | Offset-only MSE | Final unbounded source MSE | Declared <=50% capacity screen |
|---|---:|---:|---|
|Real targets|6.112975|0.105801|Pass|
|Fixed shuffled targets|6.112975|0.139685|Pass|

Strongly fitting arbitrary assignments establishes that this input/model/optimizer can fit the small source set. It does not establish emotion information, useful generalization or the ability to interpolate every448-trial assignment. Longer main training lowers source error without producing a positive held-source utility lead on this cell. This narrows the earlier basic-learning concern while preserving the remaining convergence and limited-support caveats. No automatic schedule extension, new encoder grid or protected access follows.

## Execution, verification and retained holds

The successful source gate authenticates **532** exact late intervals,348,000,000 numerical count values and full prior QC/window bytes. It records1,044,000,000 selected signal bytes,591,360 data-header bytes and38,720,146,041 opaque original-integrity bytes, exactly matching the declared39,764,737,401-byte read budget. Preparation takes185.047seconds wall/91.531seconds CPU. Retained physical windows occupy228,802,560bytes below the300,000,000-byte cap, remain transient and are not persisted.

Before target conversion, all27,720 canonical cache metadata rows pass the fixed projection/token/identity/role/fold barrier. There are532 distinct reused source targets,6,468 conversions including5,936 repeated-value checks, and zero unselected/validation/protected target or prediction conversions. No original MAT rating or protected signal is decoded.

The four real-data neural fits take **771.922seconds charged**; complete execution wall time is **959.266seconds**. Both are within their separate2hour fit/preparation caps. Peak PyTorch-reserved GPU memory is **167,772,160bytes (160MiB)**, excluding driver/context overhead. Runtime is Python3.11.13/NumPy2.4.4/Torch2.5.1/CUDA12.4 on the RTX3050 6GB laptop, with two Torch CPU threads. Five auxiliary source-context fits are separate from the four neural trajectories.

All **ten** real-data checkpoints pass tensor-only fresh CPU loading, strict geometry, finiteness, norm constraints and forward replay on fixed source/held subsets through the qualified backbone. Maximum original-scale discrepancy is **5.086263e−7**, below atol1e−4/rtol1e−5. This producer-side independent reload verifies saved-state inference on those transient windows, not an independently rewritten architecture or every waveform. All6,260 declared prediction rows and complete histories are retained.

The separately frozen [postfit audit](../../results/development/faced_baseline_independent_audit_2026-10-15/completion_v2.json) checks all6,260 rows/532 source target identities, five full/inner Ridge normal equations, source-OOF separation/offsets, category priors, the tiny permutation, ten checkpoint states/19,240 tensor scalars, source MSE, clipping-after-ensemble, every held metric and the declared learning flags. Maximum normalized Ridge-equation residual is1.385230e−15. It reads no raw recording/waveform, original target cache, validation/protected numerical value and performs no fit. It authenticates source-scaler identity/geometry/finiteness and the producer's fixed-subset forward receipts; it **does not independently recompute waveform scaler moments or replay every neural forward**. These are explicit verification limits rather than omitted successful checks.

The initial532-trial source attempt remains a separate hold: five auxiliary fits and one neural attempt that failed before model construction because bare`cuda` lacked an index, with zero optimizer steps/checkpoints. The first four-source manufactured smoke also remains failed at backward before any optimizer step/new source access. The separately frozen eight-source/four-held smoke passes two manufactured optimizer steps and two CPU checkpoint replays; this supports representative execution, not a proven diagnosis of the earlier cuDNN failure. Across actual-data attempts there are two532-trial replays of the same unique source set, ten auxiliary context fits, one pre-optimizer failed neural attempt and four completed neural fits. Synthetic work is reported separately. No hold or cost is erased.

Final preservation checks retain31 manuscript bindings/31 archive pairs,93 old activation bindings,13+21 old proposal bindings,53 source-activation bindings and the explicit229-file preservation union. All ten original confirmation gates remain false; no paper/question/candidate change or reserved access occurs.

## Scientific decision supported

This diagnostic supports an implemented, learning-capable reference and a bounded adverse utility comparison. The pattern is compatible with source overfitting; it does not uniquely diagnose its cause. It supplies **no positive lead for replacing the frozen spectral candidate or expanding a model search**, and no general claim that EEG lacks useful individual emotion information. Remaining concerns include one adaptively reused quality-complete source cell, a still-improving optimization curve, incomplete all-fold/stronger-method coverage, two unsupported held categories, catalogue/annotation/footage provenance, fixed-film precision and limited novelty. The next decision should use the completed independent audit and closest-method evidence to settle a narrow claim/protocol; it should not promote this source outcome to confirmation or change the paper automatically.
