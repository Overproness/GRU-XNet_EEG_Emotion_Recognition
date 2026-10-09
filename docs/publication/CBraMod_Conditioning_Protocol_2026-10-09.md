# Matched CBraMod embedding normalization and encoder dropout

Declared 9 October 2026 before preparing task features or fitting the new task models. This source-only development study follows the completed [learning-capacity and optimization controls](CBraMod_Learning_Protocol_2026-10-09.md). It investigates why the regularized pretrained encoder fitted its source data poorly. Conditioning or disabling dropout in an existing encoder does not establish a new conference contribution. The manuscript and its research question remain unchanged.

## Matched factorial design

Use the same DEAP and SEED-IV source panels in participant/material groupings 1 and 2, session 1, rotation 0, fold 0. Each panel compares pretrained/random42 × frozen/trainable encoder × raw/standardized pooled embeddings × encoder dropout on/off: **64 conditions**. The sixteen raw/dropout-on trajectories from the completed 1,200-update study are reused exactly, with byte-identical probabilities and histories and references to their original checkpoints. Run the other **48 trajectories** for 1,200 updates each, regardless of intermediate outcomes. All four initial/200/600/1,200 states survive: **256 states and 768 train/familiar-validation/unseen-validation probability metric sets**.

Preserve the corrected individual binary DEAP valence and existing assigned coarse three-class SEED-IV targets. Preserve the native 32/62-channel representations, prepared 200-Hz forty-second recordings, input divisor 100, and four disjoint ten-second windows. No raw-EEG preprocessing change, target relabeling, outer-test inference, new corpus, calibration or outcome-dependent exclusion is permitted. These are reused development participants and materials, with one head initialization and one random-encoder initialization, rather than independent confirmation.

## Embedding conditioning

Prepare **eight** statistic sets, one per source panel and pretrained/random initialization. Extract the initial encoder in evaluation mode on every original training trial's four windows. Average its channel/time tokens into 200-dimensional FP32 embeddings. Give every training trial and window equal weight; calculate means and population standard deviations in FP64, with `scale = max(std, 1e-6)`. Independently reconstruct statistics with scikit-learn StandardScaler and the same explicit floor. Neither validation nor outer-test recordings contribute to these statistics.

Store the mean and scale as FP32, non-trainable buffers. They remain fixed throughout fine-tuning and are shared across the frozen/trainable and dropout conditions for that panel and initialization. Do not update them from the current encoder or validation features. The raw condition bypasses affine arithmetic altogether and preserves the original operator path and checkpoint layout. This is conditioning of pooled embeddings, rather than an input EEG normalization change.

Canonical feature extraction and full-model inference use batch 16. Independently extract the same training features at batch 8 and publish the largest raw and standardized feature discrepancies as numerical diagnostics. Small encoder rounding differences can be amplified by small scales. Independent complete-state readout replay therefore uses the same canonical batch 16 and FP32 operator order; the different-batch feature diagnostic does not establish whole-model batch invariance. Scaler arrays and embeddings remain private.

## Encoder dropout and optimization

Dropout-on preserves the author's encoder configuration. Dropout-off disables all **61** encoder module-dropout and internal MultiheadAttention-dropout sites. The linear classifier has no dropout. Keep both frozen and trainable encoders in training mode during optimization, so that encoder dropout is meaningful even when its parameters are frozen. Evaluation always disables dropout.

Use precisely the preceding full-panel recipe: AdamW, head rate 0.001, trainable encoder rate 0.0001, weight decay 0.05, label smoothing 0.1, global gradient clipping at 1, and cosine decay over 1,200 updates to 1e-6. Each update samples six observations with balanced classes and replacement and one uniformly drawn window per observation. Preserve head seed 4242, random encoder seed 42, dropout seed 424242, and independent NumPy observation/window seeds 20261009/20261010. Every paired observation/window stream and initial head/encoder digest must match. Disabling dropout changes mask/RNG consumption by design, while retaining these observation/window streams. Use FP32 CUDA, mathematical attention, and no AMP.

For each new trajectory, retain mean loss and gradient/rate diagnostics every 100 updates, including the number of clipped updates and mean/maximum global gradient norm in that interval. Reused anchor histories have the original sampled gradient norms, but lack interval clipping counts; do not infer their clipping frequency. Independently reconstruct exposure at every positive saved step. The initial state is diagnostic only.

## Comparisons and verification

The **primary** comparisons hold the checkpoint step fixed at 200, 600 or 1,200. Within every panel/pretraining/trainability block and both validation roles, report normalization effects at each dropout setting, dropout effects at each scaling setting, and their interaction. Balanced log loss is primary; balanced accuracy is secondary. Retain initial and training diagnostics. Do not average binary and three-class loss values as if they described the same task, claim population confidence intervals from these panels, or choose a single global winning recipe.

Separately select each of the 64 trajectories among its three positive steps by equally weighted familiar/unseen balanced log loss, then mean balanced accuracy and stable declared order. These checkpoint-selected comparisons are **secondary**, because their selected durations can differ. No outcome-dependent budget, optimizer, threshold, clipping or numerical-tolerance change follows from an intermediate result. Aggregate full-grid interpretation waits for completion.

Strictly restore all 256 states and independently replay backbone → token mean → optional fixed affine → functional linear readout. Check fixed scaler buffers, frozen/updated encoder and head digests, dropout schema, source membership, metadata, probability metrics and all selections. The fixed probability/metric replay tolerance is 2e-6; public metric recomputation is checked at 2e-11. Hash-bind all 52 inherited/new source and test files, the completed predecessor and its sixteen anchor records/proofs, and the outcome-free GPU preflight. Recheck prepared waveform/feature hashes before declaration and completion. This does not repeat each optimizer update or authenticate external recordings/pretraining provenance.

Thirty relevant tests pass before declaration. An outcome-free synthetic GPU check covers both native channel layouts, both scaling settings and both dropout settings with three full encoder/head updates; actual memory is recorded in the declaration's resource preflight. Use an exclusive worker lock and preserve partial artifacts and failures. No concurrent GPU writer is introduced.

## Publication and decision gate

Commit and push the protocol, source and declaration before task feature preparation or fitting. Then publish verified normalizer metadata, the anchor phase, each four verified new trajectories and completion. Existing explicit author approval covers probabilities, labels and anonymous participant/trial references. Raw EEG, embeddings, normalizer arrays, coefficients, checkpoints and per-trial physical amplitudes remain local.

First-party DEAP waveform authentication, physical calibration, reused-validation limitations and the broader publication concerns remain unresolved. Results will determine whether a controlled learning diagnosis is convincing enough to justify a broader confirmatory encoder study. Any actual paper-question change still requires an evidence-backed proposal, explicit author approval, and an immediate fresh manuscript archive.
