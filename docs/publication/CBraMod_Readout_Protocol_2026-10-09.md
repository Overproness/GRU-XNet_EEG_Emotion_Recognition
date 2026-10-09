# Matched CBraMod nonlinear readouts

Declared 9 October 2026 before fitting the new task models. The author authorized the sixteen-run proposal after reviewing the [completed conditioning findings](GRU-XNet_CBraMod_Conditioning_Findings_2026-10-09.md) and [contribution decision](GRU-XNet_Contribution_Decision_2026-10-09.md). The experiment tests whether the pooled linear head limits useful prediction. It remains exploratory; the manuscript and historical joint-training question are unchanged.

## Conditions and data

Run **sixteen new fine-tuning trajectories** and reuse **eight exact pooled-linear controls**. Cross DEAP/SEED-IV, participant/material grouping 1/2 and pretrained/random42 encoder initialization with three readouts:

| Readout | Operations | Head parameters: DEAP / SEED-IV |
| --- | --- | ---: |
| Pooled linear, exact anchor | Token mean → Linear(200, 2/3) | 402 / 603 |
| Pooled MLP | Token mean → Linear(200, 200) → ELU → Dropout(0.1) → Linear(200, 2/3) | 40,602 / 40,803 |
| Flattened MLP | Channel/time/dimension flatten → Linear(64,000/124,000, 200) → ELU → Dropout(0.1) → Linear(200, 2/3) | 12,800,602 / 24,800,803 |

The pooled MLP controls adding nonlinearity before assessing flattened information. The head-family contrast also adds head dropout; flattening also increases parameter count. Neither is a pure pooling or capacity effect. No large default three-layer classifier is substituted.

Use the existing source-only session 1/rotation 0/fold 0 panels, prepared native 32/62-channel 200-Hz forty-second arrays divided by 100, and four disjoint ten-second windows. Preserve corrected individual DEAP binary valence and the assigned coarse three-class SEED-IV labels, retaining original labels. No new participant/trial exclusions, amplitude calibration, normalization, label change or outer-test inference is introduced. Native four-emotion SEED-IV is a separate future suitability experiment.

These panels reuse participants/materials and one head/random initialization. They are not new confirmatory cohorts. Training and both validation roles exclude each other's participants/trials; unseen validation also excludes training materials, while familiar validation uses training materials. All three roles and every condition remain available.

## Author-code authentication and wrapper adaptation

Use author commit `b9e961003214326972c567eff390e75b0287e32a`. Bind the already inspected FACED/SEED-V head files and `finetune_main.py` to their original SHA-256 values. An outcome-free CPU audit executes the pinned standalone two-layer heads with a synthetic identity backbone, copies matching local parameters, and checks exact outputs plus input/parameter gradients in both evaluation and training modes. The head dropout mask is matched explicitly. This verifies operators, not a published dataset score.

The FACED standalone head already supports 32 channels and ten patches; task output changes from nine to two classes. The SEED-V standalone head's input changes from 62 channels/one patch to 62 channels/ten patches, and output changes from five to three classes. Pass four-dimensional encoder tokens directly to the head. Bypass the inspected SEED-V wrapper's premature flattening, which sends two-dimensional inputs into a four-dimensional rearrangement. This limited adaptation does not establish which source version produced published scores. The pooled MLP supplies mean tokens to the same two-layer operations with a 200-dimensional input.

## Optimization and random streams

Run every new condition for exactly **1,200 updates** with balanced six-observation draws with replacement and one uniformly chosen window per observation. Preserve independent NumPy observation/window seeds 20261009/20261010 and the exact streams used by the anchors. Keep AdamW head rate 0.001, encoder rate 0.0001, weight decay 0.05, smoothing 0.1, global clipping at 1, and cosine decay over 1,200 updates to 1e-6. Use FP32 CUDA and explicitly mathematical attention, without AMP or augmentation. Encoder dropout remains on at all 61 module/internal-attention sites.

Preserve encoder initialization and head seed 4242. Head dimensions differ, so weights cannot be byte-identical across readouts. Within each dataset/readout, verify identical initial head digests across both groupings and pretraining conditions. Initial encoder digests must match across all three heads within each panel/pretraining condition.

Encoder dropout uses global seed 424242. New head dropout executes the author's functional dropout operator within a scoped RNG context at seed `424243 + update`, restoring CPU/active-CUDA global states after each call. Both new heads have the same 200-dimensional hidden shape and therefore the same head mask per update. Adding head dropout does not consume the encoder's RNG stream; different activations and parameter updates remain deliberate differences. The linear anchors consume no head RNG. Evaluation disables dropout, uses canonical batch 16, averages four window logits in FP64 and then applies softmax.

Preserve initial and 200/600/1,200 states for all 24 conditions: **96 states and 288 train/familiar/unseen probability metric sets**. Reused anchors retain byte-identical probability/history files and references to their original private checkpoints. New trajectories retain clipping frequency, global/head/encoder gradient norms, rates and mean minibatch loss every 100 updates. Do not invent missing clipping frequencies for anchors. Independently reconstruct observation/window exposure at all positive saved steps.

## Frozen analysis and verification

Balanced log loss is primary; balanced accuracy is secondary. At identical 200/600/1,200 steps, report pooled-MLP minus linear, flattened-MLP minus linear, and flattened minus pooled-MLP within every panel/pretraining block. Separately report pretrained minus random within each head. Include initial/training diagnostics and both validation roles. There are **864 fixed-step contrast points**, including **432 positive-step validation points**. Do not pool binary and three-class losses into a single score or select a global winning recipe.

Each trajectory separately selects among 200/600/1,200 using equal familiar/unseen balanced loss, then mean BA and stable first candidate. These selections are secondary because durations can differ; retain all 24 selections and 216 selected contrast points. A selected-validation score is not independent confirmation. No outcome-driven duration, optimizer, threshold, exclusion or head search is allowed.

Strictly restore all 96 model states and replay the backbone with explicit functional mean/flatten → linear → ELU → linear operations, bypassing the fitting wrapper. Use the same canonical inference batch 16 and precision/operator order. Require probability and replay metric differences at most 2e-6; public probability-derived metrics must match within 2e-11. Check state/component digests, updated encoder/head invariants, initialization/stream pairing, source boundaries, selections, metadata and exact anchor bytes. This does not repeat every optimizer update or certify external recordings/pretraining membership.

All **60 inherited/new source, analysis and test files** are bound before fitting, including the source-only selection helper and prior shape-audit source. Independently recheck prepared waveform/feature hashes before declaration and completion. Forty relevant CPU tests pass before declaration, including RNG isolation, functional gradients, positional-information behavior, small full-state fit/replay, tamper detection and closed-form contrast/metric arithmetic. Synthetic tests remain distinguishable from EEG task evidence.

Before declaration, run outcome-free full-backpropagation and canonical-inference checks for both native layouts and both new heads on the RTX 3050 6 GB. Record exact parameter counts, memory and timing. If infeasible, preserve the failure and revise the protocol before task outcomes. Use one GPU writer and an exclusive worker lock; preserve partial runs and failures.

## Publication and decision

Commit/push code, protocol and hash-bound declaration before task fitting. Publish the eight verified anchors, every four newly verified trajectories, and completion to the existing origin/main. Existing explicit author approval covers verified probabilities, labels and anonymous participant/trial references. Raw EEG, embeddings, scalers, full checkpoints and per-trial physical amplitudes remain local.

Interpretation waits for the full grid. A predictive lead must improve primary unseen loss against the matched pooled anchor in both groupings of each corpus where a benefit is claimed, survive fixed-step comparisons, and have useful performance relative to uniform and existing source-selected spectral controls. Report all contrary outcomes. A training-only gain or near-uniform calibration gain does not justify another encoder grid. If no lead emerges, stop expanding these encoder controls and reassess the dataset/objective and need for new recordings.

Any qualifying lead first needs a frozen recipe and independent repetitions. First-party DEAP signal authentication, physical calibration, actual pretraining membership, reused-panel limitations and wider publication concerns remain unresolved. Existing head changes or a wrapper correction are not scientific novelty. Any actual main-question change still requires measured evidence, a concrete proposal, explicit author approval and a fresh manuscript archive immediately before adoption.
