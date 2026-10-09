# Contribution decision after the matched conditioning grid

9 October 2026. **Complete the current grid, retain its negative generalization result, and reassess the adapter before expanding the encoder study. No conference-ready contribution or changed main question is established.** The manuscript and existing PDF remain unchanged. This is a concrete exploratory proposal presented to the author, not approval or adoption of a paper pivot.

## What the completed experiment establishes

The [complete findings](GRU-XNet_CBraMod_Conditioning_Findings_2026-10-09.md) retain all 64 conditions: sixteen exact controls and 48 new 1,200-update trajectories. Strict replay verifies 256 model states and 768 probability metric sets. The [public audit](../../results/development/cbramod_conditioning_2026-10-09/postfit_analysis/verification.json) independently reconstructs all metrics, 64 source selections, 192 sampling-exposure points, eight normalizer sets, role boundaries, and 2,400 descriptive contrasts. Thirty fitting-related tests and two analysis checks pass. Maximum public metric discrepancy is 5.86e-14. No outer-test prediction was made.

The pretrained fine-tuned encoder now fits its full declared training panels under standardization plus encoder dropout off. The following are grouping 1 / grouping 2 values at the identical final 1,200 updates. Training and validation predictions average the same four ten-second windows. Unseen validation excludes training participants and stimulus materials.

| Dataset | Raw/dropout-on training BA (%) | Standardized/dropout-off training BA (%) | Standardized/dropout-off unseen BA (%) | Its unseen balanced loss | Uniform loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEAP, individual binary valence | 57.58 / 58.02 | 93.24 / 91.34 | 50.20 / 41.90 | 0.8519 / 0.9141 | 0.6931 |
| SEED-IV, assigned coarse three-class emotion | 69.14 / 67.28 | 100.00 / 100.00 | 22.22 / 50.00 | 1.3400 / 1.3366 | 1.0986 |

Thus weak pretrained training accuracy was not an absolute inability to fit these targets. The combined intervention changes two factors and demonstrates fitting under this recipe; it does not identify a sole cause, establish optimizer convergence, or validate the original/default recipe. The factorial comparisons distinguish the tested components:

- For pretrained fine-tuning, normalization with dropout on worsens final unseen loss in all four panels. With dropout off, it worsens three of four.
- Turning dropout off on raw embeddings improves final unseen loss in one of four panels. On standardized embeddings, it improves both SEED-IV panels and worsens both DEAP panels.
- Turning dropout off improves final unseen loss of frozen random heads in all four panels at either scaling. Their losses remain above uniform, so this smaller calibration improvement is retained without being promoted to a recognition lead.
- Source-selected combined pretrained checkpoints still have unseen losses 0.8228 / 0.7392 on DEAP and 1.1892 / 1.3366 on SEED-IV, above uniform on every panel. Selection reuses these validation outcomes and is secondary to matched-duration comparisons.

Many new fits clip gradients at almost every update. This is measured behavior, not evidence that clipping causes the validation failure. Accuracy outliers and a single favorable panel do not override the declared primary balanced log loss. There is no consistent generalization lead suitable for immediate larger confirmation. The two groupings reuse people/videos, only one head/random initialization is used, and no population intervals are estimated from these small development panels.

## Proposed bounded reassessment

**Exploratory hypothesis:** the pooled linear readout may discard useful channel/time structure or provide a different learning problem from the documented CBraMod nonlinear readouts. The next diagnostic should distinguish those adapter effects before comparing more encoders. A training-only gain would count as failure to produce a predictive lead. This hypothesis remains untested by the completed conditioning grid and is not a novelty claim.

The [primary-source update](GRU-XNet_Contribution_Research_Update_2026-10-09.md) documents the difference from the published default head. A [CPU-only shape audit](../../results/development/cbramod_readout_audit_2026-10-09/verification.json) exercised pinned author averaging, one-layer and two-layer heads with synthetic token tensors. All six standalone heads accepted their documented four-dimensional inputs; all three FACED wrapper paths matched the direct heads. All three inspected SEED-V wrapper paths failed after flattening before a classifier that expects four dimensions. This is a narrow finding about that pinned code path; it does not determine which implementation produced published scores, establish a defect in every CBraMod release, or invalidate the current explicit pooling implementation.

The recommended next diagnostic is **sixteen new fine-tuning runs plus eight existing matched anchors**:

| Factor | Proposed values |
| --- | --- |
| Source panels | Existing DEAP and SEED-IV grouping 1/2 only; no new outer-test inference |
| Encoder initialization | Authenticated pretrained checkpoint and the existing random42 encoder |
| Anchor head | Existing raw, encoder-dropout-on pooled linear fine-tuning trajectory; exact bytes retained |
| New head A | Pooled 200-dimensional tokens → 200-hidden-unit ELU/dropout-0.1 MLP → unchanged current task classes (2/3) |
| New head B | Flattened channel/time tokens → the documented two-layer 200-hidden-unit ELU/dropout-0.1 head → the same task classes |
| Training | Existing 1,200-update source-only schedule, optimizer, encoder/head rates, smoothing, clipping and balanced observation/window streams |
| Primary measurements | Balanced loss at matched 200/600/1,200 steps; BA secondary; familiar/unseen panels reported separately |

This retains the current ten-second input and current two-/three-class targets. It adapts a documented head to these inputs rather than reproducing published dataset scores. Head A helps distinguish adding nonlinearity from restoring flattened information. The MLP-to-linear comparison changes the head family and its dropout together; flattening also increases parameter count, so neither contrast is a pure pooling or capacity effect. Use common head seeds with dimension-specific weights, paired encoder initialization, independently controlled dropout RNG and the exact same observation/window streams. Do not claim byte-identical weights across different head dimensions.

Before fitting, pin the exact author files/operators, explicitly document the wrapper correction, calculate parameter counts, and run outcome-free forward/backward and peak-memory checks on the RTX 3050 6 GB. The two-layer flattened heads have roughly 12.8 million DEAP and 24.8 million SEED-IV input-layer weights before biases/output weights; memory must be measured. Do not silently substitute the much larger default three-layer flattened head. If this proposed head cannot fit, revise and declare the resource decision before inspecting task outcomes. This diagnostic is about one-quarter as many conditions as the completed factorial, but wall time depends on the measured head cost; no unmeasured runtime is promised.

Freeze code, data/asset bindings and the analysis before task fitting; retain initial/fixed-step/selected probabilities, all conditions, private full states, exposure histories and failures. No score-dependent amplitude scaling, label remapping, trial exclusion, input-duration change or optimizer search belongs in this diagnostic. It is **proposed, not declared or launched** by this decision report.

Native four-emotion SEED-IV remains a separate suitability question. Metadata supports all four classes in these panels, but mixing a target change into the head comparison would confound its interpretation. A later native-target experiment needs its own protocol, matched inputs and task-specific metrics. Three- and four-class accuracy cannot be treated as the same task, and a better SEED-IV native-target score would not prove harmful joint pooling across datasets.

## Criteria for continuation or stopping

Treat a candidate as an exploratory predictive lead only if it improves primary unseen balanced loss against the matched pooled anchor in both groupings of each corpus where a benefit is claimed, survives the fixed-step comparisons, and has meaningful predictive performance relative to uniform and available source-selected spectral controls. Report all contrary outcomes. Neither almost-uniform probabilities nor an accuracy-only improvement warrants expansion.

If such a lead appears, freeze that complete adapter recipe before repeating initializations and participant/material groupings. Use source-only tuning, then reserve genuinely new participants/materials or an appropriate external cohort for confirmation. Existing panels and the already examined outer folds cannot become untouched tests through renaming. A later confirmation must quantify participant/material uncertainty, retain random/pretrained and spectral controls, and assess EEG beyond contextual priors when familiar materials are included. The sixteen-run reassessment itself would still be development evidence.

If it yields only better training fit, or losses remain weak/inconsistent, stop the encoder search on these panels. Reassess the dataset/objective and required new recordings rather than adding architecture knobs. First-party DEAP authentication and physical amplitude calibration remain unresolved, and actual checkpoint training membership is not independently certified. A central result should not depend solely on those unresolved provenance assumptions.

## What could become a paper contribution

A provisional research question worth evaluating is: **under participant and stimulus shifts, which apparent pretrained-EEG gains survive source-only selection, native task semantics and matched readouts, and can a specific controlled intervention improve generalization?** A paper would need either a distinct intervention that works beyond known preprocessing/readout recipes, or a new mechanistic finding that survives matched controls and independent confirmation. The present data establish neither.

The [updated closest-work comparison](GRU-XNet_Contribution_Research_Update_2026-10-09.md) constrains this proposal. Existing work already studies stimulus confounding, source-versus-oracle selection, heterogeneous EEG benchmarking and annotation alignment. A generic benchmark, a claim that context is predictive, a wrapper shape correction, or ordinary normalization/dropout/head changes would be insufficient novelty alone. The possible narrower gap is jointly accounting for participant/material shifts, individual versus assigned labels, readout adaptation and source-only checkpoint choice in one controlled analysis; that gap still needs a complete prior-work check and measured new finding. No claim of negative transfer from joint DEAP/SEED-IV/GAMEEMO learning follows from these single-corpus probes.

**Recommendation to the author:** retain the current research question and manuscript while considering the bounded readout reassessment above. Do not adopt a negative-transfer, foundation-model or confounding paper merely because the historical high accuracy is unsupported. There is no evidence-backed main-question replacement ready for approval yet. If a supported alternative emerges, present its concrete contribution and findings, obtain the author's explicit approval, and archive the then-current manuscript immediately before changing direction.

## Evidence bindings and remaining publication concerns

The current fitting declaration SHA-256 is `cba5c0ee9c71b58f0e1348fbf483190e06e1216553632c32146ccad41fc94e90`; its 52 source/test files remain unchanged. Numeric analysis declaration SHA-256 is `b9400859b032f75f4daa1806c841b2a176c92ae0d0f6e1b4cb8c7d9b658fc328`. Original analysis/figure bytes are preserved; a separately bound linear-axis presentation reproduces all 768 loss and 320 final validation-effect coordinates. Public probabilities, labels and anonymous identifiers are released under the author's existing approval; EEG, embeddings, scaler/weight arrays and physical per-trial amplitudes stay local.

The [readiness record](GRU-XNet_Publication_Readiness_2026-10-05.md) remains the comprehensive concern list. Full joint/neural leave-one-dataset-out evidence, broader authenticated architecture/augmentation/montage controls, GAMEEMO's separate evaluation, independent confirmation, a supported contribution, historical result reconciliation, five missing manuscript figures, manuscript rewrite/PDF verification and venue selection remain outstanding. Completing this grid does not complete the paper.
