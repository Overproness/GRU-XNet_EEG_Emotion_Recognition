# FACED EEGNet diagnostic findings — 11 October 2026

Neither readout passed the fixed engineering continuation screen. This diagnostic provides no qualifying lead for encoder-grid expansion. Preserve the complete evidence and reassess the contribution/data/target decision before further fitting.

All 20 declared neural fits completed: rotation 0 used three rates and two seeds for each readout (12 fits), followed by eight unchanged-schedule transfer fits across rotations 1–2. Each fit ran 120 epochs with checkpoints at 30, 60 and 120. The independent evidence audit passed, and all 60 private CPU-loaded checkpoint states passed hash, finiteness, geometry, schedule-counter and norm-constraint authentication. The state audit authenticates state dictionaries; independent checkpoint-to-waveform forward replay remains outside these checks.

The manuscript question remains unchanged. No confirmation outcomes or original MAT ratings were opened. The 33 reserved participants and reserved materials remain sealed. This is adaptive development on the same 20 validation participants and overlapping material panels; the three rotations provide consistency checks rather than independent replications.

The target is retrospective whole-clip item-10 valence on 0–7, predicted from offline marker-relative late EEG: seven nonoverlapping four-second windows from end−30 to end−2 seconds. The context branch is conditional on assigned elicitation category, acquisition group, catalogue duration and presentation ordinal being supplied at prediction time; assigned category is experimenter context rather than the participant's measured emotion. The population excludes neutral clips 13–16 and clip 22's unresolved version/footage correspondence. Held families separate normalized catalogue source titles. Physical footage independence and playback latency remain unauthenticated, as recorded in the frozen scope audit.

| Readout | Selected learning rate | Selected epoch | Mean gain over strongest no-EEG control | Engineering lead |
|---|---:|---:|---:|---|
| EEGNet only | 0.001 | 30 | -0.3618 | No |
| Context + EEGNet residual | 0.003 | 120 | -0.0075 | No |

Gains are participant/title-family-equal MAE differences in valence units; positive favors EEG. A qualified engineering lead requires positive gain in all three rotations, mean gain ≥0.15, and no learning flags. This threshold is a continuation rule rather than a psychometric minimum or a publishability criterion.

| Rotation | Readout | Strongest no-EEG control | Control MAE | EEG readout MAE | Gain | Learning flags |
|---:|---|---|---:|---:|---:|---|
| 0 | EEGNet only | category_median | 1.2220 | 1.6517 | -0.4297 | budget_limited_convergence |
| 0 | Context + EEGNet residual | category_median | 1.2220 | 1.1907 | +0.0314 | budget_limited_convergence |
| 1 | EEGNet only | context_only | 1.2099 | 1.6212 | -0.4113 | budget_limited_convergence |
| 1 | Context + EEGNet residual | context_only | 1.2099 | 1.3110 | -0.1011 | None |
| 2 | EEGNet only | context_only | 1.3106 | 1.5549 | -0.2443 | budget_limited_convergence |
| 2 | Context + EEGNet residual | context_only | 1.3106 | 1.2633 | +0.0473 | budget_limited_convergence |

The stronger comparator is selected from all five declared no-EEG controls: source global/coarse/category medians, the original selected context Ridge, and the fixed context kernel. Rotation 0 alone selected each neural schedule; familiar validation and rotations 1–2 did not select it.

| Rotation | Readout | Offset-only source MSE | Source MSE at 60, seeds 1/2 | Source MSE at 120, seeds 1/2 | Chosen-checkpoint unseen MAE, seeds 1/2 |
|---:|---|---:|---|---|---|
| 0 | EEGNet only | 4.6045 | 2.0986 / 2.2626 | 1.8116 / 1.8154 | 1.6365 / 1.7444 |
| 0 | Context + EEGNet residual | 2.6437 | 1.8219 / 1.7463 | 1.6067 / 1.5326 | 1.1930 / 1.2738 |
| 1 | EEGNet only | 4.9462 | 2.5187 / 2.3096 | 1.9670 / 1.8620 | 1.6548 / 1.6246 |
| 1 | Context + EEGNet residual | 2.7849 | 1.9741 / 1.8573 | 1.8054 / 2.0131 | 1.3281 / 1.3342 |
| 2 | EEGNet only | 4.8289 | 2.1883 / 2.3397 | 1.8242 / 2.0396 | 1.5950 / 1.6058 |
| 2 | Context + EEGNet residual | 2.6540 | 1.7261 / 1.6764 | 1.4568 / 1.5356 | 1.3328 / 1.2555 |

These source MSE values use unbounded eval-mode checkpoint outputs with the regression training offsets: source median for EEG-only and participant-OOF context for residual training. Full-source-context inference is reported separately. Online normalized/dropout training loss is not this learning diagnostic. Insufficient source learning means the 120-epoch source MSE exceeds 80% of its offset-only baseline; continuing >10% improvement from 60 to 120 flags the fixed budget. Chosen-checkpoint seed-MAE difference >0.15 flags validation-seed divergence; source-MSE ratio >2 (denominator floor 1e−8) flags source-seed divergence. These engineering flags do not establish biological signal absence or true negative transfer.

| Rotation | Readout | Leave-one-family-out gain range | Acquisition-group gain range |
|---:|---|---|---|
| 0 | EEGNet only | -0.4872 to -0.3521 | -1.0071 to -0.1463 |
| 0 | Context + EEGNet residual | -0.0341 to +0.0886 | -0.2471 to +0.1550 |
| 1 | EEGNet only | -0.5085 to -0.3315 | -0.5308 to -0.2964 |
| 1 | Context + EEGNet residual | -0.1331 to -0.0773 | -0.1160 to -0.0925 |
| 2 | EEGNet only | -0.3031 to -0.1782 | -0.5397 to -0.0696 |
| 2 | Context + EEGNet residual | +0.0100 to +0.0683 | -0.0907 to +0.0978 |

Sensitivity summaries are descriptive and retain the full-panel stronger comparator; they do not select another schedule or supply a tuning-corrected confidence interval. The legacy-1000, legacy-250 and modern-1000 acquisition groups contain 1, 9 and 10 validation participants respectively, so group ranges do not establish stable acquisition effects or justify exclusions. The complete group values and familiar/source metrics remain in the audit packet.

There were 980 unique source and 280 unique validation trial identities. All 980 source trials and 265 available validation trials passed exact selected-window/QC replay; the 15 fixed unavailable validation trials retained their declared fallback without EEG access. Every rotation retained 560 source-fit trials, 120 unfamiliar-validation trials and 160 familiar-validation trials. The four checkpoint/ensemble arms additionally distinguish source training-offset predictions from source full-context inference. The completed CSV denominators are 84,000 checkpoint rows, 8,400 ensemble rows and 1,800 frozen context-control rows.

The authenticated hardware preflight recorded NVIDIA GeForce RTX 3050 6GB Laptop GPU with 6.00 GiB device memory. Actual diagnostic peak PyTorch allocator reservation was 160.0 MiB. Conservatively charged neural fitting/checkpoint time was 1.275 hours; Version 2 full execution wall time, including replay and cached joins, was 1.476 hours. The frozen fit cap was six hours and the reserved-memory cap was 5,500,000,000 bytes. Allocator reservation excludes driver/context overhead and other processes; the hardware name is the preflight receipt rather than a newly asserted device inventory.

The preserved first hold used 638.859 seconds of separate execution wall time and zero neural fitting. Its time is additional to the Version 2 wall time above.

Actual runtime: Python 3.11.13, Torch 2.5.1, CUDA 12.4; cuDNN deterministic=True, benchmark=False, Torch deterministic algorithms=False.

The first attempt remains preserved. It passed source/available-validation replay and stopped before any neural fit because its cache join treated 3,360 deliberately permuted source-capacity control labels as original trial labels. Independent audit reproduced those frozen permutations and found every ordinary cached target consistent. Version 2 uses the quality cache as the authoritative true-target source, skips only the two exact known permutation-control/source-arm combinations before target conversion, and authenticates the kernel's fixed alpha from its bound plan. This was a separately activated implementation correction with the same target, participants, quality policy, grid and question; it was not an outcome-based rescue fit.

Complete evidence: [version 2 packet](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/tree/main/results/development/faced_eegnet_diagnostic_v2_2026-10-11), [independent audit](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/blob/main/results/development/faced_eegnet_diagnostic_v2_2026-10-11/independent_audit.json), [state authentication](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/blob/main/results/development/faced_eegnet_diagnostic_v2_2026-10-11/checkpoint_state_audit.json), and [preserved first hold](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/blob/main/results/development/faced_eegnet_diagnostic_2026-10-11/held_attempt_audit.json).

![Context and EEG comparison](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/raw/main/results/development/faced_eegnet_diagnostic_v2_2026-10-11/figures/diagnostic_comparison.png)

![Selected two-seed source learning](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/raw/main/results/development/faced_eegnet_diagnostic_v2_2026-10-11/figures/selected_learning.png)
