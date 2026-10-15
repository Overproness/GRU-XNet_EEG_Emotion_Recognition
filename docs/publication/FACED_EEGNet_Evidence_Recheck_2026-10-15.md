# FACED EEGNet evidence recheck for the revised exploratory paper

Rechecked 15 October 2026 from the already public continuation packet. **The earlier interpretation is supported: this implementation can learn the small source set, including a fixed target permutation, but its declared ensemble has no positive held-source increment on the inspected cell. An optimization plateau was not established.** No additional training is needed to verify those statements, and none was performed for this recheck.

## Evidence checked and numerical results

This review independently recomputed source MSE and neural held-source MAE from all relevant rows of the [6,260-row prediction export](../../results/development/faced_stronger_baseline_continuation_2026-10-15/predictions.csv), checked the [complete epoch histories](../../results/development/faced_stronger_baseline_continuation_2026-10-15/histories.json), and compared them with the [producer summary](../../results/development/faced_stronger_baseline_continuation_2026-10-15/comparison.json) and [existing independent audit](../../results/development/faced_baseline_independent_audit_2026-10-15/completion_v2.json). All seven public-result SHA-256 bindings in that audit still match. This is a public-export recheck, not a new waveform, scaler, optimizer-trajectory or complete neural-forward replication.

| Fixed epoch | Source MSE, initialization 1 | Source MSE, initialization 2 | Held two-initialization ensemble MAE |
|---:|---:|---:|---:|
| 120 | 1.341591 | 1.396618 | 1.376887 |
| 240 | 0.992146 | 1.095568 | 1.375874 |
| 360, declared primary | 0.864700 | 0.878265 | 1.407483 |

Source MSE uses the unbounded eval-mode prediction with the training offsets. Held MAE averages initialization predictions and then clips to the original 0–7 scale. The recheck reproduces the clipping-after-ensemble result exactly. The 84 held pairs comprise 14 participants by six distinct catalogue-title families, so equal participant/family aggregation agrees with the simple balanced-cell mean.

Between epochs 240 and 360, source MSE declines **12.85% and 19.83%**, respectively. The declared greater-than-10% screen therefore correctly flags budget-limited convergence. Relative to the offset-only source MSE of 3.031377, the final reductions are **71.48% and 71.03%**. The final 20 online training-loss values also have a lower mean than the preceding 20 for both initializations, but these noisy batch/dropout statistics should not be substituted for eval-mode source MSE or treated as a formal plateau test.

The matched-context and category-median MAEs in the independently audited summary are **1.295346** and **1.099392**. The primary neural ensemble is therefore worse by **0.112138** and **0.308091**, respectively. Every declared intermediate ensemble is also worse than both comparators. This recheck recalculates neural predictions directly; it cross-checks the comparator values against the prior independent normal-equation/prediction audit rather than refitting those comparators.

| 32-trial capacity control | Offset-only MSE | Recomputed final source MSE | Reduction |
|---|---:|---:|---:|
| Real targets | 6.112975 | 0.105801 | 98.27% |
| Fixed permuted targets | 6.112975 | 0.139685 | 97.71% |

Both controls use four training participants and eight titles. Their target multisets agree, supporting that the permuted control redistributes the same target values. They meet the declared capacity screen. Neither control is a test of held-person or held-material prediction; fitting the permuted assignments establishes small-set fitting capacity, not emotion information.

## Interpretation and manuscript wording

Recommended wording:

> The EEGNet residual adapter fitted both real and permuted targets in a 32-trial capacity control. On one reused source-development cell, however, its fixed 360-epoch two-initialization ensemble did not improve prediction beyond either contextual comparator. Eval-mode source MSE was still decreasing between the final fixed checkpoints, so an optimization plateau was not established.

The falling source error together with the later worsening held error is compatible with overfitting or poor transfer, but does not identify a unique cause. It does **not** imply that extending training will improve held prediction, that the architecture has no useful EEG information, or that all reasonable EEG models have been optimized. MSE training versus MAE evaluation is another qualification. The intermediate checkpoints remain diagnostics; the primary endpoint is not retrospectively changed to epoch 240.

These observations come from **448 source training trials and 84 held-source trials in one metadata-first quality-complete cell**, with two main initializations. They are not the earlier 840-pair spectral comparison, the 20-person development-validation panel, or independent confirmation in the reserved 33-person cohort. The held cell covers six categories; Fear and Inspiration have no held-material evidence here. Adaptive reuse and availability-based cell choice limit the scope even though the choice did not inspect targets or scores.

The [completed execution receipt](../../results/development/faced_stronger_baseline_continuation_2026-10-15/execution.json) records four completed neural fits. This review uses only existing public results and performs **zero fits, zero raw/private signal or rating reads, and zero reserved-cohort access**. It does not change any candidate, research question, earlier paper, confirmation gate or prior receipt. The revised paper should retain an exploratory scope and leave the 33 reserved participants sealed.
