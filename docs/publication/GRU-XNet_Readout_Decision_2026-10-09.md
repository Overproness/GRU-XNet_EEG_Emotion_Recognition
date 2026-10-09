# Decision after the matched nonlinear-readout diagnostic

Written 10 October 2026 (Asia/Karachi). The study was declared 9 October; fitting completed at `2026-10-09T19:43:14.886753+00:00`, after midnight locally. **The authorized sixteen new runs and eight exact controls are complete. No candidate meets the declared predictive-lead criterion. Stop expanding encoder grids on these development panels.** The manuscript and historical joint DEAP/SEED-IV/GAMEEMO question remain unchanged; no different main question has been approved or adopted.

The [complete findings](GRU-XNet_CBraMod_Readout_Findings_2026-10-09.md) retain every head, initialization, saved step, role, selection, gradient history and reference. The [protocol](CBraMod_Readout_Protocol_2026-10-09.md) and [declaration](../../results/development/cbramod_readout_2026-10-09/plan.json) were pushed before fitting in `6a791a4a5`. Verified runs received regular GitHub milestones; completion was pushed in `e94b9a22d`. None was discarded or rerun with outcome-selected settings.

## What the experiment resolves

The two-layer flattened head substantially changes source fitting with the pretrained encoder. At 1,200 updates it reaches **100/99.75% training balanced accuracy on DEAP and 100/100% on SEED-IV**, versus 57.58/58.02% and 69.14/67.28% for the exact pooled-linear controls. This confirms that the readout adaptation matters for source fitting under this schedule. It does not demonstrate convergence, a unique pooling/capacity mechanism, or useful recognition on new participants/materials.

Slashes denote grouping 1 / grouping 2. These groupings reuse the same underlying people and materials. Loss is primary; BA is balanced accuracy in percent. The table includes both new pretrained heads rather than choosing the most favorable one.

| Dataset/head, pretrained | Training BA (%) | Unseen BA (%) | Unseen balanced loss | Loss difference from matched linear |
| --- | ---: | ---: | ---: | ---: |
| DEAP pooled MLP | 57.63 / 60.26 | 41.96 / 51.01 | 0.7283 / 0.7091 | +0.0087 / -0.0025 |
| DEAP flattened MLP | 100.00 / 99.75 | 38.63 / 44.74 | 0.9509 / 0.8916 | +0.2313 / +0.1799 |
| SEED-IV pooled MLP | 48.77 / 74.69 | 33.33 / 50.00 | 1.1851 / 1.3688 | -0.1503 / -0.1830 |
| SEED-IV flattened MLP | 100.00 / 100.00 | 33.33 / 55.56 | 1.1828 / 0.9969 | -0.1526 / -0.5549 |

Uniform loss is 0.6931 on DEAP and 1.0986 on SEED-IV. **DEAP's pretrained flattened head is worse than the matched linear control at every positive saved step in both groupings.** The pretrained pooled head has small early loss improvements, but these remain above uniform and the final direction is inconsistent. No DEAP predictive lead emerges.

On SEED-IV the pretrained flattened head improves familiar-material loss to **0.8424/0.9371**, with familiar BA **51.85/57.41%**. Unseen performance is useful on grouping two alone: final loss 0.9969 and BA 55.56%, with selected update 600 giving 0.9730 and 61.11%. Grouping one remains above uniform at every positive saved step and ends at chance BA. Its flattened-minus-linear unseen loss is +0.0143/+0.0112/-0.1526 at 200/600/1,200; a favorable final contrast against a worsening anchor does not establish stable useful prediction. The pooled MLP's final SEED-IV improvement over linear also remains worse than uniform on both panels.

All eight existing absolute/relative spectral heads are rechecked on exactly the same trials. Their unseen losses are **DEAP absolute 0.6932/0.6932, relative 0.6966/0.6893; SEED-IV absolute 1.0990/1.1125, relative 1.1052/1.0987**. Grouping-two pretrained flattened SEED-IV beats these references, while grouping one is worse than both. Representation, optimizer and search budgets differ, so this is a predictive reference comparison rather than an architecture-only effect. The isolated grouping-two outcome fails the declared requirement for consistent benefit across both groupings of a claimed corpus.

## Random flattened heads are degenerate controls

All four random-encoder flattened trajectories finish at chance BA on training and both validation roles: 50% on DEAP and 33.33% on SEED-IV. Each of the eight final validation tables is exactly constant across observations. Three final training tables are also constant; SEED-IV grouping two has maximum probability range only **1.22e-7**. The final saved encoder-gradient norm is zero in each trajectory. Probabilities are close to uniform, and the apparently favorable loss contrasts against overconfident random pooled controls largely reflect that constant predictor.

These are optimization-degenerate outcomes under this particular adapted head and schedule. They remain in every table, plot and analysis. **A pretrained-versus-collapsed-random contrast cannot establish generally better representation quality.** Initial outputs vary across trials; parameter digests change later, but weight decay can change tensors without useful supervised learning. Saved outputs/history establish degeneracy, not its activation or optimizer cause. The operator/gradient audits passed; no assertion is made that the authors' published experiments suffered this problem. No corrective learning-rate, activation, clipping or head search is launched after seeing these outcomes.

The complete public readout and pretraining contrasts remain numerically valid descriptions of the declared trajectories. Their scientific interpretation must include the random-head learning failure. More source fitting, routine nonlinear heads, or repairing a wrapper/optimizer issue would not itself establish a conference contribution.

## Scope and checks

All **24 conditions, 96 full states and 288 probability metric sets** verify. The public audit independently recomputes all 24 selections, 72 exposure points and **1,080 readout/pretraining contrast points**; the separately bound spectral comparison preserves another 768 descriptive points. Forty relevant pre-fit tests, four boundary tests and one complete synthetic public-grid audit pass. The original sixty bound files remain exact. A separately declared pre-fit adapter corrects only the frozen analyzer's excessive validation-subject guard; trial/material exclusions, predictions and fitting/analysis choices remain unchanged.

Maximum full-state probability discrepancy is **1.11e-16**, replay metric discrepancy is zero, and public metric discrepancy is **4.22e-15**. Maximum actual tensor allocation is **1.311 GiB**, excluding desktop/driver memory. All four PNG/SVG figure pairs retain every coordinate; the PNGs were visually inspected. No outer-test inference, new people/materials, label change or manuscript change was introduced. Numerical integrity does not certify external waveform provenance, physical units or actual checkpoint pretraining membership.

SEED-IV unseen validation contains only twelve trials from three people/four materials per grouping; DEAP has thirty-two from four people/eight materials. Familiar/unseen roles deliberately share held-out people with different trials. Repeated partitions and repeatedly selected panels are development evidence, without population intervals or independent confirmation.

DEAP also inherits deterministic per-training-participant/class subsampling from the earlier exposure-arm comparison. Retained counts use eligible training-participant labels from both arms, including excluded-material labels in the alternative arm. Thus current head effects are conditional on that prepared matched population. This does not change their paired-trial comparison, but a future strictly sealed unseen-material evaluation must construct its source population without consulting labels from excluded materials, including those supplied by training participants. Existing trial/participant disjointness and source-only checkpoint selection do not by themselves establish that stronger preparation boundary.

## Decision and next work

**Do not begin a larger encoder comparison or confirm the isolated favorable panel.** The declared success requirement is not met, and the random flattened control additionally lacks a healthy source-learning result. This completes the bounded readout diagnostic; it does not complete the paper.

The next useful stage is a **data/target and contribution feasibility review** before any additional model grid:

1. Establish which available recordings have independently verifiable source identity, physical calibration and the label semantics required by a candidate question. First-party DEAP waveform comparison remains unavailable. Coarse SEED-IV assigned labels and individual DEAP/GAMEEMO valence are different targets; native four-emotion SEED-IV suitability remains untested by this diagnostic. Preserve that distinction instead of relabeling the current results.
2. Specify a source population and a genuinely reserved participant/material cohort before new fitting. Remove dependence on excluded-material labels in source-population preparation for a strict new-material claim. Previously inspected outer folds cannot be made untouched by naming them confirmation. Any eventual pretrained comparison needs healthy source-learning controls, actual overlap/membership assessment and matched task preprocessing.
3. Identify a narrower, falsifiable contribution against the [closest-work update](GRU-XNet_Contribution_Research_Update_2026-10-09.md), with a practical confirmation budget and explicit stopping rule. Existing work already covers stimulus confounding, annotation alignment, heterogeneous EEG benchmarking and source-selection pitfalls. The present grid supports no new intervention, general absence of EEG information, or joint negative-transfer claim.

No evidence-backed replacement question is ready for adoption. If the feasibility review supports one, present its measured motivation, novelty gap and experiment plan to the author. Obtain explicit approval and archive the then-current manuscript immediately before adopting it. If the historical joint-training question is retained, full target-only versus joint and train-on-two/test-on-third controls still require credible task-specific baselines and matched label semantics.

The [readiness checklist](GRU-XNet_Publication_Readiness_2026-10-05.md) retains the wider outstanding concerns: authenticated recordings/units, actual pretraining membership, full joint/neural cross-corpus evidence, GAMEEMO's separate protocol, broader authenticated architecture/montage/augmentation controls, independent confirmation, a supported contribution, historical result reconciliation, five missing manuscript figures, manuscript/PDF revision and venue selection. The current paper remains unsuitable for submission on these findings alone.
