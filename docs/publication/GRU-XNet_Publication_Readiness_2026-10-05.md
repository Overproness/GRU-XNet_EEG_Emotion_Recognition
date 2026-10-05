# Publication readiness and research decision record

Date: 5 October 2026. **The project is not ready for submission to a strong conference.** The data/pipeline repairs are substantial, but they have not produced convincing independent generalization or a demonstrated new contribution. Passing software tests is necessary for trustworthy experiments; it is not evidence of scientific novelty or recognition performance.

The author now prioritizes a strong publishable paper over retaining the original architecture or joint-training question. Exploratory experiments with alternative questions are authorized. **Adopting a different research question requires showing the findings to the author and receiving explicit approval first.** Until then, joint three-dataset training is the historical question and alternative directions remain exploratory. Before an approved change, preserve the then-current manuscript and supporting files so the change can be reversed. Commit and push validated milestones to the existing GitHub repository.

## Status of the original concerns

| Original concern or requirement | Current status | Evidence / remaining work |
| --- | --- | --- |
| Understand the paper, working folders, export, and saved results | Completed | Initial publication/code review and artifact checks are saved. |
| Research closely related work and recent advances | Initial targeted review completed; novelty assessment remains open | CNN–BiGRU–attention, montage alignment, hierarchical label alignment, and unified benchmarks already have precedents. Some summaries rely on abstracts; full methodological comparison is still required for the chosen contribution. |
| Interpret historical 95.91% correctly | Identified; manuscript correction outstanding | It is the saved pooled sample result, not proof of unseen participants or unseen datasets. The exact old run provenance cannot be reconstructed from the available synthetic lineage. |
| Prevent augmentation/window leakage | Fixed in the maintained pipeline | Original participants/trials are partitioned; windows do not overlap; augmentation is training-only. Historical arrays are excluded rather than declared repaired. |
| Replace invented synthetic subject IDs | Fixed in new experiments | Every real window retains its dataset-qualified participant and original trial. Historical synthetic ancestry remains unresolved. |
| Correct SEED-IV sadness/fear/happiness and GAMEEMO labels | Implemented and checked | All 15 SEED-IV participants/three sessions are loaded; GAMEEMO uses visually checked participant SAM valence. Neutral/midpoint exclusions are explicit. |
| Verify DEAP ratings | Mirror discrepancy found and recovery implemented | 439 valence and 439 arousal values differ by `9-r`; spreadsheet ordering is checked with dominance/liking. Corrected runs already use recovered labels. |
| Authenticate original source recordings | Partly completed | Mirror-to-bundle size/CRC checks and six SHA-256 spot checks passed. First-party DEAP signal authentication is still unavailable; spreadsheet recovery does not establish signal authenticity. |
| Align physical channels, sample rates, and durations | Implemented | Named common electrodes, masks, physical resampling, non-overlapping measured windows, and a real STFT time axis. Canonical-montage methods still need measured comparisons. |
| Independent validation/test protocol | Implemented and verified | Participant-disjoint train/validation/test and source-only validation for LODO. Existing test cohorts have been inspected during development. |
| Full subject-independent and train-on-two/test-on-third evidence | Incomplete | One full pooled subject split and diagnostic runs exist; the three LODO runs are three-epoch capped pilots. A complete repeated-fold/LOSO and full LODO study has not run. |
| Matched BiGRU/BiLSTM, attention, CNN, montage, augmentation, and reference baselines | Outstanding | Switches and some controls exist, but a completed matched ablation suite does not. Old unequal baseline comparisons cannot establish superiority. |
| Establish reliable recognition on independent participants | Unresolved for the proposed neural/joint study | DEAP EEGNet normalization controls give 46.25% and 46.67%. A five-fold SEED-IV classical control reaches 67.13% binary balanced accuracy; adding other datasets reduces it to 60.93%. These development controls do not establish a new neural method or general transfer. |
| Reconcile old confusion matrix/sample counts and augmentation claims | Findings documented; historical manuscript still unrevised | The old 6,621-prediction result and the 6,567-example figure are different evaluations. Claims of 11 active augmentations and complete ablations still need removal/replacement. New runs export consistent checkpoint-linked metrics/figures. |
| Final manuscript, full references, compiled PDF, venue selection/submission | Outstanding | `report.tex` remains historical; only a methods fragment and review reports have been written. No final manuscript or submission exists. |
| GitHub checkpoints and manuscript backtracking | Implemented | Commit `f8dd37f` was pushed to the existing repository. The current manuscript and historical PDF are preserved with hashes, and reports/selected development evidence are portable. Five active source figures are missing; the archive records this. Archive again immediately before any approved question change. |

The detailed evidence is in [implementation status](GRU-XNet_Implementation_Status_2026-10-05.md), [initial review](GRU-XNet_Publication_Review_2026-10-05.md), [dataset provenance](GRU-XNet_Dataset_Provenance_2026-10-05.md), [first-party access investigation](GRU-XNet_First_Party_DEAP_Check_2026-10-05.md), [repository code review](GRU-XNet_GitHub_Configuration_Review_2026-10-05.md), [DEAP control](GRU-XNet_DEAP_Control_2026-10-05.md), and [new exploratory findings](GRU-XNet_Exploration_Findings_2026-10-05.md).

## What can make the paper stronger

A stronger paper needs a specific research question, a contribution distinguishable from the closest work, and evidence that survives matched controls. High accuracy under an easier split would not repair the novelty or generalization problems. A negative finding can be useful if the study isolates causes and establishes a new, reproducible result across methods/datasets; the current coupled development runs do not yet do that.

Two exploratory tracks are justified before proposing a pivot:

1. **Resolve the learning/generalization failure with matched controls.** Hold model, montage, labels, split, optimizer, and budget fixed while changing normalization; inspect per-participant probability offsets using validation only. Run native-label, native-montage controls to determine whether forced binary pooling is concealing usable dataset-specific structure. These experiments diagnose possibilities, not a newly adopted paper question.
2. **Assess a focused contribution rather than adding another recurrent layer.** Possible directions include reliable transfer under electrode/subject shifts, preserving dataset-native label semantics, or a controlled integrity/leakage study. Montage alignment overlaps UBRRL and pretrained heterogeneous EEG models; coarse/fine label alignment overlaps CIHL; a generic unified benchmark overlaps LibEER and EEGain. Each direction needs a narrower demonstrable gap before it is worth adopting.

Recent LibEER already benchmarks seventeen deep models across six datasets, so a small generic benchmarking pipeline is not by itself a convincing novelty claim. [Primary LibEER paper, revised July 2025](https://arxiv.org/abs/2410.09767v3).

For each candidate, report the exact hypothesis, closest prior method, what is new, the controlled experiments, expected resource cost, failure criteria, and measured evidence. Show the author the comparison **before** changing the paper question. If neither the original question nor an alternative has a supported contribution, report that outcome rather than manufacture a positive result.

## Conference planning

An applied signal-processing or affective-computing venue is a plausible fit for a rigorous EEG method or focused empirical contribution; this is a recommendation based on topic, not an acceptance prediction. A top general machine-learning submission would need a broader methodological contribution and evidence beyond one application with these three datasets. Venue selection should follow the contribution and its evidence.

The author has no fixed conference or deadline and is willing to wait for a strong venue. Do not rush toward an already closed deadline. The current ICASSP 2027 main-paper page lists **23 September 2026**, which has passed as of this report; the journal-presentation route is not a replacement submission route for an unpublished conference paper. [Official call](https://2027.ieeeicassp.org/call-for-papers/). No future ACII deadline is assumed without an official announcement.

## Decision gate

**No research-question change has been approved or adopted.** The next deliverable is exploratory evidence and a concrete recommendation, together with regular GitHub milestones. The original manuscript remains available for backtracking. A manuscript snapshot is preservation, not approval of a pivot.
