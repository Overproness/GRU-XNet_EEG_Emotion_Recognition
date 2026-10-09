# Source-only preprocessing and EEG baseline diagnostic

Revision 2 declared 9 October 2026. This is authorized exploration, not a change to the manuscript or research question. Earlier experiments and fitting sources remain frozen. The source cohort has already been inspected; these results cannot be called untouched confirmatory evidence.

The initial declaration and source files are preserved. Its first classical panel stopped because the new context control incorrectly exponentiated the existing helper's probability output; the probability-sum check rejected it before any EEG training. All initial attempt artifacts are excluded. Revision 2 uses the probabilities directly and adds an integration test. It retains every original montage, normalization, duration, budget, grouping, initialization and selection choice. Thirteen relevant tests pass before fitting this corrected version.

The immediate problem is weak EEG validation/generalization despite demonstrated capacity to fit training examples. This study tests possible explanations involving electrode coverage, normalization and temporal input size. It does not assume that any of these causes the weakness or promise a successful method.

Use the existing first participant fold/material rotation in session 1, unexposed arm, for both fixed groupings. Each model sees only its own training rows and two disjoint validation clip panels from separate validation participants. No outer-test waveform is sent to fitting, scaling, calibration, model selection or inference. Dataset preparation may read participant files containing additional trials, and the union of source rows across the two different groupings; each model receives only the declared source subset for its own grouping.

The factorial crosses common 14 versus native 32 DEAP/62 SEED-IV electrodes, training-channel versus per-trial normalization, and four-second versus forty-second EEGNet inputs. Both use the same first forty seconds per trial. Four-second training samples one of ten nonoverlapping windows for each balanced trial draw; inference applies softmax to the mean of all ten window logits. Short and long models therefore differ in head size and sampled signal exposure; equal update counts are not equal signal/compute budgets or an isolated temporal-duration intervention.

DEAP adds two four-second, training-channel-normalized measured-baseline subtraction recipes, one per montage. The separately filtered three-second baseline is averaged across three one-second blocks and repeated over the trial waveform. This is a local diagnostic adaptation, not reproduction of a published EEG emotion score. SEED-IV has no corresponding measured baseline in the inspected release. Classical controls use mean absolute and within-channel relative log band power for both montages; DEAP also uses baseline-relative log power. The EEG-free source-video prior is retained.

There are **72 neural trajectories, 36 recipe/panel comparisons, 288 neural candidates, and 20 selected classical heads from 80 candidate fits**. The pilot fixes initialization 42; it does not add the already completed second-initialization factorial. Neural trajectories use both learning rates 0.001/0.0003 and 600 updates. Candidates at 200/600 updates retain ordinary EMA and training-only population BatchNorm states. Within each recipe/panel, equally weighted familiar/unseen validation balanced log loss selects rate, duration and BN variant; mean balanced accuracy breaks ties, then declaration order. Here duration means training updates; four/forty-second input duration is a separately reported recipe axis. Every weak candidate and complete learning curve is preserved. The finite budget cannot establish convergence.

Native inputs are regenerated from hash-checked original downloaded recordings, using the maintained physical filter/resampling and corrected labels. Every common-electrode source prefix must exactly match its earlier verified cache. DEAP's Kaggle signal provenance remains qualified, even if all internal input checks pass. Saved candidate states replay on fresh models; normalizers, draws, initializations, probability metrics and source selection are checked. Population states are reconstructed with the previously audited calibrator; this is not a second independent calibration algorithm. All classical candidates are independently refit, with coefficient/scaler and prediction checks.

Only compact derived evidence is published. Raw EEG, measured baseline waveforms, normalizer arrays and neural weights stay local. Verified milestones are committed and pushed every six new neural trajectories and on completion. The machine-readable [revision-2 plan](../../results/development/preprocessing_diagnostic_v2_2026-10-09/plan.json) and [progress](../../results/development/preprocessing_diagnostic_v2_2026-10-09/progress.json) are authoritative. The [initial failed attempt](../../results/development/preprocessing_diagnostic_2026-10-09/progress.json) remains available and is excluded from these results.

Interpret both validation panels and both groupings separately. A validation improvement here is a diagnostic lead, not a new independent test result, architecture-only comparison or publication contribution. Broader confirmation requires a separate declaration. Present any evidence-backed research-question proposal to the author; obtain explicit approval and archive the then-current manuscript before adopting it.

## Commands

```powershell
conda activate pytorch
python scripts/preprocessing_diagnostic_v2.py plan
python scripts/preprocessing_diagnostic_v2.py prepare
python scripts/preprocessing_diagnostic_v2.py run --push
```

The synthetic resource pilot tests model allocation/training only, with peak CUDA tensor allocation under 0.5 GiB before real source data and calibration. It does not measure accuracy or full-study runtime. Data and preprocessing lineage are local; reproduction requires separately obtained recordings.
