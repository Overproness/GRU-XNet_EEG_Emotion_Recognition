# Current progress and claim decision

15 October 2026, Asia/Karachi. **The execution guard and bounded source reference experiment are complete and audited. They provide no qualifying positive EEG lead. Keep the reserved cohort closed while settling the value and scope of the scientific comparison.** The saved papers, research question and four existing confirmation candidates remain unchanged.

## New empirical evidence

The [complete EEGNet findings](FACED_Stronger_Baseline_Source_Findings_2026-10-15.md) report the fixed experiment on one reused source cell: 448 training trials from 56 participants and eight titles, with 84 held source trials from 14 other participants and six other titles. This cell was chosen using existing availability metadata. It covers six held categories; Fear and Inspiration remain training only. Its scores must not be pooled with the earlier 840-pair spectral study as though they were the same evaluation.

| Predictor on this 84-trial source panel | MAE, lower is better |
| --- | ---: |
| Source-category median | 1.09939 |
| Matched contextual Ridge, alpha 100 | 1.29535 |
| EEGNet plus context, fixed two-seed ensemble after 360 epochs | 1.40748 |

The EEG ensemble's gain is **−0.11214** relative to matched context and **−0.30809** relative to category medians. All three fixed checkpoints, at 120, 240 and 360 epochs, trail both controls. Both individual final seeds also trail both controls. No checkpoint was selected from these comparisons.

The model demonstrably learns: source MSE falls from 3.03138 for the offset alone to 0.86470 and 0.87827 across the two seeds, reductions of 71.5% and 71.0%. Tiny real/permuted controls reach MSE 0.10580/0.13969 from 6.11297. Fitting shuffled labels demonstrates small-sample capacity and functioning optimization; it does not establish emotion information. Both main source curves continue improving between 240 and 360 epochs. Full convergence is therefore unproven. Improving source fit while later held MAE worsens is compatible with overfitting or poor transfer; this diagnostic does not isolate the cause or demonstrate that EEG lacks useful information.

The separately frozen [independent audit](../../results/development/faced_baseline_independent_audit_2026-10-15/completion_v2.json) passed all 6,260 prediction rows, ten saved tensor states, five contextual Ridge training equations, OOF participant separation, priors, permutations, ensemble clipping, source MSE, comparison metrics and descriptive flags. Maximum normalized Ridge-equation residual was 1.385e-15. It checked 19,240 saved tensor scalars. The producer additionally replayed all ten checkpoints on fixed source subsets. The independent audit did not reacquire transient waveforms, reconstruct channel-scaler moments from them or replay every neural forward.

Four real neural trajectories completed in 771.922 charged seconds, with a total continuation wall time of 959.266 seconds and peak Torch allocator reservation of 160 MiB. The original device-index failure and first manufactured cuDNN failure remain preserved; they supply no EEG performance evidence. A separate representative batch-eight smoke passed before source activation. Success on that geometry does not prove the cause of the earlier failure or qualify arbitrary future batch shapes. All additional failed preparation/context-fit and synthetic-test costs remain visible in the findings.

## Guard and literature status

The [durable guard](FACED_Durable_Confirmation_Qualification_Findings_2026-10-15.md) passed full manufactured traversal, canonical export, fatal-failure preservation, fresh-instance restart rejection and two actual new-Python-process restart probes. It produced 3,036 fabricated target-free predictions across the exact 759-trial metadata roster, without opening reserved originals or targets. Actual protected transport and the eventual protected lock are separate qualifications. The required future entry point is the durable wrapper and its new contract.

The [HEART follow-up](FACED_HEART_Target_Qualification_2026-10-15.md) adds primary indexed method details but still leaves exact individual FACED supervision and material reservation unverified. Avoid priority and first-regression claims. The [independent scientific review](FACED_Preconfirmation_Claim_Independent_Review_2026-10-15.md) supports a narrowly specified empirical comparison while judging Ridge-only confirmation a modest contribution for stronger venues.

## Decision and next scientific work

**Do not adopt a superior-decoder, universal EEG-failure or architectural-novelty claim. Do not automatically spend the 33-person reservation.** The defensible current claim is an exploratory observation: these specified spectral and compact neural recipes fail to improve over contextual references under the inspected source evaluations. The capacity result makes that observation more interpretable, while reused data, one neural cell and ongoing source learning constrain it.

The [limited independent benchmark proposal](FACED_Preconfirmation_Claim_Proposal_2026-10-15.md) remains an option for an applied or replication paper if adverse and inconclusive results are worth publishing. Its purpose would be to estimate the four frozen recipes' incremental error on the fixed reserved catalogue, with the practical prior and coverage contrasts fully visible. It cannot settle optimized deep learning, arbitrary-film generalization or absence of EEG information. More precise estimates alone would not remove limited novelty.

For the user's stronger-venue objective, prepare a bounded comparator/replication proposal before further fitting. Priorities are credible contextual references and the alignment between an MAE endpoint and MSE-trained predictors, plus participant/material replication with an explicit missing-source policy. These are scientific comparability questions within the accepted target/context framing. They should not become an unrestricted architecture or epoch search. Broader public-data replication would require authenticated individual targets, joins, timing and material scope before fitting; unresolved author codebooks remain genuine dependencies.

No new question is adopted here. A change of main target, dataset or contribution still requires showing the findings and obtaining the user's approval. Before any reserved signal access, complete the exact claim/baseline decision and independent pre-access review under the new inactive contract. Before reserved ratings, independently verify and commit all four methods' complete protected prediction lock and qualify the item-10-only phase. All earlier failed screens and access gates remain preserved, and all saved manuscript/archive bindings rehash unchanged.
