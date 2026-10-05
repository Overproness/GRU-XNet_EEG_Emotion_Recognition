# Matched neural negative-transfer controls

This is an exploratory investigation, not an adopted paper question. The author must review findings and explicitly approve any research-question change. The original manuscript is archived separately.

The [predeclared plan](../../results/development/neural_negative_transfer_plan_2026-10-05.json) fixes five existing SEED-IV participant folds and seeds 42/43/44. It compares target-only, target plus DEAP, target plus GAMEEMO, all three with one output head, and all three with separate binary dataset heads. Every input is the same 56-dimensional classical trial feature used in the preceding controls. The model is a two-hidden-layer MLP; it is not the historical CNN–GRU model or a proposed novel architecture.

Every condition uses a frozen scaler fitted only to SEED-IV training participants. Batches of 60 assign exact dataset and binary-class quotas. Independent dataset/class sample streams preserve the same target draw prefixes. Paired conditions share identical initial backbone and target-head weights. Validation selects by balanced accuracy, then balanced log loss, every 25 updates.

The primary budget is 600 updates in every condition. An additional exposure control continues pooled runs to 600 times the number of datasets, giving every condition exactly 36,000 SEED-IV presentations. It uses more total compute; both budgets are reported. Checkpoint selection can choose earlier steps, so equal available exposure budgets are distinguished from each selected checkpoint's actual exposure. No early stopping or test-based calibration is used.

Linear source-ablation controls use the same anchored scaler and fix effective training weight to the number of SEED-IV training trials, controlling the relative L2 penalty when source data are added. This differs deliberately from the preceding global-scaler pooling control; cross-protocol score changes do not isolate a single factor.

All test cohorts have already been inspected during development. Results do not establish unseen-dataset transfer, the cause of negative transfer, architectural superiority, or scientific novelty. Separate heads retain the same coarse binary labels; they do not test preserving SEED-IV's full native four-class supervision.

Run from the repository in the user's PyTorch environment:

```powershell
python scripts/investigate_negative_transfer.py --pack ../publication_runs/joint_seediv_diagnostic --common-cache ../publication_runs/cache_common14 --plan ../publication_runs/neural_negative_transfer_plan_2026-10-05.json --output ../publication_runs/negative_transfer_neural_seediv
```

Input hashes, original trial/label identities and source-participant exclusions are checked before fitting. Existing experiment outputs are not overwritten. Model snapshots, selected checkpoints and detailed histories remain in the local run bundle; only bounded derived summaries will be exported to GitHub after validation.
