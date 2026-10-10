# Data and target feasibility review

This is a metadata-only audit of already evaluated cohorts. No waveform is loaded, probability scored, model fitted, inference launched or different research question adopted. The complete [source/novelty decision](../../../docs/publication/GRU-XNet_Data_Target_Feasibility_2026-10-10.md) recommends source/annotation qualification before further fitting.

[Findings](metadata_findings.json) and [verification](verification.json) bind the audit script and seven existing public prediction/split files. Only label and identity fields are read; all probability columns are ignored. Reproduce from a public checkout:

```powershell
python scripts/review_data_targets.py verify
python -m unittest discover -s tests -p test_data_target_review.py
```

All 32 DEAP participants/40 videos, 15 SEED-IV participants/72 native clips, and 28 GAMEEMO participants/four games already have evaluated predictions. None of those declared participant/material identities can become fresh confirmation by repartitioning. This does not certify cross-corpus biological identity, recording authenticity or physical units.

An alternative DEAP source constructor keeps eligible rows from allowed people/materials without the older cross-exposure-arm matching dependency. All 80 previous grouping geometries have both classes in all roles, and 160 excluded-label/removal invariance checks pass. This is a preparation reference, not a retrospective change to any fitted study or newly sealed experiment.

GAMEEMO has just three positive G1 trials, five negative G2 trials, zero positive G3 trials and one negative G4 trial. Only six of twelve game allocations have cohort-level binary coverage; only six of 84 cells in a fixed identity-only seven-fold example have both classes in every role. No allocation works on all seven participant folds. The example grouping is not selected from coverage and is not a proposed fit; interactive game identity does not mean identical audiovisual content.

[Source checks](source_checks.json) and [source verification](source_verification.json) record 28 bounded metadata retrievals, including failed/empty routes and exact pinned documentation revisions. Full publisher text, original annotations and EEG remain outside Git. The local response-byte audit requires the private retrieval cache:

```powershell
python scripts/record_data_target_sources.py verify-local
```

The public qualification candidate is EmoEEG-MC; its pinned tree contains 103 raw EDF paths across 59 people, with 60 behavioral-file identities. This is a repository inventory, not proof those waveforms can all be acquired/decoded or used. No participant outcomes from the candidate were read. First-party unit/header, trigger/run and content-family checks remain a prerequisite to future fitting.

[Local annotation verification](local_label_verification.json) joins all 2,437 eligible metadata entries to 2,472 retained annotation entries, checks both DEAP spreadsheet copies and all 112 original SAM PDF hashes, and confirms no binary-label changes. This additionally needs the local annotation files and the existing optional xlrd reader:

```powershell
python scripts/audit_data_target_labels.py verify-local
```
