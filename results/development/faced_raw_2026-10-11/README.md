# Original FACED technical qualification — 11 October 2026

The user supplied the original archive and extracted recordings. All 492 extracted files verify; the redundant ZIP was then removed under explicit authorization. All 123 participants have 28 numeric video spans joined to their behavioral identity fields. All original signal bodies match pinned curator object digests directly (62 files) or after substituting only curator headers (61 files). No EEG numbers or individual rating values were decoded.

See [complete findings](../../../docs/publication/GRU-XNet_FACED_Raw_Qualification_2026-10-11.md), [summary](summary.json) and [verification bindings](verification.json).

| Declaration / collection | Scope |
| --- | --- |
| [plan](plan.json), [extraction](extraction.json), [deletion receipt](archive_deletion.json) | Frozen source/reservation/manuscript bindings; safe paths, sizes, CRC32/SHA256 and exact authorized archive removal |
| [headers](headers.json), [initial joins](joins.json) | All 246 technical headers; initial annotation-only guard failures retained |
| [channel continuation plan](annotation_channel_plan.json), [continuation](joins_annotations.json) | Seek past dummy event channel; 113 joins parse, ten fail at non-numeric text guard |
| [numeric projection plan](numeric_projection_plan.json), [complete joins](joins_numeric_projection.json) | All 123 joins; non-numeric text bytes remain opaque; protected MAT arrays skipped |
| [lineage plan](lineage_plan.json), [lineage](lineage.json) | Public bounded curator headers only; original local bodies hashed opaquely against full pinned digests |

Run from repository root using the authorized PyTorch Python:

```powershell
& 'D:\DL_Frameworks\envs\pytorch\python.exe' -m pytest -q tests/test_faced_raw.py tests/test_faced_annotation_projection.py tests/test_faced_raw_lineage.py -p no:cacheprovider
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/analyze_faced_raw.py verify
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/analyze_faced_raw.py verify --local
```

The last command requires original `Data/Data` locally. It replays technical headers and tiny original annotation/identity files, with no signal-body hash repetition. Evidence-build/collection commands refuse overwriting their outputs; the initial ZIP audit cannot be rerun after the authorized archive deletion. Full raw data, original MAT fields, recordInformation JSON, private header identities/dates and access URLs are not in this repository.

All existing participant roles and material/outcome seals remain preserved. Target/material design, preprocessing, content alignment, overlap/pretraining qualification and separate development access remain required. `fitting_ready` and `research_question_changed` remain false.
