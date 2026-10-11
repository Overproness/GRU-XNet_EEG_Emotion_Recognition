# FACED bounded original-data execution (11 October 2026)

The immutable [activation plan](activation_plan.json) binds the new runner, its synthetic tests, the earlier [bounded programme](../faced_development_protocol_2026-10-11/plan.json), original source qualification, participant/material roles and unchanged manuscript. It was created before broader original EEG or any original rating access.

First audit all 980 development-source participant/clip pairs (70 people × 14 development clips). Use exactly the qualified late 32-second, common-30, calibrated, native-50-Hz-notch pipeline and retain seven windows per trial. Complete the source audit despite review flags; any review flag holds validation EEG, ratings and fitting. Authentication/numerical failures stop and record incomplete coverage. No sample arrays are persisted.

Only if source quality passes, prepare 280 development-validation pairs, selectively decode 1,260 item-10 valence values and execute the existing 90-Ridge-fit/72-prior-estimation ceiling. All candidate, selected and source-OOF controls remain reported. All 33 confirmation people and all confirmation films remain sealed; no manuscript question is adopted.

Eight manufactured-data runner checks pass, including a complete three-rotation rehearsal, hard rating-access gate, exact fit budget, OOF separation, paired bootstrap and unavailable-validation-EEG fallbacks. Synthetic rehearsals use small artificial feature arrays, not original EEG or ratings.

```powershell
& 'D:\DL_Frameworks\envs\pytorch\python.exe' -m unittest discover -s tests -p test_faced_classical_runner.py
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/run_faced_classical.py execute
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/run_faced_classical.py verify
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/run_faced_classical.py verify --local
```

Execution never overwrites or resumes an earlier attempt. Local verification checks private feature/QC/access hashes without reopening raw recordings or ratings. Review flags are engineering checks, not validated artifact labels. An exploratory prediction lead would still need capacity/coverage review, a contribution proposal and author approval before changing the paper.
