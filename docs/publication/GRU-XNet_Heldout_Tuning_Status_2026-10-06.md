# Held-out tuning status

State: **complete**. Verified selected neural cases: **2040/2040**; context cells: **340/340**. Study completed: 2026-10-08T21:32:34.433264+00:00. Final publication verification recovered: 2026-10-09T03:41:15.825710+00:00.

The declared grid contains 4080 uninterrupted 1200-update trajectories and 24,480 source-validation candidates. GRU, EEGNet and EEGNet-plus-context independently choose their learning rate, duration and normalization in each outer fold. Two groupings, two initializations, all participant/video folds and both familiar/unseen exposure arms are retained. Context-only priors and calibrated controls use the same source partitions and two-panel selection criterion.

Source validation equally weights familiar and unseen videos from held-out validation participants. Outer test participants enter inference only after selection is sealed. This repeats development on existing cohorts; it does not add independent participants or establish convergence. Different raw/STFT representations prevent an architecture-only interpretation. The manuscript and research question are unchanged.

The [machine-readable protocol](../../results/development/heldout_tuning_2026-10-06/plan.json), [feasibility checks](../../results/development/heldout_tuning_2026-10-06/feasibility.json), [progress](../../results/development/heldout_tuning_2026-10-06/progress.json), and [portable verification](../../results/development/heldout_tuning_2026-10-06/public_verification.json) are exported with complete accepted candidate probabilities and selection/replay records. No raw EEG or model tensors are published.

All selected EEGNet states and fixed GRU sentinels are retained locally. Other selected GRU states are independently replayed before deletion, with their SHA and immediate verification certificate preserved. Such deleted weights require refitting for later raw replay. Outer aggregate findings are generated only when all cases pass verification; an in-progress export does not contain complete findings.

Worker: `D:\DL_Frameworks\envs\pytorch\python.exe scripts/heldout_tuning.py run --push-milestones`. Matching declarations and certificates permit resumption. Git checkpoints occur after every 40 newly accepted neural cases and completion. A failure stops the worker and records a local FAILURE.json; no successful split or seed is substituted.

[Complete findings](../../results/development/heldout_tuning_2026-10-06/FINDINGS.md).

The original final export ran out of memory during checksum verification after all fits and analysis completed. A separate recovery command verified the existing complete snapshot against local originals and recomputed all public candidate/selection metrics using 64-KiB checksum buffers. Frozen experiment source files and results remain unchanged. The original failure is preserved locally; [publication recovery](../../results/development/heldout_tuning_2026-10-06/publication_recovery.json) records its hash and the recovered verification.
