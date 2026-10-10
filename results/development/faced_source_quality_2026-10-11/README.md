# FACED eight-trial source-only quality pilot

11 October 2026. Eight declared source trials from four participants and two films were decoded, calibrated and processed; all 56 windows were retained. All ratings remain sealed. A separately declared fixed notch check addressed an observed 50-Hz problem using the same sample intervals. No model, supervised access or manuscript-question change occurred.

See [findings](../../../docs/publication/GRU-XNet_FACED_Source_Quality_Findings_2026-10-11.md), [original proposal](../../../docs/publication/GRU-XNet_FACED_Design_Proposal_2026-10-11.md), [trial table](trial_quality.csv) and [quality overview](quality_overview.png).

| Evidence | Purpose |
| --- | --- |
| [Execution declaration](plan.json), [synthetic independent decoder check](decoder_reference.json) | Pre-access identities, hashes, metrics and protected-data rules; manufactured PyEDFlib comparison |
| [Original records](records.json), [summary](summary.json), [bindings](verification.json) | 6.24 million unique original values, scalar signed-byte checks, calibration and 56 retained windows |
| [Original analysis verification](analysis_verification.json) | Recomputed quality table, figure bindings and 24 synthetic/boundary tests |
| [Fixed line-noise declaration](line_noise_plan.json), [execution clarification](line_noise_execution_clarification.json) | Source-informed fixed 50-Hz/Q30 control; original arrays lack historical hashes, so exact QC replay is the available historical check |
| [Line-noise records](line_noise_records.json), [bindings](line_noise_verification.json) | Same eight intervals decoded again; one fixed notch, no parameter search or new participant/material access |
| [Public packet verification](packet_verification.json) | Portable identity/hash/accounting and follow-up criterion replay; 27 focused tests passed across the two stages |

Original samples, behavioral values, private header strings and byte-range ledgers are not exported. Ledger fingerprints are public; local verification checks each declared range. The 50-Hz follow-up is observed-source-driven, not a preregistered confirmation finding. Original thresholds/proposals/results remain preserved.

For a public checkout, without original data:

```powershell
python scripts/verify_faced_source_quality_packet.py verify
```

For the original authorized workspace, with the immutable source receipts and private ledgers:

```powershell
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/run_faced_source_quality_pilot.py verify --local
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/analyze_faced_source_quality.py verify
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/check_faced_source_line_noise.py verify
```

Build/run commands preserve earlier outputs and refuse overwriting them. Supervised development still requires a separate target/schema/access declaration. This small pilot does not authenticate artifact freedom, individual target information or population-wide quality.
