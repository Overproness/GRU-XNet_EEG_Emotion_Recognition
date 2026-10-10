# FACED original-codebook reconciliation

The user supplied three downloaded documentation workbooks. Source hashes and private snapshots preserve them; raw spreadsheets/cell text remain outside Git. The [plan](plan.json) was pushed after initial documentation inspection and before the automated [audit](audit.json). [Verification](verification.json) replays the local source extraction.

All 28 catalogue entries match the preceding published projection. Twelve rating items match the pinned curator order: arousal item 9, valence item 10, scale 0–7. `trial` is presentation ordinal and `vid` is catalogue identity. No classification threshold or target is selected. Neutral placeholder differences are retained, while the task codebook explicitly names the category.

New exceptions: five participants have unrelated-task start markers; participants 036–060 have an 83-second clip 22 version instead of the catalogue's 76 seconds. These are documentation findings, not authenticated real trigger sequences. Actual participant joins, endpoints and content alignment remain outstanding. Local hashes bind user-provided copies; anonymous source metadata returned entity identities but no release-payload checksums.

The [findings](../../../docs/publication/GRU-XNet_FACED_Codebook_Findings_2026-10-10.md) distinguish resolved and remaining gates. Nine focused tests pass; initial Windows fixture-permission errors are disclosed. No individual ratings, real identity rows, triggers, EEG samples, fitting, material assignment or question change occurred. Previous declarations, reserves and fitting sources are preserved.

Public check: `python scripts/audit_faced_codebooks.py verify`. Add `--local` to replay private documentation sources and the manuscript hash.
