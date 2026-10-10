# EmoEEG-MC join and endpoint source investigation

**Completed metadata investigation; fitting remains gated.**
[Findings](../../../docs/publication/GRU-XNet_EmoEEG_MC_Join_Resolution_2026-10-10.md)
describe partial resolution and the source limitations. No numerical individual
ratings, EEG sample values, physiology features or embeddings were interpreted.

- [Historical retrieval plan](plan.json), pushed in `db1a6b93e` before the bulk
  retrieval, and [exact byte-budget addendum](retrieval_addendum.json), pushed in
  `246da9f8d`, preserve the initial 20,000-byte ceiling failure and its correction.
- [Historical authentication](historical_authentication.json) checks all sixty
  event-table Git blobs. All 103 EDF pointers and 60 behaviour-table blobs match
  the current dataset release. Marker durations are zero; raw run IDs and
  material/rating-row identities are absent.
- [Identity-map plan](identity_plan.json), pushed in `98fc6a71e`, and
  [identity authentication](identity_authentication.json) record 120 author-hosted
  files, with 118 passing the initial strict schema. The
  [single-vector supplement](identity_schema_supplements.json) admits the two
  integer-code-only files without inventing explicit trial ordinals.
- [Codebook candidates](codebook_candidates.json) and
  [complete-pilot join candidates](complete_pilot_join_candidates.json) record
  metadata consistency. Every candidate retains a closed fitting gate.
- [Crosswalk checks](archive_crosswalk_checks.json) retain conflicts, duplicate
  matches and provisional aliases without adopting participant reassignments.
- [Join/endpoint diagnostics](join_resolution.json) preserve all participant
  checks and the non-overlapping thirty-second video-window example.
- [Source discovery](source_discovery.json) records primary sources, archive
  searches and four byte-identical cross-archive behaviour identity checks.
- [Verification](verification.json) binds this phase's code/reports/evidence and
  checks the unchanged prior qualification and participant/material reservation.

Original source tables, MAT files, metadata responses, narratives, third-party
code and EEG remain in the private workspace. Stable source URLs, digests,
assigned-material identity candidates and anonymous technical timing are public.
Identities from reserved people were inspected as metadata; their outcomes remain
sealed. No actual research-question change is approved or adopted.

```powershell
conda activate pytorch
python -m unittest discover -s tests -p test_emo_mc_join_resolution.py
python scripts/verify_emo_mc_join_resolution.py verify
# Optional private-cache byte validation, still without reading outcome values:
python scripts/verify_emo_mc_join_resolution.py verify --local
```

The verifier checks integrity and source/identity/timing invariants. It does not
certify the unresolved execution log, actual playback/button endpoints, rating
column semantics, EEG quality, trained performance or conference novelty. The
[prepared metadata inquiry](../../../docs/publication/EmoEEG_MC_Metadata_Inquiry_Draft_2026-10-10.md)
has not been sent to the authors.
