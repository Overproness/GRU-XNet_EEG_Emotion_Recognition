# FACED original-documentation reconciliation

10 October 2026. **The three downloaded codebooks resolve the documentation-access dependency and confirm the rating semantics. They also document recording exceptions that must enter the trial audit.** No individual score, EEG sample, real trigger sequence or participant identity row was opened. No fitting or manuscript/research-question change occurred.

The files were found at the workspace root and preserved in a private snapshot. [The declaration](../../results/development/faced_codebooks_2026-10-10/plan.json) was pushed before building the automated reconciliation, after initial documentation inspection; it does not claim that the codebooks had never been read. The [audit](../../results/development/faced_codebooks_2026-10-10/audit.json) and [local replay](../../results/development/faced_codebooks_2026-10-10/verification.json) preserve derived findings and source bindings. Original spreadsheets and full extracted cell text remain private.

## Provenance

| Local document | Public source entity, version 1 | Local SHA-256 |
|---|---|---|
| Stimuli_info.xlsx | [syn52370955](https://www.synapse.org/Synapse:syn52370955) | `4d332c77cb784b502350f29046fbd7d16dbd001f75f39f8ab0fec4c7f9fd1b0a` |
| Task_event.xlsx | [syn52370956](https://www.synapse.org/Synapse:syn52370956) | `a80a0aad7ed58b8980716ed9751284b05d87b683b3289a8877e0860a7df9a5cb` |
| DataStructureOfBehaviouralData.xlsx | [syn52370951](https://www.synapse.org/Synapse:syn52370951) | `e0e7f01cee3cd2353f5bda0ba45afa1acccd6bbe3c3dcd0a1be3fe3bf18b48e3` |

Each has one visible documentation sheet and matches the expected public entity name and file-handle identity. Download provenance is retained as reported by the user. An anonymous metadata-only Synapse check returned all three version-1 entities but no file-handle content checksums. Local hashes freeze these copies; they do not independently authenticate them against a remotely published payload checksum. No authenticated account action, download redirect or file download was performed by the agent.

Read-only ZIP/XML extraction was used in the author's requested PyTorch environment. The spreadsheet inspection runtime failed to initialize with a sandbox-state error. No workbook was authored or original edited; formulas, hidden sheets, external workbook links and participant-file names are rejected by the documentation parser. Literal values are extracted without formula evaluation.

## Confirmed semantics

**Rating order and scale.** `DataStructureOfBehaviouralData.xlsx`, `Sheet1!C3`, specifies joy, tenderness, inspiration, amusement, anger, disgust, fear, sadness, arousal, valence, familiarity, liking. All twelve match the pinned curator conversion's `RATING_KEYS`. Arousal is item 9 and valence item 10, using one-based numbering; the scale is 0–7. `Sheet1!B13` specifies negative-to-positive endpoints for valence and absence-to-intensity endpoints for the other items. No binary threshold, outcome normalization or primary target was chosen.

**Identity joins.** `trial` is presentation ordinal; `vid` is catalogue clip identity. Both range from 1 to 28. Random presentation order means array position cannot stand in for clip identity. `Accuracy` and `ResponseTime` concern the arithmetic task. The identity-only validator requires complete, unique integer trial/video identities and rejects score payloads before inspecting them. Its tests use synthetic fixtures and do not authenticate a real participant join.

**Catalogue agreement.** All 28 indices, durations, normalized source-film fingerprints and valence assignments agree with the published projection. All eight non-neutral categories agree. The original workbook uses a backslash placeholder for neutral emotion cells; the published projection contains `/`. The task-event table explicitly names those four clips Neutral. Both literal forms are retained separately; previous projections and hashes are unchanged. Catalogue assignments are distinct from individual self-reports.

The 24 exact source-film families and repeated-film groups remain unchanged. Fear has two source films and neutral one. These codebooks cannot make an all-nine-class source/validation/confirmation film-disjoint design possible. Material assignment remains pending and all material outcomes stay sealed.

## Recording exceptions

| Exception | Source cells | Required handling before samples/outcomes |
|---|---|---|
| Participants 029, 045, 049, 059 and 060 have another task in the raw recording, with 101 markers not followed by 102. | Task_event.xlsx, Sheet1!A8–A9 | A start marker alone cannot establish an emotion trial. Audit experiment context, video identity and paired start/end markers; retain ambiguous/unpaired sequences as flags. |
| Participants 036–060 watched an 83-second clip 22 version with extra seconds at the beginning; its catalogue duration is 76 seconds. | Stimuli_info.xlsx, Sheet1!A35 | Validate actual endpoints and content-version alignment. Catalogue duration cannot universally determine trial endpoints or matched content offsets. |

The longer-version exception covers 25 released identities; the unrelated-task exception covers five. These are documented references, not new exclusions or replacements. Affected confirmation participants retain their roles. No ratings or response distributions were inspected to identify the exceptions.

The event codebook defines 100 as experiment start, including successive start markers; 101 and 102 mean clip start and end. The curator helper names 100 `task_block_start`, while its annotation builder calls it experiment start. Do not infer block boundaries without execution evidence. Its span parser replaces the current start whenever 101 occurs; the documented unrelated task makes raw-marker verification necessary. This source inspection does not establish that any actual export is wrong.

## Remaining work

The three-document access and rating-order gates are resolved. **Actual recording/event/behaviour joins remain unaudited.** The next technical phase should check identity uniqueness, marker order, start/end pairing, participant-specific clip 22 versions and agreement between original markers and curator spans, without decoding individual scores. Outcome-bearing event tables must not be opened indiscriminately: use a separately declared timing/identity projection or original marker sources. Do not treat curator span inference as authenticated execution provenance.

Remaining headers, complete-object digests, original-to-curator signal lineage, montage/preprocessing, content/pretraining overlap and a feasible target/material design still need qualification. The codebooks describe available targets; they do not select a contribution or clear fitting. A new question still requires evidence and the author's explicit approval, followed by a fresh archive immediately before an actual manuscript change.

Nine focused tests pass, covering formula-cache rejection, external-link exclusion, hidden-sheet review, participant-file exclusion, presentation/video separation, duplicate identities, score-payload rejection, exact integer identities and arousal/valence swaps. Initial runs failed in Windows temporary-directory fixture setup; a verified fresh workspace test directory with appropriate execution permissions resolved that infrastructure error. The scientific audit was built only after the successful run. Public/local replays and protected prior bindings pass.

The 70/20/33 FACED participant roles, sealed materials, EmoEEG-MC reservations, sixty earlier frozen fitting-source files and manuscript remain unchanged. The EmoEEG-MC author inquiry is separate and is not answered by these documents. The [weekly literature workflow](../../results/development/literature_watch_2026-10-10/README.md) remains active.
