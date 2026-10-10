# EmoEEG-MC qualification and protected reservation

10 October 2026. **Source qualification and an outcome-blind reservation are complete; the corpus is not yet cleared for fitting.** Author-hosted recordings are accessible and the checked complete objects match their release digests. Annotation and timing problems still prevent a certified EEG-to-material-to-rating join. No EEG sample values or individual ratings were interpreted, no models were fitted, and neither the manuscript nor its research question was changed.

The reservation was committed and pushed before the raw-file pilots in commit `341696fe8`. The complete-case follow-up declaration was pushed in `72953a7c2` before its downloads. These are development source checks, not evidence of emotion-recognition performance or a conference contribution.

## Evidence and source scope

The [qualification record](../../results/development/emo_mc_qualification_2026-10-10/qualification.json), [reservation](../../results/development/emo_mc_qualification_2026-10-10/reservation.json), [technical consistency checks](../../results/development/emo_mc_qualification_2026-10-10/technical_consistency.json) and [annotation checks](../../results/development/emo_mc_qualification_2026-10-10/annotation_join_checks.json) retain the derived evidence. Original EEG, rating tables, detailed narratives, downloaded author code and absolute recording dates remain outside Git.

The release is [OpenNeuro ds005540, version 1.0.7](https://doi.org/10.18112/openneuro.ds005540.v1.0.7), pinned to Git commit `83b5b7f28dea9964a5389509548219a27b755cc4`. Its release tag and annex pointers were checked against the public author-hosted repository. The [author code](https://github.com/ncclab-sustech/EmoEEG-MC/tree/d59aed5f9880a19df2a7db71f91caa034b8573c7) is pinned separately. Both stimulus-description spreadsheets match their complete SHA-256 annex digests. This establishes identity with the published release objects, not independent hardware calibration or correctness of every annotation.

All 103 EDF headers were retrieved with bounded HTTP ranges, pinned S3 object versions, matching object lengths and frozen Git pointer checks. The release has raw recordings for 59 participants (`sub-02`–`sub-60`), despite the [paper's 60-participant description](https://www.nature.com/articles/s41597-025-05349-2). The missing raw `sub-01` is not invented or replaced. Headers declare 64 EEG channels in a common order plus one EDF+ annotation channel. Ninety-six files declare 600 Hz and seven declare 1200 Hz.

Three complete EDF objects, totaling **1,055,080,026 bytes**, were authenticated against the pinned author-release SHA-256 digests:

- [Incomplete-case pilot](../../results/development/emo_mc_qualification_2026-10-10/pilot_authentication.json): `sub-60`, the smallest complete participant bundle in the already allocated development-source role.
- [Complete-case pilot](../../results/development/emo_mc_qualification_2026-10-10/complete_pilot_authentication.json): both `sub-37` runs, selected before retrieval by size among source participants with documented full orders and usable identity-code fields.

The full release's raw EDF inventory is about **50.63 GB**; the three checks do not authenticate every full recording. Original objects were stored privately and hashed as opaque bytes. Only their annotation channel text was interpreted; ordinary EEG samples were not decoded. Mixed behavioural-file bytes were transported transiently to verify the frozen Git blob, but only `trial_number` and `video_name` were decoded. Score fields were never interpreted or retained. These distinctions matter: the reserved files were not physically inaccessible, and this is not a claim that no dataset bytes were retrieved.

## Units, preprocessing and timing

Every header declares `uV`, physical endpoints ±100,000 µV and digital endpoints −32,768/32,767. The corresponding declared conversion is:

`physical_uV = digital_count × (200000 / 65535) + offset_uV`

The offset is about 1.526 µV and the declared gain is about 3.052 µV per count. Convert the resulting microvolts to volts exactly once when using an API expecting SI units. These are header declarations, not a measured hardware validation. The interpretation follows the [EDF specification](https://www.edfplus.info/specs/edf.html); annotation text follows the [EDF+ specification](https://www.edfplus.info/specs/edfplus.html).

All sidecars describe the reference only as `common`; they do not identify enough acquisition-reference detail for a confident cross-corpus reference claim. Header EEG prefilter fields are blank. The README/paper describe 0.1–47 Hz preprocessing, while the pinned author preprocessing example and sidecars use 0.5–47 Hz. Treat these as differing documentation/code descriptions; neither proves the analogue filter response or which operations were applied to every distributed object.

The pinned ICA example concatenates all trials within a context before fitting ICA and choosing components. A strict unseen-material study must not inherit that preprocessing as source-only processing. Its segmentation example also constructs replacement samples when a segment is short. A later protocol must specify raw conversion, verified trial endpoints, fixed filters and rejection rules before inspecting outcomes; no learned transformation should use reserved people/materials.

Concrete metadata discrepancies are preserved rather than silently repaired:

| Finding | Consequence |
|---|---|
| Both `sub-22` EDF lengths disagree with declared record geometry by 10,253,528 and 201,884,104 bytes | Initially quarantined; source identity matching alone does not make the recording structurally valid. |
| `sub-56` header declares 600 Hz; its sidecar says 1200 Hz | Never let the sidecar override a recording header without resolving the discrepancy. |
| 31 of 44 pairs have overlapping declared header-clock intervals | Calendar clocks and filenames cannot certify run stitching. This does **not** prove duplicated EEG. |
| `sub-37` header intervals overlap by 2,773 seconds | Preserve separate run-local annotation clocks until chronology is established. |
| No documented order for `sub-54` | Initially quarantined for the material join. |

## Annotation qualification

The `video_name` column is not uniformly usable. In 29 of the 59 behavioural tables it contains a constant value for every row; the other 30 have 42 distinct numeric codes. One table wraps codes in MATLAB-style brackets. Normalizing that representation is safe; interpreting a constant field as a unique material identifier is not.

The ten `score_*` columns have no semantic names in the TSV header. The README lists joy, inspiration, tenderness, sadness, fear, disgust, arousal, valence, familiarity and liking in that order; this suggests `score_8` is valence but still needs an explicit verified column-semantic join. The paper describes continuous 0–7 reports. DEAP's 1–9 scale and midpoint must not be silently reused. Individual report values, missing-score coverage and any future target threshold remain unexamined.

The authenticated `sub-60` annotations contain seven imagery and nine video starts, matching its shortened README sequences. Its behavioural table nevertheless has 42 rows. Joining the 16 EEG trials to the first 16 ratings would be invalid. The original positions of missing trials must be retained.

The authenticated `sub-37` recordings contain the expected 21 trials per context after a separately identified startup-state cluster is quarantined. Four different TypeID annotations occur simultaneously at the file origin; naïve parsing incorrectly counts one as an additional video trial. The analysis keeps both the unfiltered count and the technical exclusion. Its rule removes simultaneous multi-state origin annotations, preserves a single genuine time-zero trial marker, and never filters on class balance, ratings or desired performance.

An ordinal trigger/README alignment yields a **candidate**, not certified, mapping for all 42 numeric codes. After representation normalization, it reproduces both documented context orders in 28 of the 29 coded tables having a documented order. `sub-53` conflicts with that mapping. It remains in its reserved confirmation role with an additional join hold; no replacement participant is selected. `sub-54` cannot be checked because its documented order is absent. Constant-code tables need a different verified join.

Video-duration matching suggests a number-to-symbol correspondence, with 20 of 21 entries within 0.70 seconds after the author example's one-second guard. It also demonstrates why sorting symbolic IDs is unsafe: several joy, fear and neutral entries have a different order. This is an inference from technical timing, not content authentication. The imagery correspondence is still unverified.

One `sub-37` video interval is about **248.60 seconds** from start to rating, whereas the duration-matched catalogue entry is 150 seconds. The approximately 97.60-second discrepancy after the one-second guard is unresolved. Do not assume that the final 30 seconds before the rating are the final 30 seconds of that video. Fade markers also need an independently documented interpretation before treating them as imagery button presses.

## Stimulus overlap and reservation

The [stimulus comparison](../../results/development/emo_mc_qualification_2026-10-10/stimulus_overlap.json) narrows the earlier overlap concern. The EmoEEG-MC authors acknowledge materials obtained from the FACED collection. However, comparing its 21 video resource descriptions with all 28 film titles in the [original FACED supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41597-023-02650-w/MediaObjects/41597_2023_2650_MOESM1_ESM.docx) finds **zero normalized title matches**. Shared collection provenance does not establish that the released FACED benchmark clips are identical. Conversely, title matching alone does not prove content disjointness, rule out an expanded collection, or authenticate pretrained-model membership.

Within EmoEEG-MC, video entries 13 and 14 come from the same film, *Insidious*. They remain together. Generic platform names such as Bilibili are not treated as one content family; individual source identifiers are used. Cross-corpus clip/film-family matches with other possible source datasets remain to be audited before their EEG or pretrained checkpoints enter a protected-material experiment.

The committed identity-only allocation is:

| Role | Participants |
|---|---:|
| Future confirmation reserve | 20 |
| Development validation | 10 |
| Development source | 27 |
| Initial metadata quarantine (`sub-22`, `sub-54`) | 2 |
| Absent raw EEG (`sub-01`) | 1 |

Participant assignment uses a fixed SHA-256 ranking namespace, file integrity and documented-order availability. Material selection uses assigned elicitation categories and source-family metadata, with no individual ratings or model scores. The seven selected spreadsheet numbers are **3, 4, 8, 11, 15, 18 and 19**. They reserve seven video entries and seven imagery entries across all people. Numeric companions identify **different content in each context**; they are not the same audiovisual stimulus.

The remaining 28 context-specific entries are candidates for development, subject to a verified mapping. Outcomes from every confirmation participant, and from every reserved material for all participants, remain sealed. Unknown mappings fail closed: no waveform decoding or rating inspection is allowed. FACED EEG/embeddings and uncertified emotion-pretrained checkpoints are excluded from a protected-material source arm while overlap/membership is unresolved. These allocations are not rebalanced after looking at outcomes.

## Decision and remaining work

The source is accessible enough to continue technical qualification, but the full corpus is **not fitting-ready**. Ten focused tests check bounded retrieval, calibration geometry, score-byte exclusion, identity normalization, family closure, reservations, EDF+ parsing and run-boundary handling. The [verification record](../../results/development/emo_mc_qualification_2026-10-10/verification.json) binds the evidence and maintained source. Earlier experiments, their sixty frozen files and the original manuscript are preserved.

Next, independently establish number/symbol/imagery identities, raw-to-rating joins and true trial endpoints; resolve split-run chronology and isolate inconsistent recordings. Keep `sub-53` sealed with its join hold. Do not fill gaps, substitute ratings or reinterpret triggers merely to obtain expected counts. Only after those gates pass should an allowed development subset be used to assess target coverage and EEG baseline feasibility. A final model protocol and confirmatory analysis are still required before releasing reserved outcomes.

First-party DEAP waveform authentication remains a separate outstanding concern. The existing development panels still provide no qualifying encoder/readout lead. mdJPT and other close prior work still constrain novelty. A concrete contribution proposal and the author's explicit approval remain necessary before any research-question change; the then-current paper must be archived immediately before an approved pivot. This source audit supplies data-quality evidence, not that proposal or approval.
