# FACED original recordings: extraction, signal lineage and trial joins

11 October 2026, Asia/Karachi. **The supplied original recordings resolve the actual numeric trial joins and signal-byte lineage to the pinned NEMAR conversion.** All 123 participants have 28 complete video spans, giving 3,444 original trials. Participant outcomes remain sealed. No fitting, new primary target, material allocation or paper-question change occurred.

The evidence is in [the raw qualification index](../../results/development/faced_raw_2026-10-11/README.md) and [machine-readable summary](../../results/development/faced_raw_2026-10-11/summary.json). This supersedes the earlier absence of original markers; historical documentation and declarations remain intact.

## Extraction and authorized archive removal

The user supplied `Data.zip` and its extracted `Data/` directory. The recording root is actually `Data/Data/sub000` through `sub122`. The archive contains 616 entries: 124 directories and 492 files, four per participant (`data.bdf`, `evt.bdf`, `After_remarks.mat`, `recordInformation.json`). All extracted member paths, sizes and streamed CRC32 values match; no extra files occur under the extraction root. Extracted file bytes total 33,725,030,272.

The archive SHA-256 is `ae9951358b6cf8ea68a8b70d137e1214c95aa452d5965a3c8b15f3f39a7f7769`. Every extracted file has an individual SHA-256 receipt. Hashing reads opaque EEG/behavioral bytes without decoding numerical values. CRC agreement establishes extraction consistency, not independent author-release authenticity.

After the extraction receipt was pushed, the archive hash was checked again, its exact workspace path and absence of a reparse point were verified, and the user-authorized redundant ZIP was removed. This frees 21,974,481,658 bytes, approximately 22.0 GB or 20.5 GiB. `Data/` remains intact. The [deletion receipt](../../results/development/faced_raw_2026-10-11/archive_deletion.json) binds the verification receipt and deleted archive hash.

## Original headers and signal lineage

All 246 original data/event BDF headers pass signature and file-geometry checks. The original signal headers directly corroborate the prior curator metadata: 68 recordings at 1,000 Hz, 55 at 250 Hz, and two 32-channel orders used by 61 and 62 participants respectively. Each data header occupies 8,448 bytes. Header calibration is 0.04470348624607839 physical units per count throughout.

Ninety original recordings label their physical unit `uV`; 33 use the documented corrupted literal `?V`: participants 004, 005, 008–035 and 058–060. The original numerical calibration is unchanged in the curator's `uV` headers. Do not let an unrecognized unit silently select a library's default scaling. Preserve originals and make any virtual unit repair explicit. The header states `HP: DC, LP: 250` for the 1,000-Hz recordings and `HP: DC, LP: 62` for the 250-Hz recordings. These are acquisition-header declarations, not independently measured filter responses or a decided analysis filter/reference policy.

The two orders retain the previously documented differences: the earlier group includes T3/T4/T5/T6 and A1/A2; the later group includes T7/T8/P7/P8 and HEOR/HEOL. Channel roles, aliases, common channel selection and reference handling still require an explicit analysis policy. A sidecar's `good` status does not establish signal quality.

**All 123 local signal payloads reproduce the pinned curator object digests.** Sixty-two originals, participants 061–122, directly match complete-file SHA-256 values in the previously authenticated Git-annex pointer inventory. For the remaining 61, a separately pushed declaration permits bounded public curator-header reads and opaque local-body hashing. Replacing only the original header with the fetched curator header reproduces each complete curator digest; the original full-file and body hashes are also rechecked against the extraction receipt. No curator signal payload is downloaded or numerical EEG sample decoded.

The differing bytes are confined to patient/recording-identification fields and, for 33 recordings, channel-header unit bytes. Date/time fields, fixed technical geometry and signal bytes agree. This is cryptographic equivalence to NEMAR revision `4c37c73ece79e68702de2d8afa6b9274811a7cc8`; it is not an independently supplied author-release checksum, a signal-quality assessment, or authentication of stimulus media. Originals are not edited. Private header identity values and absolute dates are neither interpreted nor exported.

## Actual events and behavioral identity joins

Every original `evt.bdf` has two channels: `Empty Event Data`, one three-byte slot per record, and `BDF Annotations`, 100 three-byte slots per record. The initial collector required annotation-only files and refused all 123 before reading channel payloads. Its failures are preserved. A separately declared continuation seeks past the dummy channel and reads only the annotation channel. A further declared projection preserves non-numeric annotation byte lengths, hashes and relative times without interpreting their text. All three stages remain in the evidence record.

The strict event reconstruction requires video ID → 101 start → 102 end, flags unpaired/replaced starts and duplicate identities, and uses actual marker endpoints. It yields all 28 catalogue video IDs exactly once per participant, with non-overlapping spans wholly inside each data file's relative recording geometry. Every span agrees with the pinned curator's numerical span-building state machine; third-party conversion code and rating loading are not executed.

The behavioral MAT parser reads only the documented `trial` and `vid` scalar fields. It skips `score`, `Accuracy` and `ResponseTime` at their outer matrix tags, without decoding their numeric arrays. All 123 files have the documented structure and bijective 1–28 identities. The 6,888 permitted identity scalars establish that all observed video sequences agree with their behavioral presentation ordinals. A total of 10,332 protected matrices are skipped. No rating distributions, values or arithmetic performance outcomes are opened.

The source documentation warns about unrelated-task 101 starts in participants 029, 045, 049, 059 and 060. Stray unidentified starts occur in four of these original numeric streams: 029, 045, 059 and 060; participant 049 has no such observed stray start. These do not create emotion trials. No participant is excluded or reassigned. Ten recordings also have two non-numeric annotations each: twenty in total, all outside the reconstructed video spans. Their texts remain opaque and their purpose is not inferred.

Actual span durations agree with the documented catalogue/clip-version durations to within the predeclared one-second discrepancy threshold: differences range from −0.479 to +0.928 seconds across 3,444 spans, with no flagged exceptions. This threshold is an audit check, not a selected cropping rule or measured video-onset latency.

Clip 22 in participants 036–060 has marker spans of **83.909–83.928 seconds**, compared with the source note's 83-second version. The other 98 participants have spans of **75.766–76.099 seconds**, compared with 76 seconds in the ordinary catalogue. The distinction is verified, but the actual frame identities, pre-roll/termination timing and seven-second content alignment are not. Do not trim based only on nominal duration or claim identical observed content across the versions.

## Verification and research decision

Twenty-one synthetic tests pass. They cover unsafe archive paths, malformed/duplicate identities, protected-matrix poisoning invariance, TAL parsing, stray/replaced starts, seeking past dummy samples, opaque-annotation invariance, header substitution and redirect restrictions. The public replay verifies seventeen bound code/evidence files. A local replay reproduces every technical header, marker projection and identity join for all 123 participants; it does not repeat the already completed whole-body hash passes. Earlier participant reservations, manuscript source and documentation bindings remain unchanged.

The 70 development-source, 20 development-validation and 33 confirmation participant roles remain frozen. All material outcomes and participant rating values stay sealed. The existing manuscript remains byte-identical, and the research question has not changed.

The next technical decision is a **feasible target/material-family design and explicit preprocessing protocol**, including unit handling, channel/reference policy and clip-version alignment, followed by a separately declared development-access phase. The all-nine-class three-way film-disjoint design remains infeasible: the catalogue has only two fear film families and one neutral family. Media/version overlap and encoder-pretraining membership also remain unresolved. Once a defensible exploratory design is specified, any proposed paper contribution must be shown to the author and approved before adopting a changed question.

The EmoEEG-MC author inquiry remains a separate dependency. These FACED checks do not answer that inquiry or independently authenticate DEAP signals, and they do not make the project conference-ready.

Format references used for this bounded implementation: [BioSemi BDF structure](https://www.biosemi.com/faq/file_format.htm), [EDF+ annotation lists](https://www.edfplus.info/specs/edfplus.html), and [MathWorks Level 5 MAT specification](https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf). Dataset lineage is evaluated against the [pinned NEMAR repository](https://github.com/nemarDatasets/nm000112/tree/4c37c73ece79e68702de2d8afa6b9274811a7cc8); original dataset attribution remains [Chen et al., Scientific Data 10, 740 (2023)](https://www.nature.com/articles/s41597-023-02650-w).
