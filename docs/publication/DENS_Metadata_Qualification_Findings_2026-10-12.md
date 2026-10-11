# DENS metadata, annotation and timing qualification

12 October 2026, Asia/Karachi. **DENS remains on hold for supervised experiments.** This qualification establishes a release-bound inventory, non-outcome material keys and a concrete timing-export defect in three probes. It does not establish usable EEG/rating trials or a new research contribution.

The checked release is [OpenNeuro ds003751, snapshot 1.0.6](https://openneuro.org/datasets/ds003751/versions/1.0.6), pinned to commit `5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c` and tree `1552b175dffd076bb75c50c994cd154f021bfe0a`. The [evidence packet](../../results/development/dens_metadata_qualification_2026-10-12/README.md) records retrieval boundaries, hashes and replay commands.

## What is now established

| Check | Finding | Scope |
| --- | --- | --- |
| Original release inventory | 40 SET/FDT recording pairs, 40 event TSVs and 34 behaviour TSVs | All 318 pinned tree entries; filename-level 34 possible three-way participant joins |
| Behaviour schema | Eleven columns, including stimuliName, valence, arousal, dominance, liking, familiarity, relevance, emotionCateg, Quadrant, FivePointScale and MouseClick | All 34 headers; outcome columns were not decoded |
| Material key coverage | 367 filename-bearing rows, 9–11 per person, 20 distinct filenames; no within-person repeated filename | Complete authenticated behaviour bodies, only first-column values decoded |
| Acquisition declarations | All 40 sidecars report 250 Hz, 128 EEG + 1 ECG + 2 EMG, Cz reference, 50 Hz mains | Sidecars; no physical waveform calibration verified |
| Recording headers | Three continuous SET records, each 132 stored channels: E1–E129, ECG, EMG, EMG_2; reference text E129 | Bounded probes 003, 081, 108; external FDT data not read |
| Event timing | All 379 probe TSV onsets equal SET latency formatted to six decimal places; duration values are exported unchanged | Exact ordered float64 digest checks; this is raw latency export, not BIDS seconds |
| Author material catalogue | 69 film metadata entries, with matches for all 16 emotional IDs in the published methods | OSF metadata, not film bodies or played-footage authentication |
| Release license artifact | Pinned LICENSE body is ODbL, agreeing with README's database-license wording | Dataset-description CC0 and README individual-content DBCL statements still need clarification |

The six participants lacking behaviour files remain `sub-mit074`, `079`, `080`, `104`, `106`, `107` (all with the `sub-mit` prefix). **367 is a material-row count, not a usable-trial count.** Numeric target completeness, exact EEG joins, phases and endpoints remain unverified.

## The published protocol is clearer than the earlier preprint

The final IEEE Access publication describes **eleven main trials: nine emotional and two neutral clips**, each nominally sixty seconds, selected from sixteen emotional candidates plus two neutral materials. The earlier dataset manuscript describes three separate practice trials and eleven test trials. Thus nine versus eleven is explained by omission of neutral clips; the final paper also resolves the preprint's fifteen/sixteen wording. These are intended-protocol facts, not a verified release execution map. [Final paper, DOI 10.1109/ACCESS.2023.3266804](https://doi.org/10.1109/ACCESS.2023.3266804), [authors' earlier protocol](https://doi.org/10.1101/2021.08.04.455041).

Whole-clip valence, arousal and dominance are documented as retrospective 1–9 reports; liking, familiarity and relevance use 1–5. In-viewing clicks receive recalled emotion categories afterward. The latter are not continuous numeric valence. The authors' seven-second click-centred EEG extraction is an analysis choice, not a stimulus interval or calibrated emotion-onset correction.

The [methods source review](DENS_Methods_Source_Review_2026-10-12.md) separates documented intent from stored-file qualification. No observed valence, other rating, MouseClick value or recalled category was decoded here.

## Event TSV onset cannot be interpreted as seconds in the checked probes

| Probe | Events | SET recording duration, s | Largest TSV onset | Largest conventional EEGLAB event time, s |
| --- | ---: | ---: | ---: | ---: |
| sub-mit003 | 130 | 2696.356 | 501994.000789 | 2007.972003 |
| sub-mit081 | 135 | 1959.772 | 485117.695988 | 1940.466784 |
| sub-mit108 | 114 | 1469.296 | 359851.998556 | 1439.403994 |

All TSV onsets exactly match the corresponding SET latency after six-decimal formatting. Their values exceed recording duration if treated as seconds. All duration values also match the raw SET field; most are 1000, with exceptional values in two probes. We do **not** assign duration units from these values.

EEGLAB documents internal continuous latency as a one-based sample index and the conventional conversion `(latency - 1) / srate`. The converted probe onsets lie inside their recording bounds. BIDS defines TSV onset/duration in seconds, which these observed onsets do not satisfy. This is sufficient to reject directly using the TSV onset as seconds. It does not authenticate physical playback timing, identify main/practice phases, define stimulus offsets or verify duration-unit compliance. [EEGLAB event structures](https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html), [EEGLAB event fields](https://eeglab.org/tutorials/04_Import/Importing_Event_Epoch_Info.html), [BIDS 1.4.1 event specification](https://bids-specification.readthedocs.io/en/v1.4.1/04-modality-specific-files/05-task-events.html).

Only latency/duration were decoded in the separate timing pass. Type, label, description and nested event keys remain opaque because they may contain responses. The onset finding covers the **three declared probes**, not all forty recordings. Marker differences and recording discontinuities still need qualification before any extraction.

## Material identities remain unjoined

The author-hosted [AFDI Films catalogue](https://osf.io/tgcj8/) supplies 69 filenames, version metadata and reported SHA-256 values. Its 64-byte filename codebook describes `<Stimulus-id>-<Name>`. All sixteen emotional IDs in the paper have matching catalogue prefixes. Those inventory hashes authenticate metadata records here; no movie body was retrieved, so they do not authenticate the footage presented in DENS.

The release uses eighteen numbered keys plus `neutral_1.mp4` and `neutral_2.mp4`: twenty filename strings. The numeric keys run unevenly between 1 and 28 and do not directly follow the paper's catalogue IDs. Eighteen numbered keys versus sixteen published emotional candidates requires an execution/version crosswalk. It must not be resolved by guessing ordinal correspondence, assuming extra keys are practice clips, or declaring twenty independent film titles.

Neutral_1 occurs for 33 people and neutral_2 for all 34. This is a filename observation, not proof of their exact neutral-footage identity. Neither an authenticated source-title grouping nor FACED/EmoEEG-MC material-overlap check is available until the crosswalk is resolved.

## Remaining gates and decision

| Gate | Status | Required evidence |
| --- | --- | --- |
| Release inventory and header schemas | Passed within declared scope | Retain pinned tree/blob/annex bindings |
| Stored target coding | Hold | Author field/export definitions, scale and missing-value conventions without first selecting outcomes |
| Film/phase/trial crosswalk | Hold | Numbered key to original catalogue/title/version map; explicit practice/main and onset/offset meanings |
| Clock and duration semantics | Partial | Probe onset export diagnosed; qualify all required recordings, marker types, discontinuities and duration units |
| Channels and calibration | Partial | Sample rate/reference/channel labels checked; gain, physical units, montage coordinates and source-data preprocessing not authenticated |
| Participant/material reservations | Not activated | Reserve valid identity groups before target access, after the material map is established |
| Supervised fitting or paper adoption | Not activated | Qualified corpus and a concrete scientific proposal; existing approval condition remains |

The release contains no promised code/stimulus directories, behaviour/event JSON codebooks, channel TSVs or participant_details.xlsx. Bounded checks of the paper-linked OSF project and public repository metadata did not find an execution/export codebook. This does not prove such code exists nowhere.

The next dependency is an **author execution/export crosswalk**, especially film identities and actual markers. A [local metadata inquiry draft](DENS_Metadata_Request_2026-10-12.md) is prepared; it has not been sent. No external message is authorized by the present audit. Public documentation could also resolve this dependency if it supplies the same provenance.

Do not download waveform bodies or fit an EEG model to compensate for an unqualified join. Do not change the target to categories or automatically rescale DENS 1–9 with FACED 0–7. Retain all existing FACED and EmoEEG-MC reservations. DENS is a potential independent corpus, not a demonstrated research rescue or a larger-stimulus replacement.

## Access and preservation

Three SET headers, totalling 7,580,816 bytes, were bound by pinned Git-annex MD5E lengths/digests and recorded SHA-256 before selective parsing. MD5 is an integrity binding rather than strong provenance authentication. FDT geometry is compatible with 132 channels and four bytes per sample, but numerical dtype/units and waveform integrity were not established from body bytes.

Complete annotation bodies were transiently transferred for Git-blob authentication; outcome bytes were therefore transported, but outcome fields were not decoded, retained, printed or used. Initial HTTP-header reads may also buffer further bytes, so zero outcome transport is not claimed. The public packet includes only metadata, allowed material keys, numerical timing aggregates/digests and technical receipts. Three SET files remain in the ignored private working directory.

Four synthetic selective-reader checks passed, including rejection of embedded EEG.data and malformed files. Inventory, annotation and catalogue offline replay passed. Existing scientific bindings (50 activated, 94 preceding, 91 completed-diagnostic) and the original manuscript/archive hashes remain unchanged. Zero new fits, new predictions, individual rating values or numerical waveform samples were accessed; confirmation remains sealed. A separate current-progress paper draft uses existing completed FACED results only.

