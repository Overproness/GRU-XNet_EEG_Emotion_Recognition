# FACED backup qualification and evaluation constraints

10 October 2026. **The authorized metadata phase is complete; FACED is a conditional backup, not cleared for fitting.** All 123 technical inventories and four bounded BDF headers were checked without decoding a signal sample or individual rating. The manuscript and research question remain unchanged. This work can proceed while the [EmoEEG-MC author inquiry](https://github.com/ncclab-sustech/EmoEEG-MC/issues/1) awaits a response.

The [conditional evaluation protocol](Backup_Qualification_Protocol_2026-10-10.md) and [participant reservation](../../results/development/faced_backup_2026-10-10/participant_reservation.json) were pushed before header collection. The [audit index](../../results/development/faced_backup_2026-10-10/README.md) preserves declarations, failures, successful technical projections and reproducible verification. No material roles or new target have been selected.

## Source identity and access

**Author clarification, 10 October:** the author does not report existing certified Synapse access. This concerns FACED and is separate from the EmoEEG-MC GitHub inquiry, for which the author reports no reply yet. The [clarification record](../../results/development/faced_backup_2026-10-10/access_clarification.json) preserves the distinction without altering the frozen technical evidence. Synapse registration and any required certification/conditions must be completed under the author's own account. Official documentation describes certification as a 15-question data-governance quiz, typically 15–20 minutes; it is an account status, not an academic qualification. Certification does not by itself establish every file's access conditions. [Official account-type documentation](https://docs.synapse.org/synapse-docs/synapse-user-account-types).

The first-party release is [Synapse syn50614194](https://www.synapse.org/#!Synapse:syn50614194). Its original citation is Chen, Wang, Huang, Hu, Shen and Zhang, Scientific Data 10, 740 (25 October 2023), DOI 10.1038/s41597-023-02650-w. The curator README attributes that DOI to different authors and article 809; use the publisher's bibliographic record. [Original paper](https://www.nature.com/articles/s41597-023-02650-w).

The inspected public NEMAR derivative is [nm000112](https://github.com/nemarDatasets/nm000112/tree/4c37c73ece79e68702de2d8afa6b9274811a7cc8), version 1.1.3, Git revision `4c37c73ece79e68702de2d8afa6b9274811a7cc8`. The published supplementary document was retrieved directly from Springer and SHA-256 authenticated locally as `6056eaaf0679718866c5a1323d0caf895f2a3b5a994795b565adbc0d8bc52512`. Only its first three metadata tables were parsed: stimulus catalogue, original electrode positions and processed-data channel order. Later results tables and individual outcomes were not parsed. [Published supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41597-023-02650-w/MediaObjects/41597_2023_2650_MOESM1_ESM.docx).

Anonymous Synapse permission checks confirm that the original `Stimuli_info.xlsx` (`syn52370955`), `Task_event.xlsx` (`syn52370956`) and `DataStructureOfBehaviouralData.xlsx` (`syn52370951`) are publicly visible in inventory but cannot be downloaded anonymously; certification is required. Those file bodies were not requested. The author subsequently clarified that existing certified access is not available, as recorded above. The public supplement authenticates film metadata, but does not replace the trial/event and rating codebooks.

An initial HTTP-200 paper response was a challenge page and was rejected as semantic evidence. Separate publisher HTML and article XML provided readable source metadata. A successful HTTP status alone is not source authentication.

## Technical measurements

All 123 annex pointers, EEG sidecars and channel tables match the pinned Git blob identities and their retained SHA-256 values. The initial strict no-redirect collection stopped before header body reads for every participant at the public broker; all 123 failures remain preserved. A separately pushed declaration allowed one bounded public NEMAR redirect, with the temporary access URL held only in memory. Pilots `sub-000`, `sub-004` and `sub-036` passed. A subsequent metadata-discovered second layout prompted a separately declared `sub-061` pilot, which also passed. This is four headers, not four authenticated complete recordings.

| Check | Observed result | Implication |
|---|---|---|
| Sampling frequencies in 123 sidecars | 68 at 1,000 Hz; 55 at 250 Hz | Do not apply the curator prose's 92/31 split to these objects. All four sampled headers agree with their own sidecars. |
| Channel inventory | 32 entries each; two exact orders, 61 and 62 participants | The first includes A1/A2 and T3/T4/T5/T6; the second includes HEOR/HEOL and T7/T8/P7/P8. Channel position cannot be pooled blindly. |
| Channel types/units/status | All 3,936 entries marked EEG, microvolts, good | These are curator metadata claims. HEOR/HEOL need explicit role handling; `good` does not certify signal quality. |
| Header calibration | All four: 32 signals, 8,448 header bytes, `uV`, gain approximately 0.044703486 microvolts/count | Physical units are header calibration, not measured signal amplitudes. No numeric EEG was decoded. |
| Header geometry | All four agree with declared full-object size | The remaining 119 headers and every full-object digest remain unauthenticated. |
| Reference/filter sidecars | Reference, power-line frequency and software filters are `n/a` | Placeholders do not establish rereferencing or preprocessing. Header prefilters differ by rate. |
| Annex pointer total | 33,719,359,104 bytes | Curator availability says 33,721,509,453 declared and 37,811,755,149 present; preserve this accounting discrepancy without calling it corruption. |

The [channel supplement](../../results/development/faced_backup_2026-10-10/channel_supplement.json) records both complete channel groups. Its second-layout header agrees with its ordered channel table. The curator's unit repair code documents 33 originally mislabeled BDF unit strings and patches the label before reading; it does not justify a volts-to-microvolts rescaling. Original-to-curator payload equivalence remains unverified.

The pinned correction log records 94 header objects scrubbed on 7 October 2026, with history rewritten and versions updated in place. Payload identity is a curator verification claim, not our independent measurement. Version/DOI alone therefore cannot freeze bytes; retain pointer hashes and the Git pin. [Curator correction log](https://github.com/nemarDatasets/nm000112/blob/4c37c73ece79e68702de2d8afa6b9274811a7cc8/.nemar/corrections.jsonl).

## Material grouping changes what is feasible

The published catalogue's 28 clips come from **24 distinct exact source-film titles**. Fear clips 7 and 8 share *The Shining*; neutral clips 13–16 share *Blue*. Other title-normalized catalogue entries are distinct. This is a minimum content grouping; sequel/franchise, reused scenes and actual content hashes are not authenticated. The [public projection](../../results/development/faced_backup_2026-10-10/stimulus_metadata.json) retains clip identities, assigned catalogue labels, durations and source-title fingerprints, without any participant response.

| Assigned catalogue category | Distinct source films |
|---|---:|
| Anger, disgust, sadness, amusement, inspiration, joy, tenderness | 3 each |
| Fear | 2 |
| Neutral | 1 |

The supplementary table writes `/` in the targeted-emotion field for the four neutral clips; the immutable projection preserves that literal value, while this report identifies the group using its `Neutral` assigned-valence field. Assigned catalogue categories are not individual self-reports.

**A source/validation/confirmation design that holds source films disjoint and includes all nine assigned classes in each role is impossible on this catalogue.** Each class needs at least three independent film families; fear and neutral violate that necessary condition. Allocating the two Shining clips or four Blue clips to different roles would conceal shared films. This is a metadata feasibility result, not an emotion-decoding result or a new contribution.

The participant reservation remains 70 source, 20 development-validation and 33 confirmation, with no replacements. All material outcomes remain sealed and material assignment remains pending. A different estimand, a familiar-film arm or a reduced/continuous target could be considered only in an explicit proposal. This audit does not adopt any of those alternatives.

## Remaining gates and next decisions

Before opening development outcomes, authenticate the original event/rating codebooks, the BIDS-to-original clip/rating crosswalk and marker endpoint semantics. The conversion code's candidate mapping uses video identities and numbered score order, but has not been corroborated against the original workbook/readme bodies. Do not fit directly from the generic event schema.

Then select a scientifically justified and feasible target and material design from documented metadata, review exact pretrained membership/content overlap and declare the preprocessing adapter. FACED appears in training or evaluation of several existing representation studies; fresh local participant identities do not establish checkpoint independence. Verify the remaining headers/full-object hashes and joins through a newly declared technical phase before sample processing, retaining all failures and held-out exclusions.

The prepared protocol requires source-only selection, matched context/EEG/EEG-plus-context baselines, trial-level weights, participant/material uncertainty and one sealed confirmation after decisions. Its target, thresholds, effect threshold and primary contrast remain deliberately undecided until the data and novelty gates can support them. A research-question change still requires the author's explicit approval and an immediate pre-change manuscript archive.

Twenty focused tests cover byte-range enforcement, exclusion of outcomes/private header fields, reservation integrity, repeated-film grouping and literature date/error handling. Public and local technical replays pass. These checks certify the bounded metadata transformations, not classification performance or conference readiness. The main manuscript SHA-256 remains `06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0`.

An automatic approval review rejected the earlier proposed Synapse redirect check because it would store signed access links. It was not executed; safe entity metadata and the separately declared public header checks succeeded. The original codebook bodies still require certified access.
