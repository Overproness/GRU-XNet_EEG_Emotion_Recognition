# EmoEEG-MC trial joins and endpoints: source-resolution findings

Completed 10 October 2026. **The metadata investigation is complete; the corpus is still not cleared for fitting or opening individual outcomes.** Several missing sources were recovered, and the complete pilot's identity join is substantially better supported. Other joins and the actual recording endpoints remain unresolved. The manuscript and research question are unchanged.

## What was recovered and checked

The current OpenNeuro 1.0.7 snapshot omits its event tables. A [historical first-party snapshot](https://github.com/OpenNeuroDatasets/ds005540/tree/aea206971c265988c8ca16d14a69031367434d85) still contains all sixty. Every historical EDF pointer (103/103) and behaviour-table Git blob (60/60, including the person without EEG) is identical to the current release. This establishes file lineage, rather than assuming that similarly named older data are compatible.

All **60 event tables / 10,693 markers** were retrieved with exact Git-blob checks. Their only fields are onset, duration, trigger code and trigger role. Every duration is zero, and the tables supply no run IDs, material names or rating-row IDs. The marker sequences for development participants 37 and 60 agree with their already authenticated EDF annotations. Maximum time differences are approximately 0.00003334 seconds, consistent with the EDF markers' four-decimal serialization. These are redundant marker exports, not independent experiment logs or evidence of actual playback completion.

The [authors' ScienceDB archive, DOI 10.57760/sciencedb.14025](https://www.scidb.cn/detail?dataSetId=9b182864c9604255a0433d1edae88f0b), is publicly accessible. Its V6 README includes the missing, distinct imagery/video orders for participant 54. The initial quarantine role remains unchanged because additional identity conflicts remain and the frozen reservation has not been amended.

ScienceDB's `Remarks_new` files supply **trial/stimulus identities**, rather than numerical emotion ratings. All 120 primary files were retrieved using their listed sizes and stable file IDs, with retained SHA-256 hashes. The initial strict schema accepts 118 files; participant 25's two files contain only an integer `vid` vector. A separate, explicit schema supplement accepts those code vectors without inventing an explicit trial-number vector. Four additional root override files were inspected under the same identity-only gate. No original rating table, feature array or waveform values were opened or published.

## What the identity checks establish

The candidate 42-code dictionary is now derived from the separate ScienceDB identity arrays and documented per-context order for participant 37. It no longer depends on first zipping that participant's EEG trials to behaviour rows. **All thirty behaviour tables with varying stimulus codes agree with their corresponding archive identity arrays in both contexts**, including participant 25's single-vector supplement.

For the complete participant-37 pilot, the resulting 42 candidate material-code/behaviour-row joins match the raw trial sequence exactly. A [reviewable candidate manifest](../../results/development/emo_mc_join_resolution_2026-10-10/complete_pilot_join_candidates.json) retains the run-local times and every exclusion flag. This verifies metadata consistency, without supplying an original acquisition-log certificate.

Important limitations remain:

- Nineteen of sixty participants' archive identity orders disagree with at least one documented context order. This includes the person without raw EEG and several incomplete cases; it is not a count of nineteen proven unusable recordings.
- Matching both context orders suggests a participant-53/54 swap between some metadata sources. Other matches are duplicated or absent. Four separate override files resemble different participants' shortened plans. **No explicit author crosswalk was found**, so none of these aliases is adopted, and no participant is reassigned.
- Forty-seven of sixty historical marker tables have candidate start counts matching the documented orders under the existing trigger map and origin-marker rule. Thirteen do not. Missing mapping, repeated state markers, incomplete recordings and run boundaries require separate handling; matching counts alone does not prove a usable join.
- Twenty-nine of the 59 raw-participant behaviour tables have constant `video_name` values. Those fields cannot identify each material. The incomplete participant-60 case still has sixteen recorded trials versus forty-two behaviour rows; taking the first sixteen rows would remain unjustified.
- The inspected ScienceDB behaviour tables for participants 5, 37, 54 and 60 are byte-identical to their current OpenNeuro copies. Switching archives does not repair those tables.

The original 20 confirmation participants, 10 development-validation participants, 27 development-source participants, two quarantines and fourteen reserved material keys remain fixed. Participant 53 retains its confirmation role and join hold. Metadata inspections include reserved identities, but their individual outcomes remain unopened.

## Material mapping and endpoint findings

The pinned [author stimulus code](https://github.com/ncclab-sustech/EmoEEG-MC/blob/d59aed5f9880a19df2a7db71f91caa034b8573c7/tools/annotations/xuxin/xuxin_dataset.json) contains contiguous numbered segments for all 21 video symbols. Segment counts agree with unique duration/category entries in the current video catalogue and independently support every previous video symbol-to-number hypothesis. The generic VE8 labels in that code are not used as emotion labels. Original video content and clip boundaries are not authenticated by these counts.

The older video catalogue was also recovered from a public, immutable S3 object version and matches its historical annex SHA-256. **Thirteen of its 21 duration entries differ from the current catalogue.** Numbers and assigned categories remain aligned. This explains why old duration metadata cannot silently substitute for release 1.0.7. The imagery symbol-to-number map is still unverified; video correspondences are not copied into imagery. Every unmapped imagery identity remains excluded under the frozen material gate.

The endpoint concern persists independently of model outcomes. For participant 37's `dis8` video in run 02:

| Metadata quantity | Value |
| --- | ---: |
| Start-to-rating interval | 248.6017 s |
| Current catalogue duration / named segment count | 150 s |
| Difference between catalogue-end and rating-minus-one-second candidates | 97.6017 s |
| Overlap between the two candidate 30-second tails | 0 s |

The other twenty video trials agree within one second with the guarded duration comparison. The outlier is not automatically removed, relabelled or assigned a playback-end time. Playback pauses, delayed rating transitions and metadata errors remain possible explanations. The historical events add no explicit video-end marker. For imagery, `fade` is still not documented as the participant's button press or actual imagination endpoint. The author segmentation example's short-trial fallback can reach beyond the nominal rating boundary and is not adopted.

The [paper's published procedure](https://www.nature.com/articles/s41597-025-05349-2) distinguishes the last thirty seconds of video presentation from imagery before its end/button. A rating-onset proxy is not thereby an authenticated presentation endpoint. The README lists the ten rating dimensions in order, suggesting valence at position eight, but an explicitly numbered `score_1`–`score_10` export codebook remains missing.

## What is needed next

The public archive searches found two stimulus workbooks and 178 physiology-feature CSVs, but no matching `.log`, `.psyexp`, `experiment` or `rating` files. Those CSV feature values were not retrieved. This does not prove that no acquisition logs exist elsewhere.

The next dependency is a first-party execution/export specification covering **material-number/symbol correspondence, participant crosswalks, incomplete-trial row masks, numbered score-column semantics and actual video/button endpoints**. A [precise author-inquiry draft](EmoEEG_MC_Metadata_Inquiry_Draft_2026-10-10.md) is prepared; it has not been sent. Metadata recovery can continue if another public primary source appears. Further encoder fitting is not an appropriate substitute for those missing joins.

Once that specification resolves the gates, build a versioned per-run trial manifest with explicit material identity, behaviour row and endpoint provenance. Verify the protected mask before inspecting allowed development ratings or EEG values. Declare exclusions and any alternative window policy from metadata before outcomes. Freeze a bounded development experiment only after those checks pass. An actual paper-question change still requires the author's explicit approval and a fresh manuscript archive.

## Validation and publication scope

The [public evidence index](../../results/development/emo_mc_join_resolution_2026-10-10/README.md) links the plans, source authentication, schema supplement, identity candidates, crosswalk checks and endpoint diagnostics. Eighteen new focused tests exercise score/schema exclusion, context-code validity, timestamp precision, clock resets, short-trial refusal and fail-closed reserved/unknown material masks. The ten earlier qualification/annotation tests remain applicable. Byte/invariant verification separately checks the prior completed sources and frozen reservation.

There are **zero new numerical rating decodes, EEG sample-value decodes, model fits or inferences** in this phase. Original EEG, rating tables, embeddings, physiology features, source narratives and third-party code remain private. GitHub receives our audit code, technical/source hashes, anonymous identity metadata and these findings. Recovery declarations and verified milestones were committed and pushed. Passing these checks does not establish scientific novelty, a conference-ready contribution or completion of the older DEAP/manuscript concerns.
