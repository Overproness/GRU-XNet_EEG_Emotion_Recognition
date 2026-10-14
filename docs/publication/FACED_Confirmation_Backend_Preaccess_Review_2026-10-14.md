# FACED source/backend preparation: independent pre-access review

14 October 2026, Asia/Karachi. **Preparation review; protected confirmation remains inactive.** This review interprets previously public technical metadata and code only. It does not read original headers or signals, private feature/state arrays, numerical cached targets, or protected ratings. The saved papers and the approved retrospective-valence/context question remain unchanged. A separate source-state auditor and the bounded source-only worker provide numerical qualifications; their later receipts must be distinguished from this metadata review.

## Exact future scope from already qualified public metadata

The new [metadata-only scope](../../results/development/faced_confirmation_backend_2026-10-14/scope.json) uses the old inactive contract solely for frozen participant/material identities. Its historical three-method/category-primary activation is not reused. The current inactive proposal requires four methods, the matched-context primary comparison, three regional inputs and common category fallback.

| Fixed scope | Count |
| --- | ---: |
| Protected participants | 33 |
| Unfamiliar primary clips / catalogue families | 9 / 8 |
| Unfamiliar clip outcomes / participant-family cells | 297 / 264 |
| Familiar secondary clips / outcomes | 14 / 462 |
| Complete signal/prediction trial union | 759 |
| Early/middle/late region preparations | 2,277 |
| Data/event integrity inventory entries | 66 |
| Four-method prediction exports required before ratings | 3,036 |

Clips 7 and 8 share a catalogue family. The scope preserves all other arm memberships and every outcome denominator. Ratings are item 10, retrospective whole-clip valence on the original 0–7 scale. Invalid targets must stop a later declared attempt; they cannot induce exclusions or imputation. Catalogue-title separation does not authenticate disjoint footage, prior exposure or playback alignment.

Seventeen reserved recordings are at 1,000 Hz and sixteen at 250 Hz. Every native input is exactly 32 seconds and thirty ordered scalp channels. Let `A=ceil(marker_start*r)`, `B=floor(marker_end*r)` and `N=32*r`; starts are `A`, `floor((A+B-N)/2)` and `B-N`, with exclusive stop `start+N`. Each input retains local seconds 2–30, seven four-second windows of shape 7×30×512 at 128 Hz. Ranges can overlap; both representations require the same three acquisitions.

| Exact metadata-derived budget | Bytes or values |
| --- | ---: |
| Selected count values, counting overlap each time decoded | 1,391,040,000 values |
| Selected 24-bit signal payload | 4,173,120,000 bytes |
| Thirty-three data files, one opaque hash pass | 8,961,374,784 bytes |
| Thirty-three event files, one opaque hash pass | 2,448,132 bytes |
| Before-and-after hash of both file types | 17,927,645,832 bytes |
| Before data/events and after data only | 17,925,197,700 bytes |
| One technical data-header read per participant | 278,784 bytes |
| All processed regions if retained as float32 windows | 979,292,160 bytes |
| All 150-dimensional regional vectors as float64 | 2,732,400 bytes |
| Largest single-region int32 count array | 3,840,000 bytes |
| Largest single-region calibrated float64 array | 7,680,000 bytes |

Selected-payload bytes and whole-file integrity transport are separate counts. A sequential region pipeline can discard waveform windows after producing QC, hashes and spectral vectors; storing the approximately 979 MB window union is unnecessary for these frozen Ridge recipes. CPU execution suffices. Array sizes do not cap total process memory: filtering, reference-decoder temporaries, interpolation caches and imported runtimes also consume memory. A future activation must specify a process-memory ceiling, bounded wall/CPU time, explicit hash passes and cooperative checks at every read/process/export stage. The current 27-region source pilot supplies throughput evidence, not a guarantee for every protected recording. No unlimited retries or resource-driven trial exclusions are allowed.

## Quality, feature and state contract

Keep signed-24-bit selective decoding, header calibration, the documented `?V` unit repair, channel aliases, fifty-hertz notch, fixed 0.5–45 Hz bandpass, resampling, trial-local filtering, fixed interpolation and common-average reference unchanged. No ICA, learned cross-person transformation or new artifact threshold is introduced. Interpolation applies to at most three channels detected by pre-CAR core SD above 100 μV or peak-to-peak above 1,000 μV. Availability also applies the unchanged ADC-rail, stuck-count, low/high variation, high excursion and residual-line-power engineering rules. These are engineering decisions rather than authenticated artifact labels.

Late features are 150 ordered five-band/channel log-power values. Distributed features are the mean of the three already-logged 150-vectors; averaging log vectors differs from taking a log after averaging raw powers. Both append the same thirteen contextual columns: eight category indicators, three acquisition indicators, catalogue duration and presentation ordinal. Contextual column order and scalers must agree exactly between the context and both joint states. All three fitted states use common alpha 100, an intercept, source-only scaling and clipping to [0,7]. Independent replay must verify exact source identity, coefficients, intercepts, scaling, two donor means, donor masks/identities, priors and state bytes. It must not refit states or optimize a more favorable result.

Unavailable inference trials invoke no regressor and no feature callback. Category medians always use the frozen source category prior. Matched context and both joint branches use that same prior whenever any required region fails an allowed engineered rule; all outcomes remain and paired EEG/coverage differences are zero on those trials. Source-training imputation remains a separate operation: the two existing means came only from 979 available source donors while all 980 targets/context rows were retained. Technical decoding failures, nonfinite values or unknown QC reasons are fatal, not an additional missing-input category.

The earlier `predict_with_quality` helper has different routing: its context branch continues using the context model and an unavailable joint uses contextual prediction. It cannot implement this new common-category policy unchanged. Similarly, `WindowBridge` has the earlier strict clean-source/late-only policy and is not the new three-region inference/export contract. They stay preserved as historical code rather than silently adopting new semantics.

## Required source/manufactured qualification

The bounded raw replay should use the preserved eight-category/acquisition pilot plus the already documented source failure `sub-087`, clip 20: nine trials, twenty-seven regions, eight distinct recordings. This decodes exactly 19,440,000 selected count values, or 58,320,000 payload bytes. The existing eight-trial pilot alone has no missing-input case. The extra trial intentionally verifies the middle Fp2 excursion and masking of otherwise usable early/late features, without relaxing the old failed quality gate.

Check native ranges, signed scalar-decoder agreement, calibration, count/window/feature/range hashes, fixed artifact decisions and fresh QC against pinned public receipts. Replay these twenty-seven actual source regions through the same new four-method inference and serializer used for the full 980-row source qualification. Remaining available source vectors may use the independently authenticated source-only snapshot. This combined source replay produces 3,920 target-free method/trial predictions. It is not a protected prediction lock, independent performance test, or rereading of all source original waveforms.

Manufactured failures must include role/family swaps, duplicate/incomplete identities, stale state/feature order, empty/held donor contamination, nonfinite values, unavailable EEG with poisoned callbacks, changed QC flags, partial exports, modified bytes, duplicate JSON fields and concurrent/repeated publication. A complete global identity/role/schema/QC barrier must precede numeric feature callbacks. Qualifying source-mode arrays does not authorize confirmation-mode input.

## Future phased package and identity guards

1. Freeze source-state, feature/QC, raw-scope, runtime, code and serializer bindings. Independently replay fitted states and the source/manufactured extraction-to-export chain. These are preparation receipts; none changes an old false scientific or access gate.
2. Prepare a separate inactive signal/prediction declaration with exactly the 33 fixed people and 759 trials. Permit original `data.bdf` decoding only within the 2,277 frozen native/channel ranges. If event files are used, allow opaque integrity hashing only, with no annotation or behavioral decoder. Reject unknown, validation, excluded, duplicate, changed-family and changed-range identities before any original-file callback. Reject path escapes, unexpected reparse/symlink targets, stale file hashes and incomplete exports.
3. After a separately authorized signal/prediction stage, independently verify all four methods' prediction-only exports across both arms. Bind exact file bytes, numerical prediction-only projections, trained source-state bytes, roster/family/routing/QC/input digests and the complete 3,036-row denominator. Commit these actual locks before any reserved rating conversion.
4. Only a separately declared rating stage can read the qualified behavioral item-10 endpoints after checking those committed locks. It must decode no other rating item and fit no model. The fixed primary and all mandatory secondary summaries follow without changing alpha, context control, families, interval, target, thresholds or denominators. An identity/technical/resource/export/invalid-target failure stops and preserves the attempt.

The existing old confirmation helper requires three historical exports and an unmet superiority screen. A new four-method backend needs its own reviewed binding/activation contract and cannot reinterpret that old helper as passed. The current new pure inference implementation deliberately rejects confirmation/validation mode; an actual protected original-file executor remains a future separately reviewed dependency.

## Scientific interpretation and preservation

No incremental source EEG lead exists: distributed EEG remains worse than matched context and category medians, despite a small gain over late EEG. The independent benchmark could estimate favorable, adverse or inconclusive performance for these specific frozen recipes and fixed titles. It cannot establish absent EEG information, optimized deep-learning performance, arbitrary-film generalization, a peak-end mechanism or new architectural novelty.

The 33-person primary paired-t interval conditions on this catalogue and fitted states. The six-category source SD proxy is not certified precision for all eight future families; Fear and Inspiration have no material-held source transfer panel. Source reuse, shared fitted states, convenience sampling, unauthenticated footage independence and incomplete annotation ancestry remain limitations. No old 0.15 engineering screen becomes a clinical/equivalence margin or a new successful continuation gate.

Metadata rehashing confirms 31 saved-paper bindings, 51 old activation bindings plus 42 completed-evidence bindings, thirteen earlier proposal bindings, twenty-one current proposal bindings, fifty-three completed source-activation bindings and twenty-two claim-decision bindings. All ten current future activation gates remain false. The machine-readable [independent preparation receipt](../../results/development/faced_confirmation_backend_2026-10-14/independent_preparation_review.json) records checks and later source-only audit qualification separately. Source preparation cannot automatically activate protected signals/ratings or update the saved manuscripts.
