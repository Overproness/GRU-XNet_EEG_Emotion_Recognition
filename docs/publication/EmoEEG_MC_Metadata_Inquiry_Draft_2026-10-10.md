# EmoEEG-MC metadata inquiry draft

Prepared 10 October 2026. **Draft only; not sent, posted or submitted.** This requests identity/timing documentation, not identifiable participant information or copyrighted audiovisual content.

We are qualifying EmoEEG-MC for an evaluation that reserves participants and materials before opening individual ratings or EEG values. We use OpenNeuro ds005540 release 1.0.7 and ScienceDB DOI 10.57760/sciencedb.14025 V6. We recovered historical OpenNeuro event tables from commit `aea206971c265988c8ca16d14a69031367434d85`; its 103 EDF pointers and 60 behaviour-table blobs match the current OpenNeuro snapshot.

Could you provide the acquisition/export code or a metadata-only specification for these points?

1. How do symbolic material names such as `joy4`, `fear8` and `dis8` map to the video and imagery workbook numbers? Are those two mappings different? A table of identities and clip lengths/hashes would suffice; we do not need original copyrighted clips.
2. What do the integer `vid` codes in `Remarks_new` mean, and what is the explicit participant crosswalk between those files and the OpenNeuro raw recordings/README orders? Both context orders suggest a 53/54 swap in some metadata. Several other same-number files disagree, and root override files for 11/16 resemble shortened orders for other participants. We have not adopted any inferred reassignment.
3. What do `trial_number` and `video_name` identify in the behaviour TSVs? Some `video_name` columns are constant. For incomplete recordings, which behaviour rows correspond to the actual recorded trials, and what is the missing-trial mask? For example, participant 60 has seven imagery and nine video starts, while its behaviour table retains forty-two rows.
4. Which exact dimensions correspond to `score_1` through `score_10`, with their scale, endpoint meanings and missing-value conventions? The listed item order suggests position eight is valence, but we have not read numerical scores to infer that mapping.
5. Which trigger marks actual video presentation completion, the end of imagery guidance audio, imagery start/end, the participant's end button and rating-screen appearance? Does `fade` identify the button or another transition? How should repeated state annotations and split-run clocks be reconciled without relying on potentially overlapping EDF header dates?
6. In participant 37 run 02, the `dis8` start-to-rating interval is 248.6017 seconds, whereas the current catalogue and author-named segment count indicate 150 seconds. The last thirty seconds before the rating-minus-one-second proxy and the nominal catalogue end have no overlap. Is there a known pause/delay or a distinct end marker that resolves this example?
7. Can the short-trial segmentation fallback be replaced by an explicitly documented within-trial interval? The public example can extend beyond the nominal rating boundary. We want to avoid including rating activity or filling missing exposure with later samples.

We can share anonymous technical timing/code summaries if helpful. Individual participant ratings, EEG amplitudes and prediction outcomes have not been inspected for this corpus. We would appreciate a versioned codebook or metadata correction that preserves original participant/material identities.
