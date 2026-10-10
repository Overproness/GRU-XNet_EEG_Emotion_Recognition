Thank you for releasing EmoEEG-MC and the accompanying code. We are qualifying the corpus for an evaluation that reserves participants and materials before opening individual ratings or EEG sample values. Could you share the acquisition/export code, or a versioned metadata-only codebook, to resolve the following joins and trial boundaries?

We use OpenNeuro **ds005540 release 1.0.7** and ScienceDB **DOI 10.57760/sciencedb.14025, V6**. We also recovered the historical OpenNeuro event tables at commit `aea206971c265988c8ca16d14a69031367434d85`: all 103 EDF pointers and 60 behaviour-table blobs match the current snapshot. Our checks concern identities and technical annotations; we have not inspected numerical ratings or EEG amplitudes.

1. **Material identities, separately for video and imagery.** How do symbolic names such as `joy4`, `fear8` and `dis8` map to each context's stimulus-workbook numbers? Please clarify whether the two mappings differ. Identities, durations and content hashes would suffice; we are not requesting the original copyrighted audiovisual materials.

2. **Participant and code crosswalks.** What do the integer `vid` codes in ScienceDB `Remarks_new` mean, and how do the MAT filenames map to the OpenNeuro recording participants and the README presentation orders? Both context orders suggest a possible 53/54 swap in some metadata. Other same-number files disagree, and the root override files for 11/16 resemble shortened orders for other participants. We have not adopted any inferred participant reassignment. An explicit crosswalk, including duplicate/override-file meanings, would resolve this.

3. **Behaviour-row identities and incomplete trials.** What exactly do `trial_number` and `video_name` identify in the behaviour TSVs? Some `video_name` columns are constant. Which rows correspond to actually recorded trials, and is there an explicit missing-trial mask? For example, participant 60 has seven imagery and nine video starts under the current annotation mapping, but its behaviour table retains 42 rows.

4. **Numbered rating dimensions.** Please specify the dimension corresponding to every column `score_1` through `score_10`, including scale, endpoint meanings and missing-value conventions. We have kept numerical scores unopened rather than inferring this mapping from their values.

5. **Trigger meanings and split-run time.** Which markers identify actual video presentation completion, imagery guidance-audio completion, imagery start/end, the participant's end button and rating-screen appearance? What does `fade` mean? How should repeated state annotations and separate run clocks be reconciled without relying on potentially overlapping EDF header dates?

6. **A concrete video-end ambiguity.** For participant 37, run 02, `dis8`, the start-to-rating interval is 248.6017 seconds, while the current catalogue and author-named segment count indicate a 150-second video. The last 30 seconds ending at the rating-minus-one-second proxy and those ending at the nominal catalogue end do not overlap. Is there a known pause/delay or an actual end marker that resolves this case?

7. **Within-trial segmentation for short trials.** Could you document the valid interval when a trial is shorter than the standard analysis window? The public example's short-trial fallback can extend beyond the nominal rating boundary. We want to avoid including rating activity or filling missing exposure with later samples.

The [technical join/timing audit](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/blob/d2f671a07/docs/publication/GRU-XNet_EmoEEG_MC_Join_Resolution_2026-10-10.md) and [metadata-only checks](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/tree/d2f671a07/results/development/emo_mc_join_resolution_2026-10-10) preserve the supporting evidence and its limitations. We can provide a smaller technical example if helpful. No inferred relabeling or participant reassignment has been applied.

Thank you for any documentation or correction you can provide.
