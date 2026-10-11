# DENS execution/export metadata inquiry — draft only

Prepared 12 October 2026. **Not sent.** Any future external posting needs separate authorization for this exact message and a verified author contact/repository.

The original papers identify Sudhakar Mishra (`rs163@iiita.ac.in`), Mohammad Asif (`pse2017001@iiita.ac.in`) and Uma Shanker Tiwary (`ust@iiita.ac.in`) as corresponding authors. These are publication-documented addresses, not verified deliverable/current mailboxes. Sources and the bounded public codebook search are recorded in [the contact/search audit](DENS_Codebook_Search_2026-10-12.md). Proposed routing is To: Sudhakar Mishra, Cc: Mohammad Asif and Uma Shanker Tiwary. The [editable email draft](DENS_Metadata_Inquiry_2026-10-12.eml) has no sender identity and has not been dispatched. This session has no connected email-sending tool.

Subject: DENS ds003751 1.0.6 stimulus crosswalk and annotation timing

Hello,

We are qualifying the original DENS release for a reproducible EEG study, using OpenNeuro ds003751 snapshot 1.0.6 (commit 5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c). We have kept individual ratings and EEG samples unopened while checking the metadata.

Could you point us to the experiment/execution and export codebook covering:

1. The behaviour stimuliName keys (18 numbered filenames plus neutral_1.mp4 and neutral_2.mp4), their mapping to the sixteen emotional candidates in the final IEEE Access paper, original AFDI film IDs/titles/versions, and practice/main phases?
2. The event marker definitions and reliable stimulus onset/offset keys, including discontinuities and how each interval joins one behaviour row?
3. Timestamp units and export rules? In sub-mit003/081/108, TSV onset exactly equals EEG.event.latency rounded to six decimals, rather than seconds. TSV duration also equals the raw SET field (mostly 1000). What units and conversion should be used for duration and click times?
4. Behaviour field definitions: whether valence is stored on the documented 1–9 scale, allowed missing values, the roles of Quadrant/FivePointScale, and the relationship of MouseClick to recalled emotion categories?
5. Original FDT physical units/gain and the E129/Cz reference, plus which preprocessing steps precede the released files?
6. Which reuse statement governs the release? The actual LICENSE is ODbL, README mentions ODbL/DBCL, and dataset_description.json says CC0.

We found 34 potential recording/behaviour joins and 367 filename-bearing behaviour rows, but have not inferred usable trial counts or selected participants from ratings. We do not need anyone's individual scores; the execution/export definitions or scripts would be sufficient.

Thank you.

