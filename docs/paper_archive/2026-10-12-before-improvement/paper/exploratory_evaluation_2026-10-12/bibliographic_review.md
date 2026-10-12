# Bibliographic review and preprint-policy clarification

Checked 12 October 2026, Asia/Karachi. This note accompanies the approved exploratory manuscript. It clarifies the earlier frozen venue review without modifying that review, its bindings, the older papers or experimental packets. No original data, outcomes, model fitting or external correspondence was used.

## Archived preprints remain cited

JMBE's [generic author instructions](https://link.springer.com/journal/40846/submission-guidelines) distinguish published/accepted references from unpublished material. That wording does **not** by itself establish that a publicly archived preprint is prohibited. A relevant counterexample is JMBE's Original Article [Huang et al., published 24 October 2025, DOI 10.1007/s40846-025-00988-7](https://link.springer.com/article/10.1007/s40846-025-00988-7): its numbered reference list contains arXiv DOI entries for CMExam (reference 4, arXiv:2306.03030), TCMBench (reference 5, arXiv:2406.01126) and ChiMed-GPT (reference 6, arXiv:2311.06025).

This is evidence of published journal practice, rather than a universal editorial-policy guarantee. **Keep Gerster, Kong, Wang and the inspected LibEER version explicitly cited and identified as preprints.** Public availability does not establish peer review, but lack of peer review does not justify omitting the closest prior work. No editor inquiry is required to prepare the present draft, and none has been sent.

## Verified bootstrap bibliographic entries

| Entry | Primary-source confirmation | Assessment of the new bibliography |
| --- | --- | --- |
| Art B. Owen, *The pigeonhole bootstrap*, 2007 | The [author-deposited IMS electronic reprint](https://arxiv.org/pdf/0712.1111) states *The Annals of Applied Statistics*, **1(2), 386–411**, DOI **10.1214/07-AOAS122** on its first page. The [arXiv record](https://arxiv.org/abs/0712.1111) confirms those details. | `owen2007` matches author, title, journal, year, volume, issue, pages and DOI. |
| Art B. Owen and Dean Eckles, *Bootstrapping data arrays of arbitrary order*, 2012 | The [author-deposited IMS electronic reprint](https://arxiv.org/pdf/1106.2125) states *The Annals of Applied Statistics*, **6(3), 895–927**, DOI **10.1214/12-AOAS547** on its first page. The [arXiv record](https://arxiv.org/abs/1106.2125) confirms those details. | `owen2012` matches author, title, journal, year, volume, issue, pages and DOI. Use the journal year 2012, rather than the initial arXiv year 2011. |

The papers support accounting for crossed dependence under their stated conditions. Their metadata and theoretical results do not establish calibrated FACED small-material intervals or validate a future confirmation result.

## Check of the revised references.bib

The source-backed Gerster paper, Kong, Wang, Zhang/Etemad and LibEER titles, author lists and inspected version identifiers remain consistent with the preceding primary-source audit. The latest bibliography keeps Gerster as the 16 June 2026 bioRxiv v1 preprint, keeps Kong/Wang as arXiv v1 rather than asserting journal/conference publication, and identifies Zhang/Etemad as the accepted manuscript associated with DOI 10.1109/TETCI.2023.3332549. These distinctions should remain.

One attribution adjustment is recommended: `gerster2026code` currently lists only Moritz Gerster. The authenticated repository `CITATION.cff` identifies all seven associated authors and requests citation of the associated preprint. Cite that paper together with the exact August code revision; for the separate code bibliography entry, the seven-author attribution from the citation metadata avoids implying that the repository owner alone authored the associated work. Preserve revision `156d1efc335c7f399ac7e1647f0245d17afd2486`, its 10 August 2026 commit date and the 11 October 2026 initial inspection date. The [pinned citation metadata](https://github.com/moritz-gerster/faced-stimulus-confound/blob/156d1efc335c7f399ac7e1647f0245d17afd2486/CITATION.cff) is associated with the [existing authenticated source receipt](../../results/development/faced_eegnet_preflight_2026-10-11/closest_method_source_receipts.json).

Do not replace the exact-version source URLs with unqualified publisher metadata. The journal's eventual reference style can be applied while preserving these values. This bounded check does not newly authenticate every older EEGNet, FACED, Westfall or spherical-spline citation; their existing source-backed metadata is retained.

This review did not edit the manuscript, bibliography or frozen venue-review files. Its recommended attribution change is for the manuscript owner to apply before the final build and receipt.
