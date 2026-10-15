# Exploratory FACED paper: 15 October evidence update

**Author-review revision; not submitted.** The approved research question is unchanged. The [12 October main paper, supplement and evidence](../../docs/paper_archive/2026-10-15-before-paper-update/README.md) were copied and hash-verified before this revision; their original locations also remain unchanged.

**Context Controls and EEG Pairing in Retrospective Valence Prediction: An Exploratory FACED Study**

Muhammad Muntazar, then Muhammad Wasif Shakeel; NUST, Islamabad, Pakistan.

- [Main PDF](output/pdf/faced_exploratory_evaluation.pdf) and [editable source](manuscript.tex).
- [Supplement PDF](output/pdf/faced_exploratory_supplement.pdf) and [editable source](supplementary_methods.tex).
- [Independent numerical review](independent_numeric_review.json), [scientific review](../../docs/publication/FACED_Manuscript_Upgrade_Review_2026-10-15.md), [build receipt](build_receipt.json) and [visual review](visual_review.json).
- [Author checklist](author_checklist.md) and [updated cover-letter draft](cover_letter.md).

## Changes supported by completed evidence

The abstract, methods, results, discussion and conclusion now incorporate the later adverse source comparisons. Three evaluations have separate denominators and adaptation histories: the earlier 120/160-trial validation rotations, the 840-pair spectral source comparison across ten cells, and the 84-pair EEGNet diagnostic in one selected source cell. They are not pooled as replications or a before/after comparison.

The longer EEGNet schedule learns substantially, and 32-trial real/shuffled controls demonstrate fitting capacity. Its fixed 360-epoch held MAE is **1.4075**, versus matched context **1.2953** and category medians **1.0994**. Source MSE still falls between epochs 240/360, so a plateau is not established. Additional training is not claimed to improve held prediction. The [evidence recheck](../../docs/publication/FACED_EEGNet_Evidence_Recheck_2026-10-15.md) reproduces the public exports.

The source coverage table also retains the adverse 840-pair result: distributed spectral/context MAE **1.4455**, matched context **1.3665**, category medians **1.3040**. Source availability is scoped correctly: all 980 late intervals pass; 979 pass the shared three-region rule, with the remaining trial retained for fitting through source-only imputation. No held source failure occurred. The practical acquisition-free comparator is distinguished from the EEG-availability-routed matched context.

HEART is now cited as relevant prior work, using the [authors' university record](https://recerca.udg.edu/ca/publications/heart-hybrid-embedding-for-affective-and-riemannian-trajectories/) and DOI [10.1109/TAFFC.2026.3729335](https://doi.org/10.1109/TAFFC.2026.3729335), checked 15 October. Its exact original FACED target, scale and material reservation remain unverified. No first-regression, strongest-decoder or new-architecture claim is made. The earlier inspected bibliography remains preserved; the update is not a systematic literature review.

All historical result tables and the pairing figure are byte-identical to the earlier package. Source-selection values move to supplementary prose and target decomposition moves to Supplementary Table S5, retaining eight combined display items. The main document has seven pages and 2,983 conservatively counted alphanumeric words including references/displays; the structured abstract has 209. The supplement has seven pages. This maintains the earlier JMBE review-layout limits; journal requirements and suitability must be rechecked before submission.

## Scientific status

The paper remains an exploratory single-corpus evaluation. Limited novelty, reused development outcomes, few materials, a single neural source cell, MSE/MAE mismatch, incomplete optimization/stronger-comparator coverage and unresolved stimulus provenance remain scientific limitations. The revision improves completeness and interpretation; it supplies no independently confirmed finding or guaranteed acceptance advantage.

**All 33 reserved participants remain sealed.** No fitting, new original/private data access, research-question change, confirmation activation, submission or external outreach occurs in this revision. Source/manufactured-input workflow checks are engineering qualification, not empirical replication. Future fitting or reserved-cohort use is not activated by the paper.

## Rebuild and checks

Run `python scripts/verify_faced_paper_revision_2026_10_15.py` from the repository to verify the sealed package. The verifier rechecks the predecessor's 31 bindings, 30 archive pairs, seven neural and ten spectral public artifact hashes, historical display identities and all nine rows of the new source table. The independent reviewer recomputed the later scores from 8,400 spectral and 6,260 neural public prediction rows. Neither check independently replays every raw waveform or establishes scientific replication.

For a fresh, unsealed revision, the same script's `--build` uses Tectonic 0.17.0 in untrusted mode, copies final PDFs and renders every page with PyMuPDF. A populated short cache defaults to `%TEMP%/gru-xnet-tex-cache`; override with `FACED_TECTONIC_CACHE_DIR`. `--seal` requires numerical snapshot agreement and a completed visual-review record. Existing sealed revisions cannot be overwritten. The build records the harmless Windows fontconfig notice and a minor underfull supplementary paragraph; all fourteen page renders were inspected for readable type, figures, tables, boundaries and glyphs.
