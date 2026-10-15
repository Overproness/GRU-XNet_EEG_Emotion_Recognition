# Approved exploratory FACED paper

12 October 2026. **Current-progress manuscript for author review; not submitted.** The user explicitly approved this exploratory evaluation framing. Both preceding paper packages were copied and hash-verified [before the revision](../../docs/paper_archive/2026-10-12-before-approved-exploratory-revision/README.md). The original GRU-XNet source remains unchanged.

**Title:** Context Controls and EEG Pairing in Retrospective Valence Prediction: An Exploratory FACED Study

**Authors:** Muhammad Muntazar, then Muhammad Wasif Shakeel; both students at NUST, as confirmed by the author. The corresponding-author designation remains editable.

- [Main PDF](output/pdf/faced_exploratory_evaluation.pdf) and [LaTeX source](manuscript.tex).
- [Supplement PDF](output/pdf/faced_exploratory_supplement.pdf) and [LaTeX source](supplementary_methods.tex).
- [Scientific review](scientific_review.md), [independent matched-control numerical audit](independent_numeric_audit.json), [bibliographic review](bibliographic_review.md) and [build receipt](build_receipt.json).
- [Author checklist](author_checklist.md) and [JMBE cover-letter draft](cover_letter.md).

## What changed and why

The main paper focuses on a study-specific baseline information mismatch, category-supported source selection, incremental prediction and constrained EEG pairing. It uses individual retrospective 0–7 valence, new development-validation people and catalogue-title-held materials. It does not claim a novel network, the first stimulus-confound analysis, optimized EEG-null evidence or independent confirmation. The closest work remains cited explicitly, including archived preprints.

The source-selected category prior is a competitive reference. Spectral/context mean unfamiliar-title gain against it is 0.0446 MAE; the EEGNet/context gain is 0.0052. All six descriptive rotation-level intervals include zero. A newly declared **retrospective** alpha-100 matched-context sensitivity gives spectral gains −0.0006/+0.0332/+0.0995 (mean 0.0440); all three intervals include zero. This compares feature addition within a fixed prediction recipe, without identifying a causal cerebral mechanism. All six familiar/unfamiliar matched panels and 840 paired units are released.

Fifteen unavailable-EEG trials remain scored using the historical fallback. The matched sensitivity shares that fallback across both recipes, giving exactly zero gain on unavailable inputs. All fifteen historical conditions, the labelled retrospective context envelope, adverse familiar panels and all twenty-four pairing panels remain in the supplement. Pairing intervals all include zero; neural convergence and stimulus-footage authentication remain unresolved. Three rotations overlap and cannot be pooled as independent repetitions.

## Reproducibility and audit

Archive, author approval and the public-only evidence activation were committed and pushed at **fccb61c1b** before new matched arithmetic. The [producer](../../scripts/build_faced_exploratory_evidence_2026_10_12.py) used only already public cached development predictions/metadata, without new fits, original/private array access or protected-outcome access. The activation binds twenty-one files. Ten output bindings and the unchanged historical packets are rechecked by the build verifier.

The independent reviewer joined all 840 pairs from the original quality CSV, verified unchanged joint predictions and full denominators, and replayed 24,000 crossed draws with explicit Cartesian indexing. Maximum interval discrepancy was 1.11×10⁻¹⁶. All seven generated tables were checked at displayed precision (71 rows, 271 numeric values). These are identity/arithmetic audits, not independent model fitting or scientific confirmation. The producer's historical pending-audit marker remains intact; the later independent audit records its resolved disposition. A presentation-only figure derivative keeps the twelve plotted estimates/intervals unchanged while providing labels of at least 8.1 points at the intended width; its separate layout receipt binds the renderer and source evidence.

The [line-ending preservation record](line_ending_preservation.json) documents a Git packaging correction: the initial frozen commit stored normalized LF bytes for some Windows CRLF files. Their JSON/scientific contents were unchanged, and numerical execution used the reviewed original working bytes. Path-specific `-text` attributes and restaging preserve those exact reviewed bytes in the current repository, including the earlier sealed note. No historical source/evidence file was rewritten and no settings were reselected.

Build with Tectonic 0.17.0 using untrusted mode and a short writable `TECTONIC_CACHE_DIR`; main/supplement inputs compile separately. `scripts/verify_exploratory_paper_2026_10_12.py --prepare` checks bindings, numerical/format constraints, copies final PDFs and renders every page. `--seal` requires an explicit completed visual-review record. Build/temporary rendering directories are ignored; raw recordings, rating archives, media and checkpoint arrays are not included.

## Venue and remaining decisions

The package is prepared against the checked **Journal of Medical and Biological Engineering Original Article** requirements: a 150–250-word structured abstract, a conservative main-document word check below 3,000, numeric citations with full DOI links, and eight combined display items across the main paper and supplement. This is a review layout, not a claim to reproduce the publisher's final typesetting. [Checked instructions and prior work](../../docs/publication/FACED_Revised_Paper_Venue_and_Prior_Work_2026-10-12.md).

JMBE's [subscription publication route](https://link.springer.com/journal/40846/how-to-publish-with-us) has no APC and fits the author's free-first/PKR 10,000 ceiling. Optional paid open access is outside that ceiling. Suitability and acceptance remain editorial judgments; the single-corpus exploratory evidence and modest originality remain material limitations. A stronger general claim requires independently fixed evidence and/or suitably authenticated second-corpus materials, rather than another unbounded architecture search.

Before actual submission, both authors must review the work, designate the corresponding author, finalize accurate funding/conflict/contribution and institutional declarations, and approve the final venue/package. No author facts are inferred from a name, institution or repository owner. This manuscript approval does not activate the sealed confirmation packet, submit the paper or commit a fee.
