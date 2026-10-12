# FACED exploratory research-note review package

Prepared 12 October 2026. This is a separate venue-format preparation of completed FACED development results. It preserves the [seven-page empirical draft](../current_progress_2026-10-12/README.md), the original GRU-XNet manuscript and their archives. No new fitting, target access or confirmation experiment is performed.

The preparation target is a **BMC Research Notes research note**. The author confirmed on 12 October that paid journals can be included when the fit is strong. Final author approval, actual fee eligibility and the submission decision remain outstanding. Preparing this version neither submits it nor commits a publication fee. The [current venue qualification](../../docs/publication/FACED_Submission_Route_Qualification_2026-10-12.md) records official criteria, cost qualifications and alternatives.

## Review artifacts

- [Main research note](output/pdf/faced_research_note.pdf) and [editable LaTeX](manuscript.tex).
- [Supplementary methods and complete results](output/pdf/faced_supplementary_methods.pdf) and [editable LaTeX](supplementary_methods.tex).
- [Cover-letter draft](cover_letter.md).
- [Author and submission checklist](author_checklist.md).
- `build_receipt.json`: numerical, word-limit, build, preservation and public-input checks. The PDF review is technical verification, not independent scientific confirmation.

This version condenses the main argument while retaining every declared condition in the supplement. Its central comparison is whether EEG improves prediction of individual retrospective valence beyond inference-available elicitation metadata, on reused participant/title development panels. Small differences, overlapping panels, selection and continuing neural learning remain explicit. The paper does not assert a new architecture, a robust confirmed EEG advantage or an information null.

## Preparation and approval boundary

The main body is a review draft. Author-confirmed declarations, authorship consent and final venue requirements remain outstanding. The original official manuscript/question has not been replaced. The author's earlier instruction to see findings and approve an actual research-question change remains respected; this package makes the proposed framing reviewable.

DENS supplies no paper results. Its missing execution/export definitions are tracked separately and are not a dependency for this FACED-only paper. A separately authorized DENS inquiry was [submitted once through the author-linked form](../../results/development/dens_author_contact_2026-10-12/README.md); no reply or usable trial join is established.

## Rebuild

From the repository root, with a TeX installation providing the standard article and bibliography packages:

```powershell
conda activate pytorch
tectonic --untrusted --keep-logs --keep-intermediates --outdir paper/submission_note_2026-10-12/build/main paper/submission_note_2026-10-12/manuscript.tex
tectonic --untrusted --keep-logs --keep-intermediates --outdir paper/submission_note_2026-10-12/build/supplement paper/submission_note_2026-10-12/supplementary_methods.tex
python scripts/verify_submission_note.py
```

The completed figures and numerical table come from bound public development packets. No original FACED recordings, ratings archives, films or private model checkpoints are redistributed. Rendering and compiler intermediates are ignored. On Windows, a short writable `TECTONIC_CACHE_DIR` avoids long cache filenames.
