# Current-progress empirical paper draft

12 October 2026. [Compiled paper](output/pdf/current_progress_paper.pdf), [LaTeX source](manuscript.tex), [bibliography](references.bib).

**EEG Beyond Elicitation Context: An Exploratory Study of Retrospective Valence Prediction on FACED**

This is a complete separate empirical draft requested by the author. It reports the completed FACED study, including all fifteen declared main conditions, context controls, acquisition fallback, source capacity and fixed EEGNet learning diagnostics. DENS is a metadata candidate and contributes no manuscript results. The original GRU-XNet paper and its official question remain unchanged; [a fresh archive](../../docs/paper_archive/2026-10-12-before-current-progress-draft/README.md) precedes this draft.

## Claims and submission status

The evidence supports an exploratory report of the tested setting, not architecture novelty, independent confirmation, a universal EEG-information null or a guaranteed acceptance at any venue. Performance uses reused development participants and overlapping material panels. Ridge choices and part of neural schedule selection use development outcomes. Reserved outcomes remain sealed. The retrospective best-of-five context row is labelled as a benchmark, not a deployable selected model.

[Publication routes](Submission_Routes.md) checks BMC Research Notes and PLOS ONE against official policies, including required adaptation and funding considerations. These are conditional scope matches, not an acceptance prediction or a selected target.

For a modest applied conference, workshop or journal, the defensible framing is this transparent empirical comparison. Its contribution and scope still need author and editor/reviewer assessment. No venue was selected, no submission was made, and no research-question replacement was adopted. The IEEE conference style is a neutral starting format, not evidence of venue compliance.

Before actual submission:

1. Review and approve this empirical framing if it is to replace the official manuscript question.
2. Confirm both authors, affiliations, current emails, authorship consent and any funding/conflict/ethics declarations required by the venue. Draft names/affiliation/emails come from the original source; current validity has not been independently checked.
3. Select a legitimate venue accepting exploratory empirical work and follow its page limit, anonymization, artifact, AI-assistance disclosure and data/ethics policies. No conference ranking or acceptance probability is asserted here.
4. Preserve the development-only wording. A stronger claim of robust EEG benefit or absence requires a separate frozen confirmation plan and adequately precise independent assessment. The current draft does not require opening reserves simply to find a favorable result.
5. Review the closest-work discussion, references and final PDF. The primary-paper bibliography was checked online; recent preprints are identified as such. Gerster's inspected August code is not presented as numerically reproduced June-paper execution.

## Evidence map

| Manuscript evidence | Public source packet |
| --- | --- |
| Source roles, target/access contract, title groups | ../../docs/publication/GRU-XNet_FACED_Target_Contract_Findings_2026-10-11.md and GRU-XNet_FACED_Scope_Precision_Plan_2026-10-11.md |
| Spectral Ridge and fixed availability fallback | ../../results/development/faced_inference_quality_2026-10-11/ |
| Covariance/kernel, source OOF and capacity controls | ../../results/development/faced_representation_v2_2026-10-11/ |
| EEGNet fits, exact selection and audited predictions | ../../results/development/faced_eegnet_diagnostic_v2_2026-10-11/ |
| Closest-method/static-code qualification | ../../docs/publication/GRU-XNet_FACED_EEGNet_Preflight_2026-10-11.md |
| Complete paper table and figures | results_table.json and figure_receipt.json |

The manuscript's table was independently checked against the public packets: all 48 rotation values (fifteen declared models plus the retrospective context minimum), means, gains, source-learning/capacity values, intervals and runtime figures agree. This is numerical/document review, not independent scientific confirmation.

## Build and review

From the repository root, regenerate figures without raw data or fitting:

```powershell
conda activate pytorch
python scripts/build_current_progress_figures.py
```

Compile with a TeX installation providing IEEEtran, or portable Tectonic 0.17.0:

```powershell
tectonic --untrusted --keep-logs --keep-intermediates --outdir paper/current_progress_2026-10-12/build paper/current_progress_2026-10-12/manuscript.tex
```

On Windows, a short writable TECTONIC_CACHE_DIR avoids cache filenames exceeding path limits. The PDF is copied from build/manuscript.pdf into output/pdf/current_progress_paper.pdf after compilation and visual review. The build and intermediate renders stay ignored; portable compiler/dependency files are not published. No original recordings/stimuli are redistributed.

The final build receipt records source, bibliography, figures, PDF and audit hashes, page count and rendering checks. PDF pages and both standalone scientific figures are visually inspected before delivery.
