# Independent source and evidence review

Reviewed 12 October 2026. Scope: the actual research-note `manuscript.tex`, `supplementary_methods.tex` and `references.bib`, the preceding complete FACED draft and table, and completed public result packets. This is document/numerical review, not scientific confirmation or publisher acceptance. No original rating/waveform data, protected outcomes or checkpoint-to-waveform inference were accessed; no models were fitted and no submission or external message was sent.

**Assessment:** the note preserves the audited exploratory result and its material limitations. It is suitable for author review as a candidate research-note package. The initial reference-style issue is fixed, and the final supplementary PDF passes independent visual review. Actual submission remains conditional on author-approved framing and declarations, exact historical assistance-model identification and the venue's assessment.

## Scientific and numerical checks

| Item | Finding |
| --- | --- |
| Target and scope | Retrospective item-10 individual valence, 0–7, restricted non-neutral elicitation population; all permitted values retained. Assigned category is supplied context rather than an individual-label substitute. No momentary or real-time inference claim. |
| Population and roles | Seventy source, twenty reused development-validation and thirty-three reserved participants are stated correctly. Eight title families remain reserved; development uses fourteen normalized catalogue-title families. Physical footage independence and latency calibration are explicitly unqualified. |
| Exact panels and denominators | All three source/held-title rosters match the complete draft. Every rotation has 560 source, 120 held-title and 160 familiar-material trials. The unique permitted sets contain 980 source and 280 validation trials, totaling 1,260 targets. |
| Dependence and selection | Both files explain that participants, titles and targets overlap across rotations. Ridge regularization uses development outcomes; the neural schedule is selected on rotation 0 and transferred without reselection. Neither source-only fitting nor the later fixed schedules is misrepresented as independent testing. |
| Transform/offset fitting | Source-only standardization, bandwidths, source priors and five participant-disjoint source OOF context predictions agree with the declared implementation. Full-source context is correctly distinguished from OOF training offsets. |
| Input and processing | Seven nonoverlapping four-second windows from approximately end−30 to end−2, thirty channels, 128 Hz, fixed trial-local acausal processing, eight repaired source trials, no augmentation/ICA. Detailed supplement preserves calibration, channel order, filter configuration and repair limits. |
| Coverage and fallback | Fifteen of 280 validation trials have unavailable EEG; 265 remain available. Fourteen failures belong to one participant. All outcomes remain in assessment, and source-median/context fallback branches are stated correctly. The policy's development-informed origin is disclosed. |
| Complete conditions | The supplemental table contains all fifteen declared main conditions plus a clearly identified retrospective best-of-five context minimum. The main table is explicitly representative rather than a claim to list all conditions. |
| Numerical results | All 48 source rotation values and 16 means agree across the separate public classical, representation and neural audit packets. All 64 rounded supplemental table cells and nine main-table means agree with those inputs. Spectral, covariance and residual gains, interval endpoints, capacity values and learning summaries match the recorded findings at the precision shown. |
| Comparator strength | The covariance gain against weaker matched context RBF is presented beside its smaller gain against context Ridge. The neural five-control minimum is clearly retrospective rather than a deployable source-selected baseline. |
| Uncertainty | Six held titles per panel, descriptive crossed-bootstrap intervals, dependence and omitted selection uncertainty are disclosed. No significance test or independent confirmation claim is introduced. |
| Convergence and capacity | Five of six selected neural panels continue more than 10% source-MSE improvement from epochs 60–120; no optimized-EEGNet ranking or biological information-null claim follows. Real/shuffled kernel interpolation is not described as emotion information. |
| Closest methods | The note acknowledges established stimulus-confound and contextual-prior work. The supplement distinguishes the inspected August Gerster code from unverified complete June-paper execution, and explains operational target/participant/title differences without claiming numerical reproduction or generic novelty. |
| Audit scope | Exact source/window-QC replay is distinguished from numerical and state-dictionary authentication. Independent checkpoint-to-waveform forward replay is expressly outside the checks. |
| Original work and future experiments | This separate exploratory package does not replace the original GRU-XNet question or activate confirmation. DENS contributes no manuscript results. |

The new scalar-regression formula in the supplement is consistent: because the fixed target scale and number of windows are both seven, `offset + sum(window_outputs)` equals `offset + 7 * mean(window_outputs)`.

## Research-note format checks

Checked against the [official BMC research-note instructions](https://link.springer.com/journal/13104/submission-guidelines/research-note) and [general instructions](https://link.springer.com/journal/13104/submission-guidelines), retrieved 12 October 2026.

- Objective/Results abstract, no abstract references, five keywords, Introduction/Main text/Limitations structure, one table and one figure, double spacing and line-number configuration are present.
- Initial conservative regex counts were 182 abstract tokens, 1,176 body tokens without floats and 1,318 including their text. The reviewed final-source revision gives 182/1,180/1,331 respectively, comfortably below the 200/2,000 limits. Publisher/word-processor counts may differ.
- Every required declaration subsection is represented. Unknown funding, competing interests, contributions, publication consent, corresponding author and author details remain explicitly pending rather than fabricated.
- The cited FACED paper reports Tsinghua University approval THU201906 and informed consent; it does not specify written consent in the inspected Methods. The note accurately preserves that uncertainty and flags the secondary-analysis ethics statement for author confirmation.
- Repository, original-release and packet links are numbered bibliography entries rather than unsupported inline links. Access dates and scientific result revision are supplied.
- **Reference-style correction completed:** the first inspected files used `unsrt`. Both final sources now use `vancouver`; the rendered supplementary bibliography uses surname/initial formatting, numbered references, dates, source identifiers and wrapped URLs without clipping.
- OpenAI Codex is now identified; exact historical model/version remains author-pending. [Publisher assistance policy](https://www.springernature.com/gp/policies/editorial-policies/ai-manuscript-preparation).
- The supplementary PDF's twelve pages were independently visually reviewed as six original-detail page sheets. Tables S1--S3, equations, titles, margins, page/line numbers and bibliography are legible, without clipping, overlap or missing glyphs. The final page has substantial whitespace after the last long bibliography entry; this is a natural double-spaced reference continuation rather than missing content.
- The compiling agent reports visual review of all nine main-paper pages. Independent text extraction confirms nine main/twelve supplemental pages, no `[?]` reference tokens and no Unicode replacement glyphs. Main/supplement file sizes are 68,699/85,105 bytes, below the additional-file cap.
- Optional wording polish: expand FACED once and define MSE at its first body use rather than only in the later abbreviations list. This is a minor preparation item, not a change in the result or a scientific blocker.

## Final rendered supplement disposition

The final supplement reviewed above is `output/pdf/faced_supplementary_methods.pdf`, SHA-256 `8c0545a211b7a88623c8174ee4f36ac52d3232598b80ac1bfe83d145e0ee3955`. Its source SHA-256 is `cb8f4c04aa70a8afe637670027b03a43e51fcb47e2a5705b0e7948fbd8767ac9`. The six inspected images are `tmp/pdfs/sheet-supplement-{1,3,5,7,9,11}.png`, covering pages 1--12.

The main source at this final inspection places Muhammad Muntazar before Muhammad Wasif Shakeel, consistent with the author-order confirmation relayed by the coordinating agent. Exact model/version, current corresponding author and substantive author declarations remain pending. Scientific interpretations and the numerical tables retain the previously checked values.

## Decision

No scientific misstatement requiring a different result or new fitting was identified in this version. The unresolved items above prevent an actual submission-ready claim, but do not prevent presenting the concrete package to the authors for review. An exploratory submission need not manufacture a favorable outcome or open reserved data; stronger claims would require a separately frozen confirmation study.
