# FACED metadata-only design proposal

11 October 2026. A feasible continuous individual-valence experiment is proposed for 23 clips/22 source-title families, with material roles 8 source / 6 validation / 8 confirmation. Existing participant roles 70/20/33 remain unchanged. All individual outcomes and numerical EEG samples remain sealed; no model was fitted or paper question changed.

See [findings](../../../docs/publication/GRU-XNet_FACED_Design_Proposal_2026-10-11.md), [execution proposal](../../../docs/publication/FACED_Exploration_Proposal_2026-10-11.md) and [recommended structured proposal](recommended_proposal.json).

| Artifact | Purpose |
| --- | --- |
| [Initial declaration](plan.json), [initial candidate](proposal.json), [bindings](verification.json) | Four target options, family counts, fixed participant roles, three deterministic development rotations and hypothetical precision scenarios |
| [Preprocessing supplement](preprocessing_plan.json), [recommended proposal](recommended_proposal.json), [supplement bindings](supplement_verification.json) | Source-documented reference/channel handling; recommended late-viewing interval replaces initial early-viewing choice before any sample access |
| [Source quality pilot candidate](source_quality_pilot_candidate.json) | Eight proposed source-only EEG checks with all ratings sealed; not launched |

Material assignments are **proposed**, not an outcome-access authorization or an updated manuscript question. No rating-derived balance, variance, signal quality or model performance is known from this phase. Exact-title grouping does not authenticate absent shared footage or media versions. The author-method source text is private; only factual metadata and fingerprints are retained here.

From repository root:

```powershell
& 'D:\DL_Frameworks\envs\pytorch\python.exe' -m pytest -q tests/test_faced_design_proposal.py tests/test_faced_proposal_preprocessing.py tests/test_faced_late_proposal.py -p no:cacheprovider
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/prepare_faced_design_proposal.py verify
& 'D:\DL_Frameworks\envs\pytorch\python.exe' scripts/finalize_faced_design_proposal.py verify
```

Fourteen tests use synthetic signals or already-public metadata; the late-window replay checks 2,829 metadata intervals without opening raw or behavioral files. Collection/build commands preserve existing outputs. A future execution must freeze activation, permitted identities/item, source-quality and analysis rules first; confirmation stays sealed until the final locked phase.
