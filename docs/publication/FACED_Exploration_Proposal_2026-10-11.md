# Proposed FACED exploration protocol

11 October 2026. **Proposed only.** No outcome access, fitting or replacement paper question is adopted. Existing 70/20/33 participant roles are frozen. This protocol makes the next exploration reviewable within the author's authorization to investigate alternatives; any actual paper-question change still needs explicit author approval and a fresh manuscript archive.

See [design findings](GRU-XNet_FACED_Design_Proposal_2026-10-11.md), [recommended structured proposal](../../results/development/faced_design_proposal_2026-10-11/recommended_proposal.json) and [evidence index](../../results/development/faced_design_proposal_2026-10-11/README.md).

## Target and population

One retrospective individual valence target per original trial, source-documented item 10 on 0–7. No threshold, midpoint removal, or rating-based eligibility. Primary elicitation population excludes the four shared-neutral clips 13–16 and the content-version-ambiguous clip 22, across all people. There are 23 eligible clips/22 minimum exact-title film families. The claim is limited to non-neutral elicitation, not all possible valence conditions or moment-to-moment affect.

Minimum exact-title family grouping is qualified; independent stimulus-video/footage/franchise hashes are not. Use scratch models and no other corpus in the initial experiment. External/pretrained comparisons remain separate gates.

## Identity and selection boundary

The primary proposed material sets are:

- Source: 3, 4, 9, 11, 17, 20, 23, 27 (8 families).
- Validation: 2, 6, 12, 18, 25, 28 (6 families).
- Confirmation: 1, 5, 7, 8, 10, 19, 21, 24, 26 (8 families; 7/8 share one family).

Source and confirmation cover all eight non-neutral elicitation categories; validation lacks fear and inspiration. Both coarse elicitation valences occur in every role, with unknown individual-rating distributions. Three metadata-selected source/validation rotations keep confirmation identities fixed.

Source fit: 70 source people × source materials. Primary selection: 20 validation people × validation materials. Familiar-material validation is a separate diagnostic. No source preparation or matching can consult excluded films' outcomes, including outcomes from source people. All confirmation-film outcomes in all people remain sealed throughout development.

After choices are locked, refit on 90 development people × 14 development families (1,260 trials). Final protected-person familiar/unseen arms contain 462/297 trials. Confirmation roles cannot be changed after outcomes, and participant/family counts, not window count, determine dependence.

## Preprocessing and technical gate

Bind source recording/header/body receipts, use explicit ADC calibration and the qualified `?V` repair, and map the 30 common scalp channels with explicit FACED aliases. Use a sample-wise common average, a trial-local Butterworth 0.5–45 Hz SOS forward/backward filter with SciPy N=4 (eight-pole realized bandpass per pass), and 128-Hz polyphase resampling. Its two-pass response has approximately −6 dB at the specified critical frequencies. This minimal offline recipe is not the exact author-cleaned derivative.

Raw segment: 32 seconds ending at or before the original video-end marker (`floor(end × native_rate)`). Core: end−30..end−2, seven non-overlapping four-second windows. No neighbor-trial padding, recording-wide ICA/interpolation or confirmation-fitted scaler. All 2,829 eligible metadata intervals fit. Average window predictions to one trial result; average clips within film families for the primary endpoint.

The next proposed eight-trial technical pilot uses source people 001/044/085/098 and source clips 11/27, with all rating values sealed. It must qualify numerical decoder behavior and signal quality before supervised access. Nonfinite/invalid evaluated inputs require a fixed context fallback and explicit coverage; do not silently remove hard trials. Synthetic adapter tests do not establish actual artifact removal.

## Controls, contrast and budget

Context controls: source global/coarse/fine-category medians and regularized metadata regression (elicitation category, nominal duration, presentation ordinal, acquisition group). Source video medians are only for familiar videos; unseen films receive declared metadata/global fallbacks. No evaluated-film consensus. Source-prior training inputs require participant-excluded cross-fitting.

First compare regularized spectral EEG-only/context/joint regressors. Only after source learning checks add a separately verified scratch EEGNet regression adapter, keeping the previous author-checked classification code unchanged. Base neural budget: two readouts × three learning rates × two initializations = 12 fits; at most eight selected-setting repetition fits. Maximum 120 epochs per fit, 12 GPU-hours overall; measure memory/time first. No larger encoder search is selected.

Primary contrast: source-selected EEG-plus-context versus source-selected context-only, paired participant- and film-equal MAE gain on eight held film families. Engineering continuation criterion: gain at least 0.15 on 0–7 and a positive lower crossed-bootstrap interval. This is not a validated psychological minimum difference or an empirically established power threshold. Report MSE, trial-pooled MAE, familiar films, acquisition groups and leave-one-family-out sensitivity secondarily.

Pairing diagnostics: within-clip EEG exchange across participants within acquisition group, and within-person exchanges across different film families with the same assigned coarse valence. Freeze donor rules/seeds before access. If treating the primary and two pairing comparisons as hypothesis tests, apply Holm adjustment to that three-contrast family; other breakdowns are descriptive. These controls assess pairing dependence, not causal neural emotion or artifact freedom. Same-emotion/different-film swaps within a held person are unsupported by the catalogue.

Negative/zero effects are retained. An interval covering both negligible and material gains is inconclusive; it does not prove the absence of usable EEG information. Hypothetical precision scenarios remain assumptions until development variances are available; never change confirmation allocation from those variances.

## Activation and paper boundary

This document and its material sets are reviewable candidates. The next technical pilot and subsequent outcome-access/fitting phases need separate immutable execution declarations within the authorized exploration scope. Actual target-vector geometry and a selective single-item rating decoder remain to be verified. Do not use an unrestricted full `loadmat` call on protected rows.

Closest full-method comparison remains incomplete for the Gerster FACED preprint. Existing stimulus-confounding, familiar-video-prior and joint-pretraining work prevents claiming those broad ideas as novel. Use development findings to decide whether a specific contribution merits presenting to the author. Receive explicit approval before changing the paper question, archive the then-current manuscript, and lock final analysis before opening confirmation.
