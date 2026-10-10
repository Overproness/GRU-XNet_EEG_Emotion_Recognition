# EmoEEG-MC author contact: prepared request and delivery status

Updated 10 October 2026. **The approved metadata request was posted and verified as [author-repository issue #1](https://github.com/ncclab-sustech/EmoEEG-MC/issues/1) at 07:48 UTC.** GitHub accepted one submission, and a separate read confirmed the exact approved title, body and account. There were zero comments at the delivery check; an author response remains outstanding.

## Concrete request and route

The [exact request](EmoEEG_MC_Metadata_Request_2026-10-10.md) and [preparation evidence](../../results/development/emo_mc_author_contact_2026-10-10/request.json) were committed and pushed in `5e7706c91` before attempting delivery. The earlier hash-bound [inquiry draft](EmoEEG_MC_Metadata_Inquiry_Draft_2026-10-10.md) remains unchanged.

Destination: a **public issue** on [ncclab-sustech/EmoEEG-MC](https://github.com/ncclab-sustech/EmoEEG-MC/issues/1), posted through the existing `Overproness` GitHub account. Title: **Metadata/codebook request: participant/trial joins, score columns and trial end markers**.

The read-only GitHub API check at 06:35 UTC confirmed a public, unarchived repository with issues enabled, an empty complete issue listing, and a working existing sign-in. No new plugin/account connection is required. Credentials were kept in process memory and never printed or saved.

The request covers separate material mappings for video/imagery; participant and integer-code crosswalks; incomplete-trial behaviour-row identities; `score_1` through `score_10` semantics; trigger/run-clock definitions; the participant-37 `dis8` end ambiguity; and valid short-trial intervals. It links already-public technical evidence and includes anonymous participant references and timing discrepancies. The request is now publicly visible on the author repository.

## Approval and delivery record

The first proposed issue-creation command was rejected before execution by automatic approval review because the exact public payload/destination had not been explicitly approved. The [historical rejection record](../../results/development/emo_mc_author_contact_2026-10-10/approval_status.json) preserves that earlier unsent state. No workaround or alternative sending route was used.

The user subsequently explicitly approved the exact request at commit `5e7706c91`, the public author repository and the `Overproness` account. The [approval record](../../results/development/emo_mc_author_contact_2026-10-10/approval_received.json) supersedes the pending question. The approved submission returned HTTP 201; a separate issue read returned HTTP 200 with matching title/body/account. The [delivery receipt](../../results/development/emo_mc_author_contact_2026-10-10/delivery.json) records issue ID `5789310410`, creation at `2026-10-10T07:48:01Z`, and zero comments at the verification read. GitHub acceptance does not establish that an author has read or answered the inquiry. No duplicate inquiry or email was sent.

A further [unauthenticated public check](../../results/development/emo_mc_author_contact_2026-10-10/delivery_verification.json) at 07:50:44 UTC returned HTTP 200 and matched the committed approved body, title, account, issue identity and creation time. It also confirms the original hash-bound draft is unchanged. The public check found zero comments and introduced no outcome access.

## How a response will be assessed

| Required documentation | Metadata-only acceptance check before opening outcomes |
| --- | --- |
| Participant/code crosswalk | Explicit source participant and override/duplicate meanings, checked against recording paths and annotation identities; no inferred 53/54 reassignment. |
| Material identities in both contexts | Context-specific symbolic-to-workbook mapping; do not copy the video mapping into imagery. Preserve the seven reserved entries in each context. |
| Behaviour-row execution map | One-to-one recorded-trial joins and explicit missing-trial masks, including incomplete recordings; do not fill gaps by assuming row order. |
| Numbered score codebook | Dimension, scale, endpoint and missing-value specification for each column, documented without examining numerical ratings. |
| Trigger and clock definitions | Actual playback/audio/button/rating meanings and run-local time alignment, checked against the authenticated technical markers. |
| Valid EEG intervals | Documented within-trial endpoints that resolve the outlier and short trials without including rating activity or later samples. |

Archive any supplied version/date and source identity before checking it. An unresolved item remains unresolved; a useful answer to one question does not clear the other gates. The twenty reserved confirmation participants, fourteen reserved context/material keys and existing quarantine/join holds remain fixed. Numerical ratings, EEG amplitudes, fitting and prediction remain unopened for this corpus. Wider physical-calibration and stimulus/pretraining-overlap requirements also remain in force.

This contact step does not establish a new conference contribution or adopt a new research question. The manuscript is unchanged. An actual pivot still requires an evidence-backed proposal, explicit author approval and a fresh archive immediately before the change.
