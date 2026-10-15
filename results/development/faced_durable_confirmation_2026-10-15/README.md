# Durable attempt qualification

Single manufactured-only execution frozen at `3958af52a`, completed 15 October 2026. `plan_v2.json` is operative. `plan.json`/`plan_v1_unexecuted.json` and the initial inactive contract remain preserved and unexecuted; no earlier frozen executor or source result was rewritten.

- `inactive_durable_executor_contract_v2.json`: new inactive future contract retaining the already qualified source-bundle fingerprint as metadata. It requires the durable entry point and exact public attempt path; it supplies no author access approval.
- `manufactured_contracts.json`: explicitly manufactured forks using a fake model bundle, not reloaded source arrays.
- `manufactured_complete/`: durable reservation, callback journal and complete 3,036-row target-free fabricated lock. `kind=manufactured_qualification`; it cannot authorize protected ratings.
- `manufactured_failure/`: intentional first-callback fatal technical failure with persistent reservation/journal/failure and no stage completion or prediction lock.
- `completion.json`: exact counts, zero-callback fresh-instance and actual separate-process restart rejection, output hashes, timings, memory snapshots and preserved historical bindings.

Fourteen manufactured tests passed. Actual private/source-array/original/target/validation/reserved numeric reads and new fits were zero. No production activation was created. See [findings](../../../docs/publication/FACED_Durable_Confirmation_Qualification_Findings_2026-10-15.md) and [protocol](../../../docs/publication/FACED_Durable_Confirmation_Qualification_Protocol_2026-10-15.md). This packet qualifies workflow persistence with fabricated inputs, not protected throughput, future precision or scientific model performance.
