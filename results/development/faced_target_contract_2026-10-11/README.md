# FACED development target-storage contract

11 October 2026. All 1,260 inspected development score vectors are 1×12 doubles; item10 is valence. Original values remain sealed. Twenty manufactured decoder tests and eight classical-contract tests passed. No new original EEG samples or original-data model fits occurred.

See [findings](../../../docs/publication/GRU-XNet_FACED_Target_Contract_Findings_2026-10-11.md), [metadata declaration](plan.json), [qualified storage](storage.json), [storage bindings](verification.json), [public packet receipt](packet_verification.json) and [bounded development programme](../faced_development_protocol_2026-10-11/README.md).

Only 90 development participants' behavioral files were opened; only 14 development clips' score headers were inspected. The 33 confirmation participants were skipped before file access. Other scores were skipped at their outer tags; no actual item 10, other rating, accuracy or response-time number was interpreted. Opaque hashing/decompression is not numerical score decoding.

Portable replay without originals:

```powershell
python scripts/verify_faced_target_contract.py verify
python scripts/declare_faced_classical.py verify
```

The synthetic suite can be rerun with:

```powershell
python -m pytest -q tests/test_faced_targets.py tests/test_faced_classical_contract.py -p no:cacheprovider
```

Original-data access is not activated here. Missingness, actual scale adherence, variance and predictive benefit remain unknown. Build/audit commands refuse overwriting receipts.
