# Bounded DENS metadata qualification

12 October 2026. Scope: original ds003751 1.0.6 metadata, non-outcome material keys and three recording/event timing probes. **Supervised fitting remains held.** See [findings](../../../docs/publication/DENS_Metadata_Qualification_Findings_2026-10-12.md).

- inventory.json: pinned tree, authenticated metadata and annex pointers, all annotation headers.
- metadata_annotations.json: 34 behaviour files, 367 first-column material rows, 20 filename keys; three event numeric-prefix summaries.
- set_metadata.json: selective acquisition/schema decoding of three release-bound SET headers. No FDT bodies.
- set_timing.json, timing_digests.json, timing_comparison.json: exact rounded-six-decimal SET latency to TSV onset match; raw duration match. Duration units/endpoints remain unqualified.
- material_catalogue.json: retained author OSF metadata and official format documentation; 69 film entries, zero film bodies.
- license_check.json: original LICENSE length/MD5E and SHA-256, ODbL text identified. Conflicting release statements retained.
- completion/verification.json: completed public artifact bindings and independently reconstructed counts.

No individual ratings, click categories, event semantic fields or numerical waveforms are decoded. Complete annotation bodies are transiently transported for authentication; zero outcome-byte transport is not claimed. No new fitting/predictions, protected-data access or official manuscript/question change.

Public-only replay, from the repository root:

```powershell
conda activate pytorch
python scripts/qualify_dens_inventory.py --check
python scripts/qualify_dens_annotations.py --check
python scripts/qualify_dens_timing_digests.py --check
python scripts/qualify_dens_material_catalogue.py --verify
python scripts/verify_dens_metadata_qualification.py
python -m unittest discover -s tests -p test_dens_selective_metadata.py -v
```

These commands do not make HTTP requests or need private recordings. The producing commands without their check flags perform bounded metadata retrieval; do not run them as if they were offline checks. Full SET timing replay requires the retained private metadata SET files and the separately declared selective readers. No general MATLAB loader is used.
