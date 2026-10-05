# GRU-XNet dataset provenance and upstream integrity checks

Checked: 5 October 2026. The project author identified the bundled Kaggle dataset as their own and supplied the three source URLs below.

**Finding:** the DEAP rating discrepancy is already present in the supplied upstream Kaggle version. The independently downloaded `s01.dat` and participant-rating spreadsheet have the same SHA-256 hashes as the local copies, and reproduce the discrepancy. This establishes that the discrepancy in that subject file predates assembling the bundle. All other training input files also match the supplied upstream archives by file size and CRC32.

## Declared download chain

| Role | Source | Checked version | Kaggle last-updated timestamp (UTC) |
| --- | --- | ---: | --- |
| Author's bundle | [fdskjlajlkfdsa/emotion-recognition-eeg-datasets](https://www.kaggle.com/datasets/fdskjlajlkfdsa/emotion-recognition-eeg-datasets) | 1 | 2025-12-01 09:05:32 |
| DEAP source | [manh123df/deap-dataset](https://www.kaggle.com/datasets/manh123df/deap-dataset) | 1 | 2024-12-11 13:56:47 |
| GAMEEMO source | [sigfest/database-for-emotion-recognition-system-gameemo](https://www.kaggle.com/datasets/sigfest/database-for-emotion-recognition-system-gameemo) | 1 | 2021-03-09 08:03:55 |
| SEED-IV source | [phhasian0710/seed-iv](https://www.kaggle.com/datasets/phhasian0710/seed-iv) | 1 | 2019-06-05 10:14:54 |

These are the supplied download sources. The research dataset sites are [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), [GAMEEMO's Mendeley release](https://data.mendeley.com/datasets/b3pn4kwpmn/1), and [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html). Comparing a Kaggle mirror with the bundle does not independently authenticate a first-party release.

The source catalog is [dataset_sources.json](../../dataset_sources.json). Kaggle API metadata responses and source ZIP directory inventories are retained in publication_runs/provenance/ (local workspace evidence: `publication_runs/provenance/`). The author bundle was not downloaded again in full; comparisons use the existing local extraction and each independently retrieved upstream source.

## Integrity results

| Dataset | Files compared | Size + CRC32 matches | Representative files with matching SHA-256 |
| --- | --- | ---: | --- |
| DEAP | 32 Python subject files + participant-rating spreadsheet | 33 / 33 | `s01.dat`, `participant_ratings.xls` |
| GAMEEMO | 112 preprocessed AllChannels CSV recordings + 112 SAM PDFs | 224 / 224 | S01/G1 EEG CSV and SAM PDF |
| SEED-IV | 45 raw EEG MAT files + ReadMe + channel-order spreadsheet | 47 / 47 | `ReadMe.txt`, `Channel Order.xlsx` |
| Total | Training recordings and label/channel metadata | **304 / 304** | **6 / 6** |

The complete checks are in [upstream_full_integrity_comparison.json](../../results/development/provenance/upstream_full_integrity_comparison.json) and [upstream_sample_comparison.json](../../results/development/provenance/upstream_sample_comparison.json). CRC32 is an integrity checksum, not a cryptographic hash. Only the six downloaded representative files were verified by SHA-256. Unused feature exports, original GAMEEMO CSV/MAT variants, historical augmented arrays, and duplicate dataset trees are outside this comparison's scope.

The full-archive directory checks used HTTP byte ranges to retrieve the ZIP central directory, rather than downloading three large archives. Each local file was read completely to compute its CRC32. Representative source downloads are retained under source_checks/ (local workspace evidence: `publication_runs/provenance/source_checks/`).

## DEAP label finding

Across all 32 local subject files, exactly **439 trials** have both valence and arousal replaced by `9 - spreadsheet rating`. Dominance and liking match all 1,280 spreadsheet rows when joined by `Participant_id + Experiment_id`; presentation `Trial` is a different ordering. Both locally included copies of the rating spreadsheet agree.

Every changed trial has spreadsheet valence **above 5** and arousal **above 5**. Every one of these 439 binary valence labels changes from positive to negative under the project's threshold. The affected trial list, both original file values and spreadsheet values, is retained in deap_rating_discrepancies.csv (local workspace evidence: `publication_runs/provenance/deap_rating_discrepancies.csv`).

For participant S01, the independently downloaded upstream pickle and spreadsheet reproduce **13 changed valence ratings and 13 changed arousal ratings**. For example, experiment 1 has spreadsheet valence/arousal `7.71 / 7.60`, while the pickle contains `1.29 / 1.40`; dominance/liking `6.90 / 7.83` agree. The SHA-256 equality shows that this entire local subject file, including EEG and labels, matches the upstream download.

The checked source therefore already carries the altered labels. These checks do not establish who changed the source, why it was changed, or whether the mirror's EEG matches the first-party DEAP release. The pattern is systematic and should be disclosed without assigning an unverified cause.

## Consequences for experiments and publication

The corrected loader already uses the included spreadsheet ratings, checks their correspondence with the pickle, and records the original file ratings and source hashes. **This provenance result requires no label change or rerun of the completed corrected development experiments.** It does not rescue the historical accuracy claims; their evaluation and augmentation problems remain documented in [the implementation status](GRU-XNet_Implementation_Status_2026-10-05.md).

The CLI can now bind the source catalog's full contents and SHA-256 into a new audit or newly prepared cache:

```powershell
conda activate pytorch
cd GRU-XNet_EEG_Emotion_Recognition
python -m gruxnet audit --data-root ../emotion-recognition-eeg-datasets --output ../publication_runs/audit_with_sources --provenance dataset_sources.json
```

The audit with declared sources (local workspace evidence: `publication_runs/audit_with_sources/dataset_audit.json`) records the source versions alongside recovered labels. The new audit reproduces the existing label policy, trial counts, electrode orders, metadata hashes, and recovery counts. Existing caches and experiment fingerprints retain their original contents. The [supplemental cache binding](../../results/development/provenance/existing_cache_binding.json) records the cache fingerprint and hashes of the catalog, new audit, and comparison evidence without modifying the cache. New caches can use `prepare --provenance dataset_sources.json`; use a fresh output directory. Passing a catalog snapshots a declaration and does not automatically repeat the external download checks.

All 16 pipeline tests pass, including source-catalog completeness, conflicting source rejection, and hash changes when a declared source version changes. Python compilation and Git whitespace checks also pass.

Before final confirmatory experiments, compare DEAP's signal files and rating metadata against a first-party distribution, or explicitly disclose the remaining mirror limitation. The manuscript should cite the research datasets, identify the exact download versions, and explain the spreadsheet-based recovery. The model still needs successful training/validation development and matched independent evaluations.

## First-party access attempt

The [first-party DEAP check](GRU-XNet_First_Party_DEAP_Check_2026-10-05.md) records a direct attempt on 5 October 2026. The exact official EEG and metadata archive URLs return HTTP 503 after successful TLS verification; the access-request server times out. Archived official documentation confirms the video-order join, array dimensions, rating order, and all 32 electrode positions, but no first-party recording or metadata archive was obtained. The maintained `compare-deap` command is prepared to compare every signal sample and rating once an author-downloaded copy is available. The current test suite has 17 passing tests.
