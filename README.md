# FACED exploratory analysis: public code and evidence

This repository contains the curated public analysis-code and analysis-level evidence snapshot for **Context controls and EEG pairing in retrospective valence prediction: an exploratory FACED study**.

The study is exploratory and uses adaptive development evaluations. Its earlier validation panels, ten-cell spectral source comparison, and one-cell neural source diagnostic are separate analyses, not independent replications or independent confirmation. The reserved participant cohort and its outcomes are not included.

## Public contents

| Path | Contents |
|---|---|
| `gruxnet/` | 64 analysis-library modules |
| `scripts/` | 95 analysis, audit, qualification, and replay scripts |
| `tests/` | 50 test modules using synthetic inputs or public evidence |
| `paper/exploratory_evaluation_2026-10-12/` | Public evidence and numerical-audit summaries |
| `paper/exploratory_evaluation_2026-10-15/` | Public independent numerical review |
| `results/development/` | 13 curated analysis-level prediction, comparison, history, selection, and audit files |
| `requirements-publication.txt` | Python dependencies for the analysis code |

The snapshot includes historical diagnostics and inactive guarded implementations. Their presence does not mean that protected experiments were run, and it is not evidence of scientific performance by itself.

Tests that require excluded local experiment records are kept outside the public tree. Some historical analysis paths also reference auxiliary scripts and local artifacts outside this release; those paths are not supported by this snapshot.

## What is intentionally excluded

Draft and archived manuscripts, submission packages and administration, progress notes, literature-review work, historical course-project code, unrelated experiments, generated figures, model checkpoints, caches, credentials, original recordings and rating archives, private fitted states, and reserved outcomes are not part of this public tree.

The manuscript remains excluded while it is an author-review draft. Original third-party datasets must be obtained through their own access procedures; this repository does not redistribute them.

## FACED data access

The original FACED recordings and ratings are available through the authors' Synapse project, [`syn50614194`](https://www.synapse.org/Synapse:syn50614194), under its access conditions. The public evidence here contains analysis-level development exports only. It contains no raw EEG, original rating archive, private model state, or reserved outcome.

## Installation and tests

Use a suitable Python environment, then install the documented dependencies:

```bash
python -m pip install -r requirements-publication.txt
python -m pytest tests
```

PyTorch should match the hardware/CUDA environment in which the analysis will run.

## Reproduction scope

This is a source-and-evidence snapshot, not a one-click raw-data reproduction bundle. Recomputing raw-data analyses requires independently obtaining FACED, configuring local input paths, installing the required numerical dependencies, and respecting the documented source, development-validation, and reserved partitions.

Do not launch a protected or reserved-cohort runner merely because its implementation is present. Start with the focused tests and the public evidence/audit files, then inspect the specific analysis script and its command-line interface before running it.

## Authors

Muhammad Muntazar and Muhammad Wasif Shakeel, National University of Sciences and Technology (NUST), Pakistan

## License

Code is provided under the [MIT License](LICENSE). Third-party datasets and third-party source implementations remain subject to their own licenses and access terms.
