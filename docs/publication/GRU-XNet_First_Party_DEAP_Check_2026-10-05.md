# First-party DEAP access and verification attempt

Checked: 5 October 2026. **The first-party EEG-content comparison could not be completed: the official data host returns HTTP 503 and the access-request server times out.** No official EEG file or metadata archive was obtained. The previous Kaggle integrity checks remain valid within their stated scope.

## Official download routes checked

The [archived official download page](https://web.archive.org/web/20240128114615/http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) identifies the following original files and access server. It explains that credentials are obtained through the request server.

| Route | Result in this session |
| --- | --- |
| [Official Excel metadata ZIP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/metadata_xls.zip) | HTTP **503 Service Unavailable** with verified TLS |
| [Official preprocessed Python EEG ZIP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/data_preprocessed_python.zip) | HTTP **503 Service Unavailable** with verified TLS |
| [Official access-request server](https://anaxagoras.eecs.qmul.ac.uk/request.php?dataset=DEAP) | Connection timeout on HTTPS and HTTP |
| Original download, contact, and agreement pages on `www.eecs.qmul.ac.uk` | HTTP 503 after resolving a missing TLS intermediate |
| Download/README routes without `www` | Redirect to Queen Mary's general department page |
| Checked archived Excel/CSV metadata ZIP URLs | No downloadable snapshot found; archive replay returns 404 |

The initial Python requests failed certificate-chain validation on the original host. The host's public certificate identifies a Sectigo intermediate; retrieving that intermediate and adding it to the existing trusted CA bundle allowed normal hostname and chain verification to succeed. The resulting HTTP responses still report 503. These final results therefore distinguish server unavailability from the initial local certificate-chain error.

The public [EPFL research page](https://www.epfl.ch/labs/mmspg/research/page-58317-en-html/bci-2/bci_datasets/emotion_dataset/) also links to the original Queen Mary DEAP site rather than an alternative data download. Public archive recovery succeeded for the official download, README, and contact pages, but not the checked metadata archives.

Responses, timestamps, redirects, and SHA-256 hashes are retained under first_party_deap/ (local workspace evidence: `publication_runs/provenance/first_party_deap/`). The machine-readable outcome is [first_party_attempt.json](../../results/development/provenance/first_party_deap/first_party_attempt.json); [verified TLS route checks](../../results/development/provenance/first_party_deap/verified_tls_routes/route_checks.json) contain the decisive direct responses. The public probe script retains bounded response bodies and can be rerun when the service is restored.

## What could be confirmed from first-party documentation

The recovered [official README snapshot](https://web.archive.org/web/20231030222856/https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html) confirms the format used by the corrected pipeline:

- Python files contain signal arrays with dimensions `40 × 40 × 8064` and rating arrays with dimensions `40 × 4`.
- Signal trials use `Experiment_id` video order; spreadsheet `Trial` records presentation order. Joining spreadsheet ratings by participant and video ID is correct.
- Rating columns are valence, arousal, dominance, and liking, in that order.
- All 32 EEG electrode names and their order in the maintained loader match the archived official channel table, checked programmatically after case normalization.

These checks are recorded in [official_documentation_check.json](../../results/development/provenance/first_party_deap/official_documentation_check.json), including the archived response hash. They verify the interpretation of the format. They do not establish that the local signal samples or spreadsheet values match an author-issued release.

## Full comparison is prepared

The maintained CLI now accepts either an author-downloaded Python ZIP or an extracted directory containing all 32 subject files:

```powershell
conda activate pytorch
cd GRU-XNet_EEG_Emotion_Recognition
python -m gruxnet compare-deap --official ../official_deap/data_preprocessed_python.zip --data-root ../emotion-recognition-eeg-datasets --output ../publication_runs/provenance/deap_official_comparison.json
```

The example path must point to an actual author-issued copy. The comparator checks every trial, every sample, and all 40 channels, including baseline and peripheral channels. It reports serialized file hashes, decoded signal hashes, exact signal equality, maximum signal differences, and differences between the reference ratings and the included spreadsheet. A different pickle encoding can change the file hash while preserving signal values, so both forms of comparison are retained.

The output labels the reference as user-supplied; file comparison cannot itself establish download provenance. The command uses a fresh JSON output file and does not rewrite source data or prepared caches.

**Validation:** 17 pipeline tests pass. The added test distinguishes a label-only modification from a one-sample EEG change and rejects incompatible signal dimensions. A real-data control compares local S01 with the previously downloaded Kaggle S01 and correctly reports equal signals while detecting 13 valence and 13 arousal discrepancies against the spreadsheet. That control (local workspace evidence: `publication_runs/provenance/first_party_deap/comparator_control_s01.json`) is explicitly identified as a Kaggle check, not a first-party comparison. Git whitespace and Python compilation checks pass.

The remaining input is an archive already obtained directly from the authors, or restored official download service plus author-issued access. Further third-party mirrors could provide additional agreement evidence, but would not complete this first-party check.
