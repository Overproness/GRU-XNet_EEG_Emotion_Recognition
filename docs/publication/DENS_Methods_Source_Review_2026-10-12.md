# DENS methods and annotation source review — 12 October 2026

This is a public documentation qualification, not an outcome analysis. It supplements the pinned `ds003751` release inventory at commit `5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c` (snapshot 1.0.6). No original EEG samples, individual rating values, behavior rows, model predictions, or FACED confirmation data were opened for this review. Public author manuscripts, repository documentation, OSF directory metadata, and a 64-byte filename-format document were read. No external inquiry was sent.

## Published protocol resolves two apparent discrepancies

The final IEEE Access paper clarifies the design: **11 main trials comprise nine emotional and two non-emotional clips**. Emotional clips are sampled from 16 candidates; two additional neutral clips are described. Clips last 60 seconds, and an 80-second resting baseline precedes presentation. Six whole-clip ratings follow viewing; click-linked emotion categories are then recalled. Rating ranges are 1–9 for valence/arousal/dominance and 1–5 for liking/familiarity/relevance. Recall aids show the click frame and frames 20 positions before/after, at 30 frames/second. Its seven-second EEG extraction uses one second before and six after a click; this is an analysis choice, not a trial endpoint definition. These are **reported methods**, not authenticated stored-file semantics. [Final publication, DOI 10.1109/ACCESS.2023.3266804](https://doi.org/10.1109/ACCESS.2023.3266804), [accessible primary full text](https://www.researchgate.net/publication/370007995_Emotion_Recognition_using_Temporally_Localized_Emotional_Events_in_EEG_with_Naturalistic_Context_DENS_Dataset).

The earlier dataset manuscript describes **three practice trials and 11 testing trials**. It defines whole-clip valence as unpleasant-to-pleasant, arousal as inactive-to-active, dominance as submissive-to-dominant, relevance as personal relation to the situation, and later click categories as retrospective choices. It reports that most respondents perceived a two-to-three-second delay between awareness and clicking. That self-report does not supply a calibrated event-level correction. [2021 primary manuscript, DOI 10.1101/2021.08.04.455041](https://doi.org/10.1101/2021.08.04.455041), [accessible primary full text](https://www.researchgate.net/publication/353722077_Dataset_on_Emotions_using_Naturalistic_Stimuli_DENS).

The 2022 preprint's nine-trial description omits the two non-emotional trials; its stimulus prose says 15 while its table and subsequent recording description indicate 16. Prefer the final published explanation above. Its 465 extracted emotional-event arrays are a derivative analysis collection, not a count of original trials. [2022 preprint](https://arxiv.org/html/2211.02637).

## Target and timing contract

| Proposed object | Supported interpretation | Qualification still required |
| --- | --- | --- |
| Whole-clip valence | One individual's retrospective pleasantness rating after a clip | Identify actual stored field, encoding, missing values, scale transformation, and one-to-one trial join without reading rating values |
| Assigned clip category | Stimulus-selection/elicitation metadata | Authenticate category-to-material map; do not substitute for individual experienced valence |
| Click-linked emotion | Recalled category associated with an online click | Authenticate stored click identifiers, allowed categories, and per-trial association |
| Click timestamp | Participant response marker | Establish units, clock origin, alignment with EEG, recording interruption handling, and precision |
| Stimulus interval | Intended 60-second clip presentation | Verify actual onset and offset markers; do not derive an offset solely by adding the nominal duration |
| Practice/main role | Distinct phases in the documented experiment | Authenticate explicit stored phase markers or a supported execution map; do not classify phases by row order alone |

The release declares BIDS 1.4. The corresponding standard specifies event `onset` and `duration` in seconds, with an optional `sample` column. This is the **standard's intended contract**, not evidence that this release exports conforming values. Custom `trial_type` meanings and additional fields need an author codebook or an independently authenticated event mapping. [BIDS 1.4.1 task-event specification](https://bids-specification.readthedocs.io/en/v1.4.1/04-modality-specific-files/05-task-events.html).

There is no basis here for treating click categories as continuous valence trajectories, for shifting every click by a universal delay, for interpreting the seven-second derivative window as the full clip, or for automatically pooling DENS 1–9 valence with FACED item-10 0–7 valence. These would require new explicit contracts.

## Author-hosted material inventory

The paper-linked AFDI project, [OSF m8dwz](https://osf.io/m8dwz/), has a public [Films component tgcj8](https://osf.io/tgcj8/). Its anonymous [file-inventory API](https://api.osf.io/v2/nodes/tgcj8/files/osfstorage/?page%5Bsize%5D=100) returned **70 entries: 69 film files and `FileNameFormat.txt`**. The text file defines the naming form as `<Stimulus-id>-<Name>`; its current version is 1 and its inventory SHA-256 is `b7af6c73184d6a2afb9426caad16a9d368974db4a053db074a94d740ee25825f`. [Filename documentation](https://osf.io/342tn/).

The checked inventory provides material names, version numbers, byte sizes, and SHA-256 metadata for individual films. The 16 IDs in the EEG paper's emotional-material table have corresponding inventory prefixes: `10`, `26`, `40`, `51`, `54`, `67`, `92`, `98`, `109`, `113`, `152`, `198`, `199`, `201`, `210`, and `214`. This offers a plausible catalogue bridge. It **does not authenticate the exact footage played in the EEG experiment**, a behavior-file identifier mapping, neutral-material identities, practice-material selection, or playback latency. No film bodies or stimulus-validation rating tables were downloaded.

The 69-film catalogue is the wider validated stimulus collection. It should not be described as 69 EEG-recorded materials. Catalogue names also do not establish independent source-title families: repeated titles, compilations, and abbreviated names require qualification before material reservation.

## Missing execution/export documentation and access limits

The release [README](https://raw.githubusercontent.com/OpenNeuroDatasets/ds003751/1.0.6/README) describes code and stimuli directories. The preceding pinned tree audit found neither directory nor a behavior-column JSON codebook. The [dataset description](https://raw.githubusercontent.com/OpenNeuroDatasets/ds003751/1.0.6/dataset_description.json) identifies snapshot 1.0.6 and the 2023 paper, but does not supply custom event-level definitions. Its CC0 statement also differs from the README's ODbL/DBCL statement; this review does not resolve reuse terms.

The OSF root had an empty storage inventory, no wiki entries, and two public children: Films and an AFDI manuscript component (`ard6t`). The latter listed four manuscript/supplement PDFs, not experiment scripts. Targeted public GitHub repository-metadata searches for `DENS Mishra`, `DENS Asif`, and `AFDI emotion` found no author execution/export codebook. This bounded search cannot prove no such code exists elsewhere.

The browser search tool could read the primary manuscript copies and release README, but could not retrieve several publisher/OSF/API pages. Anonymous read-only PowerShell requests successfully retrieved OSF metadata and the filename-format document. Nothing was obtained by contacting authors or submitting access applications.

The next qualification should use version-bound event/schema evidence to identify phases, actual trial intervals, clock units, material keys, and behavior joins while keeping numerical outcomes sealed. Published methods resolve the intended nine/11 and fifteen/16 counts; **actual usable participant/trial counts remain unverified**, and DENS is not yet fit-ready.
