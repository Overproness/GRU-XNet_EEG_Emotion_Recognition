# Fixed FACED quality and inference policy

Declared on 11 October 2026 after the initial development experiment, before this policy's cached prediction sensitivity. This is a development-informed engineering policy, not independent preregistration, validated artifact classification or a new research contribution.

The preceding artifact-control experiment and all its source/validation predictions remain unchanged. Confirmation participants and films stay sealed. This phase reads only previously decoded technical QC/artifact caches and authorized cached prediction rows; it opens no original BDF/MAT files, waveform features, fitted model states or new rating values and performs no fitting.

## Fixed decision

Retain every declared participant, material and trial. Recompute QC from numerical technical metrics before permitting EEG inference; saved flag text alone is insufficient. A trial's EEG is unavailable if **any** channel exceeds the already fixed raw ADC-rail or stationary-count thresholds, if any existing processed/line QC rule fails, if the declared interpolation capacity is exceeded, or if EEG is numerically unavailable. This policy adds no electrode reconstruction or parameter sweep.

| Metric | Failure rule |
| --- | --- |
| Native ADC rail occupancy | Strictly greater than 0.0001 |
| Native identical-adjacent-count fraction | Strictly greater than 0.95 |
| Processed channel SD | Strictly below 0.1 µV or above 100 µV |
| Processed channel peak-to-peak | Strictly above 1,000 µV |
| Processed 49–51/1–45 Hz power ratio | Strictly above 0.10 |
| Numerics/interpolation | Nonfinite/unavailable EEG or exceeded frozen capacity |

QC requires all thirty ordered common channels, seven retained windows and no dropped windows. Missing, malformed, nonfinite or out-of-range QC metrics fail closed for review; authentication, role or join failures remain errors and are never converted into a permissive prediction fallback. Calling code must authenticate identities and files before applying this pure policy.

For an unavailable-EEG trial:

- The joint model uses the already source-fitted context model at its already selected alpha.
- EEG-only prediction uses the global median fitted from that exact source set. Source OOF diagnostics use the corresponding fold's source median.
- Context-only models and fixed source priors retain their predictions.

The EEG callback is never invoked on an unavailable trial. A context fallback assumes the experiment's declared contextual metadata are available. The source-median fallback contains only source outcomes; evaluated ratings never supply a fallback. The outcome remains in all participant/film-equal metrics. The policy conservatively forgoes the remaining good-channel information in a failed trial; it does not claim all thirty channels are physically damaged.

Before future fitting, exact source identities and all source quality metrics must pass. Source failures hold fitting and target activation for review; inference fallback does not permit dropping failed source trials, imputing training features or weakening the source gate. The existing 980-trial source set is audited under this same rule.

## Signal diagnosis and rationale

Previously scalar-checked count/QC caches show a constant negative full-scale F8 channel in all fourteen sub-030 trials and a constant negative full-scale PO3 channel in sub-100/clip-4. Their original SD is zero and rail/identical-count fractions are one. Temporal filtering leaves negligible pre-CAR variation, while CAR transfers other-channel variation into the flat channel. Thus plausible processed SD cannot authenticate the original electrode's signal. Some sub-030 Oz/PO4 intervals also contain nonconstant rail occupancy. These are original-data channel failures in the inspected intervals; the physical cause is unknown, and the audit does not diagnose an entire recording or participant.

[MNE's primary documentation](https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html) recommends inspecting raw channels early, before referencing and artifact transforms, and explains common ways to mark or interpolate bad channels. It does not prescribe our thresholds or fallback decision. Its current documentation version is 1.13.2; this policy invokes no new MNE processing and retains the previously qualified preprocessing runtime.

## Bounded cached sensitivity

Freeze the code, tests, this protocol, signal audit and preceding experiment hashes before replay. Preserve all 27,720 original candidate/prior/source-OOF prediction rows. Change only EEG-dependent predictions for unavailable trials according to the rule above. Source identities, features, fitted states, all five candidate alphas, selected alphas, priors and original target values remain unchanged. No alpha reselection is allowed even if fallback changes candidate ordering.

Report all candidate metrics, the previously selected unseen/familiar metrics, source OOF checks, trial/participant/film coverage, fallback reasons and original versus policy predictions. Keep all fifteen flagged validation trials. Recompute paired participant/film diagnostics with the original 4,000 crossed bootstrap draws and seed425401; uncertainty remains descriptive, conditional on previous development selection, six films and the same twenty validation people.

This sensitivity cannot establish independent EEG benefit, repair the original study retrospectively, trigger encoder expansion, open confirmation or adopt a new paper question. Any later fitting requires a separate bounded declaration using this reviewed policy and unchanged reservations. A question change still requires presenting findings and the user's approval, with a fresh manuscript archive beforehand.
