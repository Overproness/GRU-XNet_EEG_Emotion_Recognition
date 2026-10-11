# Selective SET timing cross-check

Declared before timing-value access on 12 October 2026. Reuse only the three length/MD5E-bound metadata SET files already checked (003, 081, 108). No additional recording-file download.

Decode only EEG.event.latency and EEG.event.duration numerical acquisition markers, plus existing acquisition scalars. Skip type, label, description, mffkeys and every other event field: they may encode responses. Never decode EEG.data, EEG.times, behavioral targets or recalled emotion categories. Keep event timestamps out of public output; retain counts, extrema and ordered numerical digests for comparison with the separately declared TSV onset/duration pass. The read scope is non-outcome marker timing; event values are accessed in this pass, unlike the initial schema-only receipt.

Test hypotheses separately: TSV onset as seconds; TSV onset equal to MATLAB one-based latency; TSV onset equal to latency minus one; TSV onset equal to (latency minus one)/srate. Never adopt a conversion merely because it is conventional. Reject ambiguous, inconsistent or out-of-record mappings. A matched clock conversion alone does not identify film onsets/offsets, phases or physical presentation delay. No model fitting or question change.
