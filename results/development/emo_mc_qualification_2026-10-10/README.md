# EmoEEG-MC source qualification and reservation

[Qualification](qualification.json) records an outcome-blind check of OpenNeuro
ds005540 release 1.0.7. [Reservation](reservation.json) allocates 20 anonymous
participants to future confirmation, 10 to development validation and 27 to
development source construction. Sub-22 and sub-54 remain quarantined;
sub-01 has no raw EEG in this release.

Seven video spreadsheet entries and their seven numeric imagery companions are
reserved across every participant. Companion numbers identify different contents
in the two contexts. This is an identity reservation, not a registered model study
or an approved change to the manuscript's research question.

The numeric stimulus entries still require a verified mapping to raw trials and
README symbolic names. No fitting, rating inspection or waveform decoding is
allowed while that gate is unresolved. Public files contain source hashes and
technical metadata, not original EEG, rating tables, narratives or embeddings.

The source reader verifies pinned Git blobs, bounds EDF HTTP range requests and
interprets only the two identity columns of behavioural files. Mixed rating bytes
are transported transiently to check the original file hash; score fields are
never decoded, inspected or retained. Reserved participant headers are technical
metadata, so this does not claim their whole files were inaccessible.

Header checks qualify the declared calibration and structure; they do not verify
whole-file annex digests or hardware accuracy. The subsequent pilot record, if
present, explicitly identifies which complete objects were authenticated.
