# DENS selective material and numeric-timing metadata pass

Declared on 12 October 2026, before this pass retrieves annotation bodies.
This extends the pinned public inventory qualification. It does not open
individual ratings, fit a model, allocate outcome-dependent roles, or adopt a
new research question.

## Pinned inputs

- OpenNeuro ds003751 release 1.0.6.
- Git commit `5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c` and tree
  `1552b175dffd076bb75c50c994cd154f021bfe0a`.
- Previously retained `inventory.json` provides the exact paths, Git blob
  SHA1 digests and lengths. The receipt is preserved.
- The 34 behaviour TSV files are read for non-outcome stimulus/material keys.
- Only `sub-mit003`, `sub-mit081` and `sub-mit108` event TSV files are read for
  numeric timing. The last two were explicitly updated in release 1.0.4.

## Permitted access and parsing

An annotation file's entire body is transported into transient application
memory so its exact length and Git blob SHA1 can be authenticated before
parsing. This means rating and event-label **bytes are transferred**. They are
not decoded, printed, saved to the repository, or used numerically. This pass
does not claim zero outcome-body transfer.

For every behaviour data row, the parser executes
`line.split(b'\t', 1)[0]` and decodes only that first field, `stimuliName`.
It records row counts, ordered material keys, unique key counts, duplicate key
counts, and empty-key counts. It never runs a full CSV/TSV decoder over a data
row, and it never decodes the remainder containing ratings and clicks.
The byte-level TSV framing must match the declared eleven-column header before
the first field is decoded. Malformed physical lines fail the pass rather than
risk treating a continuation of a response field as a material key.

For the three declared event files, the parser executes
`line.split(b'\t', 2)` and decodes only the first two fields, `onset` and
`duration`. The third byte string is discarded without decoding. It may contain
emotion or response information. The parser reports numeric/missing-field
counts, min/max timing, negative/zero counts, ordering and duplicate onset
counts, and canonical ordered numeric-field hashes. No event semantic labels
are decoded. No marker meaning is inferred.

The previous EEG JSON sidecars supply `RecordingDuration` and
`SamplingFrequency`. Any comparison between TSV onset/duration and recording
duration is explicitly conditional on TSV timing being in seconds. Field names
alone do not establish units, trial endpoints, or playback timing. The root
qualification separately checks allowed SET metadata and sample latencies.

## Published outputs and boundaries

`scripts/qualify_dens_annotations.py` creates
`results/development/dens_metadata_qualification_2026-10-12/metadata_annotations.json`.
The receipt contains only allowed material keys, numeric timing aggregates and
digests, byte/hash authentication, access logs, protocol/script/input hashes,
and unresolved questions. Complete TSV bodies and individual ratings are not
published or retained as local annotation files by this script.

No EEG/FDT waveform body is requested. No behavioural outcome, click annotation
value, event semantic label, or protected confirmation outcome is decoded.
Participant/material role reservation remains necessary before individual
outcome access. A filename join is not a verified usable trial join.
