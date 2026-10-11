# DENS numeric-timing digest addendum

Declared before the addendum's network access on 12 October 2026.

The initial selective annotation receipt uses ordered JSON numeric-field hashes.
The separately qualified SET metadata uses little-endian float64 byte hashes.
This addendum repeats **only** the same three previously declared event TSV
prefix reads (`sub-mit003`, `sub-mit081`, `sub-mit108`) to express their allowed
numeric fields in the same digest encoding. It does not reread behaviour files.

Each complete transient TSV body is authenticated against the pinned Git blob
SHA1 and length. Only onset and duration bytes are decoded. Remaining event
semantic bytes are transferred for authentication but never decoded, printed,
retained, or used. The original declaration and receipts are preserved.

SHA256 digests are computed over the ordered field arrays encoded as IEEE 754
little-endian float64 bytes (`struct.pack` with the `<d` format per value). No
timing sequence is published. Event onset strings in the initial receipt suggest
six decimal places of export precision; exact agreement with a SET digest can
therefore require the SET values to be rounded to six decimal places before
encoding. Rounding compatibility does not establish trial-marker semantics or
physical playback timing. No event semantic labels, ratings, waveforms, fitting,
role changes, or research-question changes are authorized by this addendum.
