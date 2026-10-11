# DENS bounded metadata qualification

12 October 2026, Asia/Karachi. Authorized by the author's metadata/timing request. No model fitting, label selection or research-question adoption.

Pin ds003751 snapshot 1.0.6 at 5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c. Inspect public file inventory, documentation, annotation schemas and acquisition sidecars. Read only TSV headers initially. Never decode behavioral rating values or click-emotion annotations.

A bounded recording-header probe may download at most three SET files (sub-mit003, sub-mit081, sub-mit108), at most 12 MiB combined, to an ignored private working directory. Do not request FDT waveform bodies. Authenticate each SET with the pinned Git-annex MD5E length and digest before parsing; MD5 is an integrity binding, not a cryptographic provenance guarantee.

A selective MATLAB-v5 parser may decode only acquisition scalars (nbchan, trials, pnts, srate, xmin, xmax), reference text, external-data filename and channel labels. For event/urevent structures, expose only field names, shape and count initially, without event values. All other fields and arrays are skipped. A general MATLAB loader is prohibited. Reject embedded numerical EEG.data instead of decoding it. Whole compressed metadata containers may be inflated under a 64-MiB bound; acquisition/data/outcome values outside the allowlist remain undecoded. Record file-body transfer separately from numerical waveform/outcome access.

After headers are known, any selective non-outcome marker/timing pass must be declared separately with exact allowed columns before access. Unknown timestamp units, undocumented trigger meanings or unqualified material joins cause a hold; do not infer trial endpoints from clicks. Reserve participant/material roles before any individual target access. Existing FACED and EmoEEG-MC reserves remain sealed. No external author message is authorized by this qualification.

