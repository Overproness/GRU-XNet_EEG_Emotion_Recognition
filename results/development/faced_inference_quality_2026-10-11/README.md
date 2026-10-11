# FACED fixed quality fallback

The [policy](../../../docs/publication/FACED_Inference_Quality_Policy_2026-10-11.md) defines trial-level EEG availability from the existing raw/processed engineering QC rules. Failed EEG is bypassed: joint prediction uses the source-fitted context model; EEG-only prediction uses the exact source global median. Every participant, trial and outcome is retained. Failed source QC still holds future fitting.

The preceding `faced_artifact_control_2026-10-11` experiment remains immutable. This is a separately frozen, post-hoc cached sensitivity with unchanged alphas and zero new fits, original-file reads, waveform-feature reads or rating decoding. Confirmation participants/films and manuscript/question remain unchanged. The declaration's signal audit precedes prediction replay.

```powershell
python -m unittest discover -s tests -p test_faced_inference_quality.py
python scripts/run_faced_inference_quality.py declare
python scripts/run_faced_inference_quality.py execute
python scripts/run_faced_inference_quality.py verify
```

Thirteen synthetic policy/source-boundary tests pass, including poisoned unavailable-EEG callbacks and manufactured negative-rail channels at both native rates. Execution and independent analysis receipts will record actual cached prediction effects; this packet does not authorize additional fitting or a paper pivot.
