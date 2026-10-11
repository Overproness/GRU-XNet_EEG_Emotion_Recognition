# FACED fixed quality fallback

**Completed:** all 980 source trials pass the fixed policy; fifteen of 280 validation trials use EEG-unavailable fallback. All trials, people, alphas and fitted states are retained. Replay preserves 27,720 prediction rows and changes 450 candidate predictions with zero new fits or original-data/feature reads. Mean selected joint gain is 0.042155 MAE versus 0.041689 previously, below the earlier 0.15 lead threshold. All three descriptive intervals still include zero. This is post-hoc cached sensitivity, with no independent confirmation or paper pivot.

[Signal audit](signal_audit.json), [complete candidate comparison](comparison.json), [predictions](predictions.csv), [runner verification](verification.json) and [independent replay](independent_analysis/summary.json) preserve the results. The independent audit recomputes all 226,800 signal-rule checks, verifies 2,520 technical caches and independently rebuilds eighteen exact-source/fold medians and every fallback. The focused suite passes 88 tests. Maximum independent prediction/metric discrepancy is zero.

The [policy](../../../docs/publication/FACED_Inference_Quality_Policy_2026-10-11.md) defines trial-level EEG availability from the existing raw/processed engineering QC rules. Failed EEG is bypassed: joint prediction uses the source-fitted context model; EEG-only prediction uses the exact source global median. Every participant, trial and outcome is retained. Failed source QC still holds future fitting.

The preceding `faced_artifact_control_2026-10-11` experiment remains immutable. This is a separately frozen, post-hoc cached sensitivity with unchanged alphas and zero new fits, original-file reads, waveform-feature reads or rating decoding. Confirmation participants/films and manuscript/question remain unchanged. The declaration's signal audit precedes prediction replay.

```powershell
python -m unittest discover -s tests -p test_faced_inference_quality.py
python scripts/run_faced_inference_quality.py declare
python scripts/run_faced_inference_quality.py execute
python scripts/run_faced_inference_quality.py verify
```

Thirteen new synthetic policy/source-boundary tests pass, including poisoned unavailable-EEG callbacks and manufactured negative-rail channels at both native rates. This packet establishes a fixed prospective quality policy and checks its cached sensitivity; further fitting still requires a separate bounded declaration. The source gate is not bypassed by inference fallback.
