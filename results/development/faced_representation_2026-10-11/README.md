# FACED bounded covariance and source-capacity diagnostic

Declared implementation: [protocol](../../../docs/publication/FACED_Representation_Capacity_Protocol_2026-10-11.md). The new original-signal preparation and fitting have not launched at this checkpoint. This is reused development work, with confirmation participants/films and manuscript/question unchanged.

Seven fixed classical conditions compare a single log-covariance representation, the existing spectral features and their context combinations with matched context controls. Each has a main fit and five original source participant OOF fits in three rotations. Twelve real/shuffled-label near-interpolation capacity fits test source memorization separately. Cap: 138 CPU fits/two hours; zero neural/GPU fits and no validation-dependent hyperparameter search.

The exact original decoder/preprocessing, participant/material roles, existing quality thresholds and all prior experiments are preserved. All 980 source representations/QC must pass before validation EEG or fitting. Fifteen unavailable validation trials bypass EEG reads and retain fixed context/source-median fallback; all 280 validation outcomes remain evaluated. Original ratings are not decoded again; only previously authorized published targets are reused after the source gate. Waveforms, feature arrays and model states remain private.

```powershell
python -m pytest -q tests/test_faced_representation.py tests/test_faced_representation_runner.py tests/test_faced_inference_quality.py
python scripts/run_faced_representation.py declare
python scripts/run_faced_representation.py execute
python scripts/audit_faced_representation.py
```

All results must be reported, including failed/truncated receipts. No automatic encoder expansion, confirmation access or adoption of a different paper question follows from this experiment. The declaration is development-informed, not independent preregistration.
