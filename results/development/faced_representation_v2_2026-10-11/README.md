# FACED bounded representation study: qualified preprocessing replay

**Completed:** 980 source trials pass exact reference-byte/QC checks; 265 validation trials provide EEG and fifteen bypass EEG while retaining all 280 outcomes. All 138 fixed fits and 36,120 prediction rows complete in 648.4 seconds. Covariance-plus-context RBF gains 0.06784 MAE over its matched context but only 0.02379 over original context Ridge, below the 0.15 lead screen. All three covariance-RBF descriptive intervals include zero, and source OOF error exceeds matched context. All three joint conditions fail the numerical screen. Twelve capacity controls nearly memorize both real and shuffled source targets. No qualifying lead or paper pivot is established.

The [comparison](comparison.json), [predictions](predictions.csv), [fit manifest](model_manifest.json), [feature/access manifest](feature_manifest.json), [independent audit](independent_analysis/summary.json), [verification receipt](independent_analysis/verification.json) and [comparison figure](figures/representation_comparison.png) preserve all results. The independent checker verifies 1,245 representations, 138 source-fit states, all 36,120 predictions and 270 quality fallbacks without calling the new feature/fit/quality/prediction functions. Maximum saved-state prediction discrepancy is 2.05e-12. The focused suite passes 111 tests. The figure was visually checked; its [coordinates](figures/coordinates.json) are explicit.

This is a separate fresh execution under the [continuation declaration](../../../docs/publication/FACED_Representation_Continuation_2026-10-11.md) and unchanged [scientific protocol](../../../docs/publication/FACED_Representation_Capacity_Protocol_2026-10-11.md). The [first source-only hold](../faced_representation_2026-10-11/README.md) is preserved with zero validation trials/fits. The eight-trial source diagnosis qualifies the threading correction without targets or fitting. Zero new original ratings or protected/neighbor EEG values were decoded in this completed run.

The same seven fixed conditions and twelve source-only real/shuffled capacity controls remain capped at 138 CPU fits/two hours, with no validation-dependent hyperparameter search. Preprocessing alone uses its qualified twelve-thread default; new covariance/readout computations use two threads. Every available trial must reproduce exact preceding count/window/access/reference-feature hashes. All 980 source trials must pass before validation/fitting; the fifteen unavailable validation EEG trials retain context/source-median fallback and bypass EEG access.

```powershell
python scripts/run_faced_representation_v2.py declare
python scripts/run_faced_representation_v2.py execute
python scripts/audit_faced_representation_continuation.py
```

Confirmation participants/films, previous studies and manuscript/question remain unchanged. This is reused development, not independent preregistration or a new algorithm. All outcomes and failure receipts must be retained.
