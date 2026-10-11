# FACED bounded representation study: qualified preprocessing replay

Separate fresh execution under the [continuation declaration](../../../docs/publication/FACED_Representation_Continuation_2026-10-11.md) and unchanged [scientific protocol](../../../docs/publication/FACED_Representation_Capacity_Protocol_2026-10-11.md). New preparation/fitting has not launched at this checkpoint. The [first source-only hold](../faced_representation_2026-10-11/README.md) is preserved; it contains zero validation trials/fits. The eight-trial source diagnosis qualifies the threading correction without targets or fitting.

The same seven fixed conditions and twelve source-only real/shuffled capacity controls remain capped at 138 CPU fits/two hours, with no validation-dependent hyperparameter search. Preprocessing alone uses its qualified twelve-thread default; new covariance/readout computations use two threads. Every available trial must reproduce exact preceding count/window/access/reference-feature hashes. All 980 source trials must pass before validation/fitting; the fifteen unavailable validation EEG trials retain context/source-median fallback and bypass EEG access.

```powershell
python scripts/run_faced_representation_v2.py declare
python scripts/run_faced_representation_v2.py execute
python scripts/audit_faced_representation_continuation.py
```

Confirmation participants/films, previous studies and manuscript/question remain unchanged. This is reused development, not independent preregistration or a new algorithm. All outcomes and failure receipts must be retained.
