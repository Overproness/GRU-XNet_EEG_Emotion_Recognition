# Held-out tuning findings

This development study reuses the existing participants and videos. The research question and manuscript remain unchanged.

Every neural fold independently selected learning rate, duration and BatchNorm variant using equally weighted familiar-video and unseen-video source-validation balanced log loss. Test participants were excluded from fitting and selection. Context-only calibration used the same criterion.

## SEEDIV

| Model | Video exposure | Balanced accuracy | Balanced log loss |
|---|---|---:|---:|
| eegnet | exposed | 48.06% | 1.1467 |
| eegnet | unexposed | 37.10% | 1.3619 |
| eegnet_context | exposed | 97.52% | 0.2943 |
| eegnet_context | unexposed | 35.14% | 1.4126 |
| gru | exposed | 40.34% | 1.4076 |
| gru | unexposed | 36.88% | 1.5031 |
| prior | exposed | 100.00% | 0.1823 |
| prior | unexposed | 33.33% | 1.1523 |
| context_logistic | exposed | 100.00% | 0.0998 |
| context_logistic | unexposed | 33.33% | 1.1551 |

Complete paired contrasts, all four individual grouping/initialization results and conditional uncertainty are in [comparison_seediv.json](comparison_seediv.json).
The within-video rating/EEG pairing diagnostic is in [alignment_seediv.json](alignment_seediv.json).

## DEAP

| Model | Video exposure | Balanced accuracy | Balanced log loss |
|---|---|---:|---:|
| eegnet | exposed | 51.61% | 0.7721 |
| eegnet | unexposed | 50.85% | 0.7652 |
| eegnet_context | exposed | 76.09% | 0.5542 |
| eegnet_context | unexposed | 50.51% | 0.7947 |
| gru | exposed | 49.59% | 0.7120 |
| gru | unexposed | 49.11% | 0.7052 |
| prior | exposed | 77.54% | 0.4991 |
| prior | unexposed | 49.67% | 0.7062 |
| context_logistic | exposed | 78.05% | 0.5004 |
| context_logistic | unexposed | 47.50% | 0.6953 |

Complete paired contrasts, all four individual grouping/initialization results and conditional uncertainty are in [comparison_deap.json](comparison_deap.json).
The within-video rating/EEG pairing diagnostic is in [alignment_deap.json](alignment_deap.json).

These results use different raw versus STFT representations for EEGNet and GRU; they cannot isolate an architectural effect. Training exposure and the familiar-validation panel change together across arms, so the exposure contrast describes that whole procedure. SEED labels are assigned by video; its known-video prior and exchange sanity check cannot establish individual physiological emotion prediction. DEAP waveform comparison to the first-party release remains outstanding.

Checkpoint replay verifies numerical output, rather than establishing optimization convergence. All candidate outcomes, including weak fits, are retained. Most selected GRU weights were verified and deleted according to the declared storage rule; retained sentinels and all EEGNet weights permit later state replay. No successful initialization, grouping or significant contrast was selected for reporting.

A contribution or research-question change still requires a concrete review with the author and explicit approval.
