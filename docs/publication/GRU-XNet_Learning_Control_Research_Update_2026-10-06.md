# Baseline authentication and overlapping prior work — 6 October 2026

The [source-only diagnostic protocol](Learning_Control_Protocol_2026-10-06.md) precedes fitting. Neither the main research question nor the manuscript has changed.

## EEGNet reference authentication

The [authors' EEGModels repository](https://github.com/vlawhern/arl-eegmodels) provides the original TensorFlow/Keras EEGNet architecture, with [the original paper](https://arxiv.org/abs/1611.08024). Source, README and license are downloaded at commit `4a512e503198db2010848813ead9afbf8cd54c97`; the [manifest](../../results/development/eegnet_author_audit_2026-10-06/download_manifest.json) includes immutable URLs and SHA-256 hashes. The source is executed unchanged for binary and three-class common14/40-second inputs. The [execution record](../../results/development/eegnet_author_audit_2026-10-06/tensorflow_execution.json) and [port comparison](../../results/development/eegnet_author_audit_2026-10-06/port_verification.json) show maximum probability errors below 1.8e-7, checked BatchNorm updates and both max-norm constraints. Dropout is disabled only in deterministic cross-framework QA; all accuracy fits keep 0.5. The port has 6,450 binary or 9,011 three-class trainable parameters.

The original TensorFlow environment cannot import against its installed NumPy 2.1.3. A NumPy 1.23.5 copy isolated inside `publication_runs/tf_numpy_compat` permits this CPU-only audit without modifying either installed environment. Neither this architecture check nor our differently balanced AdamW experiment reproduces author random streams, published optimizer trajectories or reported scores. The previous CBSAtt reference remains a local implementation control; authenticating EEGNet does not retroactively authenticate CBSAtt.

## New direct contextual-prior precedent

[Kong et al., arXiv:2610.03618v1](https://arxiv.org/html/2610.03618v1), submitted 2 October 2026, already studies familiar-video priors against physiology. Its full text specifies continuous valence/arousal regression on 15 aligned videos, five-fold held-out participants (24), and four external participants. It constructs training-participant median video/time trajectories and fixed prior-heavy fusion with an EEG–fNIRS branch. Reported MAE is 29.01 for internal fusion and 27.72 externally; the prior alone is worse by only 0.05 and 0.32, respectively. The physiological branch uses neighboring future-time context, so it is explicitly offline. These are their reported results, not reanalyses performed here.

This is direct prior work for a proposed “context dominates physiology” direction. Our individual trial classification, unseen-video arms and within-video exchange diagnostic differ, but simply adding a context-only comparator or learning a residual is not established as a new contribution. The manuscript has ACM conference template metadata; acceptance/publication status was not independently authenticated. We treat it as a public preprint. No author code or underlying recordings from this work were reproduced in this audit.

## Interpretation and next evidence

Our source-only learning study will retain both learning rates, full training/validation curves, final checkpoints and unsuccessful outcomes. Memorizing twelve real or permuted targets diagnoses capacity and optimization without demonstrating emotion information. Repeated source panels use the same cohorts and partial validation populations. Any later full out-of-fold confirmation must select hyperparameters using that outer fold's own source data; a globally selected recipe based on these panels could involve people later serving as test participants.

Neither reference authentication nor better training solves the unresolved manuscript contribution. We will present findings before proposing a pivot, then obtain author approval and archive the then-current paper immediately before an approved change.
