# Participant and session transfer development protocol

This is an exploratory follow-up to the fixed-duration transformer controls, not an approved change to the paper's research question. Preserve the current manuscript and obtain the author's approval before adopting a new question.

Use the existing verified common14 SEED-IV cache: all 1,080 native trials, the first ten four-second windows, absolute log-bandpower and the neutral / negative / positive target. The previous five participant folds remain fixed: nine training, three validation and three test participants. The corpus has already been inspected and is development evidence.

For each source session, fit only that session's 216 trials from training participants and select using only that session's 72 trials from validation participants. Fit normalization only to source training windows. Select once, then evaluate on each of the three sessions of the same three held-out participants (72 trials per test session). Other-session recordings and labels never enter normalization, training or checkpoint selection. No unlabeled test-participant calibration is allowed.

Compare the unchanged 19,079-parameter mean MLP and 19,075-parameter temporal transformer from the previous protocol. Three source sessions, three seeds (42/43/44) and five folds yield **90 neural fits and 270 session evaluations**. Keep the preceding 600-update AdamW budget, batch 60, three balanced coarse classes, learning rate 0.001, weight decay 0.01, clip 1 and validation every 25 steps. Select coarse three-class balanced accuracy, then balanced log loss, then earliest tie. No AMP, augmentation, schedule or early stopping. Across sources and architectures, pair the participant / original emotion / within-emotion trial-rank draw stream and model initialization for each seed/fold; actual videos differ.

Report all nine source/test cells. Compare the same-session diagonal with two fixed cyclic different-source assignments: source=(test % 3)+1 and source=((test+1)%3)+1. Each assignment evaluates every original trial once per seed. Average the two direction scores; do not ensemble probabilities or treat repeated evaluation of the same trial as independent observations. This makes the target trials identical in diagonal versus different-session comparisons.

Fit prefix-mean bandpower and scalar full-trial duration multinomial logistic controls with the same source access and folds. Choose C from 0.01/0.1/1/10 using source validation, balanced class weights and a training-only scaler: 120 candidates and 30 selected models. Duration is a diagnostic, not a proposed deployed classifier.

Primary metric is three-class trial balanced accuracy on all 1,080 trials; secondary binary balanced accuracy conditions negative/positive probabilities on the same 810 nonneutral trials. Use 10,000 paired participant-block percentile draws with seed 20261006, averaging seeds and directions inside each draw. Report exploratory, unadjusted intervals conditional on these fixed folds; seeds and repeated source directions are not additional participants.

The [SEED-IV authors' design](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) assigns different film sets to the three sessions. We identify clips by (session, trial position) from that design; original video hashes are unavailable here. Changing sessions changes recording conditions and stimulus material together. A performance change supports session robustness concerns, not a causal claim that shared video identity alone explains previous performance. See also the [EMBC 2021 stimulus-generalization study](https://www.paperhost.org/proceedings/embs/EMBC21/files/1565.pdf).

Filtering still uses the complete original trial before taking its fixed prefix; this is an offline experiment. It does not establish strictly causal 40-second acquisition, transfer to an unseen corpus, or improvement from pooled three-dataset training.

```powershell
conda activate pytorch
python scripts/session_stimulus_controls.py plan --plan ../publication_runs/session_stimulus_plan_2026-10-06.json
python scripts/session_stimulus_controls.py run --cache ../publication_runs/cache_temporal_native_seediv --output ../publication_runs/session_stimulus_seediv --plan ../publication_runs/session_stimulus_plan_2026-10-06.json --device cuda
python scripts/session_stimulus_controls.py verify --cache ../publication_runs/cache_temporal_native_seediv --output ../publication_runs/session_stimulus_seediv --device cuda
```

Commit the declaration before fitting. Verification binds source and cache hashes, confirms participant/source access and paired exposure, replays all selected neural checkpoints and linear coefficients, checks complete trial coverage and recomputes aggregates and bootstrap intervals. Heavy caches and checkpoints remain local.
