# Source-only preprocessing diagnostic

All declared fits and source-only candidate replays are complete. These small reused validation panels are development evidence; no test score or paper pivot is claimed.

| Dataset | Group | EEGNet recipe | Selected LR/step/BN | Training BA | Unseen-validation BA/loss | Familiar-validation BA/loss |
|---|---:|---|---|---:|---|---|
| DEAP | 1 | common14, source_channel, 40s, baseline=False | 0.0003/200/ema | 74.15% | 44.31%/0.6816 | 52.73%/0.6699 |
| DEAP | 1 | common14, source_channel, 4s, baseline=False | 0.0003/600/ema | 55.10% | 51.96%/0.7080 | 59.39%/0.6775 |
| DEAP | 1 | common14, source_channel, 4s, baseline=True | 0.0003/200/ema | 52.30% | 48.63%/0.7009 | 54.09%/0.6893 |
| DEAP | 1 | common14, trial_zscore, 40s, baseline=False | 0.0003/200/ema | 72.41% | 51.37%/0.6881 | 56.52%/0.6932 |
| DEAP | 1 | common14, trial_zscore, 4s, baseline=False | 0.0003/200/ema | 52.37% | 50.00%/0.6960 | 50.00%/0.6939 |
| DEAP | 1 | native, source_channel, 40s, baseline=False | 0.0003/200/ema | 74.57% | 56.08%/0.7277 | 58.79%/0.6851 |
| DEAP | 1 | native, source_channel, 4s, baseline=False | 0.0003/600/ema | 54.41% | 59.41%/0.6773 | 61.82%/0.6691 |
| DEAP | 1 | native, source_channel, 4s, baseline=True | 0.0003/200/ema | 51.17% | 58.63%/0.6997 | 56.67%/0.6906 |
| DEAP | 1 | native, trial_zscore, 40s, baseline=False | 0.0003/200/ema | 77.66% | 53.92%/0.6932 | 59.24%/0.6924 |
| DEAP | 1 | native, trial_zscore, 4s, baseline=False | 0.0003/200/ema | 50.45% | 46.67%/0.6934 | 50.00%/0.6935 |
| DEAP | 2 | common14, source_channel, 40s, baseline=False | 0.0003/200/source_population | 70.98% | 53.24%/0.6690 | 58.00%/0.6751 |
| DEAP | 2 | common14, source_channel, 4s, baseline=False | 0.0003/200/ema | 50.08% | 50.00%/0.6932 | 50.00%/0.6931 |
| DEAP | 2 | common14, source_channel, 4s, baseline=True | 0.001/200/ema | 54.52% | 50.00%/0.6871 | 50.00%/0.6965 |
| DEAP | 2 | common14, trial_zscore, 40s, baseline=False | 0.0003/200/ema | 79.64% | 49.60%/0.6902 | 52.65%/0.6893 |
| DEAP | 2 | common14, trial_zscore, 4s, baseline=False | 0.001/200/source_population | 54.89% | 46.96%/0.6919 | 53.01%/0.6933 |
| DEAP | 2 | native, source_channel, 40s, baseline=False | 0.0003/200/ema | 80.56% | 57.09%/0.6725 | 50.10%/0.6863 |
| DEAP | 2 | native, source_channel, 4s, baseline=False | 0.0003/600/ema | 55.61% | 53.85%/0.6925 | 47.09%/0.6924 |
| DEAP | 2 | native, source_channel, 4s, baseline=True | 0.0003/600/source_population | 58.06% | 42.91%/0.6842 | 45.48%/0.6785 |
| DEAP | 2 | native, trial_zscore, 40s, baseline=False | 0.0003/200/ema | 68.29% | 60.73%/0.6898 | 51.61%/0.7173 |
| DEAP | 2 | native, trial_zscore, 4s, baseline=False | 0.001/600/source_population | 59.51% | 46.76%/0.6915 | 54.73%/0.6896 |
| SEEDIV | 1 | common14, source_channel, 40s, baseline=False | 0.0003/200/source_population | 74.69% | 27.78%/1.2078 | 38.89%/1.1021 |
| SEEDIV | 1 | common14, source_channel, 4s, baseline=False | 0.0003/200/source_population | 36.42% | 33.33%/1.1015 | 33.33%/1.0996 |
| SEEDIV | 1 | common14, trial_zscore, 40s, baseline=False | 0.0003/200/ema | 96.30% | 0.00%/1.1488 | 62.96%/1.0501 |
| SEEDIV | 1 | common14, trial_zscore, 4s, baseline=False | 0.001/200/source_population | 60.49% | 27.78%/1.0976 | 35.19%/1.0985 |
| SEEDIV | 1 | native, source_channel, 40s, baseline=False | 0.001/600/source_population | 92.59% | 33.33%/1.2899 | 33.33%/1.0396 |
| SEEDIV | 1 | native, source_channel, 4s, baseline=False | 0.0003/600/source_population | 36.42% | 33.33%/1.0991 | 37.04%/1.0919 |
| SEEDIV | 1 | native, trial_zscore, 40s, baseline=False | 0.001/200/ema | 97.53% | 27.78%/1.1909 | 61.11%/0.9363 |
| SEEDIV | 1 | native, trial_zscore, 4s, baseline=False | 0.0003/600/source_population | 55.56% | 27.78%/1.0907 | 40.74%/1.0926 |
| SEEDIV | 2 | common14, source_channel, 40s, baseline=False | 0.0003/200/ema | 80.25% | 33.33%/1.1779 | 40.74%/1.0729 |
| SEEDIV | 2 | common14, source_channel, 4s, baseline=False | 0.0003/200/source_population | 46.91% | 38.89%/1.0939 | 27.78%/1.1017 |
| SEEDIV | 2 | common14, trial_zscore, 40s, baseline=False | 0.001/200/ema | 91.98% | 38.89%/1.1231 | 55.56%/1.0144 |
| SEEDIV | 2 | common14, trial_zscore, 4s, baseline=False | 0.001/200/source_population | 66.67% | 38.89%/1.0938 | 50.00%/1.0929 |
| SEEDIV | 2 | native, source_channel, 40s, baseline=False | 0.0003/200/ema | 86.42% | 33.33%/1.1238 | 48.15%/1.0276 |
| SEEDIV | 2 | native, source_channel, 4s, baseline=False | 0.0003/200/source_population | 48.15% | 61.11%/1.0935 | 25.93%/1.1012 |
| SEEDIV | 2 | native, trial_zscore, 40s, baseline=False | 0.001/200/ema | 96.91% | 44.44%/1.0931 | 68.52%/0.8530 |
| SEEDIV | 2 | native, trial_zscore, 4s, baseline=False | 0.001/600/ema | 53.09% | 44.44%/1.0931 | 35.19%/1.0957 |

Complete classical comparisons and every candidate/curve are retained in [summary.json](summary.json) and the per-fit records.

One initialization, fixed first folds/session/rotation, small reused validation panels and offline full-trial context limit interpretation. Montage changes parameter count; short-input models change head size and consumed training samples. Baseline controls are explicitly local adaptations. Fresh-model replay does not prove optimization convergence or authenticate first-party DEAP signals. A main-question change still requires a concrete proposal, author approval and a fresh manuscript archive.
