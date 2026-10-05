# GRU-XNet publication and literature review

Research cutoff: **5 October 2026**. Prepared from `report.tex`, the working implementation, the GitHub export, saved evaluations, and targeted online searches.

**Implementation update:** raw data were supplied after this initial review. The new audited pipeline and development runs are documented in [the implementation status](GRU-XNet_Implementation_Status_2026-10-05.md) and [PUBLICATION.md](../../PUBLICATION.md). Historical findings below describe the course-project snapshot, not the new experiments.

**Assessment:** GRU-XNet is a useful starting point for a publication, but the current manuscript needs new validation and a narrower novelty claim. CNN–BiGRU–attention EEG models already existed before this project. The more promising contribution is a reproducible study of heterogeneous datasets, electrode layouts, label definitions, and augmentation under genuinely independent evaluation. Publication suitability remains dependent on those results and the intended venue.

## 1. What this workspace contains

| Location | Role established from the files |
| --- | --- |
| [report.tex](../paper_archive/2026-10-05-pre-exploration/report.tex) | IEEE conference manuscript, titled “GRU-XNet: A BiGRU-Driven Self-Attentive Network for Cross-Dataset EEG Emotion Classification.” References are embedded in the file. |
| OurApproach/ (local workspace evidence: `OurApproach/`) | Working data-loading and augmentation infrastructure, with dataset adapters, PyTorch loaders, documentation, and utility scripts. |
| OurApproach/CBSAtt/ (local workspace evidence: `OurApproach/CBSAtt/`) | Working GRU-XNet implementation. The directory retains the CBSAtt name, but the model and README describe the BiGRU modification. Includes the Kaggle notebook, outputs, and baseline copies. |
| [GRU-XNet_EEG_Emotion_Recognition/](../..) | Git repository prepared for public release: model, training code, augmentation modules, baseline implementations, evaluations, and exported paper. Latest local commits are dated 17 January 2026. |
| CBSAtt_og/ (local workspace evidence: `CBSAtt_og/`) | Reproduction of the original CNN–BiLSTM–multi-head attention method. |
| AccurateEEG/ (local workspace evidence: `AccurateEEG/`) | LSTM/BiLSTM baseline implementation, with saved models and metrics. |
| EffectiveConnectivity/ (local workspace evidence: `EffectiveConnectivity/`) | Effective-connectivity and ensemble deep-learning baseline implementation. |
| WASIF/carnn.ipynb (local workspace evidence: `WASIF/carnn.ipynb`) | CA-CRNN-related experimental notebook. |
| dl papers/ (local workspace evidence: `dl%20papers/`) | Downloaded reference papers. |

The working and public copies have identical `test_results.json` and `training_history.json` contents, but their core source files are not byte-identical. The Kaggle notebook also contains its own implementation. These are related versions, and the precise code used for each published number needs to be identified.

At the time of the initial review, the accessible snapshot did **not** contain the raw `datasets/` or `OurApproach/augmented_datasets/` directories referenced by the documentation. That review inspected the code and saved results rather than rerunning training. Raw originals and historical augmented arrays have since been extracted under `emotion-recognition-eeg-datasets/`; historical augmentation lineage remains unresolved and those arrays are excluded from the corrected experiments.

## 2. What GRU-XNet actually does

The intended pipeline is:

1. Load DEAP, GAMEEMO, and SEED-IV and produce binary labels.
2. Expand the datasets offline to approximately two synthetic examples per original example.
3. Calculate channel-wise STFT magnitudes and interpolate them to a common 129 × 126 input shape.
4. Handle channel-count differences through padding.
5. Process each channel with a separate three-block CNN.
6. Feed the temporal CNN features through a two-layer BiGRU and four-head self-attention.
7. Average over time and classify with a dense head.

The saved configuration selects the **dynamic** model. In that implementation, three pooling stages reduce the common input to 16 frequency positions and 15 temporal positions. Features from the channel CNNs are concatenated at each temporal position and sent directly into the BiGRU. This is different from the manuscript’s flattened fusion vector followed by a 256-dimensional projection.

The separate **standard** implementation repeats one fused vector ten times before the GRU. That does not preserve a measured EEG time sequence. The manuscript should describe the variant that actually produced the reported results.

The reported data counts are 1,280 DEAP trials, 216 SEED-IV trials, and 13,216 GAMEEMO windows, with 29,418 synthetic examples. These sum to 44,130 examples. GAMEEMO accounts for approximately **89.83%** of the original sample count. A pooled score can therefore largely reflect GAMEEMO performance unless dataset-specific results are reported.

The working documentation explicitly reports **three loaded SEED-IV subjects**, explaining the 216 trials: 3 subjects × 3 sessions × 24 trials. The manuscript mentions the benchmark’s 15 subjects without explaining this subset. See IMPLEMENTATION_SUMMARY.md (local workspace evidence: `OurApproach/IMPLEMENTATION_SUMMARY.md`), around line 173.

## 3. Closest architectural precedents

These are the most important papers for determining what the architecture contributes. Their reported percentages are not a common leaderboard: the inputs, label spaces, training data, and evaluation protocols differ.

| Paper and publication timing | Verified relationship to GRU-XNet | Implication |
| --- | --- | --- |
| [Houssein et al., TFCNN-BiGRU with self-attention mechanism for automatic human emotion recognition using multi-channel EEG data](https://link.springer.com/article/10.1007/s10586-024-04590-5), **19 July 2024**, *Cluster Computing*. | Combines CWT scalograms, 2D CNN, BiGRU, and self-attention; evaluates SEED and GAMEEMO. The abstract reports 93.1%, 96.2%, and 92.9% for its two-, three-, and four-class tasks. | Essential missing prior work. The general CNN + BiGRU + self-attention combination cannot be claimed as new. GRU-XNet differs in STFT input and its heterogeneous training pipeline. |
| [Zhou et al., CBSAtt: a CNN-BiLSTM network with multi-head self-attention for EEG emotion recognition](https://link.springer.com/article/10.1007/s11760-025-04708-1), **2025**, *Signal, Image and Video Processing*. | Uses STFT, an independent convolution module for each channel, BiLSTM, and multi-head self-attention. Already cited in the draft and reproduced locally. | Direct architectural parent. Replacing BiLSTM with BiGRU needs a matched accuracy/efficiency comparison; the substitution alone is a limited contribution. |
| [Huang and Deng, 4D-MRSimNet](https://www.mdpi.com/2079-9292/15/1/39), **published online 22 December 2025**, *Electronics* **2026**, 15(1), 39. | Uses DE + PSD in a topology-aware 4D representation, multi-resolution CNN with SimAM, and BiGRU with temporal attention on DEAP and DREAMER. | Recent close precedent for the CNN–BiGRU–attention argument. Its input representation and attention differ. The paper identifies subject-independent recognition as future work, so its high accuracy is not evidence of unseen-subject or unseen-dataset performance. |
| [Bansiwal et al., A hybrid CNN-BiLSTM model with Multi-Head Attention mechanism for Game-based Emotion Recognition from multi-channel EEG signals](https://www.sciencedirect.com/science/article/abs/pii/S1746809426015387), **available online by the review date**, assigned to the **15 October 2026** issue of *Biomedical Signal Processing and Control*. | Uses DE features, CNN, BiLSTM, four-head attention, and a stated strict LOSO evaluation on GAMEEMO. The abstract reports 88.39% overall four-class accuracy. | Particularly relevant to the gaming dataset. Its highlighted 96.42% is the highest result for the joy category, and should not be presented as overall binary accuracy. Its issue date is later than this review cutoff, but the article is already available online. |

**Inference from these sources:** architectural novelty is weak if framed only as “CNN + BiGRU + attention.” A defensible distinction would need to come from how datasets are aligned, how augmentation is selected and validated, or what an independent evaluation reveals.

## 4. Research that directly affects the cross-dataset claim

| Paper and timing | What it contributes | Why it matters here |
| --- | --- | --- |
| [Enhanced cross-dataset electroencephalogram-based emotion recognition using unsupervised domain adaptation](https://www.sciencedirect.com/science/article/pii/S0010482524014793), *Computers in Biology and Medicine*, **January 2025**. | Reliability-guided target-sample selection and confidence-aware test-time augmentation. Reports **67.44% DEAP → SEED** and **59.68% SEED → DEAP**. | A concrete example of training on one dataset and testing on another. These results cannot be compared directly with GRU-XNet’s pooled random split. This method uses unlabeled target data, which also distinguishes its adaptation setting from zero-shot transfer. |
| [Learning unified brain region representation for cross-dataset EEG-based emotion recognition — UBRRL](https://www.sciencedirect.com/science/article/pii/S1746809426005306), *Biomedical Signal Processing and Control*, **1 September 2026** issue. | Aligns different electrode layouts through brain-region representations and adjusts feature distributions. Evaluates SEED, SEED-IV, DEAP, and HBUED. | One of the most important new competitors to the heterogeneous-channel contribution. Padding to the same channel count does not solve the anatomical correspondence problem it addresses. |
| [Cognitive-Inspired Hierarchical Learning Framework for Cross-Dataset EEG-Based Emotion Recognition — CIHL](https://www.mdpi.com/2079-9292/15/10/1971), *Electronics*, **7 May 2026**. | Handles inconsistent label spaces through shared coarse emotional categories, finer dataset-specific learning, progressive attention, and hierarchical label smoothing. | Directly relevant to merging DEAP dimensional ratings, SEED-IV discrete categories, and GAMEEMO labels. Binary outputs do not automatically give all datasets equivalent semantics. |
| [An Enhanced Source-Free Unsupervised Domain Adaptation Framework for Cross-Dataset EEG Emotion Recognition via Predictive Coding and Test-Time Training](https://arxiv.org/abs/2606.28202), **26 June 2026**, arXiv preprint. | Combines predictive self-supervised pretraining, source-free adaptation, and selective test-time training on DEAP, SEED, and DREAMER. Reports 69.56% on SEED and 63.03% on DREAMER when trained on DEAP. | Shows the continuing focus on adapting to an external target domain. Label as a preprint; target-data access and online adaptation must be disclosed when comparing it with a supervised model. |

The current manuscript’s comparison table puts reference methods trained on A and tested on B next to GRU-XNet trained and tested on mixtures containing all three datasets. This does not demonstrate superior cross-dataset transfer. Either repeat the same transfer tasks for every model, or present the protocols in separate tables with appropriately limited conclusions.

## 5. Other advances to be aware of

| Advance | Relevant primary source | Practical consequence for GRU-XNet |
| --- | --- | --- |
| Pretrained EEG representations that encode electrode position | [REVE: A Foundation Model for EEG](https://brain-bzh.github.io/reve/), **NeurIPS 2025**; code and dataset release announced **March 2026**. | REVE uses positional encoding and masked pretraining across heterogeneous recordings, and supports varied electrode arrangements. A pretrained model is now a useful additional baseline if resources permit. It is unnecessary to train a foundation model from scratch for this paper. |
| Self-supervised pretraining specifically for emotion transfer | [Masked Generative-Contrastive Representation Learning for Cross-Dataset EEG-Based Emotion Recognition — MGCRL](https://arxiv.org/abs/2607.04139), **5 July 2026**, revised **5 September 2026**, arXiv preprint. | Region-aware graph encoding, masked JEPA-style learning, and contrastive training address channel heterogeneity and generalization. Its experiments pretrain on FACED and fine-tune on SEED-series datasets; this is a different protocol from testing on a completely untouched dataset. |
| Compact models distilled from heterogeneous EEG pretraining | [BRIDGE-EEG](https://arxiv.org/abs/2609.12218), **10 September 2026**, arXiv preprint, journal submission. | Particularly close to the data-integration angle: it maps recordings with different montages and sampling rates to a 62-channel time-frequency representation and distills a pretrained teacher into smaller models. Efficiency and data harmonization are active research topics. |
| State-space temporal models and explicit subject alignment | [State Mamba](https://ojs.aaai.org/index.php/AAAI/article/view/38843), **AAAI 2026**, proceedings published **14 March 2026**. | Models coupled spatial/temporal transitions with Mamba and uses self-supervised alignment tasks; evaluates FACED, DEAP, and ISRUC. A modern temporal-model comparison is useful, but does not replace independent evaluation. |
| Frequency-specific temporal architecture design | [Frequency-Aware Neural Architecture Search with Bidirectional Mamba for EEG Emotion Recognition](https://papers.miccai.org/miccai-2026/0397-Paper0594.html), **MICCAI 2026**, accepted open-access proceedings version available at the cutoff. | Searches separate frequency-band architectures with bidirectional Mamba; reports cross-subject experiments on DEAP and DREAMER. Relevant to claims about temporal modeling and frequency information. |
| Adaptive electrode graphs with contrastive learning | [AST-CLNet](https://www.sciencedirect.com/science/article/pii/S1746809426008566), *Biomedical Signal Processing and Control*, **1 August 2026** issue. | Uses adaptive spatial/temporal graph relations and contrastive learning on SEED, SEED-IV, and DEAP. The abstract distinguishes 99.71% intra-subject accuracy from 87.94% cross-subject accuracy on SEED, illustrating why protocol matters. |
| Learned augmentation integrated with training | [A self-supervised data augmentation strategy for EEG-based emotion recognition](https://www.sciencedirect.com/science/article/abs/pii/S1566253525003525), *Information Fusion*, **November 2025** issue. | Uses PSD/DE features, GAN-based augmentation, masking, self-attention, and self-supervised fine-tuning. Existing augmentation research goes beyond applying a collection of standard transforms offline. |

These sources support updating the related-work discussion around **representation alignment, label compatibility, self-supervised transfer, and independently evaluated generalization**. Merely adding a newer recurrent layer would not resolve the central evidence gaps.

## 6. What needs to be checked before submission

### A. The main score measures a pooled random split

[data_loader.py](../../data_loader.py), lines 279–307, shuffles sample indices and takes 70/15/15 slices. The saved configuration has `use_loso_cv: false`. The normal branch is not explicitly stratified, despite the manuscript’s wording.

This score measures held-out samples from the same pooled sources. It does not establish generalization to a new person, session, stimulus, device, or dataset. Describe it as a pooled evaluation until the corresponding holdout experiments have been run.

### B. Augmentation and overlapping windows create serious leakage risk

[augment_all_datasets.py](../../augmentation_pipeline/augment_all_datasets.py), lines 44–70, augments all loaded DEAP subjects together before the training loader splits examples. Its analogous GAMEEMO and SEED-IV routines follow the same overall pattern. The augmentation pipeline includes originals in its output.

As a result, originals and their synthetic derivatives can fall into different partitions. GAMEEMO additionally uses five-second windows with 50% overlap. Randomly splitting these windows can put shared recording content into train and test. Without the original data and a lineage manifest, the exact contamination of the historical split cannot be quantified, but the code does not enforce its prevention.

Recent experimental work directly examines this issue: [Lei et al., Impact of Trial-wise and Test Data Leakage on EEG-Based Emotion Classification](https://ceur-ws.org/Vol-4115/paper7.pdf), 4DMR workshop at IJCAI 2025, proceedings published **2 December 2025**, demonstrates performance inflation under contaminated DEAP protocols. This reinforces the need to regenerate evaluations from independent source units.

**Required correction:** assign original recordings to folds first, then segment and augment only the training partition. Keep validation and test examples real and independent. Fit any learned normalization, neighbor selection, generative model, or augmentation policy using training data only. Deterministic per-example transforms can be applied to held-out inputs where appropriate.

### C. Random subject assignments invalidate synthetic provenance

The augmentation export uses `np.random.choice(subject_ids, n_augmented)` to assign subject IDs to synthetic examples. A synthetic example’s saved ID can therefore differ from the subject that generated its EEG. This is visible for DEAP around line 69, GAMEEMO around line 205, and SEED-IV around line 298 of the same script.

Turning on the existing LOSO flag is insufficient. Record the source subject, session, trial, temporal interval, and all parent examples for every augmentation. Multi-parent operations must use parents from the training partition. Use `(dataset, subject)` identifiers because subject numbers are reused between unrelated datasets. Also select validation subjects independently; the existing LOSO branch randomly divides the remaining samples for validation.

### D. The label definitions are not consistent with the manuscript

- **SEED-IV:** [config.py](../../config.py), lines 101–106, maps both label 1 and label 3 to positive. The local loader defines label 1 as sad and label 3 as happy. This contradicts the manuscript’s mapping of only happy to positive. The saved `outputs/config.json` contains the same mapping. The exact mapping used in every historical run still needs confirmation.
- **GAMEEMO:** [augmentation_pipeline/dataset_loaders.py](../../augmentation_pipeline/dataset_loaders.py), lines 254–259, assigns G1/G2 to 0 and G3/G4 to 1 using an explicitly described **arousal heuristic** instead of parsing participant SAM ratings. The manuscript interprets the common output as negative/positive valence.
- **Neutral:** assigning neutral to negative defines a happy-versus-rest task; it should not silently be treated as equivalent to a negative-versus-positive valence task.

Use documented, compatible targets and report the label policy. Options include a consistently defined valence task, retaining separate heads for different emotion dimensions, or hierarchical/multi-task labels. CIHL provides directly relevant literature for the semantic-alignment problem discussed above.

### E. Padding handles tensor sizes, not anatomical alignment

The model indexes channel CNNs by array position. There is no electrode-name alignment or positional encoding establishing that slot i represents the same scalp location in each dataset. Moreover, a padded zero channel can produce nonzero features after biased convolutions and BatchNorm; no per-example channel mask guarantees zero contribution in a mixed batch.

Useful controlled comparisons are: padding by array position, common named electrodes, canonical montage mapping with a validity mask, and brain-region pooling. This is a potential research contribution if independently evaluated, although UBRRL and REVE must be acknowledged as relevant precedents.

### F. The paper combines statistics from different evaluations

The 95.91% result is real as a saved prediction calculation, but the plotted evaluation uses different numbers:

| Artifact | Verified result |
| --- | --- |
| [outputs/test_results.json](../../outputs/test_results.json) | **6,621** predictions and labels; recomputed accuracy **95.906962694457%**. Confusion matrix: `[[3307, 98], [173, 3043]]`. |
| [outputs/metrics_summary.json](../../outputs/metrics_summary.json) | **6,567** samples; accuracy **96.223541952185%**. Confusion matrix: `[[3274, 102], [146, 3045]]`. References an epoch-30 checkpoint. |
| `report.tex`, main result versus confusion-matrix discussion | Uses 95.91% for the principal result but the 6,567-example matrix for detailed analysis. These cannot describe one identical evaluation. |
| [outputs/training_history.json](../../outputs/training_history.json) | Last training accuracy approximately **96.69%**, last validation accuracy **96.03%**, best saved validation accuracy **96.15%**; these differ from the manuscript’s table. |

Select a single identified checkpoint, dataset manifest, and split for the main evaluation. Regenerate all metrics and figures from that run and label any additional evaluation separately.

### G. Augmentation and method descriptions need reconciliation

The default configuration enables **nine** methods: Gaussian noise, time shift, window slicing, amplitude scaling, channel dropout, frequency filtering, time-frequency augmentation, Mixup, and SMOTE. CutMix, GAN, and VAE are disabled. The manuscript repeatedly claims eleven applied techniques.

The configured contribution weights sum to 2.4 and the pipeline rescales them to 2.0. The manuscript’s beginner/intermediate/advanced percentages sum to 120%, so they are not normalized mixture shares. Under the shown defaults the normalized group shares are approximately 62.5%, 29.17%, and 8.33%.

The manuscript describes same-class Mixup; the pipeline randomly chooses both parents without a same-class restriction and then retains a hard label from the first parent. The SMOTE implementation searches flattened input signals rather than a documented learned feature space. Report the actual run configuration and justify label preservation for each transform.

The Python STFT configurations also differ from the text: GAMEEMO uses `nperseg=128`, and SEED-IV remains at 200 Hz with `nperseg=400`. Spectral magnitudes are then interpolated to the target shape. The notebook has further differences. The manuscript’s claimed uniform downsampling and preprocessing need to be traced to the executed pipeline.

### H. Report only experiments with recoverable evidence

The manuscript contains extensive ablation and sensitivity results, but I did not locate corresponding experiment configurations, checkpoints, or result files in the inspected GRU-XNet/OurApproach artifacts. This does not establish that the experiments were never performed; their evidence may exist elsewhere. Recover it or rerun them before retaining the numbers.

Reference-method reproduction tables also need matched label definitions, source data, preprocessing, and evaluation protocols. Lower reproduction accuracy alone does not demonstrate overfitting or leakage in the published methods. Such claims require an identified methodological cause.

Finally, the public Python loader imports `data_loading_pipeline` from its parent directory, but the export does not include that module in the expected location. The Kaggle notebook is a separate route. Make the documented standalone training route complete, and include the exact source/configuration used for reported runs. `report.tex` also references figure paths that are not present beside the TeX file.

## 7. Prioritized experimental plan

| Priority | Experiment or correction | Evidence it would provide |
| --- | --- | --- |
| 1 | Recover raw data, participant ratings, exact source version, and source-unit manifests; correct label definitions and SEED-IV subject accounting. | Establishes what prediction task and population are actually being studied. |
| 2 | Regenerate data partitions before augmentation and segmentation; preserve full augmentation ancestry. | Removes source overlap across partitions and makes subject-level claims auditable. |
| 3 | Evaluate subject-disjoint folds for each dataset, with subject-disjoint validation, and report each dataset separately. | Measures generalization to unseen participants and avoids a GAMEEMO-dominated pooled conclusion. |
| 4 | If keeping the cross-dataset claim, train on A+B and test on untouched C for all three target datasets; optionally add directed A → B tasks. | Measures unseen-dataset generalization. Hyperparameters must be selected without target test labels. Target-adapted experiments need a separate protocol. |
| 5 | Compare CNN-only, CNN–BiLSTM–attention, CNN–BiGRU without attention, and full GRU-XNet using the same folds and seeds. | Isolates the GRU substitution, temporal processing, and attention contribution. |
| 6 | Compare no augmentation, simple noise/crop augmentation, and the justified full policy at matched sample budgets. | Tests whether the augmentation pipeline improves independent generalization rather than only sample-level interpolation. |
| 7 | Compare channel padding with anatomical mapping/masking and region pooling. | Tests the strongest potential method contribution for heterogeneous hardware. |
| 8 | Add a compact established baseline and, if feasible, one current pretrained or state-space baseline. | Places the contribution in a broader 2026 context without requiring every new model to be reproduced. |

Report accuracy, balanced accuracy, macro-F1, per-class recall, and per-dataset confusion matrices. Use several seeds and uncertainty at the independent subject/recording level; thousands of correlated windows do not create thousands of independent participants. Report parameters, inference time including preprocessing, memory use, and fixed hardware if making efficiency claims.

Maintain a clear distinction between:

- **Pooled evaluation:** all datasets represented during training and testing.
- **Cross-subject evaluation:** test participants absent from training.
- **Cross-session evaluation:** test sessions absent from training.
- **Cross-dataset generalization:** the target dataset is absent from training and adaptation.
- **Domain adaptation or fine-tuning:** specified target-domain data are used during adaptation.

The [EEGain evaluation framework paper](https://arxiv.org/abs/2505.18175), **14 May 2025**, arXiv preprint, is a useful reference for standardizing the evaluation discussion. It reviews 216 papers and provides common dataset loaders, splits, metrics, and baseline evaluation. Its existence also means a generic “first unified EEG evaluation pipeline” claim would need substantial qualification.

## 8. Recommended publication framing

**For a revision based only on recoverable existing experiments:** describe the method as a CBSAtt-inspired CNN–BiGRU–attention model for *multi-dataset* EEG classification. Restrict conclusions to the measured protocol, remove unsupported claims, and present the dataset/label limitations explicitly. That would improve accuracy of the manuscript, but would leave a limited novelty case.

**For a stronger new submission:** make the central question whether biologically aligned channels and a justified training-only augmentation policy improve subject- and dataset-independent emotion recognition across different recording systems. Retain GRU-XNet as the backbone and compare it with a matched BiLSTM alternative. This turns the existing implementation into a testable research study.

Possible revised title before transfer results exist: **“GRU-XNet: A CNN–BiGRU Attention Model for Multi-Dataset EEG Emotion Classification.”** Restore “Cross-Dataset” only when the paper reports the corresponding independent transfer experiments.

There is no need to replace GRU simply because Mamba and foundation models now exist. The first investment should be correct labels, independent partitions, matched baselines, and trustworthy artifacts. A newer model family is useful only if it answers the paper’s research question.

## 9. Scope and confidence of this review

This is a targeted literature update, not an exhaustive systematic review or a guarantee of novelty. The search covered close CNN/GRU/attention architectures, heterogeneous EEG datasets and montages, label alignment, augmentation, domain transfer, foundation models, and state-space models, with emphasis on work available by 5 October 2026.

Technical summaries above use original publisher pages, proceedings, author project pages, and author preprints. Some papers were available only through publisher abstracts or indexed article text, so fine implementation and evaluation details require full-text checking before reproducing them. Preprints and accepted conference versions are explicitly labeled; online availability is distinguished from issue dates where necessary.

Local findings distinguish directly observed code/metrics from unresolved historical provenance. I recomputed the saved 95.91% accuracy and confusion counts, compared the working/exported artifacts, and checked the relevant loader, model, augmentation, configuration, and notebook code. I did not retrain models, alter `report.tex`, or change the implementation.
