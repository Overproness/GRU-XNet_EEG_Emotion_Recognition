# Context, personal ratings and unified EEG learning: further prior work

Checked 6 October 2026 while the full-width controls were running. This supplements the earlier reviews; it adopts no research question and changes no fit or selection rule.

**Two consequential additions are EMOD (AAAI 2026) and a recent participant-identity/no-EEG control study.** Generic label harmonization, unified training and discovering confounding already have substantial precedents. A prospective contribution needs a narrower demonstrated result and matched comparison.

## EMOD and the original joint-learning objective

EMOD aligns discrete and continuous annotations in valence–arousal space using soft supervised contrastive learning and a spatial/temporal transformer. Its eight pretraining datasets include DEAP and SEED-IV; downstream evaluation fine-tunes on FACED, SEED-V and SEED. The reported model has 0.81 million parameters. SEED uses successive sessions for train/validation/test. Publication: 14 March 2026; arXiv versions date to November 2025. [AAAI record](https://ojs.aaai.org/index.php/AAAI/article/view/38796), [author manuscript](https://arxiv.org/pdf/2511.05863), [author repository](https://github.com/cyn4396/EMOD).

**Inference for our study:** using its released weights on DEAP/SEED-IV requires an overlap audit or source-only retraining; dataset-name overlap alone does not establish precisely which held-out recordings were pretrained. It is an essential comparator for a proposed unified-label/transformer contribution, with preprocessing, supervision and fine-tuning differences matched explicitly. Its reported scores are not directly comparable to our current common14/40-second grouping controls.

## Participant identity already has a no-EEG control precedent

Pandilova and colleagues re-evaluate qEEG on DEAP/DREAMER under several partitions. DEAP epoch-pooled valence AUC is .689 versus .493 with participants held out. A participant's training-label prevalence, without EEG, explains much of the pooled above-chance discrimination. They also test matched personalization and release code. Published 22 August 2026. [Full primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13567904/), [author code](https://github.com/ema-pandilova/qeeg-emotion-pipeline).

Our stimulus prior concerns **new participants with shared videos**, a different available-context setting. Finding a strong no-EEG baseline is therefore informative, but does not itself make a new contribution. Their fixed/relative targets, montage, transductive normalization and AUC differ from our controls, so their numbers must not be placed in a matched accuracy table.

## Same-stimulus alignment and contrastive learning

CL-CS (June 2025) uses inter-subject signal correlation, a three-domain encoder and contrastive pairs based on stimulus correspondence. Its abstract reports evaluation on FACED/THU-EP/SEED including new participants and novel stimuli. Exact joint-held-out and selection rules still need paper/code reproduction before comparison. [Primary publisher page](https://www.sciencedirect.com/science/article/pii/S1746809425000229), [author code](https://github.com/VCMHE/CL-CS).

CLAE (February 2026) adds attention to a contrastive approach that aligns subjects under the same stimulus, with SEED/THU-EP evaluation. This is another precedent against claiming same-video contrastive alignment as new. Full-text access was blocked; this assessment is limited to the accessible primary abstract. [Publisher abstract](https://www.sciencedirect.com/science/article/pii/S1746809425008833).

The previously reviewed GSCL also uses same-stimulus group structure. A methodological question remains whether shared stimulus should always imply matching **individual self-reported** targets. This is a question for our declared diagnostic and further literature/code checks, not an established gap or a proposed method. [Primary GSCL article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12948548/).

## Reference-model provenance

The complete local CBSAtt paper was inspected, including rendered methods pages 4–6. It declares 16 selected channels, Adam at 0.001, 30 epochs, batch 128, four attention heads and 128 BiLSTM units. Its preprocessing discussion gives 4–45 Hz and a six-second/50%-overlap example; these are not a completely specified reproduction recipe. Page 7 explicitly discusses subject-dependent evaluation. Figure 3 depicts three convolutions and two explicit pools, while our local code has three pools and global averaging before a one-step LSTM. The published temporal tensor layout and exact independent split/selection rule remain unresolved. No authenticated author repository was found in this targeted search. [Primary CBSAtt article](https://link.springer.com/article/10.1007/s11760-025-04708-1).

Local PDF: `dl papers/Wasif Papers/springer wala paper.pdf`, SHA-256 `7eba8d0f875ba95f7b2c8e64fbff74092cdad4c5098281c0a1538107c763540b`. It is not redistributed. Forward/training-mode matching of local code establishes local implementation fidelity, not fidelity to the published method; retain that distinction in the completed full-model report.

The close TFCNN-BiGRU precedent has an author-posted MATLAB File Exchange package, version 1.0.0 dated 3 May 2024. Its listing uses CWT scalograms. The download link requires sign-in and the public viewer did not expose source here; no numerical reproduction or complete training-source audit is claimed. [Author release](https://www.mathworks.com/matlabcentral/fileexchange/165126-tfcnn-bigru). This remains a useful future baseline acquisition route, without replacing the currently frozen controls.

## Uncertainty and interpretive limits

Menzel's cluster-bootstrap analysis shows that dependence and degeneracy affect bootstrap behavior; ordinary multiway resampling does not guarantee uniform validity across regimes. We should treat the current conditional percentile ranges as exploratory resampling evidence, without claiming demonstrated nominal coverage for these small fixed partitions. Final inferential claims require an appropriate statistical review. [Author's Econometrica paper](https://bpb-us-e1.wpmucdn.com/wp.nyu.edu/dist/9/2027/files/2021/09/ECTA15383.pdf).

Our within-video exchange diagnostic applies shared participant weights to both recipient and donor and retains the video weight and observed-class denominator. Correct pairing can involve participant traits, artifacts or demographics as well as emotion; an effect does not identify a causal physiological mechanism. SEED-IV's label is shared within a video, making aggregate exchange invariance a mathematical consequence, not evidence that EEG cannot help on unseen videos.

## Decision gate

Complete and replay the current full-width and context controls first. A credible proposal must explain its difference from EMOD, the identity-control work, stimulus-alignment methods and the previously reviewed material-generalization literature. Neither a transformer replacement nor a collection of corrected splits suffices to establish novelty. Show a concrete evidence-based proposal to the author; require explicit approval and archive the then-current manuscript before adopting a changed main question.
