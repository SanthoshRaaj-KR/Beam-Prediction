# PAPER DRAFT

**Working title:** GAN-Augmented Channel Synthesis for Data-Efficient mmWave Beam Prediction: An Ablation Study on Synthetic-to-Real Data Ratios

---

> ## HOW TO USE THIS DRAFT
>
> Everything that is **prose** is written and ready. Everything in `[[DOUBLE BRACKETS]]` is a placeholder that must be filled from the Phase handoff files — these are **only** numbers, figures, and tables. Do not invent values; if a number is unavailable, leave the placeholder and flag it.
>
> **Citation status:** all **14** references in Section IX were verified against arXiv at drafting time — title, full author list, and year confirmed by retrieving each paper's abstract page. None were written from memory. Section II (Related Work) is fully written and needs no drafting, only a refresh check for newer work before submission.
>
> **Target format:** IEEE conference two-column, 4–6 pages. Download the specific template from your chosen venue — IEEE templates differ slightly by conference.
>
> **Writing order:** Methodology → Results → Discussion → Introduction → Conclusion → **Abstract last**.

---

## Author Block

```
[[AUTHOR 1 NAME]], [[AUTHOR 2 NAME]], [[AUTHOR 3 NAME]]
[[DEPARTMENT]], [[INSTITUTION]]
[[CITY, COUNTRY]]
{[[email1]], [[email2]], [[email3]]}
```

---

## Abstract

> **WRITE THIS LAST.** Template below; fill the bracketed values from `HANDOFF_PHASE_8.md`, then tighten to 150–250 words.

Machine learning has emerged as a promising approach for beam selection in millimeter-wave (mmWave) communication systems, offering substantial reductions in beam-training overhead compared to exhaustive search. However, the performance of such data-driven methods is fundamentally limited by the availability of large, diverse channel state information (CSI) datasets, which are costly and time-consuming to collect through field measurement campaigns. This paper investigates whether generative adversarial networks (GANs) can mitigate this data scarcity by synthesizing additional CSI samples. We train a conditional Wasserstein GAN with gradient penalty (WGAN-GP), conditioned on user position, on a deliberately restricted seed set of [[N_REAL = 500]] real CSI samples drawn from the DeepMIMO `o1_28` ray-tracing scenario at 28 GHz. We then train an identical convolutional beam-classification network under seven synthetic-to-real data ratios spanning 0% to 200%, with three random seeds per configuration, evaluating all runs on a common held-out set of [[N_TEST = 2000]] real samples. Our results show that [[HEADLINE FINDING — e.g. "Top-1 beam prediction accuracy improves from X% to Y% at a Z% synthetic ratio, beyond which performance plateaus"]]. We additionally quantify the realism of the generated CSI using a discriminability test, in which a binary classifier distinguishes real from synthetic samples with [[GAN-TEST ACC]]% accuracy. These findings [[CONCLUDING IMPLICATION]], providing practical guidance on the extent to which synthetic channel data can substitute for costly real measurements in mmWave beam prediction.

**Index Terms** — millimeter wave, beamforming, beam prediction, generative adversarial networks, data augmentation, channel state information, deep learning, 6G.

---

## I. Introduction

Millimeter-wave (mmWave) frequency bands are central to 5G and future 6G networks because of the large contiguous bandwidths available above 24 GHz. However, mmWave propagation suffers from severe path loss and sensitivity to blockage, which necessitates highly directional transmission using large antenna arrays. Establishing and maintaining these directional links requires selecting an appropriate beamforming vector from a predefined codebook — a process that, when performed by exhaustive beam sweeping, incurs training overhead that scales with codebook size and becomes prohibitive for mobile users and large arrays.

Deep learning has been proposed as a means of circumventing this overhead by predicting the optimal beam directly from available channel or contextual information. Alrabeiah and Alkhateeb demonstrated that mmWave beam and blockage information can be inferred from sub-6 GHz channel observations using deep neural networks, achieving high prediction accuracy while eliminating explicit beam training [1]. Subsequent work has extended learning-based beam prediction to a range of input modalities and deployment scenarios. The common prerequisite across these approaches is a sufficiently large and diverse dataset of channel measurements paired with ground-truth optimal beams.

This prerequisite is difficult to satisfy in practice. Real CSI collection requires dedicated measurement campaigns with specialized hardware, repeated across the environments, frequencies, and array geometries of interest. The cost grows as carrier frequencies increase, since propagation becomes more environment-specific and measurement equipment more specialized — a trend that will intensify as systems move toward sub-terahertz 6G bands. Consequently, learning-based beam prediction faces a persistent data scarcity problem, and the practical question becomes how to train effective models when only a small quantity of real channel data is available.

Generative modeling offers one candidate solution: if a generative model can learn the distribution of channel realizations from a limited sample, it can synthesize additional training data at negligible marginal cost. Bhat *et al.* applied this idea in the mmWave sensing domain, using a GAN to generate 30,000 synthetic CSI samples and showing improved generalization for a downstream pose-classification model [2]. Related efforts have explored alternative sources of synthetic channel data, including digital twin environments [3] and diffusion-based generative models [4], the latter reporting gains over GAN-based augmentation for channel compression and beamforming tasks.

Despite this activity, an important practical question remains underexplored **in the wireless domain**: *how much* synthetic data should be used? Studies applying generative augmentation to channel data typically adopt a single augmentation configuration — often a fixed synthetic set size chosen without stated justification — and report the resulting downstream improvement. This leaves practitioners without guidance on the marginal value of additional synthetic samples, or on whether excessive synthetic data eventually degrades performance by amplifying generative artifacts.

Notably, this question *has* been examined in other application domains, with results that motivate caution. In GAN-augmented medical image classification, Afif *et al.* swept the proportion of synthetic to real training images and found that performance peaked at a **small** synthetic fraction and degraded as that fraction grew, concluding that generative augmentation functions best as a modest supplement rather than a bulk data source [11]. Comparable findings across several imaging datasets place the beneficial augmentation ratio in a similarly narrow band. Whether this behavior transfers to mmWave CSI — a signal with very different structure, dimensionality, and label semantics than natural images — is an open question, and it is the question this paper answers empirically.

**Contributions.** This paper makes the following contributions:

1. We implement a conditional WGAN-GP for mmWave CSI synthesis, conditioned on user position, trained on a deliberately scarce seed set of [[N_REAL]] real samples from the DeepMIMO `o1_28` scenario.
2. We conduct a systematic ablation across **seven** synthetic-to-real data ratios (0%, 25%, 50%, 75%, 100%, 150%, 200%), with three random seeds per configuration, isolating the effect of augmentation volume on beam-prediction accuracy while holding the model architecture, training recipe, and real data subset fixed. To our knowledge this is the first such ratio sweep reported for mmWave beam prediction.
3. We test whether the "small synthetic fraction is best" behavior observed for GAN augmentation in image domains [11] also holds for CSI-based beam prediction, providing a cross-domain point of comparison.
4. We report a quantitative realism assessment of the generated CSI via a real-versus-synthetic discriminability test and a beam-label distribution comparison, and relate this realism measure to the observed downstream augmentation gains — a linkage rarely reported in augmentation studies.
5. We release a fully specified, reproducible experimental protocol, including all architectures, hyperparameters, and data splits.

The remainder of this paper is organized as follows. Section II reviews related work. Section III describes the system model and methodology. Section IV presents results. Section V discusses implications and limitations, and Section VI concludes.

---

## II. Related Work

### A. Learning-Based mmWave Beam Prediction

The use of deep learning to reduce beam-training overhead is well established. Alrabeiah and Alkhateeb proved the existence of a mapping from sub-6 GHz channels to mmWave beam and blockage decisions and realized it with deep neural networks, reporting over 90% blockage-prediction accuracy on DeepMIMO data while eliminating explicit mmWave beam training [1]. A parallel line of work exploits contextual side information instead of, or alongside, channel measurements: Rezaie *et al.* proposed location- and orientation-aided 3D beam selection, presenting three network variants of differing capacity explicitly to accommodate **different training dataset sizes** — an acknowledgment that data availability, not only architecture, governs achievable performance [12]. Our use of user position as the GAN conditioning variable is motivated by this same position–beam relationship.

Much of this literature relies on DeepMIMO [5], a parameterized ray-tracing dataset generator that produces channel realizations for standardized scenarios and has become a common benchmark enabling reproducible comparison. More recently, DeepSense 6G [13] has provided large-scale *measured* multi-modal data (mmWave, GPS, vision, radar, LiDAR) across 40+ real-world scenarios, addressing the fidelity gap inherent to simulation. We adopt the DeepMIMO formulation for its controllability — it lets us construct a precisely sized scarce seed set and a disjoint held-out set — while noting in Section V that validation on measured data such as [13] remains future work.

### B. Data Scarcity and Augmentation in Wireless Systems

Data scarcity is a recognized obstacle across learning-based wireless tasks, not only beam prediction. In CSI feedback for massive MIMO, models trained in one environment degrade in others, motivating augmentation strategies that expand limited datasets: Zhang *et al.* introduce delay-domain augmentations to bridge the gap between training data and deployment conditions and to alleviate overfitting in outdoor scenarios [14]. Such hand-crafted, domain-knowledge-based augmentations are effective but bounded by the transformations the designer can specify in advance.

Generative augmentation removes that constraint by learning the data distribution directly. Wen *et al.* survey generative AI for data augmentation across the physical, network, and application layers of wireless systems, cataloguing consistent benefits over traditional augmentation [10]. Most directly related to our work, CSI4Free [2] trains a GAN to produce 30,000 synthetic mmWave CSI samples and demonstrates improved generalization for downstream pose classification, establishing the viability of GAN-augmented CSI. Our work differs from [2] in task — beam index prediction rather than pose classification — and, more importantly, in treating the augmentation *volume* as the experimental variable rather than fixing it at a single value.

### C. Generative Models and Alternative Synthetic Data Sources

Generative adversarial networks [6] and their conditional variants [7] underpin most learned augmentation. Training instability and mode collapse are well-known failure modes of the original objective; the Wasserstein formulation [8] and its gradient-penalty refinement [9] substantially improve stability and have become the default choice when training data is limited — precisely our regime, which is why we adopt WGAN-GP.

Synthetic channel data need not come from GANs. Khan *et al.* employ a digital twin to generate beam-prediction training data, combining it with explainability and robustness techniques and reporting a 70% reduction in real-data requirements alongside a 62% cut in beam-training overhead [3]. Lee *et al.* apply conditional denoising diffusion models to generate user-specific channels, reporting gains over GAN-based and noise-addition baselines, including an 11 dB SNR improvement for beamforming [4]. These results suggest diffusion models may ultimately surpass GANs in channel-generation fidelity. We nonetheless focus on GANs for their substantially lower sampling cost — relevant because the ratio sweep requires generating and training on many synthetic datasets — and because the ratio question we pose is orthogonal to generator choice and applies equally to diffusion-based augmentation.

### D. How Much Synthetic Data? Evidence from Other Domains

The question of augmentation *volume* has received more attention outside wireless. Afif *et al.* systematically varied the proportion of DCGAN-generated to real MRI images for brain-tumor classification and found peak accuracy at a small synthetic fraction (100 synthetic to 900 real), with performance degrading as the synthetic proportion grew; they attribute the decline to generative artifacts impairing generalization to real cases [11]. Related studies across imaging datasets report beneficial augmentation ratios concentrated in a similarly narrow low range, and note that models trained purely on synthetic data perform markedly worse — evidence that synthetic samples supplement rather than substitute for real ones.

### E. Positioning of This Work

Three observations follow from the above. First, GAN-augmented CSI is established as viable [2], but always at a single, fixed augmentation volume. Second, where augmentation volume *has* been swept, it was in image domains [11], whose signal structure and label semantics differ substantially from CSI — the optimal ratio there need not transfer. Third, no existing study, to our knowledge, characterizes the relationship between synthetic-to-real ratio and **beam-prediction** accuracy under a fixed real-data budget. This paper addresses that gap, and additionally links the observed downstream behavior to a direct measurement of generator fidelity, which augmentation studies rarely report.

---

## III. System Model and Methodology

### A. Dataset and Preprocessing

We use the DeepMIMO ray-tracing dataset [5], scenario `o1_28`, an outdoor street-canyon environment operating at a 28 GHz carrier frequency. The base station is equipped with a uniform linear array (ULA) of [[N_ANT = 64]] elements; users are single-antenna. We generate OFDM channels with [[N_SUB = 32]] subcarriers.

> `[[CONFIRM from HANDOFF_PHASE_1: the actual subcarrier count used, and the exact ChannelParameters configuration. If 32 was not supported, update every occurrence in this paper.]]`

For each user $u$, the ray-tracing generator yields a complex channel matrix $\mathbf{H}_u \in \mathbb{C}^{64 \times 32}$ and a three-dimensional position. Since scenario `o1_28` places users at approximately constant elevation, we retain only the horizontal coordinates $(x_u, y_u)$ as the conditioning variable.

Users without a valid propagation path (fully blocked or outside coverage) are discarded. Of [[N_TOTAL_USERS]] users in the scenario, [[N_DROPPED]] were removed by this criterion, leaving [[N_RETAINED]].

Each complex channel is decomposed into real and imaginary components and stacked to form a two-channel real-valued tensor $\mathbf{X}_u \in \mathbb{R}^{2 \times 64 \times 32}$, analogous to a two-channel image. Each sample is normalized by its Frobenius norm, $\tilde{\mathbf{X}}_u = \mathbf{X}_u / \|\mathbf{X}_u\|_F$, with the original norm retained for later spectral-efficiency computation. Positions are linearly mapped to $[-1,1]$ using the coordinate bounds of the retained user set.

**Data splits.** To emulate data scarcity, we randomly sample a **seed set** of [[N_REAL = 500]] users, which constitutes the entire real training budget for every experiment in this paper. A disjoint **test set** of [[N_TEST = 2000]] users is held out and used exclusively for final evaluation; it is never observed during GAN training, classifier training, or model selection. [[STATE whether the split was random or spatially disjoint — from HANDOFF_PHASE_1.]]

### B. Beam Codebook and Ground-Truth Labels

We define a discrete Fourier transform (DFT) codebook of $K = 32$ beams for the 64-element ULA. The $k$-th beamforming vector $\mathbf{w}_k \in \mathbb{C}^{64}$ has entries

$$[\mathbf{w}_k]_n = \frac{1}{\sqrt{64}} \exp\!\left(j \frac{2\pi n k}{K}\right), \quad n = 0,\dots,63.$$

For a channel $\mathbf{H} = [\mathbf{h}_1,\dots,\mathbf{h}_S]$ with $S$ subcarriers, the ground-truth optimal beam index is the codebook entry maximizing average received power across subcarriers:

$$k^\star = \arg\max_{k \in \{0,\dots,K-1\}} \frac{1}{S}\sum_{s=1}^{S} \left| \mathbf{w}_k^H \mathbf{h}_s \right|^2 .$$

This index serves as the classification label. Because per-sample Frobenius normalization scales all beams equally, labels are invariant to the normalization step.

The resulting label distribution spans [[N_POPULATED_CLASSES]] of the 32 beam classes, with the most frequent beam accounting for [[MAJORITY_CLASS_PCT]]% of samples. This majority-class frequency constitutes a trivial baseline against which all learned models must be compared.

`[[FIGURE 1: Beam label distribution histogram across the retained user set. Source: results/plots/label_hist.png (Phase 2).]]`

### C. Conditional WGAN-GP for CSI Synthesis

We adopt the Wasserstein GAN with gradient penalty [9], conditioned on user position following the conditional GAN formulation [7]. The critic $D$ and generator $G$ optimize

$$\mathcal{L}_D = \mathbb{E}_{\tilde{\mathbf{x}}\sim\mathbb{P}_g}\!\left[D(\tilde{\mathbf{x}}|\mathbf{c})\right] - \mathbb{E}_{\mathbf{x}\sim\mathbb{P}_r}\!\left[D(\mathbf{x}|\mathbf{c})\right] + \lambda\, \mathbb{E}_{\hat{\mathbf{x}}}\!\left[\left(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}}|\mathbf{c})\|_2 - 1\right)^2\right],$$

where $\hat{\mathbf{x}}$ is sampled uniformly along straight lines between real and generated samples and $\lambda = 10$.

**Generator.** A latent vector $\mathbf{z} \in \mathbb{R}^{100}$, $\mathbf{z}\sim\mathcal{N}(0,\mathbf{I})$, is concatenated with the normalized position $\mathbf{c} \in \mathbb{R}^{2}$ and projected by a fully connected layer to a $256\times4\times2$ tensor. Four transposed-convolution blocks (kernel 4, stride 2, padding 1), each with batch normalization and ReLU, successively double both spatial dimensions to $16\times64\times32$, followed by a $3\times3$ convolution to two output channels and a $\tanh$ activation. The $4\times2$ initial resolution and four doubling stages are chosen so the output dimension matches the $64\times32$ channel tensor exactly, avoiding any cropping or padding correction.

**Critic.** The position condition is broadcast as two constant spatial channels and concatenated with the two CSI channels. Four strided convolutions (kernel 4, stride 2, padding 1) with LeakyReLU(0.2) reduce the input to $256\times4\times2$, followed by a linear layer producing a scalar score with no output nonlinearity. Batch normalization is deliberately avoided in the critic, as it invalidates the per-sample gradient penalty; group normalization is used instead.

`[[TABLE I: Network architectures and training hyperparameters. Populate from HANDOFF_PHASE_4 (layer tables, parameter counts) and HANDOFF_PHASE_5 (optimizer, lambda_gp=10, n_critic=5, lr=1e-4, betas=(0.5,0.9), batch size 64, iterations).]]`

Both networks use Adam with learning rate $10^{-4}$ and $(\beta_1,\beta_2) = (0.5, 0.9)$, with five critic updates per generator update, following [9]. Training runs for [[N_GAN_ITERS]] generator iterations with batch size 64. Conditioning vectors are drawn from the empirical position distribution of the real seed set.

### D. Beam Classification Network

The downstream model is a compact CNN taking $\tilde{\mathbf{X}} \in \mathbb{R}^{2\times64\times32}$ and producing logits over the 32 beam classes: three convolutional blocks (2→32→64→128 channels, $3\times3$ kernels, batch normalization, ReLU, with max-pooling after the second and third blocks), followed by a 256-unit fully connected layer with dropout 0.3 and a final linear layer. The model has approximately [[N_PARAMS ≈ 4.3M]] parameters.

Training uses cross-entropy loss, Adam at learning rate $10^{-3}$ with cosine annealing, batch size 64, for [[N_EPOCHS = 100]] epochs. Within the 500 real samples, 400 are used for training and 100 for validation; the checkpoint achieving the highest validation accuracy is evaluated once on the held-out test set. **Critically, this architecture and training recipe are held identical across every experimental condition** — only the composition of the training set varies.

### E. Ablation Design

We define the synthetic ratio $\rho$ as the number of synthetic samples added, expressed as a percentage of the fixed real seed set size. For each $\rho \in \{0, 25, 50, 75, 100, 150, 200\}\%$, the training set comprises all [[N_REAL]] real samples plus $\rho \cdot$ [[N_REAL]] synthetic samples, as summarized in Table II. Labels for synthetic samples are assigned analytically by applying the codebook rule of Section III-B to the generated channel, rather than being predicted by any learned model.

The validation set consists exclusively of real samples in all conditions, so that model selection is never influenced by generative artifacts. Each configuration is repeated with three random seeds, which jointly randomize network initialization, data shuffling, and the synthetic sample draw. We report mean and standard deviation across seeds.

`[[TABLE II: Ablation configurations — Run ID, real count, synthetic count, total, ratio. This table is already fully determined; fill from IMPLEMENTATION_PHASES.md Phase 7 table.]]`

### F. Evaluation Metrics

We report three metrics on the held-out real test set:

1. **Top-1 accuracy:** the fraction of samples for which the predicted beam equals $k^\star$.
2. **Top-3 accuracy:** the fraction for which $k^\star$ appears among the three highest-scoring predictions — relevant in practice, since a system may probe a small candidate set.
3. **Relative spectral efficiency:** with $\hat{k}$ the predicted beam, we compute
$$\eta = \frac{1}{N}\sum_{i=1}^{N} \frac{\log_2\!\left(1 + \gamma\, g_i(\hat{k})\right)}{\log_2\!\left(1 + \gamma\, g_i(k^\star)\right)} \times 100\%,$$
where $g_i(k)$ is the subcarrier-averaged beamforming gain and $\gamma$ is a reference SNR fixed at [[SNR = 10]] dB. This captures the practical cost of a misprediction: selecting a suboptimal but well-aligned beam is far less damaging than selecting an orthogonal one, a distinction that raw accuracy obscures.

---

## IV. Results

### A. Quality of Synthesized CSI

`[[FIGURE 2: WGAN-GP training curve — Wasserstein distance estimate vs. generator iteration. Source: Phase 5.]]`

Figure 2 shows the evolution of the Wasserstein distance estimate during GAN training. [[DESCRIBE the curve: stabilization behavior, iteration at which it plateaus, any instability. From HANDOFF_PHASE_5.]]

`[[FIGURE 3: Side-by-side magnitude heatmaps |H| for real (top row) and synthetic (bottom row) CSI samples, shared color scale. Source: Phase 6.]]`

Figure 3 compares magnitude patterns of real and generated channels. [[DESCRIBE qualitatively: does the synthetic data reproduce the antenna-domain and frequency-domain structure visible in real samples?]]

To quantify realism, we train a binary CNN classifier to distinguish real from synthetic samples using a balanced set of [[N]] samples per class, repeated over three seeds. This discriminator achieves **[[GAN_TEST_ACC]]% ± [[STD]]%** accuracy. [[INTERPRET per the Phase 6 guidance: ~50–60% indicates near-indistinguishability; >90% indicates the generator captures only coarse structure. State this plainly either way — a high value is a legitimate finding that directly informs the discussion in Section V.]]

`[[FIGURE 4: Beam-label distribution, real vs. synthetic. Source: Phase 6, Test 4.]]`

Figure 4 compares the distribution of optimal beam indices computed from real and synthetic channels. [[REPORT the overlap — this indicates whether the generator captured task-relevant angular structure rather than merely matching low-order statistics.]]

### B. Effect of Synthetic-to-Real Ratio

`[[TABLE III: Main results — for each ratio (0–200%): Top-1 accuracy (mean±std), Top-3 accuracy (mean±std), relative spectral efficiency (mean±std). Include the majority-class baseline row. Source: HANDOFF_PHASE_7 / HANDOFF_PHASE_8.]]`

`[[FIGURE 5 — KEY FIGURE: Top-1 accuracy vs. synthetic ratio, error bars ±1 std over 3 seeds, dashed line for majority-class baseline. Source: Phase 8.]]`

Table III and Figure 5 present the central result. The real-data-only baseline (R0) attains [[R0_TOP1]]% Top-1 accuracy, compared with the majority-class baseline of [[MAJORITY_PCT]]%. [[STATE the trend: the best configuration, its ratio, its accuracy, and the absolute improvement in percentage points over R0.]]

[[DESCRIBE the shape of the curve — monotonic, plateau, or peak-and-decline — and identify the knee or peak if present.]]

A [[two-sample t-test / Mann–Whitney U test]] comparing the best-performing ratio against R0 across seeds yields $p =$ [[P_VALUE]]. [[State whether this meets conventional significance thresholds. With three seeds, be explicit that statistical power is limited and that differences within one standard deviation are not claimed as significant.]]

Top-3 accuracy follows [[a similar / a different]] trend, reaching [[VALUE]]% at the best configuration. Relative spectral efficiency remains above [[VALUE]]% across all conditions, indicating that [[interpretation — mispredictions typically select angularly adjacent beams, limiting throughput loss]].

---

## V. Discussion

**Interpreting the ratio curve.** [[Write 2–3 paragraphs grounded in the actual observed pattern, using the Phase 8 interpretation framework. Address: does augmentation help; where is the practical optimum; is there evidence of diminishing or negative returns at high ratios; what mechanism plausibly explains the shape — additional genuine channel diversity versus a regularization effect versus artifact amplification.]]

**Comparison with image-domain findings.** Our results can be read against the behavior reported for GAN-augmented medical imaging, where accuracy peaked at roughly a 10% synthetic fraction and declined thereafter [11]. [[STATE whether the CSI results follow the same qualitative pattern — an early peak followed by degradation — or differ. If our optimum occurs at a substantially higher ratio, one plausible explanation is that CSI at a fixed carrier and array geometry occupies a lower-complexity manifold than natural images, so the generator's samples remain useful for longer before artifacts dominate. If the pattern matches, that is evidence the "modest supplement, not bulk substitute" principle generalizes across signal modalities. Either outcome is a substantive finding and should be stated as such.]]

**Relating downstream gains to generator fidelity.** The discriminability result of Section IV-A provides a useful lens on the ablation outcome. [[If discriminability was high and gains were small, argue that the generator captured coarse rather than fine-grained structure, so synthetic samples act closer to a regularizer than to genuinely novel training data. If discriminability was low and gains were substantial, argue that the generator captured task-relevant structure. Either way, connect the two measurements explicitly — this linkage is a contribution in itself, since much of the augmentation literature reports downstream gains without characterizing generator fidelity.]]

**Practical implications.** For practitioners facing limited CSI measurement budgets, our results suggest [[recommendation — e.g. that synthetic augmentation up to approximately X% of the real dataset size offers the best accuracy-per-unit-effort, with limited benefit beyond]]. Because synthetic sample generation is computationally cheap relative to field measurement, even modest accuracy gains may be operationally worthwhile.

**Limitations.** Several constraints bound the generality of these findings and should be stated plainly:

- **Single scenario and band.** All experiments use one DeepMIMO scenario at 28 GHz with one array geometry. Whether the observed ratio behavior transfers to other environments, frequencies, or array configurations is untested.
- **Ray-traced rather than measured channels.** DeepMIMO channels are simulated. The "real" data in this study is real only in the sense of being non-generated; conclusions about true field measurements require validation on a measured dataset such as DeepSense 6G [13], which provides real mmWave collections across 40+ scenarios and would be the natural next testbed for this protocol.
- **Limited seed count.** Three seeds per configuration bounds statistical power; small differences between adjacent ratios should not be over-interpreted.
- **Fixed seed set size.** We vary the synthetic ratio at a single real-data budget of [[N_REAL]] samples. The optimal ratio may itself depend on how much real data is available.
- **Single generator family.** We evaluate WGAN-GP only. Diffusion-based generators have reported superior channel-generation fidelity [4], and the ratio curve may differ for such models.

---

## VI. Conclusion

We investigated the use of conditional WGAN-GP synthesis to alleviate data scarcity in mmWave beam prediction, and specifically the question of how much synthetic data should be used. Training an identical beam-classification network across seven synthetic-to-real ratios on a fixed 500-sample real seed set from the DeepMIMO `o1_28` scenario, we found that [[HEADLINE FINDING]]. We further quantified generator fidelity through a real-versus-synthetic discriminability test, and related that fidelity to the observed downstream gains. These results provide practical guidance on augmentation volume for data-constrained beam prediction and indicate that [[key takeaway]].

Future work includes extending the ratio analysis across multiple scenarios and carrier frequencies, varying the real-data budget to determine whether the optimal ratio scales with dataset size, evaluating diffusion-based generators under the same protocol, and exploring architectural refinements to the generator such as skip-connection designs for finer-grained channel structure.

---

## VII. Acknowledgments

`[[Optional — institution, advisor, compute resources. Kaggle's free GPU tier is worth acknowledging if the venue permits.]]`

---

## VIII. Reproducibility Statement

`[[Optional but valuable — many venues award credit for this.]]`

All architectures, hyperparameters, and data splits are specified in Section III. Code and the exact DeepMIMO configuration are available at [[REPOSITORY URL]]. Experiments were conducted on [[hardware]] using DeepMIMO version 4.0.3, scenario `o1_28`, with PyTorch [[version]].

---

## IX. References

> **All fourteen entries below were verified against arXiv during drafting** — title, full author list, and year confirmed for each by retrieving the abstract page. None are reconstructed from memory. Renumber in order of first citation if you reorder the text, and reformat to the venue's exact IEEE style. Where a preprint has since appeared in a journal or conference (e.g. [1], [5], [13]), prefer the published venue in the final bibliography.

[1] M. Alrabeiah and A. Alkhateeb, "Deep Learning for mmWave Beam and Blockage Prediction Using Sub-6GHz Channels," *arXiv preprint arXiv:1910.02900*, 2019. [Online]. Available: https://arxiv.org/abs/1910.02900

[2] N. N. Bhat, R. Berkvens, and J. Famaey, "CSI4Free: GAN-Augmented mmWave CSI for Improved Pose Classification," *arXiv preprint arXiv:2406.18684*, 2024. [Online]. Available: https://arxiv.org/abs/2406.18684

[3] N. Khan, A. Abdallah, A. Celik, A. M. Eltawil, and S. Coleri, "Digital Twin-Assisted Explainable AI for Robust Beam Prediction in mmWave MIMO Systems," *arXiv preprint arXiv:2507.14180*, 2025. [Online]. Available: https://arxiv.org/abs/2507.14180

[4] T. Lee, J. Park, H. Kim, and J. G. Andrews, "Generating High Dimensional User-Specific Wireless Channels using Diffusion Models," *arXiv preprint arXiv:2409.03924*, 2024. [Online]. Available: https://arxiv.org/abs/2409.03924

[5] A. Alkhateeb, "DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications," in *Proc. Information Theory and Applications Workshop (ITA)*, San Diego, CA, Feb. 2019. [Online]. Available: https://arxiv.org/abs/1902.06435

[6] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative Adversarial Nets," in *Advances in Neural Information Processing Systems (NIPS)*, 2014. [Online]. Available: https://arxiv.org/abs/1406.2661

[7] M. Mirza and S. Osindero, "Conditional Generative Adversarial Nets," *arXiv preprint arXiv:1411.1784*, 2014. [Online]. Available: https://arxiv.org/abs/1411.1784

[8] M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein GAN," *arXiv preprint arXiv:1701.07875*, 2017. [Online]. Available: https://arxiv.org/abs/1701.07875

[9] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved Training of Wasserstein GANs," in *Advances in Neural Information Processing Systems (NIPS)*, 2017. [Online]. Available: https://arxiv.org/abs/1704.00028

[10] J. Wen, J. Kang, D. Niyato, Y. Zhang, J. Wang, B. Sikdar, and P. Zhang, "Generative AI for Data Augmentation in Wireless Networks: Analysis, Applications, and Case Study," *arXiv preprint arXiv:2411.08341*, 2024. [Online]. Available: https://arxiv.org/abs/2411.08341

[11] M. M. Afif, A. Al Noman, K. M. T. Kabir, M. M. Ahmmed, M. M. Rahman, M. Mahmud, and M. A. Babu, "Proportional Sensitivity in Generative Adversarial Network (GAN)-Augmented Brain Tumor Classification Using Convolutional Neural Network," *arXiv preprint arXiv:2506.17165*, 2025. [Online]. Available: https://arxiv.org/abs/2506.17165

[12] S. Rezaie, E. de Carvalho, and C. N. Manchón, "A Deep Learning Approach to Location- and Orientation-aided 3D Beam Selection for mmWave Communications," *arXiv preprint arXiv:2110.06859*, 2021. [Online]. Available: https://arxiv.org/abs/2110.06859

[13] A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, N. Srinivas, U. Demirhan, and J. Morais, "DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset," *IEEE Communications Magazine*, vol. 61, no. 9, pp. 122–128, 2023. [Online]. Available: https://arxiv.org/abs/2211.09769

[14] H. Zhang, Z. Lu, X. Zhang, and J. Wang, "Data Augmentation of Bridging the Delay Gap for DL-based Massive MIMO CSI Feedback," *arXiv preprint arXiv:2308.00478*, 2023. [Online]. Available: https://arxiv.org/abs/2308.00478

---

## Appendix A — Placeholder Inventory

Track completion here. Every item must be filled or explicitly removed before submission.

| # | Placeholder | Source | Done |
|---|---|---|---|
| 1 | Author block | — | ⬜ |
| 2 | Abstract headline finding | Phase 8 | ⬜ |
| 3 | Subcarrier/antenna counts confirmed | Phase 1 | ⬜ |
| 4 | User counts (total/dropped/retained) | Phase 1 | ⬜ |
| 5 | Split type (random vs. spatial) | Phase 1 | ⬜ |
| 6 | Populated classes, majority-class % | Phase 2 | ⬜ |
| 7 | Figure 1 — label histogram | Phase 2 | ⬜ |
| 8 | Table I — architectures & hyperparameters | Phases 4, 5 | ⬜ |
| 9 | GAN iterations, parameter counts | Phases 4, 5 | ⬜ |
| 10 | Figure 2 — Wasserstein curve + description | Phase 5 | ⬜ |
| 11 | Figure 3 — real vs. synthetic heatmaps | Phase 6 | ⬜ |
| 12 | GAN-test discriminability accuracy | Phase 6 | ⬜ |
| 13 | Figure 4 — label distribution comparison | Phase 6 | ⬜ |
| 14 | Table II — ablation configurations | Phase 7 | ⬜ |
| 15 | Table III — main results | Phases 7, 8 | ⬜ |
| 16 | Figure 5 — key ablation curve | Phase 8 | ⬜ |
| 17 | p-value and significance statement | Phase 8 | ⬜ |
| 18 | Discussion paragraphs | Phase 8 | ⬜ |
| 19 | Conclusion headline | Phase 8 | ⬜ |
| 20 | Repository URL | Phase 9 | ⬜ |
| 21 | Comparison-with-image-domain paragraph (§V) | Phase 8 | ⬜ |
| 22 | Refresh literature check for work published after drafting | Phase 9 | ⬜ |

**Not placeholders — already complete:** Section II (Related Work) in full, Section IX (all 14 references, arXiv-verified), the Introduction, the Limitations list, and all methodology prose except the bracketed numeric values.
