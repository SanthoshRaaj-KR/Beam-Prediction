# How Much Synthetic Channel Data Helps? A Dose-Response Study of GAN Augmentation for mmWave Beam Prediction

**Target format:** IEEE conference, two columns. Current length is about 8 pages with all figures and tables placed. Prose is final. Only the author block, the repository URL, and one pair of numbers in Section V-C (marked `[[RANK90]]`) still need filling.

---

## Author Block

```
[[AUTHOR NAMES]]
[[DEPARTMENT]], [[INSTITUTION]]
[[CITY, COUNTRY]]
{[[emails]]}
```

---

## Abstract

Deep learning can choose the best millimeter-wave (mmWave) beam directly from channel state information (CSI) and so avoid slow exhaustive beam sweeping. The catch is that it needs thousands of labelled channel measurements, which are costly to collect. A popular fix is to train a generative adversarial network (GAN) on the small amount of real data available and add synthetic samples to the training set. Published wireless work picks one fixed amount of synthetic data and does not justify the choice. We ask the question directly: how much should you add? We train a position-conditioned Wasserstein GAN with gradient penalty on a deliberately small seed set of 500 real CSI samples from the DeepMIMO `o1_28` ray-tracing scenario at 28 GHz. We then train one fixed beam classifier at seven synthetic-to-real ratios from 0% to 200%, with three seeds each, and score all 21 runs on the same 2000 held-out real samples. Accuracy falls as the ratio grows. Top-1 accuracy drops 1.01 percentage points for every 100% of synthetic data added (Spearman ρ = −0.437, p = 0.048), and relative spectral efficiency drops 1.00 points per 100% (p = 0.012). The apparent peak at 50% sits 0.65 points above the real-only baseline and is not statistically resolvable (p = 0.53). Three measurements explain the loss: the generator covers only 9 of 32 beam classes, a classifier trained on synthetic data alone reaches 59.77% against a 53.84% majority-class floor, and the synthetic channel matrices are close to full rank while the real ray-traced ones are close to rank 4. The "small synthetic fraction is best" behaviour reported for GAN-augmented medical images does carry over, but in this setting only the penalty is large enough to measure.

**Index Terms**: millimeter wave, beam prediction, generative adversarial networks, data augmentation, channel state information, mode collapse, deep learning, 6G.

---

## I. Introduction

Millimeter-wave bands above 24 GHz give 5G and 6G systems the wide bandwidth they need, but the signal loses power quickly over distance and is easily blocked. To compensate, the base station focuses its energy into a narrow beam using a large antenna array. Picking the right beam is the problem. The textbook method, exhaustive beam sweeping, tries every beam in a predefined set called a codebook and keeps the best one. The time this costs grows with the size of the codebook, and for a moving user the answer is stale by the time it arrives.

Deep learning offers a shortcut. Instead of sweeping, a neural network predicts the best beam from information the system already has. Alrabeiah and Alkhateeb showed that mmWave beam and blockage decisions can be inferred from sub-6 GHz channel observations, with over 90% blockage-prediction accuracy and no mmWave beam training at all [1]. Later work swapped in other inputs and other scenarios. All of it needs the same thing first: a large set of channel measurements, each paired with its correct beam.

That set is expensive. Collecting real CSI means a measurement campaign with specialised hardware, repeated for every environment, frequency, and array geometry you care about. The cost rises with carrier frequency, because propagation becomes more specific to the local environment and the equipment becomes more specialised. Beam prediction therefore has a standing data scarcity problem, and the practical question is what to do when only a little real data exists.

One answer is to generate more. If a generative model learns the distribution of channels from a small sample, extra training data costs almost nothing per sample. Bhat *et al.* did this for mmWave sensing, generating 30,000 synthetic CSI samples with a GAN and improving a downstream pose classifier [2]. Others have drawn synthetic channels from digital twins [3] and from diffusion models [4].

A practical question is still open in the wireless literature: *how much* synthetic data should you add? Studies that apply generative augmentation to channel data typically fix the synthetic set at one size, report the improvement, and move on. That leaves an engineer with no idea whether the next thousand synthetic samples will help, do nothing, or cause harm.

The question has been answered elsewhere, and the answer is a warning. Afif *et al.* varied the proportion of GAN-generated to real MRI images for brain-tumour classification and found accuracy peaked at a small synthetic fraction, then fell as the fraction grew [11]. They blamed generative artifacts for the decline. Several imaging studies put the useful ratio in a similarly narrow band. Whether this carries over to mmWave CSI is not obvious, because CSI has different structure, different dimensionality, and different label semantics from a photograph. This paper answers that empirically.

**Contributions.**

1. We run what is, to our knowledge, the first synthetic-to-real ratio sweep for mmWave beam prediction: seven ratios from 0% to 200%, three seeds each, 21 runs, one fixed classifier and training recipe, one held-out real test set.
2. We report a negative result honestly. Adding synthetic CSI degrades Top-1 accuracy and spectral efficiency, with a significant downward trend across the sweep, and the small apparent gain at 50% is within noise.
3. We explain the degradation rather than just reporting it. Four independent measurements point to mode collapse and to a structural mismatch between what a convolutional generator produces and what a ray-traced channel looks like.
4. We show that the real channel matrices are nearly rank 4 while the generated ones are nearly full rank. This gives a concrete reason why GAN augmentation transfers poorly from imaging to ray-traced CSI, and it predicts that generators with the right low-rank inductive bias should do better.
5. We release the full protocol, including architectures, hyperparameters, splits, seeds, and per-run results.

Section II reviews related work, Section III describes the method, Section IV gives results, Section V explains the mechanism, Section VI discusses limits, and Section VII concludes.

---

## II. Related Work

### A. Learning-Based mmWave Beam Prediction

Using deep learning to cut beam-training overhead is well established. Alrabeiah and Alkhateeb proved a mapping exists from sub-6 GHz channels to mmWave beam and blockage decisions, then learned it with deep neural networks on DeepMIMO data [1]. A second line of work uses context instead of, or alongside, the channel. Rezaie *et al.* proposed location- and orientation-aided 3D beam selection and gave three network variants of different capacity specifically to suit **different training set sizes** [12], an admission that data availability, not only architecture, sets the ceiling. We condition our generator on user position for the same reason: position and best beam are closely linked.

Much of this work uses DeepMIMO [5], a parameterised ray-tracing dataset generator that produces channels for standard scenarios and makes results comparable between papers. DeepSense 6G [13] now provides large-scale *measured* multi-modal data across more than 40 real scenarios. We use DeepMIMO because it lets us control the experiment exactly, in particular to build a precisely sized scarce seed set and a disjoint test set. Section VI returns to what simulation costs us.

### B. Data Scarcity and Augmentation in Wireless Systems

Scarcity affects learning-based wireless tasks generally. In CSI feedback for massive MIMO, models trained in one environment degrade in another, which has motivated augmentation that stretches a limited dataset. Zhang *et al.* add delay-domain transformations to close the gap between training data and deployment and to reduce overfitting outdoors [14]. Hand-designed augmentation works, but it is bounded by the transformations the designer thinks to write down.

Generative augmentation lifts that bound by learning the distribution itself. Wen *et al.* survey generative AI for data augmentation across the physical, network, and application layers [10]. Closest to our work, CSI4Free trains a GAN to produce 30,000 synthetic mmWave CSI samples and improves downstream pose classification [2]. We differ in task, beam index rather than pose, and more importantly in treating the augmentation *volume* as the thing under test rather than a fixed setting.

### C. Generative Models and Other Sources of Synthetic Channels

Generative adversarial networks [6] and conditional GANs [7] underpin most learned augmentation. The original objective is unstable and prone to mode collapse, where the generator produces only a few distinct outputs. The Wasserstein formulation [8] and its gradient-penalty version [9] improve stability and are the usual choice when data is limited, which is why we use WGAN-GP here.

Synthetic channels need not come from a GAN. Khan *et al.* use a digital twin to generate beam-prediction training data, reporting a 70% cut in real-data requirements and a 62% cut in beam-training overhead [3]. Lee *et al.* use conditional denoising diffusion models for user-specific channels and report gains over GAN-based baselines, including 11 dB of SNR improvement for beamforming [4]. Diffusion models may well beat GANs on channel fidelity. We still study GANs, because sampling is much cheaper, which matters when a ratio sweep needs many synthetic datasets, and because the ratio question itself does not depend on the generator family. Section V argues the effect we measure would apply to diffusion too, for a reason specific to the data.

### D. How Much Synthetic Data? Evidence from Other Fields

Augmentation volume has had more attention outside wireless. Afif *et al.* varied the proportion of DCGAN-generated to real MRI images for brain-tumour classification and found the best accuracy at a small synthetic fraction, 100 synthetic to 900 real, with accuracy falling as the proportion rose [11]. They attribute the fall to generative artifacts that hurt generalisation to real cases. Related imaging studies report useful ratios in a similarly low band and note that models trained purely on synthetic data do much worse, which suggests synthetic samples supplement real ones rather than replace them. Section V-B tests that last point directly on CSI.

### E. Where This Paper Fits

Three gaps follow. GAN-augmented CSI is known to work [2], but only ever at one fixed volume. Where volume has been swept it was on images [11], whose structure and labels differ from CSI, so the best ratio there need not transfer. And no study we know of measures how the synthetic-to-real ratio affects **beam-prediction** accuracy under a fixed real-data budget. We close that gap and, unlike most augmentation studies, connect the downstream result to direct measurements of what the generator actually learned.

---

## III. System Model and Method

### A. Dataset and Preprocessing

We use DeepMIMO [5], scenario `o1_28`, an outdoor street canyon at a 28 GHz carrier. The base station has a uniform linear array (ULA) of 64 elements. Users have a single antenna. We generate OFDM channels with 32 subcarriers.

For user $u$ the generator returns a complex channel matrix $\mathbf{H}_u \in \mathbb{C}^{64 \times 32}$ and a 3D position. Users in `o1_28` sit at roughly constant height, so we keep only the horizontal coordinates $(x_u, y_u)$ as the conditioning variable.

Users with no valid propagation path, fully blocked or out of coverage, are discarded. We generated 6000 users and **none** were dropped by this rule, leaving 6000. We report this because a silent drop would bias the label distribution, and a reader cannot tell from accuracy alone whether it happened. Retained positions span 36.0 m in $x$ (242.4 m to 278.4 m) and 550.0 m in $y$ (297.2 m to 847.2 m), so the users form a long narrow strip rather than a filled area. Section VI explains why that shape matters.

Each complex channel is split into real and imaginary parts and stacked into a two-channel real tensor $\mathbf{X}_u \in \mathbb{R}^{2 \times 64 \times 32}$, which can be handled like a two-channel image. Every sample is divided by its own Frobenius norm, $\tilde{\mathbf{X}}_u = \mathbf{X}_u / \|\mathbf{X}_u\|_F$, and the original norm is stored for the spectral-efficiency metric. Positions are mapped linearly to $[-1,1]$ using the bounds of the retained set.

**Splits.** To imitate scarcity we draw a **seed set** of 500 users at random. This is the entire real training budget for every experiment in the paper. A disjoint **test set** of 2000 users is held out and used only for the final score. It is never seen during GAN training, classifier training, or model selection. The split is random over users, not spatially disjoint, so the test set covers the same strip as the seed set.

### B. Beam Codebook and Labels

We define a discrete Fourier transform (DFT) codebook of $K = 32$ beams for the 64-element ULA. A codebook is just a fixed list of beamforming vectors the system is allowed to choose from. The $k$-th vector $\mathbf{w}_k \in \mathbb{C}^{64}$ has entries

$$[\mathbf{w}_k]_n = \frac{1}{\sqrt{64}} \exp\!\left(j \frac{2\pi n k}{K}\right), \quad n = 0,\dots,63.$$

For a channel $\mathbf{H} = [\mathbf{h}_1,\dots,\mathbf{h}_S]$ with $S$ subcarriers, the correct beam is the one with the highest average received power across subcarriers:

$$k^\star = \arg\max_{k \in \{0,\dots,K-1\}} \frac{1}{S}\sum_{s=1}^{S} \left| \mathbf{w}_k^H \mathbf{h}_s \right|^2 .$$

This index is the classification label. Because per-sample normalisation scales every beam by the same factor, the label does not change when we normalise.

Over the 2500 users used in the study, all **32 of 32** beam classes are populated, but the distribution is heavily skewed: the most common beam, index 16, accounts for **53.84%** of samples. Always predicting beam 16 therefore scores 53.84%, and that majority-class figure is the floor any learned model must clear. Figure 1 shows the full histogram.

`[[FIGURE 1: Beam-label distribution over the retained users. File: Result/plots/fig_label_hist.png]]`

### C. Conditional WGAN-GP for CSI Synthesis

We use a Wasserstein GAN with gradient penalty [9], conditioned on user position in the manner of conditional GANs [7]. The critic $D$ and generator $G$ optimise

$$\mathcal{L}_D = \mathbb{E}_{\tilde{\mathbf{x}}\sim\mathbb{P}_g}\!\left[D(\tilde{\mathbf{x}}|\mathbf{c})\right] - \mathbb{E}_{\mathbf{x}\sim\mathbb{P}_r}\!\left[D(\mathbf{x}|\mathbf{c})\right] + \lambda\, \mathbb{E}_{\hat{\mathbf{x}}}\!\left[\left(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}}|\mathbf{c})\|_2 - 1\right)^2\right],$$

where $\hat{\mathbf{x}}$ is sampled uniformly on straight lines between real and generated samples and $\lambda = 10$.

**Generator.** A latent vector $\mathbf{z} \in \mathbb{R}^{100}$ drawn from $\mathcal{N}(0,\mathbf{I})$ is concatenated with the normalised position $\mathbf{c} \in \mathbb{R}^{2}$ and projected by a fully connected layer to a $256 \times 4 \times 2$ tensor. Four transposed-convolution blocks (kernel 4, stride 2, padding 1), each with batch normalisation and ReLU, double both axes to $16 \times 64 \times 32$. A final $3\times3$ convolution gives two output channels. The $4\times2$ starting shape and four doublings make the output land on $64 \times 32$ exactly, with no cropping.

The output is then divided by its own Frobenius norm, which projects it onto the unit sphere the real data occupies. This replaced a final `tanh`, and the change matters enough to record. A `tanh` bounds each element to $[-1,1]$, which puts the generated Frobenius norm one to two orders of magnitude above the real one. Real and fake were then separable by that single scalar, the critic saturated at once, and the generator received almost no usable gradient. Matching the norm removes the trivial feature and forces the critic to compare structure instead. Scale carries no information here in any case, because the label comes from an argmax that is invariant to positive scaling, and spectral efficiency uses the stored per-sample norms.

**Critic.** The position is broadcast as two constant spatial channels and concatenated with the two CSI channels. Four strided convolutions (kernel 4, stride 2, padding 1) with LeakyReLU(0.2) reduce the input to $256 \times 4 \times 2$, and a linear layer gives a scalar score with no output nonlinearity. We use group normalisation rather than batch normalisation in the critic, because batch normalisation makes the per-sample gradient penalty invalid.

Both networks use Adam at learning rate $10^{-4}$ with $(\beta_1,\beta_2) = (0.5, 0.9)$ and five critic updates per generator update, following [9]. Training runs for 3000 generator iterations at batch size 64 and takes 321 s on one NVIDIA T4. Conditioning vectors are drawn from the position distribution of the real seed set.

### D. Beam Classification Network

The downstream model takes $\tilde{\mathbf{X}} \in \mathbb{R}^{2\times64\times32}$ and outputs logits over 32 beams. It has three convolutional blocks (2→32→64→128 channels, $3\times3$ kernels, batch normalisation, ReLU, with max-pooling after the second and third), then a 256-unit fully connected layer with dropout 0.3, then a linear layer. It has about 4.3M parameters, most of them in the first linear layer.

Training uses cross-entropy loss and Adam at learning rate $10^{-3}$ with cosine annealing, batch size 64, for 100 epochs. Of the 500 real samples, 400 are used for training and 100 for validation. The checkpoint with the best validation accuracy is scored once on the held-out test set. **This architecture and recipe are identical in every condition.** Only the composition of the training set changes. No hyperparameter was tuned per run, which is what makes the comparison across ratios a fair one.

### E. Sweep Design

We define the synthetic ratio $\rho$ as the number of synthetic samples added, as a percentage of the fixed real seed set. For $\rho \in \{0, 25, 50, 75, 100, 150, 200\}\%$, the training set is the 400 real training samples plus $\rho \times 500$ synthetic samples, so totals run from 400 to 1400 (Table II). Synthetic data is always **added**, never substituted for real data, so every run has the same real content and the ratio is the only thing that changes. This is a dose-response design: one factor is varied over a range, everything else is frozen, and we look at the shape of the response.

Labels for synthetic samples come from applying the codebook rule of Section III-B to the generated channel. They are never predicted by a learned model, so a label error cannot be blamed on a second network.

The validation set is **always the same 100 real samples**, in every condition. Model selection is therefore never influenced by generative artifacts, which would otherwise let a run that overfits the synthetic distribution pick its own best checkpoint. Each configuration runs with three seeds, which jointly randomise network initialisation, data shuffling, and the synthetic draw. A fresh synthetic set is generated for each seed, so the sweep measures the generator's behaviour and not one lucky batch. We report mean and standard deviation over seeds.

A note on naming: we call this a ratio sweep rather than an ablation study. An ablation removes components to see what each contributes. Here nothing is removed, one quantity is varied, so the dose-response description is the accurate one.

`[[TABLE II: Sweep configurations. Columns: Run, real train, synthetic, total, ratio. R0 400/0/400/0%, R25 400/125/525/25%, R50 400/250/650/50%, R75 400/375/775/75%, R100 400/500/900/100%, R150 400/750/1150/150%, R200 400/1000/1400/200%.]]`

### F. Metrics

We report three numbers on the held-out real test set.

**Top-1 accuracy** is the fraction of samples where the predicted beam equals $k^\star$.

**Top-3 accuracy** is the fraction where $k^\star$ is among the three highest-scoring predictions. This matters in practice, since a system can afford to probe a short candidate list.

**Relative spectral efficiency** measures what a wrong prediction actually costs. With $\hat{k}$ the predicted beam,

$$\eta = \frac{1}{N}\sum_{i=1}^{N} \frac{\log_2\!\left(1 + \gamma\, g_i(\hat{k})\right)}{\log_2\!\left(1 + \gamma\, g_i(k^\star)\right)} \times 100\%,$$

where $g_i(k)$ is the subcarrier-averaged beamforming gain and $\gamma$ is a reference SNR fixed at 10 dB. This is a reported assumption, not a tuned value. Picking a beam next to the right one costs very little throughput; picking one pointing the other way costs almost everything. Raw accuracy hides that difference and $\eta$ exposes it.

---

## IV. Results

### A. Quality of the Synthesised CSI

`[[FIGURE 2: WGAN-GP training curve, Wasserstein estimate against generator iteration. File: Result/plots/fig_gan_training_curve.png]]`

Figure 2 shows the Wasserstein estimate during training. It falls and then settles, ending at 0.562 averaged over the final iterations. The curve oscillates, which is normal for this objective and is not by itself a sign of failure.

`[[FIGURE 3: Magnitude heatmaps |H| for real (top) and synthetic (bottom) samples on a shared colour scale. File: Result/plots/fig_real_vs_synth.png]]`

We ran four checks on the generated data, and they disagree with each other in a way that turns out to be the most useful result in this section.

The first two look good. The average power profile across antennas and subcarriers matches almost exactly: the Wasserstein distance between the real and synthetic power profiles is $6.2 \times 10^{-5}$. The beam-label distributions also overlap substantially, at 0.774 (Figure 4), so the generator has learned something about which directions matter.

The third check fails. Only **9 of 32** beam classes appear in the synthetic set, against 32 of 32 in the real data. The generator reproduces the common beams and ignores the rare ones, which is mode collapse.

The fourth check fails completely. To measure realism we train a binary CNN to tell real from synthetic, on a balanced set of 500 per class with an 80/20 split, scoring on 200 held-out samples, repeated over three seeds. It reaches **100.00% ± 0.00%**. The generated channels are perfectly separable from real ones.

This combination is the interesting part. Matching first-order statistics is not enough to fool a classifier that can look at structure, and Section V-C identifies what the classifier is seeing.

`[[FIGURE 4: Beam-label distribution, real against synthetic. File: Result/plots/fig_label_dist.png]]`

### B. Effect of the Synthetic-to-Real Ratio

`[[TABLE III: Main results. Source: Result/table_main_results.md]]`

| Run | Ratio | Top-1 (%) | Top-3 (%) | Spectral eff. (%) |
|---|---|---|---|---|
| Majority class | — | 53.84 | — | — |
| R0 | 0% | 78.23 ± 1.33 | 93.17 ± 0.64 | 84.67 ± 1.18 |
| R25 | 25% | 78.47 ± 0.90 | 94.10 ± 0.65 | 84.68 ± 0.68 |
| R50 | 50% | **78.88 ± 0.96** | 93.80 ± 0.48 | **85.30 ± 0.72** |
| R75 | 75% | 78.13 ± 1.89 | 93.43 ± 0.64 | 84.18 ± 1.47 |
| R100 | 100% | 76.55 ± 0.64 | 93.33 ± 1.00 | 82.90 ± 0.88 |
| R150 | 150% | 77.43 ± 1.93 | 93.55 ± 0.30 | 83.80 ± 1.47 |
| R200 | 200% | 76.58 ± 0.93 | 93.22 ± 1.14 | 82.92 ± 0.63 |

`[[FIGURE 5 (KEY): Top-1 and Top-3 accuracy against synthetic ratio, error bars ±1 std over 3 seeds, dashed majority-class line. File: Result/plots/fig_ablation_curve.png]]`

The real-only baseline R0 reaches 78.23% ± 1.33% Top-1, which clears the 53.84% majority-class floor by 24.39 points, so the classifier is learning the channel and not the class prior.

Across the sweep the curve rises very slightly to a peak at 50%, then declines. The honest reading is that only the decline is real.

**Primary test: the trend.** The question the paper asks is whether the ratio matters at all, so the test that answers it is a single trend test over all 21 runs, and it can be specified before looking at the curve. Top-1 accuracy falls significantly as the ratio rises: Spearman ρ = −0.437 (p = 0.048), and an ordinary least squares fit gives a slope of **−1.01 percentage points per 100% of synthetic data added** (p = 0.028, $R^2$ = 0.23).

Relative spectral efficiency gives the same answer slightly more strongly: ρ = −0.527 (p = 0.014), slope −1.00 points per 100% (p = 0.012). Since spectral efficiency is the metric closest to delivered throughput, this matters more than Top-1 for a system designer.

Top-3 accuracy does not move: ρ = −0.110, p = 0.634. We return to that asymmetry in Section V-A, because it is diagnostic rather than incidental.

**Secondary test, declared exploratory.** Splitting at the observed peak gives low ratios {25%, 50%} with mean 78.67% and high ratios {75%, 100%, 150%, 200%} with mean 77.17%, a gap of **+1.50 points** (Welch p = 0.014, Mann-Whitney p = 0.025, Cohen's d = 1.27). The split point was chosen after seeing the curve, so this is exploratory and we label it as such. We put R75 in the high group, which is the less flattering choice of the two available.

**What is not significant.** The peak at 50% is 0.65 points above R0 with p = 0.534. We do not claim a benefit from augmentation at any ratio. Pairwise tests support the decline but do not survive correction: R50 against R100 is −2.33 points (Welch p = 0.031) and R50 against R200 is −2.30 points (p = 0.041), and with seven comparisons a Bonferroni threshold of α = 0.007 rejects both. This is another reason to lead with the single trend test rather than with any pair.

One statistical detail is worth stating so a reader does not have to work it out. With three seeds per group, the smallest two-sided p-value a Mann-Whitney U test can return is 0.1. It cannot resolve any pairwise effect in this design, whatever the data. We therefore quote Welch's t-test for pairwise comparisons and give this as the reason.

**Summary.** Adding GAN-synthesised CSI beyond about half the real seed set significantly degrades beam prediction. Below that it is neutral at best. The qualitative shape reported for GAN-augmented medical imaging [11], a small peak followed by decline, does appear here, but only the decline is large enough to measure.

---

## V. Why More Synthetic Data Hurts

Four measurements point the same way. The first two come from the sweep itself, the third and fourth are separate experiments designed to test the explanation.

### A. Top-1 Falls While Top-3 Stays Flat

From R0 to R200, Top-1 drops 1.65 points while Top-3 moves by +0.05 points. That pattern is specific. If synthetic data were adding generic noise, both would fall. If it were destroying the learned representation, Top-3 would fall harder. Instead the model keeps the right beam inside its top three and loses the ability to rank it first.

This is what amplifying class imbalance looks like. The real data already concentrates 53.84% of samples on beam 16. The generator covers only 9 of 32 classes and oversamples the common ones, so adding synthetic data pushes the prior further toward the dominant beams. Rare beams get crushed in the final ranking but remain in the shortlist, and the beams next to beam 16 stay easy either way.

### B. A Classifier Trained Only on Synthetic Data

Train on Synthetic, Test on Real (TSTR) asks how much usable information the generated data carries on its own. We mirror R0's split exactly: 400 synthetic training samples and 100 synthetic validation samples, three seeds, scored on the same 2000 real test samples. No real CSI enters training or model selection.

| Training data | Top-1 (%) | Top-3 (%) | Spectral eff. (%) | Beam classes |
|---|---|---|---|---|
| Majority-class floor | 53.84 | — | — | — |
| 400 synthetic (TSTR) | 59.77 ± 1.20 | 79.75 ± 2.03 | 66.62 ± 1.38 | 9 of 32 |
| 400 real (R0) | 78.23 ± 1.33 | 93.17 ± 0.64 | 84.67 ± 1.18 | 32 of 32 |

The result splits the difference between the two obvious hypotheses. Synthetic CSI is not worthless: at 59.77% it clears the majority-class floor by 5.93 points, so the generator did capture some genuine position-to-beam structure, consistent with the 0.774 label overlap. But it is a poor substitute for real data. It sits 18.47 points below R0 and retains only **24.3%** of the gain R0 makes over the floor. In spectral-efficiency terms the gap is larger still, 66.62% against 84.67%.

So each synthetic sample carries roughly a quarter of the useful signal of a real one, while carrying a full unit of weight in the loss. Once synthetic samples outnumber real ones, the gradient is dominated by the weaker source, and the measured decline follows.

### C. The Channel Matrices Have the Wrong Rank

This explains the 100% discriminability, which the first three measurements do not.

A ray-traced channel at maximum path depth 4 is a sum of at most four propagation paths,

$$\mathbf{H} = \sum_{p \le 4} \alpha_p \, \mathbf{a}(\theta_p)\, \mathbf{d}(\tau_p)^{\mathsf{T}},$$

where $\mathbf{a}(\theta_p)$ is the array steering vector for path $p$ and $\mathbf{d}(\tau_p)$ its delay response. Each term is a rank-1 outer product, so $\mathbf{H}$ has rank at most 4 regardless of its $64 \times 32$ size. Real channels here do not fill their matrix space. They lie on a thin algebraic surface inside it.

A generator built from transposed convolutions has no such constraint. Nothing in the architecture or the loss pushes its output toward low rank, so what it emits is generically close to full rank.

Figure 6 confirms this. We take the singular values of each $64 \times 32$ complex channel matrix and normalise each sample by its own largest singular value, so the comparison is of shape and not scale. The two sets could hardly be more different. The real median falls below $10^{-2}$ by the third singular value and below $10^{-7}$ by the fifth, then flattens at about $10^{-8}$, which is numerical noise. The synthetic median declines gradually and is still above $3 \times 10^{-2}$ at index 32. Real channels effectively use four or five dimensions; synthetic ones use all of them.

`[[FIGURE 6: Singular-value spectra, real against synthetic, median with interquartile band, log scale. File: Result/plots/fig_rank_spectrum.png]]`

`[[RANK90: insert the two numbers printed by the mechanism stage, "rank@90% energy: real X ± Y, synthetic Z ± W", and the effective-rank pair. One sentence: "Averaged over samples, 90% of the energy sits in X singular values for real channels and Z for synthetic ones."]]`

A convolutional neural network detects this immediately. Local phase coherence across the antenna axis is exactly what a few steering vectors produce and what a stack of transposed convolutions does not, so the discriminator needs no subtle cue. The marginals can match perfectly, as they do here with a power-profile Wasserstein distance of $6.2 \times 10^{-5}$, and the samples remain trivially separable.

This also explains why matching the Frobenius norm fixed one problem without fixing the outcome. It removed a scalar giveaway and genuinely improved the generator, raising label overlap from 0.54 to 0.774 and classes covered from 5 to 9. Discriminability stayed at 1.000 because the remaining difference is not a scalar but the rank of the matrix.

### D. What This Implies

The failure is structural rather than a training bug, and that changes the recommendation. CSI is not a low-complexity image. Natural images occupy a diffuse manifold that a convolutional generator is well suited to cover, whereas ray-traced CSI occupies a low-rank variety that the same architecture cannot represent without being told to. Longer training, more seed data, or a larger critic would not fix a mismatch of this kind.

Three directions follow, in increasing order of how much they change. Constrain the generator's output to rank $r$ directly, for instance by generating path parameters $\{\alpha_p, \theta_p, \tau_p\}$ and composing $\mathbf{H}$ from them, which builds the physics in instead of hoping it is learned. Add a spectral penalty to the generator loss that charges for energy beyond the first few singular values. Or balance the class coverage, since the 9-of-32 collapse and the rank mismatch are separate faults and fixing one leaves the other.

The rank argument applies to any generator that produces the full matrix without a low-rank constraint, which includes the diffusion models of [4] as commonly configured. We would expect the same ceiling there, which is testable under this protocol.

---

## VI. Discussion

**What a practitioner should take from this.** On this scenario, GAN augmentation did not buy anything. If you use it anyway, keep the synthetic fraction at or below half the real set, where the effect is neutral, and do not treat synthetic samples as a way to scale up a small dataset. Measure your generator before you trust it, and measure the right things: a matching power profile told us almost nothing useful, while class coverage, TSTR, and the singular-value spectrum each predicted the downstream result correctly. Of the four, TSTR is the cheapest honest check.

**Agreement with the imaging literature.** Afif *et al.* found a peak at a small synthetic fraction then a decline for GAN-augmented MRI [11]. We see the same shape, with the peak at 50% and a significant decline after it. The principle that generative augmentation is a modest supplement rather than a bulk data source does appear to hold across these two very different signal types. The difference is in magnitude: their peak was a measurable gain, ours is not resolvable from zero. Section V gives a reason specific to CSI for why the benefit side is weaker here, and that reason does not apply to MRI images.

**Limitations.** Four points bound what we can claim.

*The channels are ray traced, not measured.* The "real" data here is real only in the sense that no generative model produced it. Conclusions about field measurements need a measured dataset such as DeepSense 6G [13]. This cuts both ways for Section V: real measured channels include diffuse scattering and noise, so they are less strictly low rank than ray-traced ones, and the rank gap we identify may be smaller in practice. The direction of the argument stands, but its size is specific to ray tracing.

*The label distribution is extreme.* One beam holds 53.84% of samples. This follows from the geometry, since the retained users form a 36 m by 550 m strip and most of them see the base station from a similar angle. A more spread-out user set would have a flatter label distribution, and mode collapse would then cost less, because the classes the generator drops would carry more of the data.

*One scenario, one band, one array, one generator.* Everything here is `o1_28` at 28 GHz with a 64-element ULA and WGAN-GP. We do not know how the ratio curve moves with environment, frequency, array geometry, or generator family. The real-data budget is also fixed at 500 samples, and the best ratio may well depend on it.

*Three seeds limit statistical power.* Three seeds per ratio is enough for the trend test over 21 runs but not for pairwise comparisons, as the Mann-Whitney floor in Section IV-B shows. Differences between adjacent ratios should not be read as real. We chose to keep three seeds across seven ratios rather than more seeds across fewer, because the trend is the question and the trend test uses every run.

---

## VII. Conclusion

We asked how much GAN-synthesised CSI helps mmWave beam prediction, and swept the answer rather than assuming it. Across 21 runs with a fixed classifier, a fixed 500-sample real seed set, and seven synthetic-to-real ratios, adding synthetic data made beam prediction worse: Top-1 accuracy fell 1.01 points and relative spectral efficiency fell 1.00 points per 100% of synthetic data added, both significant. The small peak at 50% is not distinguishable from the real-only baseline.

The reason is specific and measurable. The generator covers 9 of 32 beam classes, a classifier trained only on synthetic data keeps about a quarter of the useful signal real data provides, and the generated channel matrices are close to full rank where ray-traced ones are close to rank 4. The last point is the one we would most want others to test, because it says the problem is an architectural mismatch rather than insufficient training, and it predicts that a generator built around a small number of propagation paths should behave differently.

Future work: build that low-rank generator and rerun this sweep; repeat the protocol on measured data such as DeepSense 6G [13]; vary the real-data budget to see whether the best ratio moves with it; and apply the same four fidelity checks to diffusion-based channel generators, where the rank argument predicts a similar ceiling.

---

## VIII. Reproducibility

Every architecture, hyperparameter, and split is given in Section III. Code, the DeepMIMO configuration, and all per-run results are at [[REPOSITORY URL]]. Experiments ran on one NVIDIA T4 (Kaggle) with DeepMIMO 4.0.3, scenario `o1_28`, in PyTorch.

Three choices are worth repeating because they decide whether the comparison is fair. The validation set is 100 real samples in every condition, so model selection never sees synthetic data. Synthetic data is added, never substituted, so the real content is identical across runs. No hyperparameter was tuned per run. A fresh synthetic set is drawn for every seed, so the result describes the generator and not one sample of it.

---

## IX. References

[1] M. Alrabeiah and A. Alkhateeb, "Deep Learning for mmWave Beam and Blockage Prediction Using Sub-6GHz Channels," *arXiv preprint arXiv:1910.02900*, 2019.

[2] N. N. Bhat, R. Berkvens, and J. Famaey, "CSI4Free: GAN-Augmented mmWave CSI for Improved Pose Classification," *arXiv preprint arXiv:2406.18684*, 2024.

[3] N. Khan, A. Abdallah, A. Celik, A. M. Eltawil, and S. Coleri, "Digital Twin-Assisted Explainable AI for Robust Beam Prediction in mmWave MIMO Systems," *arXiv preprint arXiv:2507.14180*, 2025.

[4] T. Lee, J. Park, H. Kim, and J. G. Andrews, "Generating High Dimensional User-Specific Wireless Channels using Diffusion Models," *arXiv preprint arXiv:2409.03924*, 2024.

[5] A. Alkhateeb, "DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications," in *Proc. Information Theory and Applications Workshop (ITA)*, San Diego, CA, Feb. 2019.

[6] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative Adversarial Nets," in *Advances in Neural Information Processing Systems (NIPS)*, 2014.

[7] M. Mirza and S. Osindero, "Conditional Generative Adversarial Nets," *arXiv preprint arXiv:1411.1784*, 2014.

[8] M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein GAN," *arXiv preprint arXiv:1701.07875*, 2017.

[9] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved Training of Wasserstein GANs," in *Advances in Neural Information Processing Systems (NIPS)*, 2017.

[10] J. Wen, J. Kang, D. Niyato, Y. Zhang, J. Wang, B. Sikdar, and P. Zhang, "Generative AI for Data Augmentation in Wireless Networks: Analysis, Applications, and Case Study," *arXiv preprint arXiv:2411.08341*, 2024.

[11] M. M. Afif, A. Al Noman, K. M. T. Kabir, M. M. Ahmmed, M. M. Rahman, M. Mahmud, and M. A. Babu, "Proportional Sensitivity in Generative Adversarial Network (GAN)-Augmented Brain Tumor Classification Using Convolutional Neural Network," *arXiv preprint arXiv:2506.17165*, 2025.

[12] S. Rezaie, E. de Carvalho, and C. N. Manchón, "A Deep Learning Approach to Location- and Orientation-aided 3D Beam Selection for mmWave Communications," *arXiv preprint arXiv:2110.06859*, 2021.

[13] A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, N. Srinivas, U. Demirhan, and J. Morais, "DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset," *IEEE Communications Magazine*, vol. 61, no. 9, pp. 122–128, 2023.

[14] H. Zhang, Z. Lu, X. Zhang, and J. Wang, "Data Augmentation of Bridging the Delay Gap for DL-based Massive MIMO CSI Feedback," *arXiv preprint arXiv:2308.00478*, 2023.

---

## Appendix A: What Is Left To Fill

| # | Item | Where | Source |
|---|---|---|---|
| 1 | Author block | Top | You |
| 2 | Repository URL | §VIII | You |
| 3 | `[[RANK90]]` numbers | §V-C | The two lines printed by the mechanism stage |
| 4 | Table I (architecture and hyperparameters) | §III | Optional. Everything in it is already stated in the prose; add it only if you have the page budget |
| 5 | Figures 1 to 6 | Throughout | `Result/plots/`, all six already generated |
| 6 | Reference refresh | §IX | Check for work published since drafting |

All 14 references were verified against arXiv when the earlier draft was prepared. Renumber if you reorder the text and reformat to the venue's exact IEEE style. Where a preprint has since appeared in a journal or conference, prefer the published version.
