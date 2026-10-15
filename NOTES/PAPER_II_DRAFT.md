# Off-Manifold by Construction: Rethinking Generative Data Augmentation for mmWave Beam Prediction

**Status: analysis and proposal. Read this before using it.**

This is a companion to the ratio-sweep study (`PAPER_FINAL.md`). The diagnosis in Sections III and IV is backed by measurements we already have. The method in Section V has **not been run**. Nothing in Section V is a result, and every expected outcome is written as a falsifiable prediction with the protocol that would test it. Do not let a prediction drift into the past tense during editing.

**Target format:** IEEE conference, two columns, about 6 pages. Suitable for a workshop or short-paper track, where an argued negative analysis plus a concrete proposal is a normal contribution.

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

Training a generative model on scarce channel state information (CSI) and adding synthetic samples is a common answer to data scarcity in millimeter-wave (mmWave) beam prediction. In a companion study we swept the synthetic-to-real ratio across 21 controlled runs and found that accuracy falls as the ratio rises, by 1.01 percentage points per 100% of synthetic data added. This paper asks why, and argues the cause is not generator quality. We make two claims. First, when the beam label is computed from the channel by a fixed rule, the label is a closed-form function of the classifier's own input, so synthetic samples add no information. They are off-manifold inputs carrying correct labels, which teaches the network the right mapping in a region the test distribution never visits. Second, a ray-traced channel concentrates its energy in a handful of singular values, while a generator built from transposed convolutions emits generically full-rank matrices. We measure this gap directly: real channels keep almost all of their energy in about four to five singular values, synthetic ones spread it across all 32, and a discriminator separates the two with 100.00% ± 0.00% accuracy even though their average power profiles agree to within a Wasserstein distance of 6.2 × 10⁻⁵. The argument applies to any generator that emits the full matrix without a low-rank constraint, diffusion models included. We then propose a different construction: estimate the propagation paths from each measured channel, regress the path parameters on user position, and recompose channels from the physics. This reduces the learning problem from 4096 real dimensions to roughly 20 and makes the output low rank by construction. We give the protocol, the predictions, and the conditions under which each prediction would be wrong.

**Index Terms**: millimeter wave, beam prediction, data augmentation, generative models, channel modelling, low-rank channels, sample complexity.

---

## I. Introduction

Learning-based mmWave beam prediction needs labelled channel measurements, and measurements are expensive. The popular response is to train a generative adversarial network (GAN) on the small real dataset and manufacture more samples [2], [10]. In a companion study we treated the amount of synthetic data as the experimental variable rather than a fixed setting, and swept it from 0% to 200% of a 500-sample real seed set across 21 runs with a frozen classifier and recipe [15]. Accuracy declined as the ratio grew. Top-1 fell 1.01 percentage points per 100% synthetic added (p = 0.028) and relative spectral efficiency fell 1.00 points per 100% (p = 0.012).

The natural reading of such a result is that the generator was not good enough. This paper argues that reading is wrong, and that the usual remedies, longer training, a larger critic, more seed data, or a switch to diffusion, would not have changed the outcome.

We give two reasons, one about information and one about geometry, and we support both with measurements already in hand.

The information argument is simple and, once stated, hard to escape. In the standard setup the beam label is produced by applying a fixed codebook rule to the channel. The classifier is then given that same channel. So the label is a closed-form function of the network's input. There is nothing about the world that the network does not already have, and therefore nothing that an extra sample can supply. What a synthetic sample does supply is an input drawn from the wrong place, paired with a label that is correct for that wrong place. We call this being off-manifold by construction.

The geometry argument explains why the synthetic inputs are in the wrong place and why no amount of training fixes it. A ray-traced channel is a sum of a few dominant propagation paths, each contributing a rank-1 term, so its energy sits in a small number of singular values. A stack of transposed convolutions has nothing pushing it toward that structure, so it emits matrices that are generically close to full rank. The two sets of matrices are then trivially separable, which is exactly what we measure.

Taken together these say the failure is architectural and definitional rather than a question of effort. That changes what to build.

**Contributions.**

1. We show that in the standard full-CSI beam-prediction setup the label is a closed-form function of the classifier input, and we draw out what this means for augmentation: synthetic samples cannot add information, only misplaced input density.
2. We give the rank argument in general form, for $P$ paths rather than an assumed small $P$, and support it with a measured singular-value comparison between real and generated channels.
3. We argue the rank argument applies to diffusion models as commonly configured, so switching generator family is not a fix.
4. We observe that ray tracing is deterministic, so the conditional distribution of the channel given position is a point mass. The correct problem is spatial interpolation, not generation, and that removes the need for adversarial training entirely.
5. We specify a physics-parametric synthesis pipeline that is low rank by construction and reduces the learning problem by two orders of magnitude, together with a validation protocol and the conditions that would falsify each prediction.
6. We identify the task redefinition, noisy or partial CSI, that makes generative modelling meaningful again, and explain why it is the setting where the proposal should be evaluated.

---

## II. Setting and Prior Measurements

We summarise the companion setup only as far as this paper needs [15]. DeepMIMO [5] scenario `o1_28` at 28 GHz, a 64-element uniform linear array at the base station, single-antenna users, 32 OFDM subcarriers selected across a 512-bin band, and up to 25 ray-traced paths per user. Each channel $\mathbf{H} \in \mathbb{C}^{64 \times 32}$ becomes a two-channel real tensor $\mathbf{X} \in \mathbb{R}^{2 \times 64 \times 32}$, normalised by its Frobenius norm. A seed set of 500 users is the entire real training budget and 2000 disjoint users form the test set.

The beam codebook has $K = 32$ DFT vectors and the label is

$$k^\star = \arg\max_{k} \frac{1}{S}\sum_{s=1}^{S} \left| \mathbf{w}_k^H \mathbf{h}_s \right|^2 . \tag{1}$$

A position-conditioned WGAN-GP [7], [9] supplies synthetic channels, labelled by applying (1) to the generated channel rather than by any learned model.

Four measurements from that study carry into this one, and the tension between them is the reason this paper exists.

| Measurement | Value | Reading |
|---|---|---|
| Power-profile Wasserstein distance | 6.2 × 10⁻⁵ | Marginals match almost exactly |
| Beam-label overlap | 0.774 | Some task-relevant structure learned |
| Beam classes covered | 9 of 32 | Mode collapse |
| Real vs synthetic discriminability | 100.00% ± 0.00% | Trivially separable |
| Train on synthetic, test on real | 59.77% ± 1.20% | Against a 53.84% floor and 78.23% for real data |
| Top-1 trend over 21 runs | −1.01 pp per 100% (p = 0.028) | Adding synthetic data hurts |

A generator whose first-order statistics match to five decimal places, and which is nonetheless caught every single time, is not failing at fidelity in any ordinary sense. Something categorical separates the two sets. Sections III and IV identify it.

---

## III. When Augmentation Cannot Add Information

### A. The label is a function of the input

Write the classifier input as $\tilde{\mathbf{X}} = \mathbf{X} / \|\mathbf{X}\|_F$ and the label as $k^\star$ from (1). Because (1) is an argmax over beams of a quantity that scales positively with $\|\mathbf{X}\|_F^2$, the normalisation does not change the answer, so $k^\star(\tilde{\mathbf{X}}) = k^\star(\mathbf{X})$.

The label is therefore a deterministic function of the network's own input, computable in closed form: project onto the 32 codebook vectors, take squared magnitudes, average over subcarriers, take the argmax. No measurement noise enters, because none is added anywhere in the pipeline. The mapping is exact.

Two consequences follow.

**There is no information to add.** Supervised learning estimates a conditional distribution $p(y \mid x)$. Here that distribution is a point mass at a value any party can compute from $x$ alone. Additional samples, real or synthetic, cannot reduce uncertainty about the mapping, because there is none. They can only help the network approximate a known function. The task is function approximation under a sample budget, not inference under uncertainty, and conflating the two is what makes generative augmentation look like a reasonable idea here.

**A 32-number oracle feature exists.** The vector of beam gains $g_k = \frac{1}{S}\sum_s |\mathbf{w}_k^H \mathbf{h}_s|^2$, for $k = 0 \dots 31$, determines the label exactly. A model given those 32 numbers should be essentially perfect. We state this as a prediction to be checked rather than a result, in Section VII-A, because it bounds everything else in the paper.

### B. Why the baseline is not near-perfect

The real-data baseline reaches 78.23% ± 1.33%, not 100%. This is not a data limit and not an impossibility.

The classifier is a stack of $3\times3$ convolutions with max-pooling, followed by a flatten and a fully connected layer. The flatten does let it mix all 64 antennas, so the architecture is not formally incapable. But (1) needs a global transform along the antenna axis followed by a squaring, and small local kernels are a poor basis for expressing that. With 400 training samples in a 4096-dimensional input space, the network fits a rough approximation of an exactly specified operation.

So the gap from 78.23% to 100% measures the mismatch between the architecture and the function, not the scarcity of data. Adding samples of any kind addresses the wrong term.

### C. Synthetic samples as misplaced density

If the mapping is already determined, what does a synthetic sample do?

Its label is correct, because (1) is applied to the generated channel honestly. Its input, however, is drawn from the generator's distribution, which Section IV shows is disjoint from the real one in a way a small CNN detects perfectly. The sample therefore constrains the learned function at a point the test distribution assigns no mass to.

In a function-approximation problem with a fixed capacity budget, that is not neutral. Capacity and gradient steps spent fitting the function on an irrelevant region are taken from the region that matters. The expected effect is a decline that grows with the proportion of such samples, which is the trend the companion study measures.

This framing also explains the pattern of the decline better than mode collapse alone does. Top-1 fell 1.65 points from 0% to 200% while Top-3 moved by +0.05. Off-manifold constraints perturb the fine ranking among neighbouring beams, which is where the decision is close, while leaving the coarse angular region intact. Mode collapse contributes too, by skewing the prior toward the nine covered classes, but the two mechanisms are separable and the off-manifold one does not require any collapse to operate.

**The useful corollary.** Fidelity improvements do not help. A generator that halves its discriminability but remains off-manifold still places its samples in the wrong region. Only a generator whose support lies on the real manifold changes the sign of the effect, and support is a structural property, not something a fidelity metric reports. This is why we treat the rank gap in Section IV as the operative measurement and the power-profile distance of 6.2 × 10⁻⁵ as nearly uninformative.

---

## IV. The Rank Gap

### A. The structure of a ray-traced channel

A ray-traced channel is a sum over discrete propagation paths,

$$\mathbf{H} = \sum_{p=1}^{P} \alpha_p \, \mathbf{a}(\theta_p)\, \mathbf{d}(\tau_p)^{\mathsf{T}}, \tag{2}$$

with complex gain $\alpha_p$, array steering vector $\mathbf{a}(\theta_p) \in \mathbb{C}^{64}$ for angle $\theta_p$, and delay response $\mathbf{d}(\tau_p) \in \mathbb{C}^{32}$ for delay $\tau_p$. Each term is a rank-1 outer product, so

$$\operatorname{rank}(\mathbf{H}) \le \min(P, 64, 32). \tag{3}$$

We requested $P = 25$ paths, so (3) alone bounds the rank at 25 and proves little. The bound is not the point. What matters is that higher-order reflections at 28 GHz arrive with much lower power than the first few, so the singular values decay quickly and the **effective** rank, the number of singular values holding essentially all the energy, is far below the algebraic bound. We emphasise this because an earlier version of our own analysis asserted a small $P$ from the configuration, which was incorrect. The low-rank property here is empirical, and that is the stronger form of the claim.

A generator built from transposed convolutions has no term in its architecture or loss that references (2) or (3). Its output is a dense function of a latent vector, so it is generically full rank. The generator is not failing to learn the constraint. It was never asked to.

### B. Measured spectra

For each sample we take the singular values of the $64 \times 32$ complex matrix and divide by that sample's own largest singular value, so the comparison is of shape rather than scale. Scale was already matched: the generator projects its output onto the unit Frobenius sphere the real data occupies.

The two populations barely overlap. The real median falls below $10^{-2}$ by the third singular value and below $10^{-7}$ by the fifth, then flattens near $10^{-8}$, which is numerical noise rather than signal. The synthetic median declines gradually and is still above $3 \times 10^{-2}$ at index 32. Real channels use four or five dimensions of the available 32. Synthetic ones use all of them.

`[[FIGURE 1: Singular-value spectra, real against synthetic, median with interquartile band, log vertical scale. File: Result/plots/fig_rank_spectrum.png]]`

`[[RANK90: insert the rank-at-90%-energy pair and the effective-rank pair printed by the mechanism stage.]]`

### C. Why a small CNN wins immediately

A low effective rank in the antenna-delay plane shows up in the sample domain as strong local phase coherence along the antenna axis. A few steering vectors are complex exponentials in the antenna index, so neighbouring antennas have an almost constant phase increment. That is a short-range, highly regular pattern, and a $3\times3$ convolution detects it in one layer.

A transposed-convolution stack produces phase structure that is smooth but not of this form. The discriminator does not need a subtle cue, which is why discriminability sits at 100.00% with zero variance across seeds while the marginals agree to five decimal places. First-order statistics simply do not see the property that distinguishes the two sets.

This also explains an earlier observation in the companion study that otherwise looks like a failed fix. Replacing a final `tanh` with a unit-Frobenius projection removed a scalar giveaway and genuinely improved the generator, raising label overlap from 0.54 to 0.774 and class coverage from 5 to 9. Discriminability stayed at 1.000 because the remaining difference is not a scalar. It is the rank of the matrix, and normalising a matrix does not change its rank.

### D. The argument does not depend on the generator being a GAN

Nothing in Section IV-A used adversarial training. The argument needs only that the generator emits the full $64 \times 32$ matrix and that no term constrains its spectrum.

A denoising diffusion model with a U-Net backbone satisfies both conditions as usually configured. It would likely produce better marginals and more plausible samples than our WGAN-GP, and we expect it to remain separable for the same reason, because better marginals do not imply correct support. Lee *et al.* report strong results for diffusion-generated channels [4], and we are not disputing their fidelity numbers. We are saying fidelity is the wrong quantity to improve if the support is wrong, and that the test is cheap: run the spectrum comparison of Section IV-B on diffusion samples. Section VII-D states this as a prediction with its falsification condition.

---

## V. Physics-Parametric Synthesis

The diagnosis points at its own remedy. If the problem is that generated matrices violate (2), then generate the parameters of (2) and compose the matrix from them.

### A. The pipeline

**Step 1, extract paths.** For each of the 500 real seed channels, estimate $\{\alpha_p, \theta_p, \tau_p\}_{p=1}^{L}$ by sparse recovery on a two-dimensional angle-delay dictionary. Orthogonal matching pursuit is adequate here and subspace methods such as ESPRIT or SAGE are the higher-resolution alternatives. With 64 antennas, 32 subcarriers and energy concentrated in a handful of paths, the problem is well conditioned. Choose $L$ from the measured spectrum, around 5, and sort paths by delay so the parameter vector has a consistent ordering. Without a canonical order the regression target is ambiguous under permutation, which is a silent and expensive mistake.

**Step 2, learn position to paths.** Fit $f: (x, y) \mapsto \{\alpha_p, \theta_p, \tau_p\}$ with a small multilayer perceptron on the 500 seed pairs. Represent each complex gain as magnitude and phase, or as a real and imaginary pair if phase wrapping causes trouble.

**Step 3, sample and compose.** Draw new positions within the convex hull of the seed positions, evaluate $f$, and build $\hat{\mathbf{H}}$ from (2). Label with the same rule (1) used everywhere else.

### B. Why this is a different kind of object

**The output is low rank by construction.** $\hat{\mathbf{H}}$ is a sum of $L$ rank-1 terms, so it satisfies (3) exactly. The property the discriminator exploits is no longer available to it. This is a guarantee from the construction, not a behaviour to be trained in and monitored.

**The dimensionality collapses.** The raw problem asks a model to learn a distribution over $2 \times 64 \times 32 = 4096$ real numbers from 500 samples. The parametric problem asks for $4L$ numbers, about 20 for $L = 5$. That is a reduction of roughly two orders of magnitude, and it is the quieter half of why the original approach failed. Covering a 4096-dimensional distribution from 500 samples is hopeless, and collapsing to 9 of 32 beam classes is the expected symptom rather than a surprise. A 20-dimensional regression from 500 samples is routine.

**The samples land on the manifold.** Composed channels obey the same generative equation as the real ones, so they are not off-manifold in the sense of Section III-C. Their labels are computed by the same rule. They are, as far as (2) is concerned, channels.

### C. Ray tracing is deterministic, so this is interpolation

There is a sharper observation available, and it removes the need for a generative model altogether.

Ray tracing is a deterministic simulation. For a fixed position in a fixed environment there is exactly one channel. The conditional law $p(\mathbf{H} \mid x, y)$ is therefore a point mass, not a distribution with spread to be modelled. A generative model fitted to it has nothing stochastic to capture.

What creates apparent diversity is not randomness but **positions that were never measured**. The honest description of the task is spatial interpolation of the channel: given paths at 500 known locations, predict paths at new locations. That is regression plus a physics composition, and it needs no adversarial training, no critic, no gradient penalty, and no mode-coverage diagnostics.

Framing it as regression also makes the failure mode legible and bounded. Path angles and delays vary smoothly with position in open stretches of a street canyon, and they jump at blockage boundaries where a path appears or disappears. So we expect good interpolation in open regions and errors concentrated at blockage edges. That is a describable, testable, localised weakness, which is a much better position than a generator that fails everywhere for reasons its own fidelity metrics do not report.

This argument is specific to deterministic ray tracing. On measured data the conditional law is genuinely stochastic, from noise, from moving scatterers, and from hardware effects, and a generative model over path parameters becomes appropriate again. The parametric construction carries over; only the final stage changes from a regressor to a conditional density over roughly 20 dimensions, which remains a far easier problem than 4096.

### D. What this does not fix

We should be clear about the limits of our own proposal.

Physics-parametric synthesis produces on-manifold channels at new positions, which is a real improvement in sample quality. It does not repeal Section III. On the current task the label is still a closed-form function of the input, so the added samples still carry no new information about the mapping. They should help the network approximate the function over a better-covered input region, which is a genuine but bounded benefit, and we expect the representation fix of Section VI-A to capture most of the same gain more cheaply.

The place where this construction should matter is the task of Section VI-B, where information genuinely is missing. We say so in advance rather than claiming a general win.

---

## VI. Making the Task Non-Degenerate

### A. Fix the representation first

Before any augmentation, change what the classifier sees. Take the two-dimensional FFT of $\mathbf{H}$ along the antenna and subcarrier axes. In the resulting angle-delay domain a channel obeying (2) is $L$ sparse spikes rather than a dense interference pattern, and the codebook projection in (1) becomes close to a coordinate read-out.

This is cheap and it attacks the term that actually limits the baseline, namely the mismatch between local convolutions and a global transform. We predict a substantial improvement over 78.23% with no augmentation at all, and we think any augmentation study on this task should report it as the baseline. Reporting augmentation gains over a representation-limited baseline risks attributing to data what belongs to architecture.

### B. Restore the missing information

The deeper fix is to pose a task where something is actually unknown. Three options, in order of how little they change the rest of the setup.

**Noisy CSI.** Give the classifier $\mathbf{H} + \mathbf{N}$ at a stated SNR while computing the label from the clean $\mathbf{H}$. The mapping from input to label is now stochastic, $p(y \mid x)$ has real spread, and more data genuinely reduces uncertainty. This is the smallest change that makes the whole augmentation question meaningful.

**Partial CSI.** Give the classifier a subset of antennas, a subset of subcarriers, or the measured gains of a few probed beams. The last is the real operational problem: probe 4 beams, infer the best of 32. Here the information gap is large and obviously worth closing with data.

**Cross-band prediction.** Predict the mmWave beam from a sub-6 GHz channel, which is what Alrabeiah and Alkhateeb did [1]. That mapping is not closed-form, which is precisely why it is a learning problem.

Most of the beam-prediction literature predicts from position, images, LiDAR, or sub-6 GHz observations, and those are all genuine inference problems. Predicting the mmWave beam from the full mmWave channel is the degenerate corner, and augmentation studies that use it are measuring something other than what they intend.

---

## VII. Validation Protocol and Predictions

Each item states what we will run, the prediction, and what outcome would falsify it. None of these have been run.

### A. The task is degenerate

Train the existing classifier on the 32 beam gains $g_k$ instead of the raw channel, same splits and recipe.

*Prediction:* accuracy near 100%, confirming the label is exactly determined by the input and that 78.23% reflects architecture rather than data.
*Falsified if:* accuracy stays near 78%, which would mean the bottleneck is optimisation or label noise we have not accounted for, and Section III would need revisiting.

### B. The representation is the bottleneck

Train on the 2D FFT angle-delay representation with a comparable small CNN, no augmentation.

*Prediction:* a large gain over 78.23%, closing much of the gap to A.
*Falsified if:* no improvement, which would mean the input format is not the limiting factor and Section VI-A is wrong.

### C. Parametric synthesis lands on the manifold

Run the Section V pipeline, then rerun the four fidelity checks.

*Prediction:* discriminability falls sharply toward 50%, class coverage rises well above 9 of 32, the singular-value spectra of composed and real channels overlap, and train-on-synthetic-test-on-real rises substantially above 59.77%.
*Falsified if:* discriminability stays near 100%. That would mean something other than rank separates the sets, and Section IV would be wrong. This is the single most informative experiment in the paper and should be run first among the Section V items.

### D. The rank argument is not GAN-specific

Train a diffusion model on the same seed set and run the spectrum comparison.

*Prediction:* spectra resemble the WGAN-GP ones more than the real ones, and discriminability stays high.
*Falsified if:* diffusion samples show the real decay profile, which would mean the U-Net has an implicit low-rank bias we did not anticipate, and Section IV-D would be wrong.

### E. Augmentation helps once information is missing

Repeat the full 0% to 200% ratio sweep under the noisy-CSI task of Section VI-B, using parametric synthesis.

*Prediction:* the curve changes sign, showing a genuine interior optimum rather than monotone decline, because synthetic samples now supply information the classifier lacks.
*Falsified if:* the decline persists. That would be the most interesting outcome in the paper, since it would mean on-manifold synthetic data still fails to help under genuine uncertainty, and the explanation would have to lie elsewhere.

Run order: A and B first, because they are hours of work and they bound everything else. Then C. Then E. D is independent and can run whenever compute allows.

---

## VIII. Limitations

**No results for the proposal.** Sections V and VI are untested. The diagnosis in III and IV rests on measurements, the remedy rests on argument.

**Path extraction has its own error.** Orthogonal matching pursuit on a discrete dictionary suffers from off-grid angles and delays, and weak paths are recovered poorly. Extraction error propagates into every composed channel, and we have not characterised it. If it is large, the composed channels may fail the spectrum test for reasons unrelated to the argument in Section IV.

**Blockage breaks smooth regression.** Position-to-paths is smooth in open stretches and discontinuous where paths appear or vanish. A plain multilayer perceptron will blur those boundaries, and blurred boundaries are exactly where beam prediction is hardest.

**The determinism argument is specific to simulation.** Section V-C holds for ray tracing. On measured channels the conditional law is genuinely stochastic and a generative model is warranted. Our claim is about the right tool for this data, not a general claim against generative channel models.

**The closed-form argument is specific to full-CSI input.** Section III dissolves the moment the input is noisy or partial, which is the point of Section VI-B. Readers should not take Section III as an argument against generative augmentation in wireless generally. It is an argument against it for this task as commonly posed.

**One scenario, one band, one array.** Everything is `o1_28` at 28 GHz with a 64-element ULA. The effective rank we measure is a property of this environment, and a richer scattering environment would raise it and weaken the rank gap.

---

## IX. Conclusion

A measured failure of GAN augmentation for mmWave beam prediction is usually read as a generator quality problem. We argued it is not. When the label is computed from the channel by a fixed rule, the label is a closed-form function of the classifier's input, so synthetic samples cannot add information and can only add input density in the wrong place. And a ray-traced channel concentrates its energy in four or five singular values while an unconstrained convolutional generator emits full-rank matrices, which is why a discriminator separates the two perfectly while their power profiles agree to within 6.2 × 10⁻⁵.

Both arguments say the same thing in different language: the generated objects are not channels in the sense that matters, and improving how closely they resemble channels on first-order statistics does not change that.

The constructive consequence is to stop generating matrices and start generating the physics that produces them. Estimating propagation paths, regressing them on position, and recomposing turns a 4096-dimensional generative problem into a roughly 20-dimensional regression whose output satisfies the rank constraint by construction. For deterministic ray-traced data this is interpolation rather than generation and needs no adversarial machinery at all. For measured data the same parametrisation holds with a conditional density in place of the regressor.

None of that will show up as a downstream gain until the task itself is repaired, by adding receiver noise or restricting the classifier to partial observations, so that there is information worth supplying. Section VII says exactly what we will run and what would prove each claim wrong.

---

## X. References

Entries [1] to [14] follow the companion paper and were verified against arXiv at drafting time. Entry [15] is the companion study. The sparse-recovery references in Section V-A are named in prose and marked below because their bibliographic details have not been checked in this draft. **Do not submit with placeholders in the bibliography.**

[1] M. Alrabeiah and A. Alkhateeb, "Deep Learning for mmWave Beam and Blockage Prediction Using Sub-6GHz Channels," *arXiv preprint arXiv:1910.02900*, 2019.

[2] N. N. Bhat, R. Berkvens, and J. Famaey, "CSI4Free: GAN-Augmented mmWave CSI for Improved Pose Classification," *arXiv preprint arXiv:2406.18684*, 2024.

[4] T. Lee, J. Park, H. Kim, and J. G. Andrews, "Generating High Dimensional User-Specific Wireless Channels using Diffusion Models," *arXiv preprint arXiv:2409.03924*, 2024.

[5] A. Alkhateeb, "DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications," in *Proc. Information Theory and Applications Workshop (ITA)*, San Diego, CA, Feb. 2019.

[7] M. Mirza and S. Osindero, "Conditional Generative Adversarial Nets," *arXiv preprint arXiv:1411.1784*, 2014.

[9] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved Training of Wasserstein GANs," in *Advances in Neural Information Processing Systems (NIPS)*, 2017.

[10] J. Wen, J. Kang, D. Niyato, Y. Zhang, J. Wang, B. Sikdar, and P. Zhang, "Generative AI for Data Augmentation in Wireless Networks: Analysis, Applications, and Case Study," *arXiv preprint arXiv:2411.08341*, 2024.

[13] A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, N. Srinivas, U. Demirhan, and J. Morais, "DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset," *IEEE Communications Magazine*, vol. 61, no. 9, pp. 122–128, 2023.

[15] [[COMPANION PAPER: your ratio-sweep study. Fill in once it has a venue or an arXiv number.]]

[16] [[ORTHOGONAL MATCHING PURSUIT: standard reference, verify before submission.]]

[17] [[ESPRIT: standard reference, verify before submission.]]

[18] [[SAGE channel parameter estimation: standard reference, verify before submission.]]

Renumber in order of first citation once the text is final. The gaps above are left deliberately so numbering stays aligned with the companion paper during editing.

---

## Appendix A: What Is Left To Do

| # | Item | Where | Note |
|---|---|---|---|
| 1 | Author block | Top | You |
| 2 | Experiments A and B | §VII | Hours of work, and they bound every other claim. Do these first |
| 3 | Experiment C | §VII | The decisive test of Section IV |
| 4 | `[[RANK90]]` numbers | §IV-B | From the mechanism stage output |
| 5 | References [15] to [18] | §X | Verify, do not write from memory |
| 6 | Decide the framing | Whole paper | See below |

**On framing.** This draft is written as an analysis-and-proposal paper, which is honest about having no results for Section V and is a normal fit for a workshop track. If experiments A, B and C come back as predicted, the stronger move is to rewrite it as a full paper with those results in Section VII, move the predictions into a short discussion, and submit to a main track. Decide after C, not before.
