# Phase 0 — Mask-VAE shape teacher: specification and assumptions

This document records exactly what the Phase 0 specification fixed, what it left
open, and which choice was made for every open item. Anything listed under
"Assumptions" is an implementation decision, not something the specification or
the ShapeMoE paper prescribes.

## What the specification fixes

| Specification | Implementation |
|---|---|
| `M -> E_T(M) -> h_T` | `MaskVAETeacher.encoder` + global average pooling |
| `mu_T = W_mu h_T + b_mu` | `MaskVAETeacher.fc_mu` |
| `log sigma_T^2 = W_sigma h_T + b_sigma` | `MaskVAETeacher.fc_logvar` |
| `q_T(z\|M) = N(mu_T, diag(sigma_T^2))` | `MaskVAETeacher.encode` |
| `z_T = mu_T + sigma_T * eps` | `MaskVAETeacher.reparameterize` |
| `M_hat_T = D_T(z_T)` | `MaskVAETeacher.decode` |
| `L_rec = BCE(M, M_hat_T)` | `MaskVAELoss`, logit-space BCE |
| `L_prior = KL(q_T(z\|M) \|\| N(0, I))` | `gaussian_kl_to_standard_normal` |
| `L_VAE = L_rec + beta * L_prior` | `MaskVAELoss.forward` |

## Relation to the ShapeMoE paper

The paper (arXiv 2508.01664) contributes only the two-headed Gaussian encoder.
Section 3.3, Eq. (1) splits the shape encoder `E_S` into an expectation encoder
`E_mu` and a variance encoder `E_sigma` fed by a shared input, which is the
structure `fc_mu` / `fc_logvar` reproduces.

Everything else in Phase 0 is outside the paper. ShapeMoE has no teacher, no
decoder from the latent, and no KL term: its total objective is
`L = L_CE + L_CV2` (Eq. 5). The reconstruction and prior terms here come from the
Phase 0 specification, not from the paper.

Two deliberate departures:

1. **Log-variance instead of Softplus.** Eq. (2) of the paper samples
   `l_o = mu + Softplus(sigma) * eta`. Phase 0 predicts `log sigma^2` instead, so
   that the KL against `N(0, I)` has the standard closed form
   `0.5 * sum(mu^2 + sigma^2 - 1 - log sigma^2)`. Using Softplus would require
   `log(Softplus(sigma))` inside the KL, which is an avoidable source of
   numerical trouble. Keeping both the teacher and a future student on
   log-variance also keeps the Phase 2 distillation KL closed-form.
2. **`E_T` reads the mask directly.** The paper's `E_S` consumes a mask embedding
   `e_m` produced by SAM2's mask encoder `E_M`. The teacher has no `E_M` stage,
   so Phase 0 has no dependency on SAM or SAM2.

## Assumptions

Items the specification did not state, with the value chosen here.

### Latent dimension

`latent_dim = 64`. The specification gives no value, and the paper only writes
`l_o` in `R^d` without a number. This must match the router input dimension when
Phase 1 is built.

### Encoder and decoder architecture

- `E_T`: four stride-2 blocks (Conv-BatchNorm-ReLU), channels `1 -> 32 -> 64 ->
  128 -> 256`, reducing 256x256 to 16x16, then global average pooling to
  `h_T` in `R^256`.
- `D_T`: a linear layer projecting `z_T` to a 256x4x4 seed, six transposed
  convolution blocks upsampling 4 -> 256, then a 3x3 convolution to one channel.

The decoder is not an exact mirror of the encoder. Mirroring would need a linear
layer of `latent_dim x 256 x 16 x 16` (about 4.2M parameters spent on a single
projection); starting from a 4x4 seed reaches the same output resolution for
roughly a sixteenth of that.

Both are configurable through `encoder_channels`, `decoder_channels` and
`decoder_seed_size`. The constructor rejects combinations whose geometry does not
reproduce `image_size`.

### The decoder emits logits

`D_T` returns raw logits and `L_rec` is computed with
`binary_cross_entropy_with_logits`. This is mathematically identical to
`BCE(M, sigmoid(logits))` but avoids the overflow of an explicit sigmoid followed
by a log.

### Reconstruction reduction and the scale of beta

`recon_reduction = "sum"`: BCE is summed over the pixels of each mask and
averaged over the batch, the usual VAE convention, so `beta = 1` is the plain
ELBO. This matters because the specification writes `L_rec` without a reduction
and the two terms differ by four orders of magnitude at 256x256. Under
`"mean"`, `L_rec` is around 0.1 while `L_prior` is in the tens, and `beta = 1`
would collapse the posterior immediately. If `"mean"` is preferred, `beta` has to
drop by roughly the pixel count.

### Beta is constant

`beta` is a single configured number, exactly as written in `L_VAE = L_rec +
beta * L_prior`. No warm-up schedule is applied. `MaskVAELoss.forward` accepts an
optional per-call `beta` override, unused by the training script, so a warm-up
can be tried without rewriting the loss if posterior collapse shows up.

### Masks are used at full resolution, uncropped

`M` is the 256x256 mask as the dataset produces it. No crop to the lesion
bounding box and no size normalisation. This follows `M -> E_T(M)` literally.

Consequence worth knowing: `z_T` then encodes lesion position and scale as well
as shape. If Phase 1 routing is meant to depend on shape alone, crop-normalising
the mask first would be the change to make.

### Log-variance clamping

`logvar` is clamped to `[-10, 10]` before use. This is a numerical guard that is
inactive in normal training, since it corresponds to standard deviations between
about 0.007 and 148.

### Dataset

ISIC 2018 Task 1, reusing the frozen split in
`manifests/isic2018_task1_v1.json` unchanged. The teacher consumes only
`batch["mask"]`; images are loaded by the dataset and discarded, which wastes
some data-loading time but keeps `src/data/isic2018.py` untouched.

Note this is single-object binary lesion segmentation, not the amodal
instance segmentation the paper targets. There is no visible/amodal mask pair, so
`M` is simply the ground-truth lesion mask.

## Why the loss is not in `LOSS_REGISTRY`

`Trainer` and `evaluate` call `criterion(logits, targets) -> scalar`.
`MaskVAELoss` takes the whole posterior and returns three tensors, so it cannot
satisfy that contract. Registering it would let a segmentation config select a
loss that would then fail at run time, so it is exported from `src.losses` by
name only.

## Reading the training log

- `reconstruction`, `kl`, `loss` — the three terms of the objective.
- `dice` — Dice between the thresholded reconstruction and `M`. Diagnostic only:
  it says how much shape survives the bottleneck.
- `active_units` — number of latent dimensions whose `mu` varies across the batch
  (variance above 1e-2). If this stays near zero while `kl` goes to zero, the
  posterior has collapsed and `z_T` carries nothing for later phases.

## Open questions for later phases

Deliberately not decided here, because Phase 0 does not depend on them:

- Whether `z_T` will be distilled into a student posterior, clustered to supply
  router supervision, or used to initialise expert decoders. That choice
  determines which of the metrics above is the real stopping criterion.
- Number of experts and routing sparsity. The paper's ablations settle on 4
  experts with top-1 selection (Tables 2 and 3).
- The `L_CV2` balancing loss and its weight, which the paper writes without a
  coefficient.

---

# Phase 1 — ShapeMoE segmenter: what maps to the paper, what does not

Phase 1 implements Sec. 3.3 to 3.6 of ShapeMoE on the UNet trunk this repository
already has. The teacher of Phase 0 is frozen and supplies a distillation target.

## Correspondence with Sec. 3

| Paper | Phase 1 | Faithful? |
|---|---|---|
| 3.2 stage 1: SAM image encoder + `E_M` mask embedding | UNet trunk; no `E_M`, no visible-mask prompt | Substituted |
| 3.3 Eq. (1): `E_S` = `E_mu` + `E_sigma` | `ShapeDistributionEncoder`, two heads on the pooled bottleneck | Structure yes, input differs |
| 3.4 Eq. (2): `l_o = mu + Softplus(sigma) * eta` | `ShapeAwareSparseRouter.sample_latent`, log-variance form | Deliberate deviation |
| 3.4 Eq. (3): `s = W * l_o` | `router.gate`, `nn.Linear(d, K, bias=False)` | Yes |
| 3.4 Eq. (4): `pi = Softmax(TopK(s, k))` | `router.forward` | Yes |
| 3.5 replicate only the lightweight hyper-network | `ExpertMaskHeads`, K 1x1 convolutions replacing `out_conv` | Analogue |
| 3.6 Eq. (5): `L = L_CE + L_CV2` | `ShapeMoELoss`, configurable segmentation loss | Yes |
| — | `KL(q_S(z\|I) \|\| q_T(z\|M))` distillation | Not in the paper |

The last row is this project's own contribution: the privileged-information link
between the Phase 0 teacher and the student.

## Where the student's shape posterior comes from

The paper derives the shape distribution from a *visible mask* prompt through
SAM's mask encoder. There is no prompt here, and letting the student read any
ground-truth mask would destroy the whole point of a privileged teacher. So `E_S`
reads the UNet bottleneck, average-pooled to a vector. The two-headed structure of
Eq. (1) is preserved; the trunk feeding it is not the paper's.

## Expert design

In SAM the hyper-network emits mask weights `w` that multiply the refined feature
`F`. In a UNet the exact structural counterpart is `out_conv`, the 1x1 projection
from the final decoder feature map to logits. Phase 1 replicates that projection K
times and drops the original, so the encoder and decoder stay shared and only a
few hundred parameters per expert are duplicated, matching the paper's stated
motivation for not duplicating whole decoders.

Dispatch is really sparse: each head runs only on the sub-batch routed to it.
`test_unselected_head_receives_no_gradient` pins that down.

## Assumptions

### The balancing loss consumes dense probabilities, not `pi`

This one is worth understanding before reading any training log.

Eq. (4) sets every non-selected score to `-inf` before the softmax. With `top_k=1`
that makes `pi` identically `1.0` for the selected expert — a constant, with no
gradient to `W`. `L_CV2(pi)` as literally written would therefore be a constant
too, which contradicts the paper's own Table 4 showing the balancing loss changes
results. So `expert_balance_cv2_loss` is applied to the dense `softmax(s)`, which
is differentiable.

The consequence is that the loss penalises *concentration of routing
probability*, not *imbalance of hard assignments*. Those are correlated but not
the same: the smoke run showed `cv2 = 0.001` while all samples went to a single
expert. The per-expert hard shares are therefore logged separately every epoch
(`expert_k_share`), and they, not the loss value, are what says whether routing is
balanced.

If real training shows persistent collapse onto one expert, the Switch
Transformer balancing loss is the known fix: `K * sum_k f_k * P_k`, where `f_k` is
the (constant) fraction of samples routed to expert k and `P_k` the mean dense
probability. It keeps a gradient while penalising hard imbalance directly. That is
a departure from the paper's cited CV², so it is not the default.

### Which terms train which blocks

Worth checking against intuition, because the split is unusual:

- **Experts and trunk** — trained by the segmentation loss.
- **Router `W`** — `s` affects the output only through the non-differentiable
  top-k selection and through the constant `pi`, so at `top_k=1` the segmentation
  loss gives it no gradient. `W` is trained by the balancing loss alone.
- **`E_S`** — its output reaches the loss only through the router, so it is
  trained by the distillation KL alone.

That means **without a teacher checkpoint the shape encoder receives no gradient
from any term** and its posterior stays at initialization. The training script
prints a warning when `--teacher` is omitted. `test_full_objective_reaches_every_trainable_block`
asserts all three blocks receive non-zero gradients when a teacher is present.

### Sampling at evaluation time

Eq. (2) samples `eta ~ N(0, I)`, which would make routing, and therefore
predictions, non-deterministic at inference. `ShapeMoESegmenter.forward` samples
while training and uses `mu` in eval mode. The paper does not discuss this.

### Latent dimension must match the teacher

`model.latent_dim` has to equal the Phase 0 teacher's, or the distillation KL has
no common space. The training script checks this when loading the checkpoint and
fails with an explicit message.

### Expert counts

`num_experts = 4`, `top_k = 1`, from the paper's Tables 2 and 3. These are the
only Phase 1 hyperparameters the paper actually pins down.

### `balance_weight` and `distillation_weight`

Eq. (5) writes `L = L_CE + L_CV2` with no coefficient, and the distillation term
does not exist in the paper. Defaults here are `balance_weight = 0.01` and
`distillation_weight = 1.0`, both untuned starting points rather than anything
derived.

---

# Decisions log

Recorded as they were made, with the evidence available at the time.

## Fig. 3 does specify the encoder architecture

An early note in this document claimed the paper never describes `E_mu` and
`E_sigma`. That was drawn from the body text, which indeed says only "two
separate encoders". The figures say more, and were extracted from the PDF to
check:

- **Fig. 3** draws each branch as `Conv2D -> Conv2D -> Avg Pool -> Linear`, with
  `e_m` fanning out to two branches that share no convolutions.
- **Fig. 2** draws `e_m` as a spatial feature map, marks the Mask Embedding
  Encoder and the Image Feature Encoder as frozen, and marks only the two
  Gaussian encoders, the router and the experts as trainable.

Still absent everywhere, text and figures alike: whether `Avg Pool` is global or
strided, the channel counts, the kernel sizes, and `d`.

## `E_mu` and `E_sigma` share a trunk

Fig. 3 gives each branch its own pair of convolutions; the Phase 0 specification
writes `mu_T = W_mu h_T + b_mu` and `log sigma^2_T = W_sigma h_T + b_sigma`, a
single shared `h_T` feeding two linear heads. The two sources conflict.

**Decision: follow the specification.** The convolution trunk stays shared and
the split happens at the linear heads, as implemented. The consequence is that
`sigma` cannot attend to different evidence than `mu`; separating the branches is
listed as a later experiment.

## `Avg Pool` is treated as global

Undetermined by the paper. Global pooling followed by one `Linear` is the
shortest path to the vector output that Eq. (1) requires; a strided pool would
force a `Linear` over a flattened `C x h x w`, which nothing in the figure
suggests.

## The Phase 0 deliverable is `E_M`, cut before pooling

Phase 0 exists to produce, unsupervised, the pretrained mask encoder that
ShapeMoE gets for free from SAM2. The cut is at the convolution stack's output,
before global pooling, because that is where Fig. 2 draws the boundary between
the frozen `E_M` and the trainable branches, and because `e_m` there is still a
feature map as the figure shows.

`MaskVAETeacher.embed` returns it, `mask_embedding_state_dict` and
`embedding_geometry` are written into every checkpoint, and
`load_mask_embedding_encoder` rebuilds it frozen from a checkpoint path.

Note this pretrained `E_M` can only be used where a mask is available at
inference. ShapeMoE takes a visible mask as a prompt, so its student can use
`E_M`; a student that only sees the image cannot.

## Freezing has to survive `.train()`

`E_M` contains BatchNorm, so `requires_grad_(False)` does not freeze it: in
training mode those layers keep updating running statistics, and the block's
output drifts even with no gradient reaching it. Calling `.eval()` fixes that but
does not stick, because `nn.Module.train()` recurses into every child, so the
next epoch's `model.train()` silently unfreezes it.

Two guards, both tested: `MaskVAETeacher.train` re-applies `.eval()` to a frozen
`E_M`, and `load_mask_embedding_encoder` returns the encoder wrapped in
`FrozenModule`, whose `train()` is a no-op.

## The decoder still hangs off `z_T`

`D_T(z_T)` as the specification writes it, not `D_T(e_m)`. Reconstructing from
`e_m` would train a stronger `E_M`, since it skips the 64-dimensional
bottleneck, but would leave `z_T` under-trained. Revisit experimentally.

## Still open

Before `E_M` is frozen, Phase 0 needs an acceptance criterion. Reconstruction
Dice and `active_units` are logged, but no threshold has been agreed on, and
after freezing there is no way back.

## AMP is off for Phase 0 and on for Phase 1

Measured at 256x256 on an RTX GPU, 20 optimizer steps each:

| Setting | Steps applied | Final loss scale |
|---|---|---|
| Phase 1, AMP on | 20 / 20 | 65536, never reduced |
| Phase 0, AMP on, default scale | 6 / 20 | 4 |
| Phase 0, AMP on, init scale 16 | 18 / 20 | 4 |
| Phase 0, AMP off | 20 / 20 | not applicable |

`recon_reduction="sum"` puts the Phase 0 loss around 4e4, and GradScaler multiplies
it again before the backward pass, so the float16 gradients overflow and most
early steps are skipped. The scaler recovers by driving the loss scale down to 4,
at which point it is no longer protecting against gradient underflow, which is
the only reason to scale at all. The teacher has 1.4M parameters, so fp32 costs
little.

Phase 1 is unaffected: its loss is O(1), the scale never has to move.

`pretrain_teacher_vae.py` prints a warning if AMP is enabled together with sum
reduction, since the failure is silent otherwise: training still runs and the
loss still falls, just with most of the early updates thrown away.
