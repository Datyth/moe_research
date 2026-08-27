# Implementation Plan — Shape Teacher Pretraining (Clean vs. Denoising)

**Purpose.** Build and evaluate a *mask-only* Shape Teacher that learns a compact latent representation of lesion shape. This is a self-supervised reconstruction task over already-available segmentation masks, not a segmentation-network training task.

This document is deliberately self-contained so it can be handed to an implementation agent without relying on the proposal diagrams.

---

## 1. Frozen scope

### In scope

Train the same *Gaussian posterior Shape Teacher* in two configurations:

| Experiment ID | Teacher input | Reconstruction target | Intent |
|---|---|---|---|
| `teacher_clean` | clean binary ground-truth mask `M` | clean mask `M` | ordinary shape autoencoder baseline |
| `teacher_denoise` | corrupted view `C(M)` | clean mask `M` | denoising shape-prior teacher |

For each experiment, save the best checkpoint and report reconstruction metrics on both clean and corrupted validation inputs.

### Explicitly out of scope

- No image input, image encoder, segmentation model, or segmentation loss.
- No Shape Student, knowledge distillation, MoE, router, experts, or joint training.
- **No consistency loss `L_cons`.** Do not make two augmented views for a consistency objective.
- No KL loss, adversarial loss, classifier, Shape Student, or any other auxiliary objective.

The posterior mechanism shown in the reference architecture is **in scope and must be retained**:

```text
mask -> encoder E_T -> shape feature h_T -> mean head mu_T + scale head sigma_T
     -> q_T(z | mask) = N(mu_T, diag(sigma_T^2))
     -> z_T = mu_T + sigma_T ⊙ epsilon, epsilon ~ N(0, I) -> shape decoder
```

There is intentionally no KL term because the requested ablation trains only reconstruction loss `L_shape`. The implementation must log the scale statistics and report if `sigma_T` collapses toward zero; this is an outcome to observe, not a reason to silently replace the requested posterior with a deterministic autoencoder.

---

## 2. Objective and terminology

Let `M` be a clean ground-truth binary mask and `M_in` be the model input.

```text
M_in -> ShapeEncoder E_T -> h_T -> MeanHead(mu_T) + ScaleHead(sigma_T)
     -> q_T(z | M_in) -> reparameterization -> z_T -> ShapeDecoder D_shape -> logits -> M_hat
```

```math
M_in = M                    for teacher_clean
M_in = C(M)                 for teacher_denoise
q_T(z | M_in) = N(mu_T, diag(sigma_T^2))
epsilon ~ N(0, I)
z_T = mu_T + sigma_T ⊙ epsilon
M_hat = sigmoid(D_shape(z_T))

L_shape = BCEWithLogits(logits, M) + (1 - Dice(M_hat, M))
```

- The decoder output is **logits**, so use `BCEWithLogitsLoss`; do not apply sigmoid before that loss.
- `Dice` is soft Dice, evaluated per sample then averaged over the batch.
- Start with equal coefficients: `lambda_bce = 1.0`, `lambda_dice = 1.0`. Keep them configurable.
- The clean mask `M` is always the target, including the denoising run.

The work is best described as **mask-based self-supervised reconstruction / denoising pretraining**. It is not fully unsupervised because the masks are needed during training, but it needs neither input images nor extra class labels.

---

## 3. Data contract

### Required input

The implementation should accept either of these minimal formats:

1. Three text/CSV manifests (`train`, `val`, `test`) with one `mask_path` per row; or
2. Three directories of mask images, one directory per split.

Do **not** infer a train/validation/test split from filenames if split manifests are available. Never mix masks across the supplied splits.

### Mask preprocessing

1. Read a single-channel mask (if the source is RGB, convert it to one channel).
2. Convert foreground to `float32` `0.0/1.0` with a configurable threshold (default source-pixel threshold: `> 0`).
3. Resize to `image_size × image_size` using nearest-neighbor interpolation only (default `256 × 256`).
4. Return shape `[1, H, W]`.

Do not normalize a mask like RGB image data. Do not use bilinear interpolation for target masks.

### Edge cases to log before training

- Counts of empty masks and all-foreground masks in each split.
- Original dimensions and foreground-area statistics.
- Duplicate path across splits (hard error).
- Unreadable file or non-binary source (hard error unless an explicit conversion policy is configured).

Empty masks may remain in the experiment if they are legitimate data. Dice must use epsilon smoothing so they never produce NaN; report the empty-mask count separately.

---

## 4. Model contract

Implement `ShapeTeacher` with the exact public behavior below. Exact internal style may vary, but preserve input/output dimensions and the Gaussian posterior interface.

```python
class ShapeTeacher(nn.Module):
    def forward(self, masks: Tensor, sample: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # masks: [B, 1, 256, 256], values in [0, 1]
        # returns: logits [B, 1, 256, 256], z_T [B, latent_dim],
        #          mu_T [B, latent_dim], sigma_T [B, latent_dim]
        # sample=True: z_T = mu_T + sigma_T * eps
        # sample=False: z_T = mu_T (deterministic evaluation / visualization)
```

### Recommended lightweight architecture

Use `feature_dim=256` and `latent_dim=128` by default; make both configurable.

**Encoder**

```text
[B, 1, 256, 256]
Conv(1, 32, k=3, s=2, p=1) + GroupNorm + SiLU       -> 128×128
Conv(32, 64, k=3, s=2, p=1) + GroupNorm + SiLU      -> 64×64
Conv(64, 128, k=3, s=2, p=1) + GroupNorm + SiLU     -> 32×32
Conv(128, 256, k=3, s=2, p=1) + GroupNorm + SiLU    -> 16×16
AdaptiveAvgPool2d(1) -> flatten = h_T [B, 256]
```

The posterior heads from `h_T`, required by the diagram, are:

```text
AdaptiveAvgPool2d(1) -> flatten = h_T [B, 256]
MeanHead:  Linear(256, 128) -> mu_T
ScaleHead: Linear(256, 128) -> softplus(.) + 1e-4 -> sigma_T
```

**Decoder**

```text
Linear(128, 256×16×16) -> reshape [B, 256, 16, 16]
ConvTranspose(256, 128, k=4, s=2, p=1) + GroupNorm + SiLU -> 32×32
ConvTranspose(128, 64,  k=4, s=2, p=1) + GroupNorm + SiLU -> 64×64
ConvTranspose(64, 32,   k=4, s=2, p=1) + GroupNorm + SiLU -> 128×128
ConvTranspose(32, 16,   k=4, s=2, p=1) + GroupNorm + SiLU -> 256×256
Conv(16, 1, k=3, s=1, p=1) -> logits
```

- `GroupNorm` (for example 8 groups where divisible) is preferred to BatchNorm, because medical-mask batch sizes may be small.
- Do not use dropout in the teacher for this first controlled comparison.
- `softplus + sigma_floor` is required for a valid positive scale. Use `sigma_floor=1e-4` by default.
- Do not apply sigmoid to reconstruction logits in `forward`; expose logits to the training loss. Use sigmoid only for visualisation/metrics.
- At training time call `sample=True`. For the primary clean/corrupted validation metrics use `sample=False` so measurements are deterministic and comparable. Optionally log a Monte-Carlo metric (e.g. 8 samples) separately, but do not use it as the checkpoint-selection metric.

If implementation uses a different `image_size`, make the downsample factor and decoder seed spatial size derived from the configuration rather than hard-coded.

---

## 5. Denoising corruption `C(M)`

`teacher_clean` **must not** corrupt its training input. It may use only resizing and deterministic input formatting.

`teacher_denoise` applies a stochastic composition of morphology-like errors to the **input only**. Target remains untouched.

### Training-time corruption policy (default)

For every training mask, sample 1–3 operations without replacement from the following list; each operation is independently skipped according to its probability. Clamp output to `[0, 1]` and optionally binarize after each operation.

| Operation | Probability | Parameters | Motivation |
|---|---:|---|---|
| Erosion | 0.35 | disk/ellipse radius 1–5 px | missing boundary / under-segmentation |
| Dilation | 0.35 | disk/ellipse radius 1–5 px | boundary bleed / over-segmentation |
| Random holes | 0.25 | 1–5 small disks, radius 2–10 px, placed within foreground | fragmented interiors |
| Foreground blob removal | 0.20 | remove 1–3 small connected regions or disks | missed lesion regions |
| Boundary jitter | 0.25 | affine translate ≤4 px; scale 0.95–1.05; rotate ≤10° | contour imprecision |

Implementation notes:

- Morphological operations should be implemented reproducibly (e.g. OpenCV or Kornia); treat masks as binary when applying morphology.
- For affine transformation, use nearest-neighbor interpolation and border fill `0`.
- Skip an operation if it makes a nonempty mask empty or all foreground; record that it was skipped. This avoids a training set dominated by degenerate corruptions for small lesions.
- Do not use additive Gaussian pixel noise as the primary corruption: it does not model plausible binary-mask segmentation errors.
- Each sample needs a seeded RNG based on the run seed plus worker/sample state, so a run can be reproduced.

### Fixed validation/test robustness set

At evaluation time, make two input modes:

- `clean`: input `M`.
- `corrupted`: input `C_eval(M)` using the same *types* of corruption but a milder, **fixed seeded** policy (suggested radius 1–3 px, ≤2 operations).

Generate this evaluation corruption deterministically per `(global_seed, split_name, mask_path)`; never resample it between epochs. This makes best-checkpoint selection and the two teachers comparable.

---

## 6. Training protocol

### Shared settings

Use identical architecture, splits, image size, optimizer, augmentation policy (except input corruption), seed list, and training budget for both experiments.

Suggested initial configuration:

```yaml
seed: 42
image_size: 256
latent_dim: 128
batch_size: 32             # reduce if GPU memory requires it
epochs: 150
optimizer: adamw
lr: 0.0003
weight_decay: 0.00001
lr_scheduler: cosine
warmup_epochs: 5
num_workers: 4
amp: true
lambda_bce: 1.0
lambda_dice: 1.0
dice_epsilon: 0.000001
early_stopping_patience: 25
selection_metric: val_clean_dice
```

If resource allows, run both configurations for seeds `[42, 43, 44]`; otherwise run seed `42` first and record that the comparison is one-seed preliminary evidence.

### Per-epoch procedure

```python
for clean_mask, metadata in train_loader:
    input_mask = clean_mask if config.input_mode == "clean" else corrupt(clean_mask)
    logits, z_t, mu_t, sigma_t = model(input_mask, sample=True)
    loss_bce = bce_with_logits(logits, clean_mask)
    loss_dice = 1 - soft_dice(sigmoid(logits), clean_mask)
    loss = lambda_bce * loss_bce + lambda_dice * loss_dice
    loss.backward()
    optimizer.step()
```

- The training dataloader must return/retain the uncorrupted target separately from the input.
- Do not accidentally call `corrupt()` on the target tensor in place.
- Log at least `mean(sigma_T)`, `median(sigma_T)`, and the percentage of scales below `1e-3` every epoch. With no KL term, sigma collapse is possible and must be visible in the results.
- Use AMP with `GradScaler` only if CUDA is available; provide CPU-safe fallback.
- Choose the best model by `val_clean_dice`, breaking ties with lower `val_clean_loss`.
- Also log `val_corrupted_dice`, but do not choose a denoising checkpoint solely by it; clean reconstruction prevents a robustness-only degenerate model.

### Reproducibility requirements

- Set and log Python, NumPy, and PyTorch seeds; enable deterministic behavior where practical.
- Store the fully resolved config, git revision if available, dependency versions, device/GPU name, split manifests, and command line next to each run.
- Never reuse the same output run directory silently; include run ID, experiment ID, seed, and timestamp.

---

## 7. Metrics, comparisons, and decision rule

### Report for each checkpoint

For validation and final test, report mean and standard deviation where multiple seeds exist:

| Metric | Clean input → clean target | Corrupted input → clean target |
|---|---:|---:|
| BCE | required | required |
| Soft Dice | required | required |
| IoU | recommended | recommended |
| Boundary F1 / HD95 | optional | optional |

For the corrupted-input columns, apply the same fixed `C_eval` to the input of both teachers. The target is always original clean `M`.

Also export a small qualitative grid per experiment: `clean target | teacher input | reconstructed probability | thresholded reconstruction`, including small, large, smooth, irregular, and difficult masks.

### Acceptance checks

Implementation is accepted only if all are true:

1. `teacher_clean` input equals target exactly (after common preprocessing); no corruption code runs in that mode.
2. `teacher_denoise` input differs from target for a visible nonzero fraction of its training samples; save a corruption preview grid to prove this.
3. Both runs complete without NaN/Inf and can reload their best checkpoints to reproduce recorded validation metrics.
4. The clean and denoising run use the same model parameter count and same number of training steps.
5. No `L_cons`, KL, MoE, student, router, or image tensor is present in the teacher training graph. Conversely, the encoder feature, `mu_T` head, positive `sigma_T` head, Gaussian sampling, and shape decoder must all be present.

### Research decision after the two runs

Prefer `teacher_denoise` as the future teacher only if, on the held-out test set:

- `test_corrupted_dice(teacher_denoise)` exceeds `test_corrupted_dice(teacher_clean)` by **at least 0.03 absolute**, and
- `test_clean_dice(teacher_denoise)` is no worse than `teacher_clean` by more than **0.01 absolute**.

Otherwise retain `teacher_clean` as the primary teacher and report the denoising result as a negative/neutral robustness ablation. These thresholds are decision thresholds, not claims of statistical significance; with three seeds, add mean ± SD and a paired per-mask comparison.

---

## 8. Suggested repository structure

The remote agent may adapt names to the existing project, but should keep responsibilities separated.

```text
shape_teacher/
  configs/
    teacher_clean.yaml
    teacher_denoise.yaml
  data/
    mask_dataset.py          # discovery/manifests, preprocessing, validation
    corruptions.py           # C_train and deterministic C_eval
  models/
    shape_teacher.py         # encoder, h_T, mu/sigma heads, sampling, decoder
  losses.py                  # BCE + stable soft Dice
  metrics.py                 # Dice, IoU, optional boundary metrics
  train_teacher.py           # train/validate/checkpoint entry point
  evaluate_teacher.py        # clean + fixed-corruption test evaluation
  visualize_teacher.py       # qualitative reconstruction/corruption grids
  tests/
    test_dataset.py
    test_corruptions.py
    test_model.py
    test_loss.py
```

Suggested CLI:

```bash
python -m shape_teacher.train_teacher --config shape_teacher/configs/teacher_clean.yaml
python -m shape_teacher.train_teacher --config shape_teacher/configs/teacher_denoise.yaml
python -m shape_teacher.evaluate_teacher --checkpoint <best_checkpoint> --split test --input-mode clean
python -m shape_teacher.evaluate_teacher --checkpoint <best_checkpoint> --split test --input-mode corrupted
```

Each config must differ only in `experiment_id`, `input_mode`, `corruption.enabled`, and output directory:

```yaml
# teacher_clean.yaml
experiment_id: teacher_clean
input_mode: clean
corruption:
  enabled: false

# teacher_denoise.yaml
experiment_id: teacher_denoise
input_mode: denoise
corruption:
  enabled: true
```

---

## 9. Minimal automated tests before any long run

1. **Preprocessing:** a known 2D binary mask becomes `[1, H, W]`, preserves binary values, and uses nearest resize.
2. **Clean mode contract:** sampled `input_mask` and `target_mask` are identical tensors.
3. **Denoising contract:** with a fixed seed, corruption is reproducible; target remains bit-identical to source; at least one configured corruption changes a nonempty test mask.
4. **Forward shape:** model maps `[B,1,256,256]` to logits `[B,1,256,256]`, `z_T` `[B,128]`, `mu_T` `[B,128]`, and positive `sigma_T` `[B,128]`.
5. **Gradient test:** one optimizer step produces finite `L_shape` and nonzero gradients in encoder and decoder.
6. **Sampling test:** `sample=False` returns `z_T == mu_T`; `sample=True` is reproducible with a fixed RNG seed and positive finite `sigma_T`.
7. **Checkpoint test:** save → reload → same fixed batch gives numerically equal outputs in eval mode using `sample=False`.
8. **Smoke run:** one epoch over a tiny subset writes metrics, checkpoint, scale-statistics, and a four-column visual grid for each configuration.

---

## 10. Required implementation handoff

The coding agent should return all of the following, not only source files:

1. Exact commands used for the two smoke runs and full runs.
2. Resolved configs for `teacher_clean` and `teacher_denoise`.
3. Model parameter count and device/training-time summary.
4. Best validation metrics and final test table in clean/corrupted input modes.
5. Paths to best checkpoints and qualitative reconstruction figures.
6. Epoch-level `sigma_T` statistics and a statement on whether the posterior scales collapse without KL regularization.
7. Confirmation that excluded components (`L_cons`, KL, MoE/student/router) were not implemented in this module, while the required posterior heads and reparameterization were retained.

## 11. Non-decisions that must be surfaced, not silently assumed

The remote agent must ask or expose a config default for these project-specific items if the repository does not already define them:

- mask dataset root and authoritative train/validation/test manifests;
- foreground convention (white/255, 1, or another value);
- appropriate `image_size` for the dataset;
- whether empty masks are valid examples;
- available GPU memory / batch size;
- whether the implementation must plug into an existing experiment tracker.

Do not start any future student/MoE stage until the two teacher runs and their clean-vs-corrupted comparison are complete.
