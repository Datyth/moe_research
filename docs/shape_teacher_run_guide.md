# Shape Teacher run guide

The implementation follows `docs/teacher_plan.md`. It trains a mask-only
Gaussian posterior teacher with reconstruction loss only. It does not invoke the
ShapeMoE, student, router, experts, image encoder, KL loss, or consistency loss.

## Data sources

The existing single JSON manifest remains supported. For a generic dataset,
configure one authoritative source per split. A source can be a CSV manifest
with a `mask_path` column, a TXT/list file with one path per line, or a mask
directory:

```yaml
dataset:
  root: /path/to/dataset
  manifest:
    train: {manifest: /path/to/train.csv}
    val: {manifest: /path/to/val.txt}
    test: {directory: /path/to/test_masks}
```

Paths inside CSV/TXT manifests are resolved relative to `dataset.root`.
Directory discovery is recursive and sorted. The loader never infers or mixes
splits, and duplicate mask paths across splits are a hard error.

Masks are converted to one grayscale channel, checked as binary (`0/1` or
`0/255` unless `allow_non_binary_source` explicitly enables thresholding),
thresholded to `float32` `0/1`, and resized with nearest-neighbor interpolation.

## Commands

Activate the repository environment first:

```bash
conda activate medical-seg
```

Train the clean-input teacher:

```bash
python scripts/training/train_shape_teacher.py \
  --config configs/teacher_clean_isic2018.yaml
```

Train the denoising teacher:

```bash
python scripts/training/train_shape_teacher.py \
  --config configs/teacher_denoise_isic2018.yaml
```

Evaluate either best checkpoint with deterministic `z_T = mu_T`:

```bash
python scripts/evaluation/evaluate_shape_teacher.py \
  --checkpoint runs/<experiment>/<run-id>/best.pt \
  --split test \
  --input-mode clean

python scripts/evaluation/evaluate_shape_teacher.py \
  --checkpoint runs/<experiment>/<run-id>/best.pt \
  --split test \
  --input-mode corrupted
```

## Run artifacts

Each run contains the resolved config, environment/data-source metadata,
preflight data audit, epoch history, best/last checkpoints, and deterministic
validation/test metrics for clean and fixed-corrupted inputs.

Qualitative outputs are selected from the whole test split with deterministic
clean reconstruction (`sample=False`):

- `qualitative_selection.json`: selected `small`, `large`, `smooth`,
  `irregular`, and lowest-clean-Dice `difficult` records, including statistics
  and paths.
- `qualitative_clean.png`: `clean target | clean input | probability | binary
  reconstruction`.
- `qualitative_corrupted.png`: the same selected rows with fixed corrupted
  inputs.
- `reconstruction_grid.png`: compatibility grid matching the run's training
  input mode.
- `corruption_preview.png`: stochastic input/target proof for denoising runs.

The history records BCE, soft Dice, hard Dice, IoU, posterior-scale mean and
median, percentage of scales below `1e-3`, and corruption statistics. A run is
marked as posterior-scale collapsed when at least 95 percent of validation
scales are below `1e-3`.

The two full configs intentionally share the same model and optimization
settings. They differ only in experiment identity, input mode, and whether
training-input corruption is enabled.
