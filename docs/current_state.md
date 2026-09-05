## Current data flow

At clean HEAD `33a0b95`, the pipeline is:

```text
YAML config
  ↓ resolve `extends`, validate paths/options
Tracked split manifest
  ↓
ISIC image (.jpg) + sparse mask (.npz)
  ↓ load and densify
Joint resize/augmentation/normalization
  ↓
Batch:
  image [B, 3, 256, 256]
  mask  [B, 1, 256, 256]
  ↓
UNet, promptless SAM, or E-SAM
  ↓
SegmentationOutput.logits [B, 1, 256, 256]
  ↓
BCE + Dice loss → backward → AdamW
  ↓
Validation → best.pt / last.pt / history.json
  ↓ reload best.pt
Test metrics → test_metrics.json
```

### 1. Experiment setup

[scripts/run_experiment.py](/home/teama/projects/project_01/long/moe_research/scripts/run_experiment.py:25) loads a fresh YAML configuration or a saved resume configuration. The current E-SAM experiments inherit shared settings from [isic2018_common.yaml](/home/teama/projects/project_01/long/moe_research/configs/isic2018_common.yaml:5).

[execute_experiment()](/home/teama/projects/project_01/long/moe_research/src/experiment.py:232) then:

- Seeds Python, NumPy, and PyTorch.
- Creates the run directory and metadata.
- Builds train, validation, and test loaders.
- Builds the registered model, loss, AdamW optimizer, and scheduler.
- Runs training.
- Reloads the best validation checkpoint for final testing.

### 2. Dataset path

The native 8-organ Synapse CT dataset is additionally described by
`manifests/synapse_ct_v1.json`: 1,280 audited NPZ training slices from 18
cases and 12 held-out HDF5 volumes. `SynapseCTDataset` expands HDF5 volumes
into 2D slices at evaluation time. Its `val.txt` is used for both validation
and final evaluation because no separate validation cohort exists; final
metrics therefore are not independent of checkpoint selection.

The frozen ISIC manifest contains:

- Training: 2,075 samples
- Validation: 519 samples
- Test: 100 samples

Each record contains `image`, `label`, and `pseudo`, but the current dataset only consumes `image` and `label`; `pseudo` is unused.

[ISIC2018Dataset.__getitem__()](/home/teama/projects/project_01/long/moe_research/src/data/isic2018.py:68) performs:

```text
JPEG → RGB PIL image
NPZ sparse mask → dense float mask [1, H, W]
             ↓
resize image bilinearly
resize mask with nearest-neighbor
optional synchronized horizontal flip for training
ImageNet/SAM normalization
mask thresholding at 0.5
```

The returned dictionary contains `image`, `mask`, `sample_id`, `image_path`, and `mask_path`. Transform behavior is defined in [transforms.py](/home/teama/projects/project_01/long/moe_research/src/data/transforms.py:44).

### 3. SAM-family model paths

The adapter-free promptless baseline is registered as `sam` and configured by
`acdc_sam.yaml`, `amos22_sam.yaml`, `isic2018_sam.yaml`, and
`synapse_sam.yaml`. It uses plain checkpoint-compatible SAM ViT blocks, no
Adapter/MoE/LPEG modules, an empty sparse prompt, and the learned
`no_mask_embed` dense prompt. By default the image and prompt encoders are
frozen and only the task-specific decoder is optimized. The decoder emits one
binary channel or exactly `dataset.num_classes` multiclass channels, so this is
a supervised promptless SAM baseline rather than the original prompted
zero-shot SAM interface.

The E-SAM experiments are the E0..E3 ablation rows of paper Table 2, one set per prepared dataset (`acdc`, `amos22`, `isic2018`, `synapse`), plus a `<dataset>_smoke.yaml` per dataset:

- `*_e0`: SAM + Adapters, `use_moe: false`, `use_lpeg: false`.
- `*_e1`: + four-expert sparse MoE-FEB, selecting 50% of tokens per expert.
- `*_e2`: + LPEG only.
- `*_e3`: full MoE-SAM (MoE-FEB + LPEG).

`EsamModel` defaults both switches to `True`, so every E-SAM config states `use_moe` and `use_lpeg` explicitly; `tests/test_configs.py` fails a config that omits either.

The E-SAM forward path is:

```text
[B, 3, 256, 256]
  ↓ patch embedding
[B, 16, 16, 768]
  ↓ 12 ViT Adapter blocks
final embedding: [B, 256, 16, 16]
intermediate embeddings: 12 × [B, 16, 16, 768]
```

For E1, the intermediate embeddings follow this additional branch:

```text
[B, 12, 16, 16, 768]
  ↓ sparse noisy top-k routing
4 expert MLPs, each selecting 50% of all batch/layer/spatial tokens
  ↓ attention fusion and mean across 12 layers
[B, 768, 16, 16]
  ↓ projection
[B, 256, 16, 16]
  ↓ residual addition to final ViT embedding
```

The fused embedding goes through the promptless SAM mask decoder and is upsampled to `[B, 1, 256, 256]`. The wrapper returns raw logits plus IoU predictions and expert indices in `diagnostics`: [EsamModel.forward()](/home/teama/projects/project_01/long/moe_research/src/models/esam/__init__.py:70).

The training loss and evaluator currently ignore those diagnostics.

### 4. Training and evaluation

The batch loop in [trainer.py](/home/teama/projects/project_01/long/moe_research/src/engine/trainer.py:241) does:

```text
model(images)
  → raw logits
  → 0.5 BCEWithLogits + 0.5 soft Dice
  → AMP-scaled backward
  → gradient clipping
  → AdamW step
  → cosine scheduler after each epoch
```

Validation runs every epoch. Higher validation Dice replaces `best.pt`; `last.pt` is always replaced. After training, the best model is tested using:

- Loss
- Dice
- IoU
- HD95
- ASSD
- Boundary F1

See [evaluator.py](/home/teama/projects/project_01/long/moe_research/src/engine/evaluator.py:121).

### Current blockers/risks

- E-SAM configs resolve their dataset root two different ways: `acdc_common.yaml` uses the absolute server path `/home/teama/projects/project_01/dataset/acdc`, while `amos22_common.yaml`, `isic2018_common.yaml` and `synapse_common.yaml` use repo-relative `dataset/<name>`. On the training machine one of the two is wrong unless `<repo>/dataset` is a symlink to the shared data root; `export ACDC_DATA_ROOT=...`-style env vars (documented in README) are the portable fix.
- `checkpoints/sam_vit_b_01ec64.pth` is not in the repo, so promptless SAM and E-SAM cannot load their configured pretrained weights until it is downloaded on the training machine.
- `src/metrics/__init__.py` exposes two implementations of `compute_multiclass_dice_iou`/`compute_multiclass_surface_metrics` (from `segmentation.py` and `multiclass.py`); the `segmentation.py` import wins, so classes absent from a slice are scored 1.0 instead of being excluded from the mean. See the module docstring — switching is a measurement decision.
- The only completed run presently stored is the older UNet run, with test Dice `0.86959` and IoU `0.78625`; there are no E-SAM run artifacts yet.