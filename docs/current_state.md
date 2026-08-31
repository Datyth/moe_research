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
UNet or E-SAM
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

The frozen manifest contains:

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

### 3. E-SAM/MoE model path

The current experiments are:

- [E0](/home/teama/projects/project_01/long/moe_research/configs/isic2018_e0.yaml:8): SAM + Adapters, `use_moe: false`.
- [E1](/home/teama/projects/project_01/long/moe_research/configs/isic2018_e1.yaml:8): SAM + Adapters + four-expert sparse MoE, selecting 50% of tokens per expert.

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

- E0/E1 currently resolve their dataset root to `moe_research/dataset/isic2018_task1`, which is absent. The actual prepared dataset is at `/home/teama/projects/project_01/dataset/isic2018_task1`.
- `checkpoints/sam_vit_b_01ec64.pth` is also absent, so E-SAM construction currently cannot load its configured pretrained weights.
- The noisy MoE router samples random noise even during `model.eval()`. Consequently, E1 validation and test predictions are stochastic.
- The only completed run presently stored is the older UNet run, with test Dice `0.86959` and IoU `0.78625`; there are no E0/E1 run artifacts yet.