# Agent Implementation Plan
# Phase-A Small-CNN Mask Reconstruction Baseline

## Environment

```
conda activate moe_research
```
## Repository

```text
https://github.com/Datyth/moe_research
```

---

# 1. Objective

Implement the first working baseline for:

```text
Phase A: Shape Representation Pretraining
```

The baseline task is:

\[
M
\rightarrow
E_M
\rightarrow
P_M
\rightarrow
h_M
\rightarrow
D_M
\rightarrow
\tilde M
\]

where:

- \(M\): ground-truth binary lesion mask
- \(E_M\): Small-CNN shape encoder
- \(P_M\): spatial bottleneck projector
- \(h_M\): compact shape latent
- \(D_M\): latent-only reconstruction decoder
- \(\tilde M\): reconstructed lesion mask

The training objective is:

\[
\mathcal L_{\mathrm{rec}}
=
0.5\mathcal L_{\mathrm{BCE}}
+
0.5\mathcal L_{\mathrm{Dice}}.
\]

For this first baseline, report only:

```text
loss
dice
```

The purpose of this implementation is to establish a clean and reliable Phase-A training pipeline before introducing:

```text
ResNet-18
SAM2
geometry analysis
latent-shuffle analysis
Phase B
Phase C
```

---

# 2. Scope Discipline

Keep this implementation minimal.

Do not implement functionality merely because it might be useful later.

Implement abstractions only when they are necessary for:

1. preserving the existing segmentation pipeline;
2. running the new mask-reconstruction task cleanly.

## Explicitly Out of Scope

Do NOT implement:

```text
SAM2 shape encoder
ResNet-18 shape encoder
shape encoder registry
geometry preservation metrics
latent-shuffle evaluation
latent effective-rank analysis
linear probing
shape-representation export
Phase B
Phase C
posterior routing
prior routing
MoE changes
hierarchical routing
multi-dataset Phase-A training
multi-seed experiment infrastructure
mask-only dataset
new augmentation framework
new reconstruction loss
hyperparameter search
Weights & Biases
TensorBoard integration
Hydra
PyTorch Lightning
Accelerate
distributed training
callback framework
plugin framework
generic model factory for shape models
```

Do not implement deferred items after finishing this task.

Stop when the Small-CNN reconstruction baseline is verified.

---

# 3. Software Architecture Principle

Use the following separation:

\[
\texttt{engine}
=
\text{how optimization and evaluation run}
\]

\[
\texttt{tasks}
=
\text{what learning objective is solved}
\]

\[
\texttt{models}
=
\text{what neural network computes}
\]

\[
\texttt{experiments}
=
\text{how one research experiment is assembled}
\]

The generic engine must not know whether a model solves:

```text
image -> segmentation mask
```

or:

```text
mask -> reconstructed mask
```

Do NOT create:

```text
shape_trainer.py
shape_evaluator.py
posterior_trainer.py
distillation_trainer.py
```

There should remain one generic Trainer and one generic Evaluator.

---

# 4. Target Project Structure

After implementation, the relevant project structure should approximately be:

```text
src/
├── engine/
│   ├── __init__.py
│   ├── trainer.py
│   └── evaluator.py
│
├── tasks/
│   ├── __init__.py
│   ├── base.py
│   ├── segmentation.py
│   └── mask_reconstruction.py
│
├── models/
│   ├── ...
│   └── shape/
│       ├── __init__.py
│       ├── small_cnn.py
│       ├── projector.py
│       ├── decoder.py
│       └── autoencoder.py
│
├── experiments/
│   ├── __init__.py
│   └── shape_pretraining.py
│
├── metrics/
│   ├── __init__.py
│   └── segmentation.py
│
└── experiment.py
```

Add:

```text
scripts/
└── run_shape_pretraining.py
```

Add:

```text
configs/
└── phase_a/
    ├── isic2018_shape_common.yaml
    ├── isic2018_s0_small_cnn.yaml
    └── isic2018_s0_small_cnn_smoke.yaml
```

Tests:

```text
tests/
├── existing tests...
├── test_shape_models.py
├── test_tasks.py
└── test_shape_pretraining.py
```

Important:

Do NOT move:

```text
src/experiment.py
```

in this task.

The existing segmentation experiment path must remain functional.

---

# 5. Pre-Implementation Repository Check

Before modifying code:

1. Verify the current Git branch.
2. Record the current commit SHA.
3. Run:

```bash
git status
```

4. Do not discard or overwrite user changes.
5. Do not automatically pull remote changes.
6. Pull only if:
   - the worktree is clean;
   - pulling is appropriate;
   - authorization is clear.

Inspect the latest versions of:

```text
src/engine/trainer.py
src/engine/evaluator.py
src/experiment.py
scripts/evaluation/evaluate.py
src/configs/experiment.py
src/losses/
src/metrics/
src/models/base.py
tests/
.github/workflows/tests.yml
```

Do not assume the repository is identical to an earlier review.

---

# 6. Implementation Order

Implement in this exact order:

```text
Step 1
Define generic Task contract

Step 2
Move existing segmentation semantics into SegmentationTask

Step 3
Generalize Evaluator

Step 4
Generalize Trainer

Step 5
Update existing segmentation experiment/evaluation callers

Step 6
Run existing tests and verify segmentation compatibility

Step 7
Implement MaskReconstructionTask

Step 8
Implement Small-CNN ShapeAutoencoder

Step 9
Implement Phase-A experiment runner

Step 10
Add Phase-A configs

Step 11
Add/update tests

Step 12
Run smoke experiment

Step 13
Verify resume

Step 14
Perform tiny-overfit sanity check
```

Do not implement the Small-CNN model until the existing segmentation path works correctly through the new task abstraction.

This separation is important so failures can be attributed either to:

```text
engine refactor
```

or:

```text
new Phase-A model
```

rather than both simultaneously.

---

# 7. Generic Task Contract

Create:

```text
src/tasks/base.py
```

Use a minimal contract.

Recommended structure:

```python
from dataclasses import dataclass
from typing import Protocol, Any

from torch import Tensor, nn


@dataclass
class TaskStepOutput:
    loss: Tensor
    metrics: dict[str, Tensor | float]
    batch_size: int


class Task(Protocol):
    criterion: nn.Module

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        device,
    ) -> TaskStepOutput:
        ...

    def evaluation_step(
        self,
        model: nn.Module,
        batch: Any,
        device,
    ) -> TaskStepOutput:
        ...
```

Do not create:

```text
TaskBaseClass
TaskRegistry
TaskLifecycle
TaskHooks
Callbacks
TaskFactory
```

unless they become strictly necessary.

A Protocol and dataclass are sufficient.

---

# 8. TaskStepOutput Contract

The contract must be precise.

## 8.1 Loss

`TaskStepOutput.loss` must be:

```text
scalar
finite
batch mean
```

Required:

```python
loss.ndim == 0
torch.isfinite(loss)
```

The generic engine automatically exposes this value as:

```text
"loss"
```

Therefore:

```text
TaskStepOutput.metrics MUST NOT contain "loss"
```

If a task returns `"loss"` inside `metrics`, raise a clear error.

---

## 8.2 Metrics

Every metric value must be:

```text
scalar
finite
batch mean
```

Examples:

```python
{
    "dice": 0.91,
}
```

or:

```python
{
    "dice": tensor(0.91),
    "iou": tensor(0.84),
}
```

are acceptable.

Per-sample vectors must be reduced to a batch mean inside the task before returning.

Metric keys must remain consistent across all batches in a single evaluation.

Example invalid behavior:

```text
batch 1:
{"dice"}

batch 2:
{"dice", "iou"}
```

The generic evaluator must raise a clear error if metric keys change across batches.

---

## 8.3 Batch Size

`batch_size` must be:

```text
positive integer
```

representing the number of samples summarized by:

```text
loss
metrics
```

---

# 9. Aggregation Contract

All epoch/evaluation aggregation must be weighted by sample count.

For a metric \(m\):

\[
\bar m
=
\frac{
\sum_b n_bm_b
}{
\sum_b n_b
}.
\]

Loss follows the same rule:

\[
\bar L
=
\frac{
\sum_b n_bL_b
}{
\sum_b n_b
}.
\]

Conceptually:

```python
total_loss += step.loss.item() * step.batch_size

for name, value in step.metrics.items():
    total_metrics[name] += float(value) * step.batch_size

total_samples += step.batch_size
```

Final:

```python
mean_loss = total_loss / total_samples
```

and:

```python
mean_metric = total_metric / total_samples
```

Do NOT average batch means equally.

---

# 10. Uneven Final Batch Test

Add a required test demonstrating correct aggregation.

Example:

```text
dataset size = 5
batch size = 2
```

Batches:

```text
2 samples
2 samples
1 sample
```

Verify:

\[
\bar m
=
\frac{
2m_1+2m_2+m_3
}{5}
\]

and NOT:

\[
\frac{
m_1+m_2+m_3
}{3}.
\]

Test both:

```text
loss
one metric
```

using uneven batch sizes.

---

# 11. Criterion Ownership

The criterion belongs to the Task.

Example:

```python
class SegmentationTask:
    def __init__(
        self,
        criterion,
        threshold,
        boundary_tolerance,
    ):
        self.criterion = criterion
        ...
```

and:

```python
class MaskReconstructionTask:
    def __init__(
        self,
        criterion,
        threshold,
    ):
        self.criterion = criterion
        ...
```

The engine must ensure the criterion is moved to the resolved device:

```python
task.criterion.to(resolved_device)
```

This matters because loss modules can contain registered buffers such as:

```text
pos_weight
```

Do NOT add:

```text
Task.to()
Task.setup()
TaskDeviceManager
LifecycleManager
```

Device handling should remain minimal.

---

# 12. Task Configuration Metadata

Trainer must receive task-specific configuration explicitly.

Add:

```python
Trainer(
    ...,
    task=task,
    task_config: dict[str, Any] | None = None,
)
```

The Trainer treats this dictionary as opaque metadata.

Trainer must:

```text
store a shallow copy
serialize it to the checkpoint
never inspect task type
never read thresholds from it
never use isinstance(task, ...)
```

Canonical checkpoint location:

```python
checkpoint["task_config"]
```

Example segmentation task config:

```python
{
    "name": "segmentation",
    "threshold": 0.5,
    "boundary_tolerance": 2.0,
}
```

Example Phase-A task config:

```python
{
    "name": "mask_reconstruction",
    "threshold": 0.5,
}
```

Do not create task-config classes or a task-config registry.

If `task_config` is provided and is not a dictionary, raise a clear error.

---

# 13. SegmentationTask

Create:

```text
src/tasks/segmentation.py
```

Implement:

```python
class SegmentationTask:
```

It owns:

```text
criterion
threshold
boundary_tolerance
```

Trainer must no longer own these segmentation semantics.

---

# 14. Segmentation Training Step

Require:

```text
batch["image"]
batch["mask"]
```

Move tensors to device as float32.

Conceptually:

```python
images = batch["image"].to(
    device,
    dtype=torch.float32,
    non_blocking=True,
)

targets = batch["mask"].to(
    device,
    dtype=torch.float32,
    non_blocking=True,
)
```

Run:

```python
output = model(images)
```

Require:

```python
isinstance(output, SegmentationOutput)
```

Use:

```python
raise TypeError(...)
```

not:

```python
assert isinstance(...)
```

Compute:

```python
loss = self.criterion(
    output.logits,
    targets,
)
```

Return:

```python
TaskStepOutput(
    loss=loss,
    metrics={},
    batch_size=images.shape[0],
)
```

Do not compute training Dice unless it already exists and is strictly needed.

---

# 15. Segmentation Evaluation Step

Evaluation must preserve the existing segmentation metrics:

```text
dice
iou
hd95
assd
boundary_f1
```

Compute the loss using the task criterion.

Use the existing metric implementations.

Return only batch means:

```python
TaskStepOutput(
    loss=loss,
    metrics={
        "dice": dice.mean(),
        "iou": iou.mean(),
        "hd95": hd95.mean(),
        "assd": assd.mean(),
        "boundary_f1": boundary_f1.mean(),
    },
    batch_size=images.shape[0],
)
```

Do not move segmentation metric computation back into the engine.

---

# 16. Binary Dice/IoU Metric Location

Make:

```text
src/metrics/segmentation.py
```

the canonical home of:

```python
compute_binary_dice_iou(...)
```

If the function currently lives in the evaluator, move it to metrics.

Update:

```text
src/metrics/__init__.py
```

If existing imports through:

```text
src.engine
```

must remain functional, preserve a simple compatibility re-export.

Do not duplicate the implementation.

There should remain one canonical Dice/IoU implementation.

---

# 17. Generic Evaluator

Refactor:

```text
src/engine/evaluator.py
```

into generic evaluation infrastructure.

Desired interface:

```python
evaluate(
    *,
    model,
    loader,
    task,
    device,
) -> dict[str, float]
```

The evaluator must not know:

```text
SegmentationOutput
ShapeAutoencoderOutput
Dice
IoU
HD95
Boundary F1
threshold
boundary tolerance
```

Those belong to tasks.

---

# 18. Evaluator Runtime Safety

Preserve the safety behavior of the existing evaluator.

Resolve:

```python
resolved_device = torch.device(device)
```

If CUDA is requested but unavailable:

```python
raise RuntimeError(...)
```

Reject empty loader:

```python
if len(loader) == 0:
    raise ValueError(...)
```

Move model and criterion:

```python
model.to(resolved_device)
task.criterion.to(resolved_device)
```

Preserve original model mode:

```python
was_training = model.training
```

Use:

```python
try:
    model.eval()

    with torch.inference_mode():
        ...
finally:
    model.train(was_training)
```

The original training/evaluation state must be restored even if:

```python
task.evaluation_step(...)
```

raises an exception.

After iteration, if zero samples were produced:

```python
raise ValueError(...)
```

---

# 19. Evaluator Validation

For every `TaskStepOutput`, validate:

```text
loss is scalar
loss is finite
metrics do not contain "loss"
all metrics are scalar
all metrics are finite
batch_size is a positive integer
metric keys remain consistent
```

Return:

```python
{
    "loss": mean_loss,
    **mean_metrics,
}
```

---

# 20. Evaluator Mode-Restoration Tests

Add tests for:

```text
model initially train()
-> after evaluate(), model is still train()
```

```text
model initially eval()
-> after evaluate(), model is still eval()
```

Also test:

```text
task.evaluation_step raises exception
-> model mode is still restored
```

Also test:

```text
empty loader
-> clear ValueError
```

and:

```text
loader produces zero samples
-> clear ValueError
```

---

# 21. Generic Trainer

Refactor:

```text
src/engine/trainer.py
```

Trainer now receives:

```text
model
task
task_config
optimizer
scheduler
train_loader
val_loader
config
checkpoint_metadata
```

Trainer must own only generic concerns:

```text
optimization
AMP
GradScaler
gradient clipping
scheduler
checkpointing
resume
history
logging
monitoring
```

Trainer must not contain:

```text
image/mask preparation
SegmentationOutput logic
Dice computation
IoU computation
prediction threshold
boundary tolerance
```

---

# 22. TrainerConfig

TrainerConfig should contain engine-level settings only.

Required fields approximately:

```python
epochs
device
last_checkpoint_path
best_checkpoint_path
history_path
use_amp
amp_dtype
log_interval
gradient_clip_norm
monitor
monitor_mode
```

Remove runtime ownership of:

```text
prediction_threshold
boundary_tolerance
```

Existing YAML files may continue storing these under:

```yaml
training:
```

for compatibility.

The experiment builder reads them and passes them into the Task.

Do NOT migrate all existing configs to a new `task:` section in this PR.

---

# 23. TrainerConfig Validation

Validate directly inside:

```python
TrainerConfig.__post_init__()
```

because callers/tests may instantiate it without using the YAML resolver.

Require:

```python
if not isinstance(self.monitor, str) or not self.monitor:
    raise ValueError(...)
```

Require:

```python
if self.monitor_mode not in {"min", "max"}:
    raise ValueError(...)
```

Keep existing validation for:

```text
epochs
amp_dtype
log_interval
gradient_clip_norm
checkpoint paths
```

as appropriate.

---

# 24. Generic Training Loop

Conceptually:

```python
for batch in train_loader:

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(...):
        step = task.training_step(
            model,
            batch,
            device,
        )

    validate_step_output(step)

    loss = step.loss

    scaler.scale(loss).backward()

    if gradient clipping:
        scaler.unscale_(optimizer)
        clip_grad_norm_(...)

    scaler.step(optimizer)
    scaler.update()

    total_loss += (
        loss.detach().item()
        * step.batch_size
    )

    total_samples += step.batch_size
```

Keep current:

```text
AMP
bfloat16 support
GradScaler behavior
gradient clipping
scheduler support
```

Do not rewrite these systems unnecessarily.

---

# 25. Generic History

Do not hard-code:

```text
val_loss
val_dice
val_iou
```

Create history dynamically.

Example:

```python
entry = {
    "epoch": epoch,
    "train_loss": train_loss,
}
```

If validation exists:

```python
for name, value in validation_metrics.items():
    entry[f"val_{name}"] = value
```

Segmentation therefore creates:

```text
epoch
train_loss
val_loss
val_dice
val_iou
val_hd95
val_assd
val_boundary_f1
```

Phase A creates:

```text
epoch
train_loss
val_loss
val_dice
```

If no validation loader exists, history should contain only:

```text
epoch
train_loss
```

Do NOT insert:

```text
val_loss: null
val_dice: null
val_iou: null
```

Update old tests if they currently expect those null fields.

---

# 26. Generic Logging

Do not hard-code:

```text
Val Dice
Val IoU
```

Print whatever validation metrics are available.

Example:

```text
Epoch 10/100 completed
Train Loss : 0.123456
Val Loss   : 0.134567
Val Dice   : 0.901234
```

For segmentation, additional metrics may also be printed.

Do not add a logging framework.

Simple deterministic stdout logging is sufficient.

---

# 27. Generic Monitoring

Trainer configuration supports:

```text
monitor
monitor_mode
```

Segmentation default:

```text
monitor = dice
monitor_mode = max
```

Phase A:

```text
monitor = loss
monitor_mode = min
```

After validation:

```python
if config.monitor not in validation_metrics:
    raise ValueError(...)
```

Do not silently skip a missing metric.

For:

```text
mode = max
```

higher is better.

For:

```text
mode = min
```

lower is better.

---

# 28. Config Resolver Monitoring Validation

Update:

```text
src/configs/experiment.py
```

Defaults:

```python
training.setdefault("monitor", "dice")
training.setdefault("monitor_mode", "max")
```

Validate:

```text
training.monitor is a non-empty string
```

and:

```text
training.monitor_mode ∈ {"min", "max"}
```

Do not attempt to validate metric existence statically.

The Trainer checks metric existence after actual evaluation.

---

# 29. Generic Checkpoint Schema

Keep:

```text
format_version = 2
```

unless an actual implementation blocker requires changing it.

New checkpoint structure should contain:

```python
{
    "format_version": 2,
    "epoch": epoch,

    "model_class": ...,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,

    "metrics": dict(history_entry),

    "monitor_name": config.monitor,
    "monitor_mode": config.monitor_mode,
    "best_monitor_value": ...,

    "trainer_config": ...,
    "task_config": ...,
    "metadata": ...,
}
```

Do not add top-level task-specific metric fields for new tasks.

---

# 30. Segmentation Checkpoint Compatibility Alias

For segmentation-compatible runs using:

```text
monitor = dice
monitor_mode = max
```

retain:

```python
"best_val_dice": best_monitor_value
```

as a compatibility alias.

Trainer does not need to inspect the task type.

It may retain the alias whenever:

```text
monitor == "dice"
monitor_mode == "max"
```

This preserves compatibility without making the engine segmentation-aware.

---

# 31. Legacy Version-2 Resume Compatibility

Legacy version-2 segmentation checkpoints may contain:

```text
best_val_dice
```

but not:

```text
monitor_name
monitor_mode
best_monitor_value
```

Interpret them as:

```text
monitor_name = dice
monitor_mode = max
best_monitor_value = best_val_dice
```

Resume is allowed only if current configuration is also:

```text
dice/max
```

Examples:

```text
saved: dice/max
current: dice/max
=> allowed
```

```text
saved: dice/max
current: loss/min
=> reject
```

For new checkpoints:

```text
saved monitor_name/mode
```

must exactly match current monitor configuration.

If not:

```python
raise ValueError(...)
```

with a clear message.

---

# 32. Checkpoint Metrics on Resume

For new checkpoints, use:

```python
checkpoint["metrics"]
```

for the saved epoch metrics.

Do not rely on fixed keys such as:

```text
val_iou
val_dice
```

for generic resume logic.

Legacy checkpoint compatibility may read legacy fields only where needed.

---

# 33. Completed Run metadata.json Schema

When an experiment finishes successfully, `metadata.json` must contain generic best-model information:

```python
{
    ...
    "best_epoch": ...,
    "monitor_name": ...,
    "monitor_mode": ...,
    "best_monitor_value": ...,
}
```

For segmentation runs using:

```text
dice/max
```

also retain:

```python
"best_val_dice": ...
```

as a compatibility alias.

Do not populate `metadata.json` with every metric.

Detailed epoch metrics belong in:

```text
history.json
```

and:

```text
checkpoint["metrics"]
```

---

# 34. Scheduler Behavior

Preserve existing scheduler behavior.

For:

```text
CosineAnnealingLR
```

step per epoch as currently done.

For:

```text
ReduceLROnPlateau
```

continue monitoring:

```text
validation_metrics["loss"]
```

because every Task always exposes a validation loss.

If no validation exists and ReduceLROnPlateau is used:

```python
raise ValueError(...)
```

Do not generalize scheduler monitoring further in this PR.

---

# 35. Update Existing Segmentation Experiment Runner

Update:

```text
src/experiment.py
```

Construct:

```python
criterion = build_loss(...)
```

then:

```python
task = SegmentationTask(
    criterion=criterion,
    threshold=...,
    boundary_tolerance=...,
)
```

Create:

```python
task_config = {
    "name": "segmentation",
    "threshold": ...,
    "boundary_tolerance": ...,
}
```

Pass:

```python
Trainer(
    model=model,
    task=task,
    task_config=task_config,
    ...
)
```

For test evaluation:

```python
evaluate(
    model=model,
    loader=test_loader,
    task=task,
    device=...,
)
```

Existing:

```text
UNet
SAM baseline
MoE-SAM
```

semantics must remain unchanged.

---

# 36. Standalone Segmentation Evaluation CLI

Update:

```text
scripts/evaluation/evaluate.py
```

to construct:

```python
SegmentationTask(...)
```

instead of passing criterion and segmentation parameters directly to the generic evaluator.

Task-specific settings must resolve in this priority:

```text
1. CLI explicit override
2. checkpoint["task_config"]
3. legacy checkpoint["trainer_config"]
4. default
```

For example:

```text
threshold
boundary_tolerance
```

CLI overrides must remain highest priority.

Preserve all existing visualization behavior.

Do not redesign the visualization system.

---

# 37. Legacy Evaluation Compatibility

Old checkpoints may not contain:

```text
task_config
```

For them, continue reading:

```text
trainer_config.prediction_threshold
trainer_config.boundary_tolerance
```

as fallback.

Add tests for:

```text
new task_config checkpoint
legacy trainer_config checkpoint
CLI override
```

---

# 38. Resume Requirement

Phase-A CLI must support resume.

This is required.

Fresh run:

```bash
python scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn.yaml
```

Resume:

```bash
python scripts/run_shape_pretraining.py \
  --resume runs/phase_a_s0_small_cnn/<run-id>
```

Follow the existing repository resume semantics where practical.

Do not invent a second checkpoint/resume format.

Resume must restore:

```text
model
optimizer
scheduler
scaler
history
epoch
best monitor state
```

---

# 39. MaskReconstructionTask

Create:

```text
src/tasks/mask_reconstruction.py
```

Implement:

```python
class MaskReconstructionTask:
```

Own:

```text
criterion
threshold
```

No image-specific behavior belongs here.

---

# 40. Mask Reconstruction Training Step

Require:

```text
batch["mask"]
```

Do not require the RGB image for model forward.

Move:

```python
masks = batch["mask"].to(
    device,
    dtype=torch.float32,
    non_blocking=True,
)
```

Run:

```python
output = model(masks)
```

Require:

```python
isinstance(
    output,
    ShapeAutoencoderOutput,
)
```

Use explicit:

```python
raise TypeError(...)
```

if contract is violated.

Compute:

```python
loss = self.criterion(
    output.reconstruction_logits,
    masks,
)
```

Return:

```python
TaskStepOutput(
    loss=loss,
    metrics={},
    batch_size=masks.shape[0],
)
```

Do not calculate training Dice.

---

# 41. Mask Reconstruction Evaluation Step

Evaluation reports only:

```text
loss
dice
```

Reuse the existing Dice implementation.

If the existing helper is:

```python
compute_binary_dice_iou(...)
```

it is acceptable to call:

```python
dice, _ = compute_binary_dice_iou(
    logits,
    masks,
    threshold=self.threshold,
)
```

Do not return or report IoU.

Return:

```python
TaskStepOutput(
    loss=loss,
    metrics={
        "dice": dice.mean(),
    },
    batch_size=masks.shape[0],
)
```

Do not implement a second Dice function.

---

# 42. Shape Model Package

Create:

```text
src/models/shape/
```

with:

```text
__init__.py
small_cnn.py
projector.py
decoder.py
autoencoder.py
```

Do not create a shape registry yet.

There is only one model implementation in this milestone.

---

# 43. ShapeAutoencoderOutput

Define in:

```text
src/models/shape/autoencoder.py
```

or another obvious shape-model module:

```python
@dataclass
class ShapeAutoencoderOutput:
    reconstruction_logits: Tensor
    latent: Tensor
```

The reconstruction output must be raw logits.

No sigmoid inside the model.

---

# 44. ShapeAutoencoder Input Contract

The Phase-A baseline is fixed to:

\[
[B,1,256,256].
\]

Fail early in:

```python
ShapeAutoencoder.forward(...)
```

if input is invalid.

Validate:

```python
if masks.ndim != 4:
    raise ValueError(...)
```

```python
if masks.shape[1] != 1:
    raise ValueError(...)
```

```python
if masks.shape[-2:] != (256, 256):
    raise ValueError(...)
```

Do not repeat the same validation inside every encoder/projector/decoder layer.

Do not build a generalized arbitrary-resolution interface in this task.

---

# 45. Small-CNN Encoder Architecture

Create:

```text
src/models/shape/small_cnn.py
```

Input:

\[
[B,1,256,256].
\]

Use exactly four stages.

Each stage:

```text
Conv2d(
    kernel_size=3,
    stride=2,
    padding=1,
    bias=False
)

GroupNorm(
    num_groups=8,
    num_channels=out_channels
)

GELU

Conv2d(
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

GroupNorm(
    num_groups=8,
    num_channels=out_channels
)

GELU
```

Channel progression:

```text
1 -> 32 -> 64 -> 128 -> 256
```

Spatial progression:

```text
256
-> 128
-> 64
-> 32
-> 16
```

Final output:

\[
F_M
\in
\mathbb R^{B\times256\times16\times16}.
\]

Do not add:

```text
residual blocks
attention
dropout
SE blocks
extra convolutions
```

---

# 46. Spatial Bottleneck Projector

Create:

```text
src/models/shape/projector.py
```

Use exactly:

```python
Conv2d(
    256,
    64,
    kernel_size=1,
    bias=True,
)

AdaptiveAvgPool2d((4, 4))

Flatten()

Linear(
    64 * 4 * 4,
    256,
    bias=True,
)
```

Data flow:

\[
[B,256,16,16]
\]

\[
\downarrow
\]

\[
[B,64,16,16]
\]

\[
\downarrow
\]

\[
[B,64,4,4]
\]

\[
\downarrow
\]

\[
[B,1024]
\]

\[
\downarrow
\]

\[
h_M
\in
\mathbb R^{B\times256}.
\]

Do NOT use global average pooling.

Do NOT add activation after the final bottleneck Linear.

---

# 47. Reconstruction Decoder

Create:

```text
src/models/shape/decoder.py
```

Input:

\[
h_M
\in
\mathbb R^{B\times256}.
\]

Decoder must receive only the latent.

There must be:

```text
no encoder skip connections
no spatial bypass
no encoder feature injection
```

---

# 48. Decoder Initial Projection

Use:

```python
Linear(
    256,
    128 * 8 * 8,
    bias=True,
)

GELU()
```

Then reshape:

\[
[B,128,8,8].
\]

---

# 49. Decoder Upsampling Block

Each block must be exactly:

```text
Upsample(
    scale_factor=2,
    mode="bilinear",
    align_corners=False
)

Conv2d(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

GroupNorm(
    num_groups=8,
    num_channels=out_channels
)

GELU
```

---

# 50. Decoder Channel Schedule

Use:

```text
128 -> 128    # 8   -> 16
128 -> 64     # 16  -> 32
64  -> 32     # 32  -> 64
32  -> 16     # 64  -> 128
16  -> 16     # 128 -> 256
```

Final layer:

```python
Conv2d(
    16,
    1,
    kernel_size=1,
    bias=True,
)
```

Output:

\[
[B,1,256,256].
\]

Do not use:

```text
ConvTranspose
skip connections
dropout
residual blocks
attention
sigmoid
```

---

# 51. ShapeAutoencoder Composition

Implement:

```python
features = self.encoder(masks)

latent = self.projector(features)

reconstruction_logits = self.decoder(latent)

return ShapeAutoencoderOutput(
    reconstruction_logits=reconstruction_logits,
    latent=latent,
)
```

Expose:

```python
model.encoder
model.projector
model.decoder
```

as explicit attributes.

Do not implement Shape Teacher export yet.

---

# 52. No Shape Registry Yet

Do NOT create:

```text
src/models/shape/registry.py
```

Instantiate:

```text
SmallCNN
SpatialProjector
ReconstructionDecoder
ShapeAutoencoder
```

directly in the Phase-A experiment builder.

A registry may be reconsidered only after:

```text
ResNet-18
SAM2
```

are actually implemented.

---

# 53. Loss

Reuse the existing:

```text
BCEDiceLoss
```

Do not implement another reconstruction loss.

Configuration:

```yaml
loss:
  name: bce_dice
  bce_weight: 0.5
  dice_weight: 0.5
```

The model outputs raw logits.

The loss handles logits appropriately.

---

# 54. Dataset

Reuse the existing:

```text
ISIC2018Dataset
```

Do not create a mask-only dataset.

Existing sample structure remains:

```python
{
    "image": ...,
    "mask": ...,
    "sample_id": ...,
    ...
}
```

MaskReconstructionTask simply ignores:

```text
image
```

This small inefficiency is intentionally accepted for the first baseline.

Only optimize data loading if profiling later proves it necessary.

---

# 55. Experiment Lifecycle Reuse

Create:

```text
src/experiments/shape_pretraining.py
```

Implement:

```python
execute_shape_pretraining(...)
```

Reuse existing lifecycle helpers from:

```text
src/experiment.py
```

where practical.

Prefer reusing:

```text
seed setup
JSON save
YAML save
config fingerprint
manifest hashing
run-directory creation
Git metadata
device metadata
dataset config creation
data loader creation
optimizer creation
scheduler creation
```

Do not copy hundreds of lines from `src/experiment.py`.

If a currently private helper must be shared:

```text
make a minimal public rename
```

or:

```text
extract only that genuinely shared helper
```

Do NOT redesign the whole experiment framework.

Do NOT move `src/experiment.py`.

---

# 56. Phase-A Experiment Construction

The Phase-A experiment builder should approximately:

```text
load resolved config
set seed
create/resume run directory
build ISIC2018 loaders
build ShapeAutoencoder
build existing BCE+Dice criterion
build MaskReconstructionTask
build optimizer
build scheduler
build generic Trainer
train/resume
load best checkpoint
evaluate test split
save test metrics
update metadata
```

Do not force the shape model through the existing segmentation model registry.

Direct model construction is acceptable.

---

# 57. Optimizer

Reuse current AdamW construction where practical.

Initial baseline:

```yaml
optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.0001
```

Optimize all trainable model parameters.

Do not implement parameter groups.

---

# 58. Scheduler

Use:

```yaml
scheduler:
  name: cosine
  eta_min: 0.0
```

for full training.

Smoke configuration:

```yaml
scheduler:
  name: none
```

Reuse existing scheduler infrastructure.

---

# 59. Phase-A Common Config

Create:

```text
configs/phase_a/isic2018_shape_common.yaml
```

Use:

```yaml
seed: 42

dataset:
  name: isic2018
  root: dataset/isic2018_task1
  manifest: manifests/isic2018_task1_v1.json
  version: isic2018-task1-v1
  task: binary
  num_classes: 1
  in_channels: 3
  image_size: [256, 256]
  image_mean: [0.485, 0.456, 0.406]
  image_std: [0.229, 0.224, 0.225]
  mask_threshold: 0.5

loss:
  name: bce_dice
  bce_weight: 0.5
  dice_weight: 0.5

optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.0001

scheduler:
  name: cosine
  eta_min: 0.0

training:
  epochs: 100
  batch_size: 32
  num_workers: 8
  device: cuda
  amp: true
  amp_dtype: bfloat16
  prediction_threshold: 0.5
  log_interval: 20
  gradient_clip_norm: 1.0
  monitor: loss
  monitor_mode: min
```

`prediction_threshold` stays under:

```yaml
training:
```

for compatibility with the current config structure.

At runtime it belongs to:

```text
MaskReconstructionTask
```

not Trainer.

These values are baseline defaults, not claimed optimal hyperparameters.

---

# 60. Small-CNN Experiment Config

Create:

```text
configs/phase_a/isic2018_s0_small_cnn.yaml
```

Example:

```yaml
extends: isic2018_shape_common.yaml

experiment:
  name: phase_a_s0_small_cnn
  output_root: runs

model:
  name: shape_autoencoder

  encoder:
    name: small_cnn

  projector:
    channels: 64
    spatial_size: 4
    bottleneck_dim: 256

  decoder:
    start_channels: 128
    start_size: 8
```

The implementation architecture remains fixed according to this plan.

Do not create a generic model builder solely for this nested config.

Validate only the fields necessary to prevent obvious config/model inconsistency.

---

# 61. Smoke Config

Create:

```text
configs/phase_a/isic2018_s0_small_cnn_smoke.yaml
```

Use:

```yaml
extends: isic2018_s0_small_cnn.yaml

experiment:
  name: phase_a_s0_small_cnn_smoke

scheduler:
  name: none

training:
  epochs: 1
  batch_size: 4
  num_workers: 0
  amp: false
```

Do not add:

```text
max_samples
special smoke dataset
new dataset subset abstraction
```

---

# 62. Phase-A CLI

Create:

```text
scripts/run_shape_pretraining.py
```

Follow the style of the existing experiment CLI.

Fresh run:

```bash
python scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn.yaml
```

Resume:

```bash
python scripts/run_shape_pretraining.py \
  --resume runs/phase_a_s0_small_cnn/<run-id>
```

Do not create a new CLI framework.

---

# 63. Run Artifacts

Completed Phase-A runs must contain:

```text
runs/phase_a_s0_small_cnn/<run-id>/
├── config.yaml
├── metadata.json
├── history.json
├── best.pt
├── last.pt
└── test_metrics.json
```

Initial:

```text
test_metrics.json
```

contains only:

```json
{
  "checkpoint": "best.pt",
  "split": "test",
  "loss": 0.0,
  "dice": 0.0
}
```

Do not save:

```text
geometry metrics
latent files
latent statistics
shape_representation.pt
```

yet.

---

# 64. Best Checkpoint Selection

For Phase A:

\[
\boxed{
\min \mathcal L_{\mathrm{rec,val}}
}
\]

Configuration:

```yaml
monitor: loss
monitor_mode: min
```

Existing segmentation experiments default to:

```yaml
monitor: dice
monitor_mode: max
```

---

# 65. Existing Test Compatibility

The authoritative CI command is:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Do not replace the project test framework with pytest.

Do not add pytest as a dependency.

If the local environment cannot execute tests because Python/dependencies are unavailable:

1. report the exact error;
2. do not fabricate successful results;
3. allow repository CI to become the authoritative execution environment.

---

# 66. Existing Segmentation Regression Check

After implementing:

```text
Task contract
SegmentationTask
generic Evaluator
generic Trainer
```

run the existing test suite before implementing the ShapeAutoencoder.

Existing:

```text
dataset tests
loss tests
metric tests
trainer tests
experiment tests
model tests
evaluation CLI tests
```

must continue to pass after any necessary API updates.

Do not weaken assertions merely to accommodate the refactor.

---

# 67. Shape Model Tests

Create:

```text
tests/test_shape_models.py
```

Required tests follow.

---

# 68. Shape Forward Test

Input:

```python
masks = torch.randn(
    2,
    1,
    256,
    256,
)
```

Verify:

```python
output.latent.shape == (
    2,
    256,
)
```

Verify:

```python
output.reconstruction_logits.shape == (
    2,
    1,
    256,
    256,
)
```

---

# 69. Invalid Shape Input Tests

At minimum test:

```text
wrong ndim
wrong channel count
wrong spatial resolution
```

Examples:

```text
[B, 256, 256]
[B, 3, 256, 256]
[B, 1, 128, 128]
```

Each should raise a clear:

```text
ValueError
```

Do not add excessive defensive tests.

---

# 70. Raw Logit Contract

Do not assert:

```text
0 <= reconstruction_logits <= 1
```

The output is raw logits.

No sigmoid belongs inside the ShapeAutoencoder.

---

# 71. Backward Gradient Test

Create binary target masks.

Compute:

```python
loss = BCEDiceLoss()(
    output.reconstruction_logits,
    targets,
)

loss.backward()
```

For every parameter where:

```python
parameter.requires_grad is True
```

require:

```python
parameter.grad is not None
```

and:

```python
torch.isfinite(
    parameter.grad
).all()
```

Additionally verify that each module:

```text
encoder
projector
decoder
```

contains at least one parameter whose gradient has non-zero magnitude.

This catches disconnected gradient paths.

---

# 72. Task Tests

Create/update:

```text
tests/test_tasks.py
```

Test:

```text
SegmentationTask
MaskReconstructionTask
```

---

# 73. SegmentationTask Tests

Verify:

```text
image is passed to model
mask is target
SegmentationOutput is required
criterion is used
existing evaluation metrics remain available
```

Invalid model output must raise:

```text
TypeError
```

not assertion failure.

---

# 74. MaskReconstructionTask Tests

Verify:

```text
mask is passed to model
mask is reconstruction target
RGB image is not required by forward
criterion is used
```

Evaluation returns exactly:

```text
loss
dice
```

where `"loss"` comes from the generic evaluator and:

```text
metrics == {"dice": ...}
```

inside the TaskStepOutput.

Do not return IoU.

Calling the shared Dice/IoU helper and discarding IoU is allowed.

---

# 75. TaskStepOutput Validation Tests

Test invalid outputs:

```text
non-scalar loss
non-finite loss
metrics contains "loss"
non-scalar metric
non-finite metric
batch_size <= 0
```

Do not create an excessively complex validation framework.

One helper in engine code is sufficient.

---

# 76. Generic Evaluator Tests

At minimum test:

```text
sample-weighted loss aggregation
sample-weighted metric aggregation
uneven final batch
metric-key consistency
non-finite rejection
empty loader rejection
model mode restoration
mode restoration after exception
```

Do not duplicate detailed Dice/HD95/etc metric tests.

Those remain in metric-specific tests.

---

# 77. Generic Trainer Tests

Verify the same Trainer can optimize:

```text
SegmentationTask
MaskReconstructionTask
```

Do not duplicate the entire trainer test suite for both.

One task-independence example plus existing trainer tests is sufficient.

---

# 78. Monitor Tests

Test:

```text
monitor = dice
monitor_mode = max
```

and:

```text
monitor = loss
monitor_mode = min
```

Test missing metric:

```text
monitor = does_not_exist
```

must raise clearly after validation.

Test invalid:

```text
monitor = ""
monitor_mode = invalid
```

both through:

```text
TrainerConfig
```

and config resolver where applicable.

---

# 79. Generic History Tests

With validation:

```text
history includes dynamic val_* metrics
```

Without validation:

```python
{
    "epoch": ...,
    "train_loss": ...,
}
```

must be valid.

Do not require null validation keys.

Update existing tests if necessary.

---

# 80. Checkpoint Schema Tests

New checkpoint must contain:

```text
metrics
monitor_name
monitor_mode
best_monitor_value
trainer_config
task_config
metadata
```

Phase-A task_config:

```python
{
    "name": "mask_reconstruction",
    "threshold": 0.5,
}
```

Segmentation task_config:

```python
{
    "name": "segmentation",
    "threshold": ...,
    "boundary_tolerance": ...,
}
```

---

# 81. Legacy Resume Compatibility Tests

Create a fabricated legacy format-version-2 segmentation checkpoint containing:

```text
best_val_dice
```

but not:

```text
monitor_name
monitor_mode
best_monitor_value
```

Verify:

```text
current monitor = dice/max
=> resume succeeds
```

Verify:

```text
current monitor = loss/min
=> resume raises clear error
```

Also verify new generic checkpoints resume correctly.

---

# 82. Evaluation CLI Compatibility Tests

Test resolution priority:

```text
CLI override
>
new checkpoint task_config
>
legacy trainer_config
>
default
```

Test:

```text
threshold
boundary_tolerance
```

Do not alter visualization behavior unless required by the API refactor.

---

# 83. Phase-A Experiment Smoke Test

Create:

```text
tests/test_shape_pretraining.py
```

Do not require real ISIC2018 files in CI.

Use:

```text
temporary dataset
synthetic data
patched dataset builder
```

following existing experiment-test style.

Verify:

```text
run directory created
config.yaml created
metadata.json created
history.json created
best.pt created
last.pt created
test_metrics.json created
```

Verify:

```text
test_metrics.json contains loss
test_metrics.json contains dice
```

Do not test scientific model quality in CI.

---

# 84. metadata.json Tests

Completed new runs must contain:

```text
best_epoch
monitor_name
monitor_mode
best_monitor_value
```

For:

```text
dice/max
```

also verify compatibility alias:

```text
best_val_dice
```

For Phase A:

```text
loss/min
```

`best_val_dice` does not need to exist.

---

# 85. Manual Validation Sequence

After unit tests pass, validate the implementation manually in this order.

---

# 86. Manual Check A — Full Unit Tests

Run:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Requirement:

```text
all existing and new tests pass
```

If execution is impossible because dependencies are unavailable:

```text
report exact environment limitation
do not fabricate results
```

---

# 87. Manual Check B — Tensor Flow

Verify:

\[
[B,1,256,256]
\]

\[
\downarrow E_M
\]

\[
[B,256,16,16]
\]

\[
\downarrow P_M
\]

\[
[B,64,4,4]
\]

\[
\downarrow
\]

\[
[B,256]
\]

\[
\downarrow D_M
\]

\[
[B,128,8,8]
\]

\[
\downarrow
\]

\[
[B,1,256,256].
\]

Debug-print these once during development if needed.

Remove unnecessary debug prints before completion.

---

# 88. Manual Check C — Gradient Flow

Verify:

```text
finite scalar loss
all trainable parameters receive gradients
all gradients are finite
encoder has non-zero gradient
projector has non-zero gradient
decoder has non-zero gradient
```

---

# 89. Manual Check D — Real-Data Smoke Run

Run:

```bash
python scripts/run_shape_pretraining.py \
  --config configs/phase_a/isic2018_s0_small_cnn_smoke.yaml
```

Expected:

```text
one epoch completes
validation completes
best.pt exists
last.pt exists
test evaluation completes
test_metrics.json exists
```

If real data or GPU is unavailable:

```text
do not fabricate results
report exact limitation
```

---

# 90. Manual Check E — Resume

Resume the smoke/full run.

Verify restoration of:

```text
model
optimizer
scheduler
GradScaler
history
completed epoch
best monitor value
monitor name
monitor mode
```

Verify training continues from the next epoch.

---

# 91. Manual Check F — Tiny Overfit

This is a scientific sanity check, not necessarily a CI test.

Train on a very small subset of masks.

Expected qualitative behavior:

```text
reconstruction loss decreases strongly
reconstruction Dice increases strongly
```

The purpose is to prove:

\[
E_M
\rightarrow
h_M
\rightarrow
D_M
\]

has enough capacity and correct gradient flow.

Do NOT add permanent dataset-subsetting infrastructure only for this check.

A temporary manifest or development-only script/command is acceptable.

---

# 92. Acceptance Criteria

The implementation is complete only when the following conditions hold.

## Engine

```text
one generic Trainer
one generic Evaluator
no shape-specific Trainer
no shape-specific Evaluator
generic sample-weighted aggregation
generic history
generic logging
generic monitor selection
generic checkpoint metrics
generic task_config storage
```

Trainer contains no:

```text
SegmentationOutput handling
image/mask semantics
Dice logic
IoU logic
threshold logic
boundary tolerance logic
```

---

# 93. Segmentation Compatibility Criteria

Existing:

```text
UNet
SAM baseline
MoE-SAM
```

experiment paths remain functional.

SegmentationTask reproduces current semantics.

Standalone evaluation CLI remains functional.

Legacy checkpoints remain evaluation-compatible.

Legacy version-2 resume works when:

```text
monitor semantics match
```

and rejects incompatible monitor configuration.

---

# 94. Phase-A Task Criteria

Mask reconstruction:

```text
uses batch["mask"] as input
uses batch["mask"] as target
does not require RGB image for forward
reuses existing BCE+Dice
reuses existing Dice metric
reports only loss + dice
```

---

# 95. Shape Model Criteria

Input:

\[
[B,1,256,256].
\]

Encoder:

\[
[B,256,16,16].
\]

Latent:

\[
[B,256].
\]

Reconstruction:

\[
[B,1,256,256].
\]

Required:

```text
decoder receives only latent
no skip connection
no encoder spatial bypass
raw logits output
no sigmoid
```

---

# 96. Gradient Criteria

Every trainable parameter must have:

```text
grad is not None
finite gradient
```

Each of:

```text
encoder
projector
decoder
```

must have at least one non-zero gradient.

---

# 97. Monitoring Criteria

Segmentation:

```text
dice/max
```

Phase A:

```text
loss/min
```

Missing monitored metrics raise errors.

Invalid monitoring config raises errors.

Resume rejects incompatible monitor semantics.

---

# 98. Reproducibility Criteria

A completed Phase-A run contains:

```text
config.yaml
metadata.json
history.json
best.pt
last.pt
test_metrics.json
```

Checkpoint contains:

```text
metrics
monitor_name
monitor_mode
best_monitor_value
trainer_config
task_config
metadata
```

metadata.json contains:

```text
best_epoch
monitor_name
monitor_mode
best_monitor_value
```

---

# 99. Deferred Work

After completing this task, do NOT continue to:

```text
S1 ResNet-18
S2 frozen SAM2
S3 reconstruction-adapted SAM2
geometry metrics
HD95 for Phase A
ASSD for Phase A
Boundary F1 for Phase A
latent shuffle
latent statistics
effective rank
Shape Teacher export
Phase B
Phase C
```

These are separate future milestones.

---

# 100. Final Coding-Agent Deliverable

When implementation is complete, report:

1. Starting branch.
2. Starting commit SHA.
3. Whether the working tree was initially clean.
4. Files added.
5. Files modified.
6. Generic engine/task architecture implemented.
7. Changes made to preserve existing segmentation behavior.
8. Exact TaskStepOutput aggregation semantics.
9. Exact Small-CNN architecture.
10. Exact spatial projector architecture.
11. Exact decoder architecture.
12. Shape input validation behavior.
13. Confirmation that existing BCE+Dice was reused.
14. Confirmation that existing Dice metric was reused.
15. Exact checkpoint schema.
16. Exact `task_config` behavior.
17. Exact metadata.json schema.
18. Legacy checkpoint compatibility behavior.
19. Monitor/resume behavior.
20. Unit-test command executed.
21. Unit-test results.
22. Real-data smoke-run command.
23. Smoke-run result or exact reason it could not run.
24. Resume test result.
25. Tiny-overfit result if environment permits.
26. Any unresolved issue.
27. List of intentionally deferred features.

Do not proceed to another research milestone automatically.

Stop after the Small-CNN Phase-A reconstruction baseline is implemented and verified.