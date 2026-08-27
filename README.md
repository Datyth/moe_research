# Medical Image Segmentation Research Framework

A lightweight research framework for reproducible medical image segmentation experiments. The current baseline targets binary lesion segmentation on ISIC 2018 with UNet, while keeping datasets, models, losses, and training components independently extensible.

## Overview

The framework provides:

- YAML-based experiment configuration.
- Dataset, model, and loss registries.
- A strict model output contract based on `SegmentationOutput`.
- A versioned and Git-tracked dataset split manifest.
- Reproducible run folders with configuration, metadata, history, checkpoints, and test metrics.
- Resume support for model, optimizer, AMP scaler, scheduler, epoch, and best metric.
- Basic cosine and reduce-on-plateau learning-rate schedulers.
- Multi-seed summaries with mean and sample standard deviation.
- CPU unit tests and a minimal GitHub Actions workflow.

The main workflow is:

```text
Tracked manifest -> YAML config -> training and validation
                 -> best and last checkpoints
                 -> final test using the best checkpoint
                 -> per-run artifacts and multi-seed summary
```

## Project structure

```text
configs/                    Experiment YAML files
manifests/                  Versioned dataset split manifests
scripts/
├── data/                   Dataset conversion and preparation
├── evaluation/             Checkpoint evaluation and visualization
├── training/               Legacy-compatible training wrappers
├── run_experiment.py       Main experiment entrypoint
└── summarize_experiments.py
src/
├── configs/                Dataset and experiment configuration
├── data/                   Dataset API, registry, and transforms
├── engine/                 Trainer and evaluator
├── losses/                 Loss implementations and registry
├── models/                 Model API, registry, and architectures
└── experiment.py           Experiment builders and lifecycle utilities
tests/                      CPU unit and integration tests
docs/                       Detailed usage documentation
/home/teama/projects/project_01/dataset/
                            External local dataset root
runs/                       Generated experiment artifacts ignored by Git
```

## Setup and data

Python 3.10 is recommended. Create a Conda environment and install the project dependencies:

```bash
conda create --name medical-seg python=3.10 pip -y
conda activate medical-seg
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download and prepare ISIC 2018:

```bash
bash scripts/02_prepare_isic2018_dataset.sh
```

Raw archives and extracted source files are stored under `/home/teama/projects/project_01/dataset/raw/isic2018`. Converted images and sparse masks are stored under `/home/teama/projects/project_01/dataset/isic2018_task1`.

The frozen train, validation, and test split is defined by:

```text
manifests/isic2018_task1_v1.json
```

The training seed in an experiment YAML never changes this split.

## Running experiments

Start a new experiment:

```bash
python scripts/run_experiment.py --config configs/unet.yaml
```

Resume an interrupted run using the configuration saved in its run folder:

```bash
python scripts/run_experiment.py --resume runs/unet_isic2018/<run-id>
```

Each run is stored under `runs/<experiment>/<run-id>/`:

```text
config.yaml
metadata.json
history.json
best.pt
last.pt
test_metrics.json
```

Change the loss by replacing only the `loss` block in the YAML. For example:

```yaml
loss:
  name: dice
  smooth: 1.0
  epsilon: 1.0e-7
```

Available loss names are `bce`, `dice`, and `bce_dice`.

The ShapeMoE pipeline runs in two phases. Both write a run folder with the same
file names as an ordinary experiment, but their runs are **not** picked up by
`summarize_experiments.py`: that script filters on a `status` field and requires
the full surface-metric set, neither of which these phases produce yet.

Phase 0 pretrains the mask-VAE shape teacher on ground-truth masks only:

```bash
python scripts/training/pretrain_teacher_vae.py \
  --config configs/teacher_vae_isic2018.yaml
```

Phase 1 trains the shape-routed segmenter under that frozen teacher, which
supplies the shape posterior the student must reproduce from the image alone:

```bash
python scripts/training/train_shapemoe.py \
  --config configs/shapemoe_isic2018.yaml \
  --teacher runs/teacher_vae_isic2018/<run-id>/best.pt
```

`model.latent_dim` in the Phase 1 config must match the teacher's, and omitting
`--teacher` disables distillation, which leaves the shape encoder untrained.
Which parts follow the ShapeMoE paper, which are substitutions, and every choice
the specification left open are recorded in
[ShapeMoE implementation assumptions](docs/shapemoe_assumptions.md).

Summarize completed runs across multiple seeds:

```bash
python scripts/summarize_experiments.py \
  --runs-root runs \
  --experiment unet_isic2018 \
  --output runs/unet_isic2018/summary.csv
```

Create optional test visualizations from the best checkpoint:

```bash
python scripts/evaluation/evaluate.py \
  --checkpoint runs/unet_isic2018/<run-id>/best.pt \
  --data-root /home/teama/projects/project_01/dataset/isic2018_task1 \
  --split test \
  --output-dir runs/unet_isic2018/<run-id>/visualization \
  --device cuda \
  --num-visualizations 12
```

## Tests and documentation

Run the CPU test suite:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The detailed guide covers configuration fields, module contracts, dataset manifests, training, resume behavior, evaluation, summaries, extension points, and troubleshooting:

[Experiment framework and segmentation run guide](docs/segmentation_run_guide.md)
