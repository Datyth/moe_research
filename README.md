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
├── training/               Legacy CLI wrappers and per-model convenience scripts
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
checkpoints/                Pretrained model weights (e.g. SAM) ignored by Git
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
