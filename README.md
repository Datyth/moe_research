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

### ACDC (cardiac MRI, multiclass)

The ACDC pipeline targets the MoE-SAM paper setup (see `docs/research_papers/1526_paper.pdf`): labeled end-diastole/end-systole frames are converted to 2D slices, z-score normalized and clipped to [-3, 3] before 8-bit rescaling, and resized to 256x256 at training time. Labels keep the four ACDC classes (background, RV, MYO, LV). Because the official 50 test patients have no public labels, the 100 labeled patients are re-split by patient (60/20/20 by default) so no slice leaks across splits.

Place the raw ACDC archive (`acdc.zip`, requires registration at the challenge site) under `dataset/raw/acdc/`, then run:

```bash
pip install nibabel
bash scripts/03_prepare_acdc_dataset.sh
```

Converted 2D samples are stored under `dataset/acdc/` and the frozen patient-level split is defined by `manifests/acdc_v1.json`. Train the UNet baseline with the paper recipe (256x256, CE+Dice, AdamW lr 5e-4, DSC/HD metrics):

```bash
python scripts/run_experiment.py --config configs/acdc_unet.yaml
```

### Synapse CT (8 organs, native NPZ/HDF5)

The audited TransUNet-style dataset at
`/home/teama/projects/project_01/dataset/synapse_ct` contains 1,280 training
NPZ slices from 18 cases and 12 held-out HDF5 volumes with labels 0–8. Its
tracked descriptor is `manifests/synapse_ct_v1.json`; the loader expands
`train.txt` into 2D training samples and each `val.txt` HDF5 volume into 2D
evaluation slices.

There is no separate validation cohort: `val.txt` is deliberately used for
both validation/checkpoint selection and final evaluation. Consequently, the
final metric is not an independent test estimate. Run the 224×224 promptless
SAM configuration with:

```bash
python scripts/run_experiment.py --config configs/synapse_ct_sam.yaml
```

The frozen train, validation, and test splits are defined by:

```text
manifests/acdc_v1.json             acdc-v1            1,075 / 363 / 403   slices (60/20/20 patients)
manifests/amos22_ct_v1.json        amos22-ct-v1      21,288 / 4,699 / 4,292 slices (210/45/45 cases)
manifests/isic2018_task1_v1.json   isic2018-task1-v1  2,075 / 519 / 100    images
manifests/synapse_btcv_v2.json     synapse-btcv-v2    1,542 / 242 / 394   slices (20/4/6 cases)
manifests/synapse_ct_v1.json        synapse-ct-8organ-v1  1,280 train slices / 12 shared val-test volumes (18/12 cases)
```

All five datasets passed the full integrity audit (every image and compressed mask opens, paths resolve, image/mask sizes match, declared class IDs are exactly the ones present, no duplicate, missing, unreferenced or leaked samples).

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

### Promptless SAM baseline

Each prepared dataset has an adapter-free SAM ViT-B baseline: `acdc_sam`,
`amos22_sam`, `isic2018_sam`, and `synapse_sam`. It uses no point, box, or
mask prompt; the sparse prompt is empty and SAM's learned `no_mask_embed` is
the dense prompt. The pretrained image and prompt encoders are frozen while
the dataset-specific mask decoder is trained. This is therefore a
**promptless fine-tuned SAM** semantic-segmentation baseline, not SAM's
original prompted zero-shot interface.

```bash
python scripts/run_experiment.py --config configs/isic2018_sam_smoke.yaml
python scripts/run_experiment.py --config configs/isic2018_sam.yaml
```

The official ViT-B checkpoint must be available at
`checkpoints/sam_vit_b_01ec64.pth`. The baseline configs use the same data,
loss, optimizer, schedule, training settings, and seed as E0, so the
comparison isolates the Adapter contribution.

### E-SAM ablation configs

Every prepared dataset carries the same four-row MoE-SAM ablation (paper Table 2) plus a 1-epoch sanity config, each extending its own `<dataset>_common.yaml`:

| Row | `use_moe` | `use_lpeg` | ACDC | AMOS22 CT | ISIC 2018 | Synapse/BTCV |
|---|---|---|---|---|---|---|
| E0 — SAM + Adapter | off | off | `acdc_e0` | `amos22_e0` | `isic2018_e0` | `synapse_e0` |
| E1 — + MoE-FEB | on | off | `acdc_e1` | `amos22_e1` | `isic2018_e1` | `synapse_e1` |
| E2 — + LPEG | off | on | `acdc_e2` | `amos22_e2` | `isic2018_e2` | `synapse_e2` |
| E3 — full MoE-SAM | on | on | `acdc_e3` | `amos22_e3` | `isic2018_e3` | `synapse_e3` |

`acdc_e3.yaml` replaces the former `acdc_moesam.yaml` (same settings, renamed for consistency). Rows within one dataset differ only in the `model` block, so the ablation measures the components and nothing else. `tests/test_configs.py` and `tests/test_esam_ablation_matrix.py` enforce that and build all 16 models on CPU. The complete progression is promptless SAM → E0 Adapter → E1/E2 component additions → E3 full MoE-SAM.

```bash
python scripts/run_experiment.py --config configs/synapse_e3.yaml
python scripts/run_experiment.py --config configs/acdc_smoke.yaml   # quick pre-flight check
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

Available loss names are `bce`, `dice`, and `bce_dice` for binary tasks, and `ce_dice` for multiclass tasks (ACDC), which combines cross-entropy and soft Dice over the foreground classes.

### Dataset paths across machines

`dataset.root`, `dataset.manifest` and `experiment.output_root` resolve as follows:

- **Relative paths** (e.g. `dataset/acdc`) are resolved against the project root, so configs are portable as long as data lives inside the repository.
- **Absolute paths** are used as-is.
- **`${ENV_VAR}`** references are expanded at load time — use them to point at machine-specific data locations without editing YAML:

```yaml
dataset:
  root: ${ACDC_DATA_ROOT}   # export ACDC_DATA_ROOT=/data/acdc on the server
```

An undefined environment variable raises a clear error at config load instead of failing later on a missing file.

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
