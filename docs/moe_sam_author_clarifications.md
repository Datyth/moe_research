# MoE-SAM (MICCAI 2025) — protocol details confirmed by the authors

Source: email reply from Ruocheng Li (first author), 2026-09-07, answering our
2026-09-04 reproducibility questions. This file is the source of truth for the
choices below; the paper and the released repository
(github.com/Asphyxiate-Rye/E-SAM) do not specify any of them.

## 1. Evaluation metric

- The `HD` column in Table 1 is **HD95**, computed with
  `medpy.metric.binary.hd95`; Dice is `medpy.metric.binary.dc`.
- The `hd95` call is made **without voxel spacing**, so the reported figure is
  in **voxel units on the evaluation grid**, not millimetres, and does not use
  the original NIfTI physical spacing.
- Evaluation **assembles a 3D prediction array and scores each foreground class
  over that whole volume** — per-volume, per-class, not per-2D-slice.
- The released `calculate_metric_percase` computes Dice only. Their local copy
  additionally has `calculate_metric_percase_val`, called by `val_single_volume`,
  which is what produces the reported HD95.

What this means here: `src/metrics/volumetric.py` matches this protocol and
calls the same two medpy functions, so its `hd95` key is directly comparable to
the paper's `HD` column. `src/engine/evaluate()`'s per-slice metrics are **not**
comparable and should not be quoted against Table 1.

## 2. Intensity preprocessing, per dataset

The window is **not** the same for every CT benchmark:

| Dataset | Preprocessing |
| --- | --- |
| BTCV (13 organs) | clip/normalize equivalent to mapping **[-150, 500] -> [0, 1]** |
| Synapse CT (8 organs) | clip to **[-125, 275]**, normalize to [0, 1], labels remapped to 8 foreground classes |
| MMWHS | map **[-750, 750] -> [0, 1]**, seven anatomical labels remapped to consecutive class IDs |
| ACDC | **min-max normalization per loaded sample** — per *slice* during training, per *volume* during evaluation |

This repo previously applied the TransUNet/SAMed window [-125, 275] to BTCV as
well. That is now corrected: `scripts/data/synapse_conversion.py` defines
`HU_WINDOW = (-150.0, 500.0)` for BTCV and `SYNAPSE_CT_HU_WINDOW` for the
8-organ benchmark, and `scripts/data/prepare_synapse.py` takes `--hu-window`.
AMOS22 is not one of the paper's four datasets and keeps its existing window.

Consequence: BTCV slice data converted before this change is stale. Rebuild it
(manifest version `synapse-btcv-v3`, which records the window in
`split_metadata`) before quoting BTCV numbers against Table 1.

## 3. Train/test splits

Both datasets were "randomly divided into training and testing sets" — there is
no published fixed split and no case ID list to match. Our seeded case-level
random split is therefore methodologically equivalent, but the specific test
patients differ, so small deviations from Table 1 are expected and not a bug.

## 4. Training stability

They did **not** use gradient clipping on any of the four datasets.

Note this conflicts with our own observation: running the released code without
clipping, we saw training diverge partway through on more than one dataset
(loss spikes in a single step, validation Dice collapses toward 0 and does not
recover). Our configs currently set `gradient_clip_norm: 1.0`. Keeping it is a
deliberate deviation from the paper for stability; see the run guide.

## 5. MoE router `top_k` and short volumes

The authors offered a workaround rather than a fix: reduce **both** `evl_chunk`
and the `batch_size` argument passed when constructing the model, since the
latter is what determines the router's fixed `top_k`.

This repo does not need that workaround. `src/models/esam/_vendor/moe.py`
replaces the fixed count with `top_k_ratio`, resolved against the actual token
count on every forward call, so undersized batches and short evaluation volumes
cannot overflow `topk()`.
