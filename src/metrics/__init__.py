"""Metrics for segmentation experiments.

`compute_multiclass_dice_iou` and `compute_multiclass_surface_metrics` exist in
two implementations after the `long/moe-sam` and ACDC branches were merged:

* `segmentation.py` (imported last, so it wins the public name): accepts
  `[B, 1, H, W]` or `[B, H, W]` targets and scores a class absent from both
  prediction and target as perfect (1.0);
* `multiclass.py`: requires `[B, H, W]` targets and *excludes* an absent class
  from that sample's mean (the convention described in
  docs/segmentation_run_guide.md).

The evaluator therefore reports the first convention today. Switching the
public name to the `multiclass.py` implementation changes reported Dice/IoU on
slices where a class does not appear (ACDC, AMOS22, Synapse/BTCV), so it is a
measurement decision, not a cleanup - keep the two in sync deliberately.
"""

from .multiclass import compute_multiclass_dice_iou, compute_multiclass_surface_metrics
from .segmentation import (
    compute_binary_boundary_f1,
    compute_binary_hd,
    compute_binary_hd95_assd,
    compute_binary_surface_distances,
    compute_binary_surface_metrics,
    compute_multiclass_dice_iou,
    compute_multiclass_surface_metrics,
    extract_binary_surface,
)

__all__ = [
    "compute_binary_boundary_f1",
    "compute_binary_hd",
    "compute_binary_hd95_assd",
    "compute_binary_surface_distances",
    "compute_binary_surface_metrics",
    "compute_multiclass_dice_iou",
    "compute_multiclass_surface_metrics",
    "extract_binary_surface",
]
