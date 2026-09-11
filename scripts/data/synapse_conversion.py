#!/usr/bin/env python3
"""Synapse/BTCV-specific labels; conversion logic lives in
scripts/data/ct_conversion.py (shared with AMOS22).

BTCV's official test split (RawData/Testing) has no public labels, so
scripts/data/prepare_synapse.py pools every labeled case and carves out its
own case-level train/val/test split. MoE-SAM's authors confirmed by email
(2026-09-07) that they also "randomly divided" both datasets into train and
test rather than using a published split, so a seeded random case-level
split is the closest match available to us.
"""

from scripts.data.ct_conversion import (
    convert_case,
    convert_cases,
    window_and_normalize,
    write_dataset_json,
)

# HU window for the 13-organ BTCV protocol. The MoE-SAM authors confirmed by
# email (2026-09-07) that BTCV used "intensity clipping and normalization
# equivalent to mapping [-150, 500] to [0, 1]", which is NOT the [-125, 275]
# TransUNet/SAMed window this repo originally used here. That narrower window
# is correct for their separate 8-organ "Synapse" benchmark, which the same
# reply lists as clipping to [-125, 275]; see SYNAPSE_CT_HU_WINDOW below.
HU_WINDOW = (-150.0, 500.0)

# The paper reports "Synapse CT" (8 organs, 18/12 split) and "BTCV" (13
# organs) as separate columns even though both come from the same 30 labeled
# scans, and the authors gave each its own window.
SYNAPSE_CT_HU_WINDOW = (-125.0, 275.0)

# Standard 13-organ Synapse/BTCV label map (TransUNet/SAMed convention).
SYNAPSE_LABELS = {
    "0": "background",
    "1": "spleen",
    "2": "right kidney",
    "3": "left kidney",
    "4": "gallbladder",
    "5": "esophagus",
    "6": "liver",
    "7": "stomach",
    "8": "aorta",
    "9": "inferior vena cava",
    "10": "portal vein and splenic vein",
    "11": "pancreas",
    "12": "right adrenal gland",
    "13": "left adrenal gland",
}

__all__ = [
    "HU_WINDOW",
    "SYNAPSE_CT_HU_WINDOW",
    "SYNAPSE_LABELS",
    "convert_case",
    "convert_cases",
    "window_and_normalize",
    "write_dataset_json",
]
