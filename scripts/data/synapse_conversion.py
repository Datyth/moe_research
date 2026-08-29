#!/usr/bin/env python3
"""Synapse/BTCV-specific labels; conversion logic lives in
scripts/data/ct_conversion.py (shared with AMOS22).

BTCV's official test split (RawData/Testing) has no public labels, so
scripts/data/prepare_synapse.py pools every labeled case and carves out its
own case-level train/val/test split, exactly like AMOS22.
"""

from scripts.data.ct_conversion import (
    convert_case,
    convert_cases,
    window_and_normalize,
    write_dataset_json,
)

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
    "SYNAPSE_LABELS",
    "convert_case",
    "convert_cases",
    "window_and_normalize",
    "write_dataset_json",
]
