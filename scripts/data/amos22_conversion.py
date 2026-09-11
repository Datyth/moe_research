#!/usr/bin/env python3
"""AMOS22-specific labels; conversion logic lives in scripts/data/ct_conversion.py
(shared with Synapse/BTCV)."""

from scripts.data.ct_conversion import (
    convert_case,
    convert_cases,
    window_and_normalize,
    write_dataset_json,
)

AMOS_LABELS = {
    "0": "background",
    "1": "spleen",
    "2": "right kidney",
    "3": "left kidney",
    "4": "gall bladder",
    "5": "esophagus",
    "6": "liver",
    "7": "stomach",
    "8": "aorta",
    "9": "postcava",
    "10": "pancreas",
    "11": "right adrenal gland",
    "12": "left adrenal gland",
    "13": "duodenum",
    "14": "bladder",
    "15": "prostate/uterus",
}

__all__ = [
    "AMOS_LABELS",
    "convert_case",
    "convert_cases",
    "window_and_normalize",
    "write_dataset_json",
]
