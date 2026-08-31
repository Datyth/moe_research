"""ACDC conversion utilities.

Converts the raw ACDC archive (patient folders with cine MRI NIfTI files and
``Info.cfg`` descriptors) into 2D image/label pairs matching the MoE-SAM
paper's experimental setup:

- Only the annotated end-diastole (ED) and end-systole (ES) frames are used.
- Each short-axis slice with enough foreground becomes one 2D sample
  (MoE-SAM is built on the 2D SAM ViT-B image encoder).
- Intensities are z-score normalized per frame, clipped to [-3, 3] and
  rescaled to [0, 255] before saving as 8-bit PNGs.
- Labels keep the four ACDC classes: {0: background, 1: RV, 2: MYO, 3: LV}.
- No resizing happens here; the 256x256 crop happens at training time via the
  dataset transform (bilinear image / nearest-neighbor mask), matching the
  paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = {0: "background", 1: "right ventricle", 2: "myocardium", 3: "left ventricle"}


def parse_info_cfg(path: Path) -> dict[str, str]:
    """Parse an ACDC ``Info.cfg`` file into a string mapping.

    Typical content::

        Diastole:  46
        Esistole:  78
        Weight:    82
        Height:    171

    Returns keys lowercased with stripped values, e.g. ``{"diastole": "46",
    "esistole": "78", ...}`` (the official files misspell "Systole").
    """

    if not path.is_file():
        raise FileNotFoundError(f"Info.cfg not found: {path}")

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        values[key.strip().lower()] = value.strip()
    return values


def annotated_frame_numbers(info: dict[str, str]) -> list[int]:
    """Return the ED and ES frame numbers declared in a parsed ``Info.cfg``."""

    frames: list[int] = []
    for key in ("diastole", "esistole", "systole"):
        value = info.get(key)
        if value is None:
            continue
        try:
            frames.append(int(value))
        except ValueError as error:
            raise ValueError(
                f"Info.cfg key '{key}' must contain an integer frame number, "
                f"got '{value}'."
            ) from error
    if not frames:
        raise ValueError("Info.cfg is missing 'Diastole'/'Esistole' frame numbers.")
    return frames


def normalize_slice(slice2d: np.ndarray) -> np.ndarray:
    """Z-score normalize an MRI slice and rescale it to 8-bit [0, 255].

    Per-slice z-score with a small std floor, clipped to [-3, 3] so outlier
    intensities cannot dominate the 8-bit dynamic range.
    """

    array = np.asarray(slice2d, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got {array.ndim} dimensions.")

    mean = float(array.mean())
    std = float(array.std())
    if std < 1e-6:
        std = 1e-6

    normalized = (array - mean) / std
    normalized = np.clip(normalized, -3.0, 3.0)
    return np.asarray((normalized + 3.0) / 6.0 * 255.0, dtype=np.uint8)


def _load_nifti(path: Path):
    try:
        import nibabel
    except ImportError as error:  # pragma: no cover - environment guard
        raise ImportError(
            "ACDC conversion requires nibabel. Install it with "
            "`pip install nibabel`."
        ) from error
    return np.asarray(nibabel.load(str(path)).get_fdata())


def _resolve_frame_pairs(patient_dir: Path) -> list[tuple[str, str]]:
    """Return (image_stem, gt_path) pairs for the pre-extracted ED/ES frames.

    Only patients with ``patientXXX_frameYY_gt.nii.gz`` annotation files can
    be converted; the 50 official test patients ship without public labels.
    """

    gt_paths = sorted(patient_dir.glob("*_gt.nii.gz"))
    if not gt_paths:
        raise FileNotFoundError(
            f"No annotated *_gt.nii.gz frames found in {patient_dir}; "
            "only labeled ED/ES frames are supported."
        )
    return [
        (gt_path.name.removesuffix("_gt.nii.gz"), gt_path)
        for gt_path in gt_paths
    ]


def extract_patient_samples(
    patient_dir: Path,
    *,
    min_foreground_pixels: int = 10,
) -> list[dict[str, np.ndarray]]:
    """Extract 2D samples from one ACDC patient directory.

    Returns a list of dicts with keys ``image`` (uint8 [H, W]), ``mask``
    (uint8 class indices [H, W]) and ``sample_id``
    (``patientXXX_frameYY_sliceZZZ``). Slices without at least
    ``min_foreground_pixels`` labeled pixels are dropped.
    """

    if min_foreground_pixels < 0:
        raise ValueError("min_foreground_pixels must be non-negative.")

    patient_id = patient_dir.name
    samples: list[dict[str, np.ndarray]] = []

    for image_stem, gt_path in _resolve_frame_pairs(patient_dir):
        image_volume = _load_nifti(patient_dir / f"{image_stem}.nii.gz")
        gt_volume = _load_nifti(gt_path)

        if image_volume.ndim != 3 or gt_volume.ndim != 3:
            raise ValueError(
                f"Expected 3D image/label volumes for {image_stem}, got "
                f"{image_volume.ndim}D and {gt_volume.ndim}D."
            )
        if image_volume.shape != gt_volume.shape:
            raise ValueError(
                f"Image/label shape mismatch for {image_stem}: "
                f"{image_volume.shape} vs {gt_volume.shape}."
            )

        frame_number = int(image_stem.rsplit("frame", maxsplit=1)[-1])
        for slice_index in range(image_volume.shape[2]):
            image_slice = image_volume[:, :, slice_index]
            mask_slice = np.rint(gt_volume[:, :, slice_index]).astype(np.uint8)

            if int((mask_slice > 0).sum()) < min_foreground_pixels:
                continue

            samples.append(
                {
                    "image": normalize_slice(image_slice),
                    "mask": mask_slice,
                    "sample_id": f"{patient_id}_frame{frame_number:02d}_slice{slice_index:03d}",
                }
            )

    return samples


def save_sample(
    sample: dict[str, np.ndarray],
    output_root: Path,
    split_name: str,
) -> dict[str, str]:
    """Persist one sample as an RGB PNG image and a compressed .npz label."""

    image_dir = output_root / "images" / split_name
    label_dir = output_root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    sample_id = sample["sample_id"]

    image_path = image_dir / f"{sample_id}.png"
    image = Image.fromarray(sample["image"], mode="L").convert("RGB")
    if not image_path.exists():
        image.save(image_path)

    label_path = label_dir / f"{sample_id}.npz"
    if not label_path.exists():
        np.savez_compressed(label_path, mask=sample["mask"].astype(np.uint8))

    return {
        "image": str(Path("images") / split_name / f"{sample_id}.png"),
        "label": str(Path("labels") / split_name / f"{sample_id}.npz"),
    }


def convert_patients(
    patient_dirs: list[Path],
    output_root: Path,
    split_name: str,
    *,
    min_foreground_pixels: int = 10,
) -> list[dict[str, str]]:
    """Convert patient directories into manifest records for one split."""

    records: list[dict[str, str]] = []
    for patient_dir in sorted(patient_dirs):
        for sample in extract_patient_samples(
            patient_dir,
            min_foreground_pixels=min_foreground_pixels,
        ):
            records.append(save_sample(sample, output_root, split_name))
    return records
