#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_sample_id(mask_path: Path) -> str:
    """Convert ISIC_xxx_segmentation.png -> ISIC_xxx."""
    stem = mask_path.stem
    suffix = "_segmentation"

    if stem.lower().endswith(suffix):
        stem = stem[: -len(suffix)]

    return stem


def build_image_index(image_dir: Path) -> dict[str, Path]:
    index = {}

    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.stem] = path

    return index


def materialize_image(
    source: Path,
    destination: Path,
    copy_images: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        return

    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def convert_split(
    image_dir: Path,
    mask_dir: Path,
    output_root: Path,
    split_name: str,
    make_pseudo: bool,
    copy_images: bool,
) -> list[dict[str, str]]:
    image_index = build_image_index(image_dir)
    mask_paths = sorted(mask_dir.rglob("*.png"))

    if not image_index:
        raise RuntimeError(f"No images found in {image_dir}")

    if not mask_paths:
        raise RuntimeError(f"No masks found in {mask_dir}")

    records = []

    image_output_dir = output_root / "images" / split_name
    label_output_dir = output_root / "labels" / split_name
    pseudo_output_dir = output_root / "pseudo" / split_name

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    if make_pseudo:
        pseudo_output_dir.mkdir(parents=True, exist_ok=True)

    missing_images = []

    for mask_path in mask_paths:
        sample_id = get_sample_id(mask_path)

        if sample_id not in image_index:
            missing_images.append(sample_id)
            continue

        source_image_path = image_index[sample_id]

        with Image.open(source_image_path) as image:
            image = image.convert("RGB")
            width, height = image.size

        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"))

        mask = (mask > 127).astype(np.uint8)

        if mask.shape != (height, width):
            raise ValueError(
                f"Shape mismatch for {sample_id}: "
                f"image={(height, width)}, mask={mask.shape}"
            )

        # Store the foreground label with shape [1, H, W].
        label = mask[np.newaxis, ...]

        destination_image_path = (
            image_output_dir / source_image_path.name
        )
        materialize_image(
            source_image_path,
            destination_image_path,
            copy_images=copy_images,
        )

        shape_string = f"(1,{height},{width})"
        label_filename = f"{sample_id}.{shape_string}.npz"
        destination_label_path = label_output_dir / label_filename

        # Store [1, H * W] sparse matrix.
        sparse_label = sparse.csr_matrix(label.reshape(1, -1))
        sparse.save_npz(destination_label_path, sparse_label)

        record = {
            "image": destination_image_path.relative_to(
                output_root
            ).as_posix(),
            "label": destination_label_path.relative_to(
                output_root
            ).as_posix(),
        }

        if make_pseudo:
            # Temporary proxy:
            # background = -1, lesion instance = 1.
            #
            # This lets the original training code run, but it is derived
            # from GT and is NOT a faithful reconstruction of the paper's
            # pseudo-label generation pipeline.
            pseudo = np.full((1, height, width), -1, dtype=np.int16)
            pseudo[0, mask == 1] = 1

            destination_pseudo_path = (
                pseudo_output_dir / f"{sample_id}.npy"
            )
            np.save(destination_pseudo_path, pseudo)

            record["pseudo"] = destination_pseudo_path.relative_to(
                output_root
            ).as_posix()

        records.append(record)

    if missing_images:
        print(
            f"[WARNING] {len(missing_images)} masks have no matching image."
        )
        print("First missing IDs:", missing_images[:10])

    print(
        f"[{split_name}] Converted {len(records)} samples."
    )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_images", type=Path, required=True)
    parser.add_argument("--train_masks", type=Path, required=True)
    parser.add_argument("--val_images", type=Path, required=True)
    parser.add_argument("--val_masks", type=Path, required=True)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/isic2018_task1"),
    )

    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images instead of creating symbolic links.",
    )

    args = parser.parse_args()

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    training_records = convert_split(
        image_dir=args.train_images.resolve(),
        mask_dir=args.train_masks.resolve(),
        output_root=output_root,
        split_name="train",
        make_pseudo=True,
        copy_images=args.copy_images,
    )

    test_records = convert_split(
        image_dir=args.val_images.resolve(),
        mask_dir=args.val_masks.resolve(),
        output_root=output_root,
        split_name="test",
        make_pseudo=False,
        copy_images=args.copy_images,
    )

    dataset_description = {
        "labels": {
            "0": "background",
            "1": "skin lesion",
        },
        "training": training_records,
        "test": test_records,
    }

    dataset_json_path = output_root / "dataset.json"

    with open(dataset_json_path, "w", encoding="utf-8") as file:
        json.dump(
            dataset_description,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Dataset JSON: {dataset_json_path}")
    print(f"Training samples: {len(training_records)}")
    print(f"Test samples: {len(test_records)}")


if __name__ == "__main__":
    main()
