#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def write_yaml(dataset_root: Path, class_names: list[str], yaml_path: Path) -> None:
    names_map = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_names)])
    text = (
        f"path: {dataset_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names_map}\n"
    )
    yaml_path.write_text(text, encoding="utf-8")


def convert_split(labels_src: Path, labels_dst: Path, num_classes: int, min_area: float) -> None:
    labels_dst.mkdir(parents=True, exist_ok=True)
    mask_files = sorted(labels_src.glob("*.png"))
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")
        if mask.ndim != 2:
            raise RuntimeError(f"Mask must be single-channel index map: {mask_path}")

        height, width = mask.shape
        lines: list[str] = []

        for class_id in range(1, num_classes + 1):
            binary = (mask == class_id).astype(np.uint8)
            if binary.max() == 0:
                continue

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    continue
                if cv2.contourArea(contour) < min_area:
                    continue

                points = contour.reshape(-1, 2).astype(np.float32)
                points[:, 0] /= float(width)
                points[:, 1] /= float(height)
                coords = " ".join(f"{v:.6f}" for v in points.reshape(-1))
                yolo_class = class_id - 1
                lines.append(f"{yolo_class} {coords}")

        out_file = labels_dst / f"{mask_path.stem}.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO segmentation dataset from indexed PNG masks.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/spectralwaste_segmentation"),
        help="Source dataset root containing rgb/ and labels_rgb/ folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/spectralwaste_yolo_seg"),
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=3.0,
        help="Minimum contour area (pixels) to keep.",
    )
    args = parser.parse_args()

    source_root = args.source_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    meta = json.loads((source_root / "meta.json").read_text(encoding="utf-8"))
    categories = meta["categories"]
    if len(categories) < 2:
        raise RuntimeError("Expected background + at least one foreground class in meta.json")
    class_names = categories[1:]
    num_classes = len(class_names)

    for split in ("train", "val", "test"):
        images_src = source_root / "rgb" / split
        images_dst = output_root / "images" / split
        labels_src = source_root / "labels_rgb" / split
        labels_dst = output_root / "labels" / split

        images_dst.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(images_src.glob("*.png")):
            link_path = images_dst / image_path.name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(image_path.resolve())

        convert_split(labels_src=labels_src, labels_dst=labels_dst, num_classes=num_classes, min_area=args.min_area)

    write_yaml(output_root, class_names, output_root / "spectralwaste_seg.yaml")
    print(f"Prepared dataset at: {output_root.resolve()}")
    print(f"YAML: {(output_root / 'spectralwaste_seg.yaml').resolve()}")


if __name__ == "__main__":
    main()

# python prepare_yolo_seg_dataset.py --source-root data/spectralwaste_segmentation --output-root data/spectralwaste_yolo_seg