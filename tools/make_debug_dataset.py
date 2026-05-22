#!/usr/bin/env python3
"""Create a tiny height-classification dataset from a large source map.

The generated files follow the Dataframes/Images layout consumed by
``dataloaders/HCDataset.py`` and are intended for smoke tests, parser checks,
and quick debugging. The labels are synthetic; use the original datasets for
paper results.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image


Image.MAX_IMAGE_PIXELS = None

DEFAULT_SOURCE_CANDIDATES = [
    Path("/root/sxy1/Dataset/QingDao/@map@120.42118549346924@36.60643328438966@120.4841423034668@36.573836401969416@.jpg"),
    Path("/root/sxy1/Dataset/BeiJing/CT01@116.35551452636700@40.09815882135800@116.44632339477501@40.15118932709900@.tif"),
    Path("/root/sxy1/Dataset/ShangHai/CT02@121.34485244750999@31.08564938117800@121.43737792968800@31.12900748947800@.tif"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a compact debug dataset in the altitude-classification layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-map", type=Path, default=None, help="Large map image used as the crop source.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/sxy1/Dataset"), help="Fallback root used to search for a source map.")
    parser.add_argument("--output-root", type=Path, default=Path("data/debug/height_classification"), help="Output root containing train/ and test/.")
    parser.add_argument("--dataset-name", type=str, default="debug_height", help="Dataset folder and CSV stem.")
    parser.add_argument("--heights", type=float, nargs="+", default=[100, 150, 200, 250], help="Synthetic flight heights in meters.")
    parser.add_argument("--samples-per-height", type=int, default=16, help="Training samples generated for each height.")
    parser.add_argument("--test-samples-per-height", type=int, default=4, help="Test samples generated for each height.")
    parser.add_argument("--image-size", type=int, nargs=2, default=[336, 448], metavar=("HEIGHT", "WIDTH"), help="Output image size.")
    parser.add_argument("--bin-width", type=int, default=50, help="Altitude-bin width used to derive the CSV flight_class column.")
    parser.add_argument("--source-downsample", type=int, default=4, help="JPEG decoder downsample factor for very large source maps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing debug dataset with the same name.")
    return parser.parse_args()


def find_source_map(source_map: Path | None, dataset_root: Path) -> Path:
    if source_map is not None:
        if not source_map.exists():
            raise FileNotFoundError(f"source map does not exist: {source_map}")
        return source_map

    for candidate in DEFAULT_SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate

    image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in image_suffixes:
            return path

    raise FileNotFoundError(f"no image source found under {dataset_root}")


def open_source(path: Path, downsample: int) -> Image.Image:
    image = Image.open(path)
    if downsample > 1 and image.format == "JPEG":
        target_size = (max(1, image.size[0] // downsample), max(1, image.size[1] // downsample))
        image.draft("RGB", target_size)
    return image.convert("RGB")


def crop_size_for_height(height: float, heights: list[float], image_size: tuple[int, int], source_size: tuple[int, int]) -> tuple[int, int]:
    out_h, out_w = image_size
    source_w, source_h = source_size
    min_h = min(heights)
    max_h = max(heights)
    ratio = 0.0 if max_h == min_h else (height - min_h) / (max_h - min_h)
    crop_w = int(out_w * (1.0 + 0.9 * ratio))
    crop_h = int(crop_w * out_h / out_w)
    if crop_w > source_w or crop_h > source_h:
        scale = min(source_w / crop_w, source_h / crop_h)
        crop_w = max(1, int(crop_w * scale))
        crop_h = max(1, int(crop_h * scale))
    return crop_w, crop_h


def prepare_split_dirs(root: Path, dataset_name: str, overwrite: bool) -> tuple[Path, Path]:
    dataframe_dir = root / "Dataframes"
    image_dir = root / "Images" / dataset_name
    csv_path = dataframe_dir / f"{dataset_name}.csv"

    if (csv_path.exists() or image_dir.exists()) and not overwrite:
        raise FileExistsError(f"{root} already contains {dataset_name}; rerun with --overwrite")

    dataframe_dir.mkdir(parents=True, exist_ok=True)
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    return csv_path, image_dir


def write_split(
    split_root: Path,
    dataset_name: str,
    source: Image.Image,
    source_name: str,
    heights: list[float],
    samples_per_height: int,
    image_size: tuple[int, int],
    bin_width: int,
    rng: random.Random,
    overwrite: bool,
) -> int:
    csv_path, image_dir = prepare_split_dirs(split_root, dataset_name, overwrite)
    out_h, out_w = image_size
    source_w, source_h = source.size
    rows = []

    min_height = min(heights)
    for height in heights:
        crop_w, crop_h = crop_size_for_height(height, heights, image_size, source.size)
        max_x = max(0, source_w - crop_w)
        max_y = max(0, source_h - crop_h)
        for _ in range(samples_per_height):
            x = rng.randint(0, max_x) if max_x else 0
            y = rng.randint(0, max_y) if max_y else 0
            rotation = 0.0
            filename = f"@{dataset_name}@{height:.2f}@{rotation:.2f}@{x}@{y}@.png"
            crop = source.crop((x, y, x + crop_w, y + crop_h)).resize((out_w, out_h), Image.Resampling.BICUBIC)
            crop.save(image_dir / filename)

            rows.append(
                {
                    "year": dataset_name,
                    "origin_img": source_name,
                    "flight_height": f"{height:.2f}",
                    "flight_class": int((height - min_height) // bin_width),
                    "rotation_angle": f"{rotation:.2f}",
                    "loc_x": x,
                    "loc_y": y,
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "origin_img", "flight_height", "flight_class", "rotation_angle", "loc_x", "loc_y"])
        writer.writeheader()
        writer.writerows(rows)

    return sample_id


def main():
    args = parse_args()
    source_path = find_source_map(args.source_map, args.dataset_root)
    source = open_source(source_path, args.source_downsample)
    rng = random.Random(args.seed)
    image_size = (args.image_size[0], args.image_size[1])

    train_count = write_split(
        args.output_root / "train",
        args.dataset_name,
        source,
        source_path.name,
        args.heights,
        args.samples_per_height,
        image_size,
        args.bin_width,
        rng,
        args.overwrite,
    )
    test_count = write_split(
        args.output_root / "test",
        args.dataset_name,
        source,
        source_path.name,
        args.heights,
        args.test_samples_per_height,
        image_size,
        args.bin_width,
        rng,
        args.overwrite,
    )

    print(f"source={source_path}")
    print(f"source_size={source.size}")
    print(f"train_images={train_count}")
    print(f"test_images={test_count}")
    print(f"output_root={args.output_root}")


if __name__ == "__main__":
    main()
