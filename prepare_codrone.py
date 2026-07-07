from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


CODRONE_NAMES = (
    "car",
    "truck",
    "traffic-sign",
    "people",
    "motor",
    "bicycle",
    "traffic-light",
    "tricycle",
    "bridge",
    "bus",
    "boat",
    "ship",
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "val", "test")


@dataclass
class SplitStats:
    split: str
    images: int = 0
    xmls: int = 0
    labels: int = 0
    objects: int = 0
    ignored: int = 0
    skipped_unknown: int = 0
    skipped_invalid: int = 0
    missing_images: list[str] = field(default_factory=list)
    unmatched_images: list[str] = field(default_factory=list)


def normalize_name(name: str) -> str:
    """Normalize known CODrone filename noise for fallback image matching."""
    stem = Path(name).stem.strip()
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem.replace("-", "_").lower()


def build_image_index(image_dir: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for suffix in IMAGE_SUFFIXES:
        for image_path in image_dir.glob(f"*{suffix}"):
            keys = {image_path.stem.lower(), normalize_name(image_path.name)}
            for key in keys:
                image_index.setdefault(key, image_path)
    return image_index


def find_image(xml_path: Path, root: ET.Element, image_index: dict[str, Path]) -> Path | None:
    candidates = [xml_path.stem]
    filename = root.findtext("filename")
    if filename:
        candidates.append(filename)

    for candidate in candidates:
        for key in (Path(candidate).stem.lower(), normalize_name(candidate)):
            image_path = image_index.get(key)
            if image_path is not None:
                return image_path
    return None


def read_size(root: ET.Element) -> tuple[float, float]:
    width = root.findtext("size/width")
    height = root.findtext("size/height")
    if width is None or height is None:
        raise ValueError("missing image size")
    image_width = float(width)
    image_height = float(height)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image size: {image_width}x{image_height}")
    return image_width, image_height


def object_to_yolo_line(
    obj: ET.Element,
    class_to_id: dict[str, int],
    image_width: float,
    image_height: float,
) -> tuple[str | None, str]:
    name = (obj.findtext("name") or "").strip()
    if name == "ignored":
        return None, "ignored"
    if name not in class_to_id:
        return None, "unknown"

    box = obj.find("bndbox")
    if box is None:
        return None, "invalid"

    points: list[tuple[float, float]] = []
    for index in range(4):
        x_text = box.findtext(f"x{index}")
        y_text = box.findtext(f"y{index}")
        if x_text is None or y_text is None:
            return None, "invalid"
        points.append((float(x_text), float(y_text)))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min = min(max(min(xs), 0.0), image_width)
    x_max = min(max(max(xs), 0.0), image_width)
    y_min = min(max(min(ys), 0.0), image_height)
    y_max = min(max(max(ys), 0.0), image_height)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None, "invalid"

    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height
    width /= image_width
    height /= image_height
    return f"{class_to_id[name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}", "ok"


def clean_split(output_root: Path, split: str) -> None:
    for kind in ("images", "labels"):
        split_dir = output_root / kind / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / "labels" / f"{split}.cache"
    if cache_path.exists():
        cache_path.unlink()


def convert_split(source_root: Path, output_root: Path, split: str, clean: bool) -> SplitStats:
    source_split = source_root / split
    source_image_dir = source_split / "images"
    source_xml_dir = source_split / "labels"
    if not source_image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {source_image_dir}")
    if not source_xml_dir.is_dir():
        raise FileNotFoundError(f"XML label directory not found: {source_xml_dir}")

    if clean:
        clean_split(output_root, split)

    output_image_dir = output_root / "images" / split
    output_label_dir = output_root / "labels" / split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: index for index, name in enumerate(CODRONE_NAMES)}
    image_index = build_image_index(source_image_dir)
    used_images: set[Path] = set()
    stats = SplitStats(split=split, images=len(set(image_index.values())), xmls=len(list(source_xml_dir.glob("*.xml"))))

    for xml_path in sorted(source_xml_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        image_path = find_image(xml_path, root, image_index)
        if image_path is None:
            stats.missing_images.append(xml_path.name)
            continue

        used_images.add(image_path)
        shutil.copy2(image_path, output_image_dir / image_path.name)
        image_width, image_height = read_size(root)
        lines: list[str] = []

        for obj in root.findall("object"):
            line, status = object_to_yolo_line(obj, class_to_id, image_width, image_height)
            if status == "ok" and line is not None:
                lines.append(line)
                stats.objects += 1
            elif status == "ignored":
                stats.ignored += 1
            elif status == "unknown":
                stats.skipped_unknown += 1
            else:
                stats.skipped_invalid += 1

        (output_label_dir / f"{image_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats.labels += 1

    stats.unmatched_images = sorted(path.name for path in set(image_index.values()) - used_images)
    return stats


def write_classes(output_root: Path) -> None:
    (output_root / "classes.txt").write_text("\n".join(CODRONE_NAMES) + "\n", encoding="utf-8")


def write_dataset_yaml(path: Path, dataset_root: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CODRONE_NAMES))
    content = (
        "# CODrone horizontal-box detection dataset converted from VOC-style OBB annotations.\n"
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 12\n"
        "names:\n"
        f"{names}\n"
    )
    path.write_text(content, encoding="utf-8")


def validate_output(output_root: Path, split: str) -> tuple[int, int, list[str], list[str]]:
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_stems = {path.stem for suffix in IMAGE_SUFFIXES for path in image_dir.glob(f"*{suffix}")}
    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)
    return len(image_stems), len(label_stems), missing_labels, extra_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"D:\temp\CODrone"), help="Raw CODrone root")
    parser.add_argument("--output", type=Path, default=Path(r"D:\datasets\CODrone"), help="Ultralytics dataset root")
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("ultralytics/cfg/datasets/CODrone.yaml"),
        help="Dataset YAML to write",
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS, help="Splits to convert")
    parser.add_argument("--clean", action="store_true", help="Remove output split directories before conversion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"source dataset root not found: {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)

    write_classes(args.output)
    write_dataset_yaml(args.yaml, args.output)

    all_ok = True
    for split in args.splits:
        stats = convert_split(args.source, args.output, split, args.clean)
        image_count, label_count, missing_labels, extra_labels = validate_output(args.output, split)
        all_ok = all_ok and not stats.missing_images and not missing_labels and not extra_labels
        print(
            f"{split}: source_images={stats.images}, xmls={stats.xmls}, output_images={image_count}, "
            f"output_labels={label_count}, objects={stats.objects}, ignored={stats.ignored}, "
            f"unknown={stats.skipped_unknown}, invalid={stats.skipped_invalid}"
        )
        if stats.missing_images:
            print(f"  missing images for XML: {len(stats.missing_images)}; first={stats.missing_images[:5]}")
        if stats.unmatched_images:
            print(f"  images without XML: {len(stats.unmatched_images)}; first={stats.unmatched_images[:5]}")
        if missing_labels:
            print(f"  images without labels: {len(missing_labels)}; first={missing_labels[:5]}")
        if extra_labels:
            print(f"  labels without images: {len(extra_labels)}; first={extra_labels[:5]}")

    print(f"classes: {args.output / 'classes.txt'}")
    print(f"yaml: {args.yaml}")
    if not all_ok:
        raise SystemExit("CODrone conversion finished with consistency warnings")


if __name__ == "__main__":
    main()
