import csv
import os
from collections import Counter
from typing import Optional, Sequence

import yaml
from PIL import Image

from corridor_classifier.turn_detection import (
    bridge_short_turning_gaps,
    replace_post_turn_label_islands,
)


def class_index_from_one_hot(
    values: Sequence[int],
    num_classes: int,
) -> Optional[int]:
    if len(values) != int(num_classes):
        return None
    normalized = [int(value) for value in values]
    if any(value not in (0, 1) for value in normalized):
        return None
    if sum(normalized) != 1:
        return None
    return normalized.index(1)


def resize_for_dataset(image: Image.Image, input_size: Sequence[int]) -> Image.Image:
    height, width = int(input_size[0]), int(input_size[1])
    return image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)


class DatasetSessionWriter:
    def __init__(
        self,
        session_dir: str,
        class_names: Sequence[str],
        input_size: Sequence[int],
        image_format: str,
        jpeg_quality: int,
        metadata: dict,
    ):
        self.session_dir = os.path.abspath(session_dir)
        self.images_dir = os.path.join(self.session_dir, "images")
        if os.path.exists(self.session_dir) and os.listdir(self.session_dir):
            raise FileExistsError(
                f"dataset session already exists and is not empty: {self.session_dir}"
            )
        os.makedirs(self.images_dir, exist_ok=True)

        self.class_names = tuple(str(name) for name in class_names)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.image_format = str(image_format).lower()
        self.jpeg_quality = int(jpeg_quality)
        self.sample_count = 0
        self.class_counts = Counter()
        self._closed = False

        self._metadata = dict(metadata)
        self._metadata.update(
            {
                "class_names": list(self.class_names),
                "input_size": list(self.input_size),
                "image_format": self.image_format,
            }
        )
        self._manifest = open(
            os.path.join(self.session_dir, "samples.csv"),
            "w",
            newline="",
        )
        self._csv_writer = csv.writer(self._manifest)
        self._csv_writer.writerow(
            ("filename", "stamp", "class_index", "class_name")
        )
        self._manifest.flush()
        self._write_metadata()

    def _write_metadata(self) -> None:
        metadata = dict(self._metadata)
        metadata["sample_count"] = int(self.sample_count)
        metadata["class_counts"] = {
            name: int(self.class_counts.get(index, 0))
            for index, name in enumerate(self.class_names)
        }
        with open(os.path.join(self.session_dir, "metadata.yaml"), "w") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)

    def save(self, image: Image.Image, class_index: int, stamp: float) -> str:
        if self._closed:
            raise RuntimeError("dataset writer is already closed")
        if class_index < 0 or class_index >= len(self.class_names):
            raise ValueError(f"invalid class index: {class_index}")

        extension = "jpg" if self.image_format == "jpeg" else self.image_format
        filename = f"{self.sample_count:06d}.{extension}"
        relative_path = os.path.join("images", filename)
        output_path = os.path.join(self.session_dir, relative_path)
        resized = resize_for_dataset(image, self.input_size)
        save_kwargs = {}
        if extension == "jpg":
            save_kwargs["quality"] = self.jpeg_quality
        resized.save(output_path, **save_kwargs)

        self._csv_writer.writerow(
            (
                relative_path,
                f"{float(stamp):.9f}",
                int(class_index),
                self.class_names[class_index],
            )
        )
        self._manifest.flush()
        self.class_counts[class_index] += 1
        self.sample_count += 1
        return output_path

    def close(self) -> None:
        if self._closed:
            return
        self._write_metadata()
        self._manifest.close()
        self._closed = True


def replace_post_turn_labels_in_session(
    session_dir: str,
    class_names: Sequence[str],
    turn_class_index: int,
    maximum_seconds: float,
) -> int:
    """Refine a completed manifest and update its class-count metadata."""
    session_dir = os.path.abspath(session_dir)
    manifest_path = os.path.join(session_dir, "samples.csv")
    with open(manifest_path, "r", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"stamp", "class_index", "class_name"}
    if not required.issubset(fieldnames):
        raise ValueError(
            f"manifest is missing required columns {sorted(required)}: "
            f"{manifest_path}"
        )

    original = [int(row["class_index"]) for row in rows]
    refined = replace_post_turn_label_islands(
        stamps=[float(row["stamp"]) for row in rows],
        labels=original,
        turn_class_index=turn_class_index,
        maximum_seconds=maximum_seconds,
    )
    changed = sum(before != after for before, after in zip(original, refined))
    counts = Counter()
    for row, class_index in zip(rows, refined):
        row["class_index"] = str(class_index)
        row["class_name"] = str(class_names[class_index])
        counts[class_index] += 1

    temporary = manifest_path + ".tmp"
    with open(temporary, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest_path)

    metadata_path = os.path.join(session_dir, "metadata.yaml")
    with open(metadata_path, "r") as stream:
        metadata = yaml.safe_load(stream) or {}
    metadata["class_counts"] = {
        str(name): int(counts.get(index, 0))
        for index, name in enumerate(class_names)
    }
    metadata["post_turn_next_label"] = {
        "maximum_seconds": float(maximum_seconds),
        "changed_samples": int(changed),
    }
    with open(metadata_path, "w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return int(changed)


def bridge_turning_gaps_in_session(
    session_dir: str,
    class_names: Sequence[str],
    turn_class_index: int,
    maximum_seconds: float,
) -> int:
    """Bridge short turning gaps and update manifest metadata counts."""
    session_dir = os.path.abspath(session_dir)
    manifest_path = os.path.join(session_dir, "samples.csv")
    with open(manifest_path, "r", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"stamp", "class_index", "class_name"}
    if not required.issubset(fieldnames):
        raise ValueError(
            f"manifest is missing required columns {sorted(required)}: "
            f"{manifest_path}"
        )

    original = [int(row["class_index"]) for row in rows]
    refined = bridge_short_turning_gaps(
        stamps=[float(row["stamp"]) for row in rows],
        labels=original,
        turn_class_index=turn_class_index,
        maximum_seconds=maximum_seconds,
    )
    changed = sum(before != after for before, after in zip(original, refined))
    counts = Counter()
    for row, class_index in zip(rows, refined):
        row["class_index"] = str(class_index)
        row["class_name"] = str(class_names[class_index])
        counts[class_index] += 1

    temporary = manifest_path + ".tmp"
    with open(temporary, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest_path)

    metadata_path = os.path.join(session_dir, "metadata.yaml")
    with open(metadata_path, "r") as stream:
        metadata = yaml.safe_load(stream) or {}
    metadata["class_counts"] = {
        str(name): int(counts.get(index, 0))
        for index, name in enumerate(class_names)
    }
    metadata["turning_gap_bridge"] = {
        "maximum_seconds": float(maximum_seconds),
        "changed_samples": int(changed),
    }
    with open(metadata_path, "w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return int(changed)
