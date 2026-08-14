import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from corridor_classifier.models import depth_to_tensor


@dataclass(frozen=True)
class CorridorSample:
    image_path: str
    class_index: int
    session_name: str
    stamp: float = 0.0
    depth_path: Optional[str] = None


def session_directories(dataset_dir: str) -> List[str]:
    dataset_dir = os.path.abspath(dataset_dir)
    if not os.path.isdir(dataset_dir):
        return []
    return [
        os.path.join(dataset_dir, name)
        for name in sorted(os.listdir(dataset_dir))
        if os.path.isfile(os.path.join(dataset_dir, name, "samples.csv"))
    ]


def load_session_samples(
    session_dir: str,
    num_classes: int,
) -> List[CorridorSample]:
    manifest_path = os.path.join(session_dir, "samples.csv")
    samples = []
    with open(manifest_path, "r", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "class_index"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"manifest is missing required columns {sorted(required)}: "
                f"{manifest_path}"
            )
        for row_number, row in enumerate(reader, start=2):
            class_index = int(row["class_index"])
            if class_index < 0 or class_index >= int(num_classes):
                raise ValueError(
                    f"invalid class_index at {manifest_path}:{row_number}"
                )
            image_path = os.path.join(session_dir, row["filename"])
            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"dataset image was not found: {image_path}"
                )
            depth_filename = str(row.get("depth_filename", "") or "").strip()
            depth_path = (
                os.path.join(session_dir, depth_filename)
                if depth_filename
                else None
            )
            if depth_path is not None and not os.path.isfile(depth_path):
                raise FileNotFoundError(
                    f"dataset depth map was not found: {depth_path}"
                )
            samples.append(
                CorridorSample(
                    image_path=image_path,
                    class_index=class_index,
                    session_name=os.path.basename(session_dir),
                    stamp=float(row.get("stamp", 0.0) or 0.0),
                    depth_path=depth_path,
                )
            )
    if not samples:
        raise ValueError(f"dataset session contains no samples: {session_dir}")
    return samples


def _load_sessions(
    directories: Sequence[str],
    num_classes: int,
) -> List[CorridorSample]:
    samples = []
    for directory in directories:
        samples.extend(load_session_samples(directory, num_classes))
    return samples


def load_dataset_samples(
    dataset_dir: str,
    num_classes: int,
    session_names: Optional[Sequence[str]] = None,
) -> List[CorridorSample]:
    sessions = session_directories(dataset_dir)
    if not sessions:
        raise ValueError(
            f"no dataset sessions found under {os.path.abspath(dataset_dir)}"
        )
    if session_names:
        requested = [str(name) for name in session_names]
        by_name = {os.path.basename(path): path for path in sessions}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(
                "dataset session(s) not found under "
                f"{os.path.abspath(dataset_dir)}: {', '.join(missing)}"
            )
        sessions = [by_name[name] for name in requested]
    return _load_sessions(sessions, num_classes)


def class_counts(samples: Sequence[CorridorSample], num_classes: int) -> List[int]:
    counts = Counter(sample.class_index for sample in samples)
    return [int(counts.get(index, 0)) for index in range(int(num_classes))]


class CorridorDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[CorridorSample],
        input_size: Sequence[int],
    ):
        self.samples = list(samples)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            expected_size = (self.input_size[1], self.input_size[0])
            if image.size != expected_size:
                raise ValueError(
                    f"dataset image must be {expected_size}, got {image.size}: "
                    f"{sample.image_path}"
                )
            tensor = self.transform(image)
        return tensor, sample.class_index


class CorridorMultiInputDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[CorridorSample],
        input_size: Sequence[int],
        variant_config: dict,
    ):
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.sequence_length = int(variant_config["sequence_length"])
        self.maximum_gap_seconds = float(
            variant_config.get("maximum_gap_seconds", 0.4)
        )
        self.use_depth = bool(variant_config["use_depth"])
        self.depth_min_m = float(variant_config.get("depth_min_m", 0.1))
        self.depth_max_m = float(variant_config.get("depth_max_m", 10.0))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        sessions = defaultdict(list)
        for sample in samples:
            sessions[sample.session_name].append(sample)
        self.sequences = []
        for session_samples in sessions.values():
            session_samples.sort(key=lambda sample: sample.stamp)
            for end in range(self.sequence_length - 1, len(session_samples)):
                sequence = session_samples[
                    end - self.sequence_length + 1 : end + 1
                ]
                if any(
                    later.stamp - earlier.stamp > self.maximum_gap_seconds
                    or later.stamp < earlier.stamp
                    for earlier, later in zip(sequence, sequence[1:])
                ):
                    continue
                if self.use_depth and any(
                    sample.depth_path is None for sample in sequence
                ):
                    continue
                self.sequences.append(tuple(sequence))
        if not self.sequences:
            requirement = " with depth maps" if self.use_depth else ""
            raise ValueError(
                "dataset contains no valid sequences"
                f" of length {self.sequence_length}{requirement}"
            )

    def __len__(self):
        return len(self.sequences)

    def _load_rgb(self, sample: CorridorSample):
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            expected = (self.input_size[1], self.input_size[0])
            if image.size != expected:
                raise ValueError(
                    f"dataset image must be {expected}, got {image.size}: "
                    f"{sample.image_path}"
                )
            return self.transform(image)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        inputs = {
            "rgb": torch.stack(
                [self._load_rgb(sample) for sample in sequence]
            )
        }
        if self.use_depth:
            inputs["depth"] = torch.stack(
                [
                    depth_to_tensor(
                        np.load(sample.depth_path, allow_pickle=False),
                        self.depth_min_m,
                        self.depth_max_m,
                    )
                    for sample in sequence
                ]
            )
        return inputs, sequence[-1].class_index
