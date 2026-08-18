import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

from corridor_classifier.models import depth_to_tensor
from corridor_classifier.passage_directions import passage_target_from_index


@dataclass(frozen=True)
class CorridorSample:
    image_path: str
    class_index: int
    session_name: str
    stamp: float = 0.0
    depth_path: Optional[str] = None
    bev_path: Optional[str] = None


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
            bev_filename = str(row.get("bev_filename", "") or "").strip()
            bev_path = (
                os.path.join(session_dir, bev_filename) if bev_filename else None
            )
            if bev_path is not None and not os.path.isfile(bev_path):
                raise FileNotFoundError(
                    f"dataset BEV scan was not found: {bev_path}"
                )
            samples.append(
                CorridorSample(
                    image_path=image_path,
                    class_index=class_index,
                    session_name=os.path.basename(session_dir),
                    stamp=float(row.get("stamp", 0.0) or 0.0),
                    depth_path=depth_path,
                    bev_path=bev_path,
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
        augmentation_config: Optional[dict] = None,
        sequence_step: int = 1,
    ):
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.sequence_length = int(variant_config["sequence_length"])
        self.frame_stride = int(variant_config.get("frame_stride", 1))
        self.sequence_step = int(sequence_step)
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.sequence_step <= 0:
            raise ValueError("sequence_step must be positive")
        self.required_span = (
            (self.sequence_length - 1) * self.frame_stride + 1
        )
        self.maximum_gap_seconds = float(
            variant_config.get("maximum_gap_seconds", 0.4)
        )
        self.use_depth = bool(variant_config["use_depth"])
        self.output_mode = str(variant_config.get("output_mode", "class"))
        self.class_names = tuple(variant_config.get("class_names", ()))
        self.turning_class_name = str(
            variant_config.get("turning_class_name", "turning")
        )
        if self.output_mode not in ("class", "passage_directions"):
            raise ValueError(f"unsupported output_mode: {self.output_mode}")
        if self.output_mode == "passage_directions" and not self.class_names:
            raise ValueError(
                "passage_directions output requires configured class_names"
            )
        self._excluded_turning_index = None
        if self.output_mode == "passage_directions":
            if self.turning_class_name not in self.class_names:
                raise ValueError(
                    "turning_class_name must exist in class_names: "
                    f"{self.turning_class_name}"
                )
            self._excluded_turning_index = self.class_names.index(
                self.turning_class_name
            )
        self.depth_min_m = float(variant_config.get("depth_min_m", 0.1))
        self.depth_max_m = float(variant_config.get("depth_max_m", 10.0))
        augmentation = dict(augmentation_config or {})
        self.sequence_consistent_augmentation = bool(
            augmentation.get("sequence_consistent", False)
        )
        self.horizontal_flip_probability = float(
            augmentation.get("horizontal_flip_probability", 0.0)
        )
        self.rgb_dropout_probability = float(
            augmentation.get("rgb_dropout_probability", 0.0)
        )
        if not 0.0 <= self.rgb_dropout_probability <= 1.0:
            raise ValueError("rgb_dropout_probability must be in [0, 1]")
        self.depth_scale_jitter = float(
            augmentation.get("depth_scale_jitter", 0.0)
        )
        color_jitter = augmentation.get("color_jitter", {})
        self.color_jitter = transforms.ColorJitter(
            brightness=float(color_jitter.get("brightness", 0.0)),
            contrast=float(color_jitter.get("contrast", 0.0)),
            saturation=float(color_jitter.get("saturation", 0.0)),
            hue=float(color_jitter.get("hue", 0.0)),
        )
        self.grayscale_probability = float(
            augmentation.get("grayscale_probability", 0.0)
        )
        self.blur_probability = float(
            augmentation.get("blur_probability", 0.0)
        )
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
            for end in range(
                self.required_span - 1,
                len(session_samples),
                self.sequence_step,
            ):
                window = session_samples[end - self.required_span + 1 : end + 1]
                if any(
                    later.stamp - earlier.stamp > self.maximum_gap_seconds
                    or later.stamp < earlier.stamp
                    for earlier, later in zip(window, window[1:])
                ):
                    continue
                sequence = window[:: self.frame_stride]
                if self.use_depth and any(
                    sample.depth_path is None for sample in sequence
                ):
                    continue
                if sequence[-1].class_index == self._excluded_turning_index:
                    continue
                self.sequences.append(tuple(sequence))
        if not self.sequences:
            requirement = " with depth maps" if self.use_depth else ""
            raise ValueError(
                "dataset contains no valid sequences"
                f" of length {self.sequence_length}, frame stride "
                f"{self.frame_stride}{requirement}"
            )

    def __len__(self):
        return len(self.sequences)

    def _sample_rgb_augmentation(self):
        fn_indices, brightness, contrast, saturation, hue = (
            transforms.ColorJitter.get_params(
                self.color_jitter.brightness,
                self.color_jitter.contrast,
                self.color_jitter.saturation,
                self.color_jitter.hue,
            )
        )
        return {
            "fn_indices": tuple(int(index) for index in fn_indices),
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "grayscale": random.random() < self.grayscale_probability,
            "blur_sigma": (
                random.uniform(0.1, 2.0)
                if random.random() < self.blur_probability
                else None
            ),
        }

    @staticmethod
    def _augment_rgb(image: Image.Image, parameters: dict) -> Image.Image:
        operations = (
            (transform_functional.adjust_brightness, parameters["brightness"]),
            (transform_functional.adjust_contrast, parameters["contrast"]),
            (transform_functional.adjust_saturation, parameters["saturation"]),
            (transform_functional.adjust_hue, parameters["hue"]),
        )
        for index in parameters["fn_indices"]:
            operation, value = operations[index]
            if value is not None:
                image = operation(image, value)
        if parameters["grayscale"]:
            image = transform_functional.rgb_to_grayscale(
                image,
                num_output_channels=3,
            )
        if parameters["blur_sigma"] is not None:
            sigma = float(parameters["blur_sigma"])
            image = transform_functional.gaussian_blur(
                image,
                kernel_size=[3, 3],
                sigma=[sigma, sigma],
            )
        return image

    def _load_rgb(self, sample: CorridorSample, augmentation_parameters=None):
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            expected = (self.input_size[1], self.input_size[0])
            if image.size != expected:
                raise ValueError(
                    f"dataset image must be {expected}, got {image.size}: "
                    f"{sample.image_path}"
                )
            parameters = (
                self._sample_rgb_augmentation()
                if augmentation_parameters is None
                else augmentation_parameters
            )
            return self.transform(self._augment_rgb(image, parameters))

    def __getitem__(self, index):
        sequence = self.sequences[index]
        flip = random.random() < self.horizontal_flip_probability
        rgb_augmentation = (
            self._sample_rgb_augmentation()
            if self.sequence_consistent_augmentation
            else None
        )
        inputs = {
            "rgb": torch.stack(
                [
                    self._load_rgb(sample, rgb_augmentation)
                    for sample in sequence
                ]
            )
        }
        if (
            self.rgb_dropout_probability > 0.0
            and random.random() < self.rgb_dropout_probability
        ):
            inputs["rgb"].zero_()
        if self.use_depth:
            sequence_depth_scale = (
                random.uniform(
                    1.0 - self.depth_scale_jitter,
                    1.0 + self.depth_scale_jitter,
                )
                if self.sequence_consistent_augmentation
                and self.depth_scale_jitter > 0.0
                else None
            )
            inputs["depth"] = torch.stack(
                [
                    depth_to_tensor(
                        np.load(sample.depth_path, allow_pickle=False)
                        * (
                            sequence_depth_scale
                            if sequence_depth_scale is not None
                            else random.uniform(
                                1.0 - self.depth_scale_jitter,
                                1.0 + self.depth_scale_jitter,
                            )
                            if self.depth_scale_jitter > 0.0
                            else 1.0
                        ),
                        self.depth_min_m,
                        self.depth_max_m,
                    )
                    for sample in sequence
                ]
            )
        label = sequence[-1].class_index
        if flip:
            inputs = {
                key: torch.flip(value, dims=(-1,))
                for key, value in inputs.items()
            }
            label = {2: 3, 3: 2, 5: 7, 7: 5}.get(label, label)
        if self.output_mode == "passage_directions":
            return inputs, passage_target_from_index(label, self.class_names)
        return inputs, label
