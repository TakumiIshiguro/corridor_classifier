import csv
import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class CorridorSample:
    image_path: str
    class_index: int
    session_name: str


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
            samples.append(
                CorridorSample(
                    image_path=image_path,
                    class_index=class_index,
                    session_name=os.path.basename(session_dir),
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
) -> List[CorridorSample]:
    sessions = session_directories(dataset_dir)
    if not sessions:
        raise ValueError(
            f"no dataset sessions found under {os.path.abspath(dataset_dir)}"
        )
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
